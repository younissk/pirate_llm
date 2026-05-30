"""Source registry — the reusable building blocks datasets are composed from.

A *source* is one piratized corpus (a `DatasetDict` with `text` columns), built
once and cached under `data/sources/<name>/`. Datasets (see `build.py`) combine
one or more sources into a tokenizer + train.bin/val.bin.

Add a new source by writing a builder fn and registering it in REGISTRY.

Run (build/cache a single source on its own):
  uv run python -m nanobeard.dataset_pipeline.sources --source tiny_stories_pirate
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import TypedDict

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk

SOURCES_DIR = Path("data/sources")

# Subsample size for the cosmopedia 'stories' subset (~5M rows total).
# materialize() caches by name only — bump this then rebuild with `--force`.
COSMOPEDIA_STORIES_ROWS = 1_500_000

# Public-domain pirate-themed books pulled from Project Gutenberg (name -> ebook id).
GUTENBERG_BOOKS = {
    "pyle_book_of_pirates": 973,
    "treasure_island": 120,
    "general_history_of_pyrates": 40580,
    "peter_pan": 16,
    "coral_island": 646,
}


class SourceSpec(TypedDict):
    builder: Callable[[], DatasetDict]
    origin: str


def build_tiny_stories_pirate() -> DatasetDict:
    """roneneldan/TinyStories, run through the `arrr` piratizer."""
    from nanobeard.dataset_pipeline.piratize import piratize

    raw = load_dataset("roneneldan/TinyStories")
    return DatasetDict({split: piratize(ds, split) for split, ds in raw.items()})


def build_cosmopedia_wikihow(val_rows: int = 5000) -> DatasetDict:
    """HuggingFaceTB/cosmopedia 'wikihow' subset, piratized.

    Only ships a train split (179k synthetic how-to articles), so we carve a
    small validation set for parity with the other sources.
    """
    from nanobeard.dataset_pipeline.piratize import piratize

    raw = load_dataset("HuggingFaceTB/cosmopedia", "wikihow", split="train")
    raw = raw.select_columns(["text"])  # drop prompt/seed_data/etc; keep schema lean
    split = raw.train_test_split(test_size=val_rows, seed=1337)
    return DatasetDict(
        {
            "train": piratize(split["train"], "train"),
            "validation": piratize(split["test"], "validation"),
        }
    )


_STORIES_TOTAL_SHARDS = 43  # HuggingFaceTB/cosmopedia 'stories' parquet shards
_STORIES_ROWS_PER_SHARD = 116_000  # ~5M / 43


def _stories_shard_files(n_rows: int) -> list[str]:
    """Pick just enough parquet shards to cover n_rows, spread evenly across the
    corpus (the subset is topic-ordered, so adjacent shards share themes)."""
    k = min(_STORIES_TOTAL_SHARDS, math.ceil(n_rows / _STORIES_ROWS_PER_SHARD) + 2)
    if k >= _STORIES_TOTAL_SHARDS:
        ids = list(range(_STORIES_TOTAL_SHARDS))
    elif k <= 1:
        ids = [0]
    else:
        ids = sorted({round(i * (_STORIES_TOTAL_SHARDS - 1) / (k - 1)) for i in range(k)})
    return [f"data/stories/train-{i:05d}-of-{_STORIES_TOTAL_SHARDS:05d}.parquet" for i in ids]


def build_cosmopedia_stories(
    n_rows: int = COSMOPEDIA_STORIES_ROWS, val_rows: int = 2000, seed: int = 1337
) -> DatasetDict:
    """A subsample of HuggingFaceTB/cosmopedia 'stories' (~5M rows), piratized.

    Downloads only the shards needed to cover n_rows (resumable via the HF
    cache — far more robust than row-streaming, which dies on any network blip),
    then shuffles + selects locally. Arrow is memory-mapped, so RAM stays low.
    """
    from nanobeard.dataset_pipeline.piratize import piratize

    full = load_dataset(
        "HuggingFaceTB/cosmopedia", data_files=_stories_shard_files(n_rows), split="train"
    )
    full = full.select_columns(["text"])
    sample = full.shuffle(seed=seed).select(range(min(n_rows, len(full))))
    split = sample.train_test_split(test_size=val_rows, seed=seed)
    return DatasetDict(
        {"train": piratize(split["train"], "train"), "validation": piratize(split["test"], "validation")}
    )


_PG_START = re.compile(r"\*\*\*\s*START OF TH(?:E|IS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.I | re.S)
_PG_END = re.compile(r"\*\*\*\s*END OF TH(?:E|IS) PROJECT GUTENBERG EBOOK", re.I)


def _fetch_gutenberg(book_id: int) -> str:
    """Download a book's canonical UTF-8 plain text (PG 403s without a User-Agent)."""
    url = f"https://www.gutenberg.org/ebooks/{book_id}.txt.utf-8"
    req = urllib.request.Request(url, headers={"User-Agent": "nanoBeard/0.1"})
    with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310 (trusted host)
        return resp.read().decode("utf-8", errors="replace")


def _strip_pg(text: str) -> str:
    """Drop the Project Gutenberg license header/footer around the actual book."""
    start = m.end() if (m := _PG_START.search(text)) else 0
    end = m.start() if (m := _PG_END.search(text)) else len(text)
    return text[start:end]


def _paragraphs(text: str, min_chars: int = 200) -> list[str]:
    """Split into paragraphs on blank lines, unwrap hard-wrapped lines, drop short bits."""
    out = []
    for para in re.split(r"\n\s*\n", text):
        para = re.sub(r"\s+", " ", para).strip()
        if len(para) >= min_chars:
            out.append(para)
    return out


def build_gutenberg_books(val_rows: int = 500, min_chars: int = 200) -> DatasetDict:
    """5 public-domain pirate books from Project Gutenberg, paragraph-chunked.

    NOT piratized — kept as authentic source prose (unlike the cosmopedia/
    TinyStories sources). Each paragraph becomes one `text` row.
    """
    texts: list[str] = []
    for book_id in GUTENBERG_BOOKS.values():
        texts.extend(_paragraphs(_strip_pg(_fetch_gutenberg(book_id)), min_chars))

    ds = Dataset.from_dict({"text": texts}).train_test_split(test_size=val_rows, seed=1337)
    return DatasetDict({"train": ds["train"], "validation": ds["test"]})


REGISTRY: dict[str, SourceSpec] = {
    "tiny_stories_pirate": {
        "builder": build_tiny_stories_pirate,
        "origin": "roneneldan/TinyStories piratized via arrr",
    },
    "cosmopedia_wikihow": {
        "builder": build_cosmopedia_wikihow,
        "origin": "HuggingFaceTB/cosmopedia (wikihow subset) piratized via arrr",
    },
    "cosmopedia_stories": {
        "builder": build_cosmopedia_stories,
        "origin": f"HuggingFaceTB/cosmopedia (stories subset, {COSMOPEDIA_STORIES_ROWS} subsample) piratized via arrr",
    },
    "gutenberg_books": {
        "builder": build_gutenberg_books,
        "origin": "Project Gutenberg, NOT piratized: "
        + ", ".join(f"{name} (#{bid})" for name, bid in GUTENBERG_BOOKS.items()),
    },
}


def source_dir(name: str) -> Path:
    return SOURCES_DIR / name


def is_cached(name: str) -> bool:
    return (source_dir(name) / "dataset_dict.json").exists()


def materialize(name: str, force: bool = False) -> DatasetDict:
    """Return a source's DatasetDict, building + caching it on first use."""
    if name not in REGISTRY:
        raise KeyError(f"Unknown source {name!r}. Known: {sorted(REGISTRY)}")

    out = source_dir(name)
    if is_cached(name) and not force:
        return load_from_disk(str(out))

    spec = REGISTRY[name]
    ds = spec["builder"]()
    if out.exists():  # clear a stale cache so old shards can't linger
        shutil.rmtree(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(out))
    (out / "source.json").write_text(
        json.dumps(
            {
                "name": name,
                "origin": spec["origin"],
                "rows": {split: len(part) for split, part in ds.items()},
            },
            indent=2,
        )
    )
    return ds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, choices=sorted(REGISTRY))
    parser.add_argument("--force", action="store_true", help="Rebuild even if cached")
    args = parser.parse_args()

    ds = materialize(args.source, force=args.force)
    print(f"{args.source}: { {k: len(v) for k, v in ds.items()} } -> {source_dir(args.source)}")


if __name__ == "__main__":
    main()
