"""Tokenize piratized corpus into train.bin / val.bin under data/<version>/.

Run:
  uv run python -m nanobeard.dataset_pipeline.tokenize_corpus --data-dir data/sloop
"""

import argparse
from pathlib import Path

import numpy as np
from datasets import load_from_disk
from tokenizers import Tokenizer
from tqdm import tqdm

BATCH = 1000


def encode_split(ds_split, tokenizer: Tokenizer, eot_id: int, out_path: Path) -> int:
    n_rows = len(ds_split)
    estimated = n_rows * 500
    arr = np.memmap(str(out_path), dtype=np.uint16, mode="w+", shape=(estimated,))
    idx = 0

    for start in tqdm(range(0, n_rows, BATCH), desc=f"Tokenizing → {out_path}"):
        texts = ds_split[start : start + BATCH]["text"]
        encs = tokenizer.encode_batch(texts)
        for e in encs:
            ids = e.ids
            if not ids:
                continue
            n = len(ids)
            arr[idx : idx + n] = ids
            arr[idx + n] = eot_id
            idx += n + 1

    arr.flush()
    del arr
    with open(out_path, "r+b") as f:
        f.truncate(idx * np.dtype(np.uint16).itemsize)
    return idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, help="Data directory, e.g. data/sloop")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    pirate_ds = load_from_disk(str(data_dir / "tiny_stories_pirate"))

    tokenizer = Tokenizer.from_file(str(data_dir / "pirate_bpe.json"))
    eot_id = tokenizer.token_to_id("<|endoftext|>")
    assert eot_id is not None, "Tokenizer must define <|endoftext|>"
    assert tokenizer.get_vocab_size() < 2**16, "vocab too large for uint16"
    print(f"Vocab size: {tokenizer.get_vocab_size()}")
    print(f"<|endoftext|> ID: {eot_id}")

    train_count = encode_split(pirate_ds["train"], tokenizer, eot_id, data_dir / "train.bin")
    val_count = encode_split(pirate_ds["validation"], tokenizer, eot_id, data_dir / "val.bin")
    print(f"Train tokens: {train_count:,}")
    print(f"Val tokens:   {val_count:,}")
    print(f"\nSaved {data_dir / 'train.bin'} and {data_dir / 'val.bin'}")

    loaded = np.memmap(str(data_dir / "train.bin"), dtype=np.uint16, mode="r")
    print(f"\nFirst 20 tokens of train.bin: {loaded[:20].tolist()}")
    print(f"Decoded: {tokenizer.decode(loaded[:20].tolist())}")


if __name__ == "__main__":
    main()
