"""Pull TinyStories, piratize, save under data/<version>/tiny_stories_pirate/.

Run:
  uv run python -m nanobeard.dataset_pipeline.tiny_stories --data-dir data/sloop
"""

import argparse
from pathlib import Path

from datasets import DatasetDict, load_dataset

from nanobeard.dataset_pipeline.piratize import piratize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, help="Output directory, e.g. data/sloop")
    args = parser.parse_args()

    out_dir = Path(args.data_dir) / "tiny_stories_pirate"
    out_dir.parent.mkdir(parents=True, exist_ok=True)

    tiny_stories_ds = load_dataset("roneneldan/TinyStories")
    pirate_ds = DatasetDict({k: piratize(v) for k, v in tiny_stories_ds.items()})
    pirate_ds.save_to_disk(str(out_dir))
    print(f"Saved piratized dataset to {out_dir}")


if __name__ == "__main__":
    main()
