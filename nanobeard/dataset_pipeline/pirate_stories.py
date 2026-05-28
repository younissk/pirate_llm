"""Print a random piratized story.

Run:
  uv run python -m nanobeard.dataset_pipeline.pirate_stories --data-dir data/sloop
"""

import argparse
import random
from pathlib import Path

from datasets import load_from_disk


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, help="Data directory, e.g. data/sloop")
    args = parser.parse_args()

    pirate_ds = load_from_disk(str(Path(args.data_dir) / "tiny_stories_pirate"))
    i = random.randint(0, len(pirate_ds["train"]) - 1)
    print(pirate_ds["train"][i]["text"])


if __name__ == "__main__":
    main()
