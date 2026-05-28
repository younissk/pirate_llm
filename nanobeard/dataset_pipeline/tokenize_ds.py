"""Train BPE tokenizer from piratized corpus, save to data/<version>/pirate_bpe.json.

Run:
  uv run python -m nanobeard.dataset_pipeline.tokenize_ds --data-dir data/sloop
"""

import argparse
from pathlib import Path

from datasets import load_from_disk
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, help="Data directory, e.g. data/sloop")
    parser.add_argument("--vocab-size", type=int, default=8192)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    pirate_ds = load_from_disk(str(data_dir / "tiny_stories_pirate"))

    tokenizer = Tokenizer(BPE(unk_token=None))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=args.vocab_size,
        special_tokens=["<|endoftext|>"],
        initial_alphabet=ByteLevel.alphabet(),
        show_progress=True,
    )

    def batch_iter(ds, batch_size=1000):
        for i in range(0, len(ds), batch_size):
            yield ds[i : i + batch_size]["text"]

    tokenizer.train_from_iterator(
        batch_iter(pirate_ds["train"]),
        trainer=trainer,
        length=len(pirate_ds["train"]),
    )

    out_path = data_dir / "pirate_bpe.json"
    tokenizer.save(str(out_path))
    print(f"Vocab size: {tokenizer.get_vocab_size()}")
    print(f"Saved tokenizer to {out_path}")


if __name__ == "__main__":
    main()
