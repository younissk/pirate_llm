import numpy as np
from datasets import load_from_disk
from tokenizers import Tokenizer
from tqdm import tqdm

BATCH = 1000

pirate_ds = load_from_disk("dataset/tiny_stories_pirate")

tokenizer = Tokenizer.from_file("pirate_bpe.json")
EOT_ID = tokenizer.token_to_id("<|endoftext|>")
assert tokenizer.get_vocab_size() < 2**16, "vocab too large for uint16"
print(f"Vocab size: {tokenizer.get_vocab_size()}")
print(f"<|endoftext|> ID: {EOT_ID}")


def encode_split(ds_split, out_path: str) -> int:
    n_rows = len(ds_split)
    # Over-estimate token count, then truncate file at the end.
    # ~300 tokens/story is generous for TinyStories.
    estimated = n_rows * 500
    arr = np.memmap(out_path, dtype=np.uint16, mode="w+", shape=(estimated,))
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
            arr[idx + n] = EOT_ID
            idx += n + 1

    arr.flush()
    del arr
    # Truncate the file to the actual number of tokens written.
    with open(out_path, "r+b") as f:
        f.truncate(idx * np.dtype(np.uint16).itemsize)
    return idx


train_count = encode_split(pirate_ds["train"], "train.bin")
val_count = encode_split(pirate_ds["validation"], "val.bin")
print(f"Train tokens: {train_count:,}")
print(f"Val tokens:   {val_count:,}")
print("\nSaved train.bin and val.bin")

loaded = np.memmap("train.bin", dtype=np.uint16, mode="r")
print(f"\nFirst 20 tokens of train.bin: {loaded[:20].tolist()}")
print(f"Decoded: {tokenizer.decode(loaded[:20].tolist())}")
