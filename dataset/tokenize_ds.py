from datasets import load_from_disk
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

pirate_ds = load_from_disk("dataset/tiny_stories_pirate")

tokenizer = Tokenizer(BPE(unk_token=None))
tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
tokenizer.decoder = ByteLevelDecoder()

trainer = BpeTrainer(
    vocab_size=8192,  # 8k vocab — small, GPU-friendly
    special_tokens=["<|endoftext|>"],  # separator between stories
    initial_alphabet=ByteLevel.alphabet(),  # all 256 bytes available from the start
    show_progress=True,
)


def batch_iter(ds, batch_size=1000):
    for i in range(0, len(ds), batch_size):
        yield ds[i : i + batch_size]["text"]


tokenizer.train_from_iterator(
    batch_iter(pirate_ds["train"]),
    trainer=trainer,
    length=len(pirate_ds["train"]),  # lets it show a progress bar
)

tokenizer.save("pirate_bpe.json")
print(f"Vocab size: {tokenizer.get_vocab_size()}")
print("Saved tokenizer to pirate_bpe.json")
