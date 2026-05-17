import random
from dataclasses import dataclass

import torch
from datasets import load_dataset
from dotenv import load_dotenv
from tokenizers import Tokenizer

from training.config import Config

load_dotenv()

PROMPT_WITH_CONTEXT = (
    "### Instruction:\n{instruction}\n\n### Context:\n{context}\n\n### Response:\n"
)
PROMPT_NO_CONTEXT = "### Instruction:\n{instruction}\n\n### Response:\n"

IGNORE_INDEX = -100


def render_prompt(example: dict) -> tuple[str, str]:
    """Return (prompt_text, response_text). Loss is masked on prompt_text."""
    ctx = (example.get("context") or "").strip()
    instr = example["instruction"].strip()
    if ctx:
        prompt = PROMPT_WITH_CONTEXT.format(instruction=instr, context=ctx)
    else:
        prompt = PROMPT_NO_CONTEXT.format(instruction=instr)
    response = example["response"].strip()
    return prompt, response


@dataclass
class SFTExample:
    input_ids: list[int]  # length == block_size, right-padded with eos
    labels: list[int]  # length == block_size, IGNORE_INDEX on prompt + padding


def encode_example(
    example: dict,
    tokenizer: Tokenizer,
    block_size: int,
    eos_id: int,
) -> SFTExample | None:
    """Tokenize one example. Returns None if the prompt alone overflows block_size."""
    prompt_text, response_text = render_prompt(example)

    prompt_ids = tokenizer.encode(prompt_text).ids
    response_ids = tokenizer.encode(response_text).ids + [eos_id]

    # If the prompt already fills the window, there's no room to learn anything.
    if len(prompt_ids) >= block_size:
        return None

    input_ids = prompt_ids + response_ids
    labels = [IGNORE_INDEX] * len(prompt_ids) + response_ids

    # Truncate response if needed.
    input_ids = input_ids[:block_size]
    labels = labels[:block_size]

    # Right-pad with eos for inputs, IGNORE_INDEX for labels.
    pad_len = block_size - len(input_ids)
    input_ids = input_ids + [eos_id] * pad_len
    labels = labels + [IGNORE_INDEX] * pad_len

    return SFTExample(input_ids=input_ids, labels=labels)


def build_sft_dataset(
    config: Config,
    val_fraction: float = 0.02,
    seed: int = 1337,
) -> tuple[list[SFTExample], list[SFTExample], Tokenizer]:
    tokenizer = Tokenizer.from_file(config.tokenizer_path)
    eos_id = tokenizer.token_to_id("<|endoftext|>")
    assert eos_id is not None, "Tokenizer must have <|endoftext|>"

    raw = load_dataset("TeeZee/dolly-15k-pirate-speech", split="train")

    examples: list[SFTExample] = []
    skipped = 0
    for row in raw:
        ex = encode_example(row, tokenizer, config.block_size, eos_id)
        if ex is None:
            skipped += 1
            continue
        examples.append(ex)

    rng = random.Random(seed)
    rng.shuffle(examples)
    n_val = max(1, int(len(examples) * val_fraction))
    val = examples[:n_val]
    train = examples[n_val:]

    print(
        f"SFT dataset: {len(train)} train / {len(val)} val "
        f"(skipped {skipped} prompts longer than block_size={config.block_size})"
    )
    return train, val, tokenizer


def get_sft_batch(
    split_examples: list[SFTExample],
    config: Config,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a batch. Returns (x, y) shaped (B, T-1) like pretraining get_batch."""
    idxs = torch.randint(0, len(split_examples), (config.batch_size,))
    rows = [split_examples[i] for i in idxs.tolist()]

    input_ids = torch.tensor([r.input_ids for r in rows], dtype=torch.long)
    labels = torch.tensor([r.labels for r in rows], dtype=torch.long)

    # Causal shift: predict labels[t+1] from input_ids[:t+1].
    x = input_ids[:, :-1].contiguous()
    y = labels[:, 1:].contiguous()

    if config.device == "cuda":
        x = x.pin_memory().to(config.device, non_blocking=True)
        y = y.pin_memory().to(config.device, non_blocking=True)
    else:
        x = x.to(config.device)
        y = y.to(config.device)
    return x, y
