# training/sample.py
"""
Sampling / generation from a trained Tiny Pirate GPT checkpoint.

Pipeline (Lesson 5):
  prompt text → tokenizer → token IDs
  → repeatedly: forward → softmax(logits / T) → top-k filter → sample → append
  → token IDs → tokenizer → text

Run:
  uv run python -m training.sample --prompt "Once upon a time" --max-tokens 200
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from .config import Config
from .model import GPT


@torch.no_grad()
def generate(
    model: GPT,
    idx: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 0.8,
    top_k: int | None = 40,
) -> torch.Tensor:
    """
    Autoregressive sampling loop.

    Args:
        model: trained GPT in eval mode
        idx: starting token IDs, shape (B, T)
        max_new_tokens: how many tokens to generate
        temperature: <1.0 sharpens (greedy-ish), >1.0 flattens
        top_k: keep only the k most likely tokens at each step (None = disabled)

    Returns:
        idx extended with max_new_tokens new tokens, shape (B, T + max_new_tokens)
    """
    block_size = model.config.block_size

    for _ in range(max_new_tokens):
        # Crop the context to the last block_size tokens. The model has
        # nothing to say about anything older than its context window.
        idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]

        # Forward pass — logits shape (B, T, vocab_size).
        logits, _ = model(idx_cond)

        # We only care about the prediction for the next token, i.e. the
        # logits at the last position. Shape: (B, vocab_size).
        logits = logits[:, -1, :] / temperature

        # Top-k filtering: keep only the k highest-probability tokens,
        # set the rest to -inf so they get zero mass after softmax.
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("inf")

        # Softmax to probabilities, sample one token per batch row.
        probs = F.softmax(logits, dim=-1)
        next_idx = torch.multinomial(probs, num_samples=1)  # shape (B, 1)

        # Append and continue.
        idx = torch.cat((idx, next_idx), dim=1)

    return idx


def load_checkpoint(ckpt_path: str, device: str) -> GPT:
    """Load a saved checkpoint into a fresh GPT model."""
    print(f"Loading checkpoint from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    config: Config = checkpoint["config"]
    config.device = device  # override in case we're loading on a different device

    model = GPT(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    print(
        f"Model: {model.num_parameters() / 1e6:.2f}M params, "
        f"trained for {checkpoint['iter_num']} iters, "
        f"val loss {checkpoint['val_loss']:.4f}"
    )
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="out/ckpt.pt", help="path to checkpoint")
    parser.add_argument(
        "--tokenizer", default="pirate_bpe.json", help="path to BPE tokenizer"
    )
    parser.add_argument("--prompt", default="Once upon a time", help="prompt text")
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument(
        "--num-samples", type=int, default=3, help="how many completions"
    )
    parser.add_argument(
        "--device", default=None, help="cuda / mps / cpu (auto-detect if None)"
    )
    args = parser.parse_args()

    # Auto-detect device.
    if args.device is None:
        if torch.cuda.is_available():
            args.device = "cuda"
        elif torch.backends.mps.is_available():
            args.device = "mps"
        else:
            args.device = "cpu"
    print(f"Device: {args.device}")

    # Load tokenizer + model.
    tokenizer = Tokenizer.from_file(args.tokenizer)
    model = load_checkpoint(args.ckpt, args.device)

    # Encode prompt.
    prompt_ids = tokenizer.encode(args.prompt).ids
    idx = torch.tensor(prompt_ids, dtype=torch.long, device=args.device).unsqueeze(0)
    print(f"\nPrompt: {args.prompt!r} ({len(prompt_ids)} tokens)")
    print(f"Sampling: temperature={args.temperature}, top_k={args.top_k}")
    print("=" * 60)

    # Generate num_samples completions.
    for i in range(args.num_samples):
        out_idx = generate(
            model,
            idx,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )
        out_text = tokenizer.decode(out_idx[0].cpu().tolist())
        print(f"\n--- Sample {i + 1} ---")
        print(out_text)
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
