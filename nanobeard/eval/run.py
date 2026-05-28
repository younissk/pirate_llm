"""Entrypoint: run perplexity + gallery for a given config / ckpt.

uv run python -m nanobeard.eval.run --config configs/sloop.py
uv run python -m nanobeard.eval.run --config configs/sloop.py --prompts evals/prompts.jsonl
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from dataclasses import asdict
from pathlib import Path

import torch
from tokenizers import Tokenizer

from nanobeard.config import load_config
from nanobeard.eval.gallery import load_prompts, render_markdown, run_gallery
from nanobeard.eval.perplexity import compute_perplexity
from nanobeard.models.naming import display_name
from nanobeard.sample import load_checkpoint

DEFAULT_PROMPTS = Path("evals/prompts.jsonl")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config .py file")
    parser.add_argument("--ckpt", default=None, help="Override ckpt path")
    parser.add_argument(
        "--prompts", default=str(DEFAULT_PROMPTS), help="JSONL prompt file for gallery"
    )
    parser.add_argument(
        "--n-batches", type=int, default=200, help="Perplexity batches (set lower for quick check)"
    )
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Override output dir (default: evals/results/<date>/<model_name>/)",
    )
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    ckpt_path = args.ckpt or cfg.ckpt_path

    if args.device is None:
        if torch.cuda.is_available():
            args.device = "cuda"
        elif torch.backends.mps.is_available():
            args.device = "mps"
        else:
            args.device = "cpu"

    model = load_checkpoint(ckpt_path, args.device, tokenizer_path=cfg.tokenizer_path)

    print("Computing perplexity...")
    ppl = compute_perplexity(model, cfg, n_batches=args.n_batches)
    print(f"  loss       : {ppl.loss:.4f}")
    print(f"  perplexity : {ppl.perplexity:.2f}")
    print(f"  tokens     : {ppl.n_tokens:,}")

    print(f"\nRunning gallery from {args.prompts}...")
    prompts = load_prompts(args.prompts)
    tokenizer = Tokenizer.from_file(cfg.tokenizer_path)
    samples = run_gallery(
        model,
        tokenizer,
        prompts,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )

    today = dt.date.today().isoformat()
    out_dir = Path(args.out_dir) if args.out_dir else Path("evals/results") / today / cfg.model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    name = display_name(cfg, model)
    (out_dir / "gallery.md").write_text(render_markdown(name, samples))
    (out_dir / "metrics.json").write_text(
        json.dumps(
            {
                "display_name": name,
                "model_name": cfg.model_name,
                "ckpt": ckpt_path,
                "perplexity": asdict(ppl),
                "n_gallery_samples": len(samples),
                "temperature": args.temperature,
                "top_k": args.top_k,
            },
            indent=2,
        )
    )

    print(f"\nWrote {out_dir / 'gallery.md'}")
    print(f"Wrote {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
