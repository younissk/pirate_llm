"""Generate completions for a fixed prompt list — qualitative side-by-side."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from tokenizers import Tokenizer

from nanobeard.sample import generate


@dataclass
class GallerySample:
    prompt: str
    completion: str
    n_input_tokens: int
    n_output_tokens: int


def load_prompts(path: str | Path) -> list[dict]:
    """Load a JSONL file of prompt records. Each line: {"prompt": "...", ...}."""
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            out.append(json.loads(line))
    return out


def run_gallery(
    model: nn.Module,
    tokenizer: Tokenizer,
    prompts: list[dict],
    device: str = "cpu",
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int | None = 40,
) -> list[GallerySample]:
    results: list[GallerySample] = []
    for record in prompts:
        prompt = record["prompt"]
        prompt_ids = tokenizer.encode(prompt).ids
        idx = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
        out = generate(
            model, idx, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k
        )
        completion = tokenizer.decode(out[0].cpu().tolist())
        results.append(
            GallerySample(
                prompt=prompt,
                completion=completion,
                n_input_tokens=len(prompt_ids),
                n_output_tokens=out.size(1) - len(prompt_ids),
            )
        )
    return results


def render_markdown(model_name: str, samples: list[GallerySample]) -> str:
    """Render samples to a markdown doc — easy to diff between model versions."""
    lines = [f"# Gallery — {model_name}", ""]
    for i, s in enumerate(samples, 1):
        lines += [
            f"## {i}. `{s.prompt}`",
            "",
            "```",
            s.completion,
            "```",
            "",
            f"- input tokens: {s.n_input_tokens}",
            f"- output tokens: {s.n_output_tokens}",
            "",
        ]
    return "\n".join(lines)
