"""Held-out perplexity on val.bin.

PPL is exp(mean cross-entropy). Lower = better. Comparable only within the
same tokenizer + same eval corpus — useful for "did v2 actually beat v1" but
not for comparing across tokenizer changes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from nanobeard.config import Config


@dataclass
class PerplexityResult:
    loss: float
    perplexity: float
    n_tokens: int
    n_batches: int


@torch.no_grad()
def compute_perplexity(
    model: nn.Module,
    config: Config,
    bin_path: str | None = None,
    n_batches: int = 200,
    seed: int = 1337,
) -> PerplexityResult:
    """Average loss over `n_batches` random crops from a .bin file.

    Mirrors the random-crop strategy `get_batch` uses during training so the
    metric matches what the training loop already reports as `val/loss`.
    """
    path = bin_path or config.val_bin
    data = np.memmap(path, dtype=np.uint16, mode="r")

    rng = np.random.default_rng(seed)
    n_positions = len(data) - config.block_size - 1
    if n_positions <= 0:
        raise ValueError(f"{path} too short for block_size={config.block_size}")

    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for _ in range(n_batches):
        ix = rng.integers(0, n_positions, size=config.batch_size)
        x = np.stack([data[i : i + config.block_size].astype(np.int64) for i in ix])
        y = np.stack([data[i + 1 : i + 1 + config.block_size].astype(np.int64) for i in ix])
        x_t = torch.from_numpy(x).to(config.device)
        y_t = torch.from_numpy(y).to(config.device)
        _, loss = model(x_t, y_t)
        total_loss += loss.item() * y_t.numel()
        total_tokens += y_t.numel()

    mean_loss = total_loss / total_tokens
    return PerplexityResult(
        loss=mean_loss,
        perplexity=math.exp(mean_loss),
        n_tokens=total_tokens,
        n_batches=n_batches,
    )
