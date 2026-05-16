import numpy as np
import torch

from training.config import Config


def get_batch(split: str, config: Config) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample a random batch from train.bin or val.bin.

    Args:
        split: 'train' or 'val'
        config: the run Config

    Returns:
        x: input tokens of shape (batch_size, block_size)
        y: target tokens of shape (batch_size, block_size), shifted left by 1
    """
    # Re-create memmap each call to avoid a memory leak in long-running loops.
    # See nanoGPT issue #154 — the cost of recreation is ~negligible.
    if split == "train":
        data = np.memmap(config.train_bin, dtype=np.uint16, mode="r")
    elif split == "val":
        data = np.memmap(config.val_bin, dtype=np.uint16, mode="r")
    else:
        raise ValueError(f"split must be 'train' or 'val', got {split!r}")

    # Pick batch_size random starting indices.
    # The -1 ensures we have room for y to read one token past x.
    ix = torch.randint(len(data) - config.block_size - 1, (config.batch_size,))

    # Build the batch. Cast to int64 because PyTorch embedding layers expect long.
    x = torch.stack(
        [torch.from_numpy(data[i : i + config.block_size].astype(np.int64)) for i in ix]
    )
    y = torch.stack(
        [
            torch.from_numpy(data[i + 1 : i + 1 + config.block_size].astype(np.int64))
            for i in ix
        ]
    )

    # Move to device. pin_memory + non_blocking gives a small speedup on CUDA;
    # on MPS / CPU it's a no-op (and harmless).
    if config.device == "cuda":
        x = x.pin_memory().to(config.device, non_blocking=True)
        y = y.pin_memory().to(config.device, non_blocking=True)
    else:
        x = x.to(config.device)
        y = y.to(config.device)

    return x, y
