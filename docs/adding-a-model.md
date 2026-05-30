# Adding a model

Recipe to ship a new ship-class (e.g. Brig). Order matters.

## 1. Pick a codename + dispatch key

Naming convention: **codename** is the human-readable ship class (Brig, Frigate,
Galleon, Man-o-War). **Dispatch key** is the lowercase version used inside
`Config.model_name` and the registry.

## 2. Write the architecture module

`nanobeard/models/brig.py`:

```python
"""nanoBeard Brig — v2 architecture."""

import torch.nn as nn
from nanobeard.config import Config

ARCH_FIELDS = ("vocab_size", "block_size", "n_layer", "n_head", "n_embd", "dropout", "bias")
# Add new fields here if Brig introduces them (e.g. "rope_theta", "ffn_mult").

class GPT(nn.Module):
    def __init__(self, config: Config):
        ...
    def forward(self, idx, targets=None):
        ...
    def num_parameters(self) -> int:
        ...
```

Contract enforced by `tests/test_model_contract.py`:

- accepts `Config`
- exposes `config.block_size`
- `num_parameters() > 0`
- `forward(idx)` returns `(logits, None)` of shape `(B, T, V)`
- `forward(idx, targets)` returns finite scalar loss
- rejects `T > block_size` with `AssertionError`
- causal mask: mutating position `k` must not change logits at positions `< k`

## 3. Register the spec

`nanobeard/models/__init__.py`:

```python
from .brig import GPT as GPTBrig
from .brig import ARCH_FIELDS as BRIG_ARCH_FIELDS

MODEL_REGISTRY = {
    "sloop": ModelSpec(..., cls=GPTSloop, ...),
    "brig": ModelSpec(
        dispatch_key="brig",
        codename="Brig",
        hf_repo="younissk/nanoBeard-Brig",
        cls=GPTBrig,
        arch_fields=BRIG_ARCH_FIELDS,
    ),
}
```

## 4. Add a config

`configs/brig.py` mirroring `configs/sloop.py`, with `model_name="brig"`,
`data_dir="data/datasets/<dataset>"` (the dataset Brig trains on),
`run_dir="runs/brig"`, and Brig-specific hyperparameters / arch fields.

Models and datasets are decoupled: a config just points `data_dir` at a built
dataset, so multiple models can share one dataset (and one model can be
retrained on different datasets).

## 5. Build the dataset

```bash
make dataset DATASET=<dataset>
```

To use a bigger corpus, register a new source in
`nanobeard/dataset_pipeline/sources.py` and add it to the dataset's
`recipe.json`. To change tokenizer vocab size, set `vocab_size` in that same
`recipe.json` — each dataset trains its own tokenizer.

## 6. Train

```bash
make train CONFIG=brig CONFIG_VARIANT=gpu
```

## 7. Verify the contract tests pass on Brig

```bash
make test
```

`test_model_contract.py` is parametrized over `MODEL_REGISTRY`, so it
automatically picks up Brig and runs every architectural invariant on it.
If a test fails, Brig violates a contract that Sloop honored — decide
whether to fix Brig or document the deviation by forking the failing test.

## 8. Eval side-by-side

```bash
make eval CONFIG=sloop
make eval CONFIG=brig
diff -r evals/results/<date>/sloop/ evals/results/<date>/brig/
```

## 9. Publish

```bash
make publish CONFIG=brig    # -> younissk/nanoBeard-Brig
```

The Space picks up Brig automatically because `NANOBEARD_REPOS` defaults to
all `hf_repo`s in `MODEL_REGISTRY`. A model-switcher dropdown appears once
more than one model is registered.
