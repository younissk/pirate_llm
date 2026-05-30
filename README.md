---
language: en
tags:
  - gpt
  - nanogpt
  - pirate
  - tinystories
  - text-generation
pipeline_tag: text-generation
library_name: pytorch
---

![nanoBeard Banner](./banner.png)

# nanoBeard ☠️

A tiny pirate-themed GPT trained from scratch on a piratized version of TinyStories,
then SFT-tuned. Built as a learning project — closer to nanoGPT than to a production LM.

The repo is structured for **multiple ship-class versions** under one codebase:

| Codename | Status | HF repo |
|---|---|---|
| **Sloop** (v1) | shipped | [`younissk/nanoBeard`](https://huggingface.co/younissk/nanoBeard) |
| Brig (v2) | planned | `younissk/nanoBeard-Brig` |

- **Source code:** https://github.com/younissk/pirate_llm
- **Docs:** see `docs/` or run `make docs-serve`

## Layout

```
nanobeard/             # package: training, sampling, SFT, publish, eval
  models/              # one file per architecture, registered via MODEL_REGISTRY
  eval/                # perplexity + sample-gallery harness
  dataset_pipeline/    # sources (piratized corpora) + recipe-driven dataset builds
configs/               # one .py per model version
data/sources/<name>/   # reusable piratized corpora (cached arrow + source.json)
data/datasets/<name>/  # composed datasets: recipe.json + tokenizer + bins + metadata
runs/<version>/        # per-version checkpoints
evals/                 # prompt set + per-eval reports
space/                 # Gradio playground (multi-version dropdown)
tests/                 # 80+ pytest tests
docs/                  # mkdocs site
```

## Model details (Sloop, v1)

| Field | Value |
|---|---|
| Architecture | Decoder-only Transformer (GPT-style) |
| Parameters | ~13.8M |
| Layers / heads / embd | 6 / 6 / 384 |
| Context length | 256 tokens |
| Vocab size | 8192 (custom BPE) |
| Bias in Linear/LN | False |
| Tokenizer | `pirate_bpe.json` (HuggingFace `tokenizers` BPE) |

## Training (Sloop)

- **Pretraining** on piratized TinyStories.
- **SFT** on `TeeZee/dolly-15k-pirate-speech`.
- AdamW, warmup + cosine decay. `bfloat16` on CUDA.
- See `training_metadata.json` in the HF repo for the exact run config + losses.

## Quick start

```bash
make install                       # uv sync
uv sync --dev                      # dev tooling (pytest, ruff, mypy, mkdocs)
make env                           # .env from example
pre-commit install                 # format/lint on commit

make dataset DATASET=tiny_pirate_stories   # build -> data/datasets/tiny_pirate_stories/
make train   CONFIG=sloop          # local smoke
make train   CONFIG=sloop CONFIG_VARIANT=gpu   # GPU run
make sample  CONFIG=sloop PROMPT='Ahoy matey'
make eval    CONFIG=sloop          # perplexity + gallery -> evals/results/<date>/sloop/
make publish CONFIG=sloop          # push to HF model repo
```

## Tests

```bash
make test            # fast (~1.5s)
make test-all        # include slow integration
```

Highlights:
- `tests/test_model_contract.py` — **parametrized over `MODEL_REGISTRY`**, so any
  new architecture is automatically checked for the same invariants (causal mask,
  weight tying, shape contracts, finite grads).
- `tests/test_publish.py` — HF Hub mocked; verifies `config.json` carries
  `model_name`, `codename`, `display_name`, `num_parameters`, and all arch fields.
- `tests/test_tokenizer_hash.py` — tokenizer fingerprint in every ckpt.

## Loading a published model

`nanoBeard` is **not** a `transformers` model — load via the `nanobeard` package:

```python
import json, torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_model
from tokenizers import Tokenizer

from nanobeard.config import Config
from nanobeard.models import build_model

repo = "younissk/nanoBeard"      # Sloop
cfg_dict = json.load(open(hf_hub_download(repo, "config.json")))
cfg = Config(**{k: v for k, v in cfg_dict.items() if k in Config.__dataclass_fields__})
model = build_model(cfg).eval()
load_model(model, hf_hub_download(repo, "model.safetensors"))

tok = Tokenizer.from_file(hf_hub_download(repo, "pirate_bpe.json"))
ids = torch.tensor([tok.encode("Once upon a time").ids])
with torch.no_grad():
    for _ in range(80):
        logits, _ = model(ids[:, -cfg.block_size:])
        next_id = torch.multinomial(torch.softmax(logits[:, -1] / 0.8, -1), 1)
        ids = torch.cat([ids, next_id], dim=1)
print(tok.decode(ids[0].tolist()))
```

## Adding a new model version

See `docs/adding-a-model.md`. TL;DR:

1. Write `nanobeard/models/<key>.py` (new arch, frozen contract).
2. Register a `ModelSpec` in `nanobeard/models/__init__.py`.
3. Drop a config in `configs/<key>.py`.
4. `make dataset DATASET=<name>` / `make train CONFIG=<key>`.
5. `make test` — contract suite parametrizes automatically.
6. `make publish CONFIG=<key>` to its own HF model repo.

## Limitations

- Trained on a tiny synthetic corpus. Vocabulary, grammar, and world knowledge
  are extremely narrow.
- Short context (256 tokens).
- No safety tuning. Pirate-flavored nonsense at best.
- Educational artifact, not a useful chat model.

## Data

- Base corpus: [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)
  (CDLA-Sharing-1.0), transformed by `nanobeard/dataset_pipeline/piratize.py`.
- SFT corpus (Sloop): `TeeZee/dolly-15k-pirate-speech`.
