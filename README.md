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

- **Source code:** https://github.com/younissk/pirate_llm
- **Model on the Hub:** https://huggingface.co/younissk/nanoBeard

## Model details

| Field | Value |
|---|---|
| Architecture | Decoder-only Transformer (GPT-style) |
| Parameters | ~13.9M (approx) |
| Layers | 6 |
| Heads | 6 |
| Embedding dim | 384 |
| Context length | 256 tokens |
| Vocab size | 8192 (custom BPE) |
| Bias in Linear/LN | False |
| Tokenizer | `pirate_bpe.json` (HuggingFace `tokenizers` BPE) |

## Training

- **Pretraining:** TinyStories, piratized via a rule-based transform.
- **SFT stage:** stage=`sft`, iters=`1400`, val_loss=`4.2816` (best `4.2485`).
- **Optimizer:** AdamW, lr=2e-5, weight_decay=0.0, betas=(0.9, 0.95), grad_clip=1.0.
- **LR schedule:** linear warmup (50 steps) → cosine decay to min_lr=2e-6 over 1500 steps.
- **Hardware/dtype:** trained on `cuda` in `bfloat16`.

## Files in the released repo

- `model.safetensors` — model weights.
- `config.json` — architecture config (load into `training.config.Config`).
- `pirate_bpe.json` — tokenizer (load with `tokenizers.Tokenizer.from_file`).
- `training_metadata.json` — full training config + metrics snapshot.
- `banner.png` — the banner above.

## Usage

This model is **not** a `transformers` model — it uses the custom `GPT` class
from this repo.

```python
import json, torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_model
from tokenizers import Tokenizer

from training.config import Config
from training.model import GPT

repo = "younissk/nanoBeard"
cfg_path = hf_hub_download(repo, "config.json")
weights_path = hf_hub_download(repo, "model.safetensors")
tok_path = hf_hub_download(repo, "pirate_bpe.json")

cfg_dict = json.load(open(cfg_path))
cfg = Config(**{k: v for k, v in cfg_dict.items()
                if k in Config.__dataclass_fields__})
model = GPT(cfg).eval()
load_model(model, weights_path)

tok = Tokenizer.from_file(tok_path)
ids = torch.tensor([tok.encode("Once upon a time").ids])
with torch.no_grad():
    for _ in range(80):
        logits, _ = model(ids[:, -cfg.block_size:])
        next_id = torch.multinomial(torch.softmax(logits[:, -1] / 0.8, -1), 1)
        ids = torch.cat([ids, next_id], dim=1)
print(tok.decode(ids[0].tolist()))
```

## Limitations

- Trained on a small synthetic corpus (TinyStories, piratized). Vocabulary,
  grammar, and world knowledge are extremely narrow.
- Short context window (256 tokens).
- No safety tuning. Outputs are pirate-flavored nonsense at best.
- Intended as an educational artifact, not a useful chat model.

## Data

- Base corpus: [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)
  (CDLA-Sharing-1.0), transformed by the `dataset/piratize.py` script in this repo.
