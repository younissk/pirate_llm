# nanoBeard

Tiny pirate LLM you train from scratch. Currently ships **Sloop** (v1); the
repo is structured so additional ship-class versions (Brig, Frigate, …) plug
in via a model registry without touching shared training plumbing.

## What's in the box

- **`nanobeard/`** — model code, training loop, SFT, sampling, publish pipeline.
- **`nanobeard/models/`** — one file per architecture, registered in `__init__.py`.
- **`configs/`** — Python config files per model version.
- **`data/<model>/`** — corpora + tokenizer + tokenized bins, one dir per version.
- **`runs/<model>/`** — checkpoints and run metadata.
- **`nanobeard/eval/`** — perplexity + sample gallery harness for side-by-side comparisons.
- **`space/`** — Gradio playground that loads any registered model from its HF repo.

## Quick links

- [Getting started](getting-started.md) — install, build dataset, train.
- [Architecture](architecture.md) — package layout + how dispatch works.
- [Adding a model](adding-a-model.md) — recipe for shipping v2 in one PR.
- [Eval harness](eval.md) — how to compare versions numerically + qualitatively.
- [Vast.ai](vast-ai.md) — provisioning and training on rented GPUs.
- [TODO](todo.md) — open follow-ups.
