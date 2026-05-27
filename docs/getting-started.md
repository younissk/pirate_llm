# Getting started

## Install

```bash
make install              # uv sync
uv sync --dev             # dev tooling: pytest, ruff, mypy, pyright, mkdocs
make env                  # copy example.env -> .env, fill in HF_TOKEN and WANDB_API_KEY
pre-commit install        # one-time, enables format/lint on commit
```

## Build the dataset

Each model version owns its own dataset directory under `data/<version>/`.
For Sloop that's `data/sloop/`.

```bash
make data       CONFIG=sloop   # piratize TinyStories
make tokenizer  CONFIG=sloop   # train BPE
make tokens     CONFIG=sloop   # tokenize corpus -> train.bin, val.bin
# or do all three in one shot:
make dataset    CONFIG=sloop
```

## Train

```bash
make train CONFIG=sloop                       # smoke train on local CPU/MPS
make train CONFIG=sloop CONFIG_VARIANT=gpu    # full GPU run (vast.ai etc.)
```

Checkpoints land in `runs/<version>/ckpt.pt` and (if `hf_ckpt_repo` is set)
roll to a private HF Hub repo on every `eval_interval`.

## Sample

```bash
make sample CONFIG=sloop PROMPT='Ahoy matey'
```

## Eval

```bash
make eval CONFIG=sloop          # full perplexity + gallery
make eval-quick CONFIG=sloop    # 20 batches, short samples
```

Results land in `evals/results/<date>/<version>/`. Compare two versions by
diffing two directories.

## Publish to Hugging Face

```bash
make publish CONFIG=sloop        # ckpt -> younissk/nanoBeard
make publish-space               # update playground Space
```

## Test

```bash
make test            # fast suite (~1.5s)
make test-all        # include slow integration
```
