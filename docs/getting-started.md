# Getting started

## Install

```bash
make install              # uv sync
uv sync --dev             # dev tooling: pytest, ruff, mypy, pyright, mkdocs
make env                  # copy example.env -> .env, fill in HF_TOKEN and WANDB_API_KEY
pre-commit install        # one-time, enables format/lint on commit
```

## Build the dataset

Datasets are composed from reusable **sources**. A *source* is one piratized
corpus, cached under `data/sources/<name>/`. A *dataset* lives under
`data/datasets/<name>/`, defined by a `recipe.json` that lists the sources to
combine; building it produces a tokenizer (`pirate_bpe.json`), `train.bin` /
`val.bin`, and a `metadata.json` (provenance + token counts).

```bash
# Build (and cache) one source — piratizes TinyStories the first time.
make source  SOURCE=tiny_stories_pirate

# Build a dataset from its recipe: combine sources -> tokenizer + bins + metadata.
make dataset DATASET=tiny_pirate_stories
```

`make dataset` materializes any sources its recipe needs, so the explicit
`make source` step is optional. To compose a new dataset, create
`data/datasets/<name>/recipe.json` (see `pirate_enhanced/` for the pattern) and
run `make dataset DATASET=<name>`.

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
