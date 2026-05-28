# Architecture

## Package layout

```
nanobeard/
  config.py                  # Config dataclass + load_config(path)
  data.py                    # get_batch(split, config)
  train.py                   # pretraining loop, ckpt save/resume, HF sync
  sft.py                     # instruction-tuning loop
  sft_data.py                # prompt rendering + tokenization for SFT
  sample.py                  # generation + CLI
  publish.py                 # ckpt -> HF model repo
  tokenizer_hash.py          # SHA-256 gate against tokenizer drift
  models/
    __init__.py              # MODEL_REGISTRY, build_model(cfg), spec_for(cfg)
    naming.py                # display_name(cfg, model)
    sloop.py                 # Sloop architecture — frozen
  eval/
    perplexity.py            # PPL on val.bin
    gallery.py               # prompt -> completion -> markdown
    run.py                   # eval entrypoint
  dataset_pipeline/          # piratize, tokenize, dump bins

configs/
  sloop.py                   # Sloop hyperparams (smoke / gpu / sft variants)

data/<version>/              # per-version corpus + tokenizer + bins
runs/<version>/              # per-version checkpoints + metadata
evals/results/<date>/<ver>/  # per-eval reports

space/                       # Gradio playground (loads from HF Hub)
scripts/                     # publish_space.py, vast_*
tests/                       # pytest suite (72+ tests)
docs/                        # this site
```

## Dispatch

Three pieces of state determine which architecture is used:

1. **`Config.model_name`** — string key (`"sloop"`, future `"brig"`, etc.).
2. **`MODEL_REGISTRY[model_name]`** — `ModelSpec(dispatch_key, codename, hf_repo, cls, arch_fields)`.
3. **`build_model(cfg)`** — single function used everywhere: `model = build_model(cfg)`.

Every loop (train, SFT, sample, publish, eval, Space) calls `build_model`.
Adding a new model means registering a new spec; no caller changes.

## Checkpoint contract

Every saved ckpt contains:

| Key | Purpose |
|---|---|
| `model` | `state_dict` of the raw (un-compiled) module |
| `optimizer` | `state_dict` for resume |
| `config` | the full `Config` instance — arch is reconstructable from this alone |
| `model_name` | duplicate of `config.model_name`, top-level for fast dispatch |
| `iter_num`, `val_loss`, `best_val_loss` | run progress |
| `tokenizer_sha256` | hash of `pirate_bpe.json` at train time; verified on load |
| `stage` | `"pretrain"` (omitted) or `"sft"` |

## Tokenizer hash gate

Silent tokenizer drift is the worst-class tiny-LM bug: model emits gibberish
and no metric flags it. Mitigation:

- `train.save_checkpoint` records the SHA-256 of `config.tokenizer_path`.
- `sft.load_pretrained` **raises** on mismatch — wrong tokenizer would corrupt SFT.
- `sample.load_checkpoint` **warns** on mismatch — useful when intentionally exploring.

See `nanobeard/tokenizer_hash.py`.

## Backwards-compat shim

Legacy checkpoints (from before the refactor) pickled `Config` under
`training.config.Config`. `nanobeard/__init__.py` aliases the old module path
to the new one so `torch.load` keeps working. Remove the shim once all
legacy ckpts have been migrated.
