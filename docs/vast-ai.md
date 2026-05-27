# Vast.ai workflow

Training on rented GPUs without clicking through the web UI.

## The flow we optimized for

1. **Local** — build the dataset once, push to a private HF dataset repo.
2. **CLI** — `make vast-launch CONFIG=sloop` provisions a cheap instance, bootstraps it, starts training in `tmux`.
3. **Remote** — training auto-saves checkpoints to a private HF *ckpt* repo every `eval_interval`. Disconnecting (or losing the spot instance) loses **at most one eval window** of progress.
4. **Local** — `make sample` / `make eval` / `make publish` pull the latest ckpt from HF and act on it.

## One-time setup

```bash
pip install vastai
vastai set api-key <YOUR_VAST_API_KEY>

# Tokens (also need to be on the remote instance — vast_launch.sh forwards them).
export HF_TOKEN=<your-hf-token>
export WANDB_API_KEY=<your-wandb-key>          # optional

# Push the local dataset once so future instances can download it in seconds
# instead of re-running the piratization pipeline.
make push-data CONFIG=sloop                    # -> younissk/nanobeard-data-sloop (dataset, private)
```

## Launching a training run

```bash
make vast-launch CONFIG=sloop
```

Knobs (env vars):

| Var | Default | Purpose |
|---|---|---|
| `CONFIG` | `sloop` | which config preset to train |
| `VARIANT` | `gpu` | `gpu` / `sft` etc. — sets `CONFIG_VARIANT` on the box |
| `GPU` | `RTX_4090` | Vast.ai gpu_name filter |
| `MAX_DPH` | `0.40` | hard cap on $/hour |
| `INET_DOWN` | `200` | min Mbps (matters because we pull a 1GB dataset on boot) |
| `DISK_GB` | `50` | instance disk |
| `IMAGE` | `pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime` | container image |
| `REPO_URL` | this repo | source the bootstrap pulls from |

The script writes the new instance id to `.vast_instance`. Use it:

```bash
make vast-ssh          # opens an SSH url
make vast-logs         # tails the cloud-init log
make vast-destroy      # tears the instance down
```

## What runs on the remote box

`scripts/vast_bootstrap.sh` is fired via Vast's `--onstart`. It:

1. installs `git`, `tmux`, `uv`
2. clones the repo (or pulls if already there)
3. `uv sync --no-dev`
4. `huggingface-cli download younissk/nanobeard-data-<CONFIG> --local-dir data/<CONFIG>` — falls back to running the full piratize/tokenize pipeline if the dataset repo is empty
5. starts `uv run python -m nanobeard.train --config configs/<CONFIG>.py` inside a detachable `tmux` session named `nanobeard-<CONFIG>`

Reattach with `tmux attach -t nanobeard-sloop`. Detach with `Ctrl-b d`.

## Spot / interruptible instances

Resume is robust:

- Every `eval_interval` iterations the loop writes `runs/<CONFIG>/ckpt.pt` locally *and* (if `hf_ckpt_repo` is set in the config) uploads it.
- `try_resume()` first checks local, then pulls from HF.
- Spot instances are typically 30–60% cheaper. Worth it for runs that take more than a couple hours.

Add `--bid <price>` to the `vastai create instance` call in `vast_launch.sh` to use the spot market.

## Cost reference (May 2026, approximate)

| GPU | $/hr (on-demand) | $/hr (spot) |
|---|---|---|
| RTX 3090 | 0.15–0.25 | 0.10–0.20 |
| RTX 4090 | 0.30–0.45 | 0.20–0.35 |
| A5000 | 0.30–0.45 | 0.20–0.30 |
| A100 40GB | 0.80–1.50 | 0.50–1.00 |

Sloop (~14M params, 20k iters, block_size=256) is a small enough run that any
RTX-class card finishes in well under an hour. A4000 / 3090 is the sweet spot.

## Why this beats the web UI workflow

- **No clicking.** Every step is a make target or env var. PR-reviewable.
- **No 30-min piratize on cold start.** Dataset pulls from HF in ~30 seconds.
- **No state lost on disconnect.** ckpts roll to a separate HF repo on every eval.
- **Reproducible.** `vast_launch.sh` captures the exact image, disk, dph cap,
  and bootstrap script. Re-launching v2 next month is the same command.

## Open improvements

See `docs/todo.md` for:

- Pre-built Docker image (`Dockerfile.vast`) — drops cold-start by another minute.
- `vastai search offers` ranked by inet_down × dph_total (current rank is dph alone).
- Webhook on training completion that auto-destroys the instance.
