"""nanoBeard Sloop — v1 preset.

Two factory variants:
  - make_config()          — full GPU training run (default if --config used)
  - make_config_smoke()    — M1 smoke test (~50 iters)

Pick via CONFIG_VARIANT env var or call directly.
"""

import os

from nanobeard.config import Config


def make_config_smoke() -> Config:
    """Tiny config for an M1 Mac sanity-check run. Will NOT produce a good model."""
    return Config(
        run_name="sloop-m1-smoke",
        model_name="sloop",
        data_dir="data/datasets/tiny_pirate_stories",
        run_dir="runs/sloop-smoke",
        hf_model_repo="younissk/nanoBeard-sloop-14M",
        device="mps",
        dtype="float32",
        compile=False,
        batch_size=4,
        max_iters=50,
        eval_interval=25,
        eval_iters=5,
        warmup_iters=10,
        lr_decay_iters=50,
    )


def make_config_gpu() -> Config:
    """Full Sloop config for a real GPU."""
    return Config(
        run_name="sloop-gpu",
        model_name="sloop",
        data_dir="data/datasets/tiny_pirate_stories",
        run_dir="runs/sloop",
        hf_model_repo="younissk/nanoBeard",
        hf_ckpt_repo="younissk/pirate-llm-ckpts",
        device="cuda",
        dtype="bfloat16",
        compile=True,
        batch_size=64,
        max_iters=20000,
        wandb_project="pirate-llm",
    )


def make_config_sft() -> Config:
    """SFT-on-Sloop config."""
    return Config(
        run_name="sloop-sft",
        model_name="sloop",
        data_dir="data/datasets/tiny_pirate_stories",
        run_dir="runs/sloop-sft",
        hf_model_repo="younissk/nanoBeard-sloop-14M",
        hf_ckpt_repo="younissk/pirate-llm-sft-ckpts",
        # SFT loads the *pretraining* ckpt.pt from here (NOT the model repo,
        # which holds safetensors). Override with --pretrained-repo.
        pretrained_ckpt_repo="younissk/pirate-llm-ckpts",
        device="cuda",
        dtype="bfloat16",
        compile=False,
        dropout=0.0,
        block_size=256,
        learning_rate=2e-5,
        min_lr=2e-6,
        weight_decay=0.0,
        warmup_iters=50,
        lr_decay_iters=1500,
        max_iters=1500,
        batch_size=16,
        eval_interval=100,
        eval_iters=20,
        log_interval=20,
        resume=False,
    )


def make_config() -> Config:
    """Variant dispatcher. CONFIG_VARIANT=smoke|gpu|sft (default: smoke locally, gpu otherwise)."""
    variant = os.environ.get("CONFIG_VARIANT", "smoke")
    return {
        "smoke": make_config_smoke,
        "gpu": make_config_gpu,
        "sft": make_config_sft,
    }[variant]()
