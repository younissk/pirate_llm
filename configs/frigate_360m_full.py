"""Frigate 358M — 1.6 epochs on pirate_enhanced_full (~4.75B tokens). A100 / RTX 4090.

27 layers / 16 heads / 1024 embd -> 358,345,088 params. head_dim = 1024/16 = 64
(RoPE-even + flash path). A scaled-up Frigate: ~2.85x the 126M preset's depth*width.

Memory: batch sized down for a 24GB 4090; comfortable on an A100 (40/80GB).
If OOM on the 4090, halve batch_size and double gradient_accumulation_steps
(keeps the 49,152 tokens/iter constant so the horizon math is unchanged).

Variants: CONFIG_VARIANT=smoke|gpu (default: smoke).
"""

import os

from nanobeard.config import Config

DATA_DIR = "data/datasets/pirate_enhanced_full"

ARCH = dict(
    block_size=512,
    n_layer=27,
    n_head=16,
    n_embd=1024,
    use_rope=True,
    use_swiglu=True,
    use_rmsnorm=True,
    use_qk_norm=True,
)


def make_config_smoke() -> Config:
    """Tiny M1 Mac sanity-check. Real arch but few iters — NOT a good model."""
    return Config(
        run_name="frigate-360m-full-smoke",
        model_name="frigate",
        data_dir=DATA_DIR,
        run_dir="runs/frigate-360m-full-smoke",
        hf_model_repo="younissk/nanoBeard-frigate-360M-full",
        dropout=0.05,
        optimizer="muon",
        device="mps",
        dtype="float32",
        compile=False,
        batch_size=1,
        gradient_accumulation_steps=1,
        max_iters=50,
        eval_interval=25,
        eval_iters=5,
        warmup_iters=10,
        lr_decay_iters=50,
        **ARCH,
    )


def make_config_gpu() -> Config:
    """Full 1.6-epoch Frigate-358M run. A100 recommended (40/80GB); fits a 4090."""
    return Config(
        run_name="frigate-360m-full",
        model_name="frigate",
        data_dir=DATA_DIR,
        run_dir="runs/frigate-360m-full",
        hf_model_repo="younissk/nanoBeard-frigate-360M-full",
        # Own ckpt repo — keeps the 358M full-corpus weights separate.
        hf_ckpt_repo="younissk/frigate-360M-full-ckpts",
        dropout=0.05,
        # Muon for the 2D hidden matrices; embeddings/head/norms stay on AdamW.
        optimizer="muon",
        muon_lr=0.02,
        muon_momentum=0.95,
        device="cuda",
        dtype="bfloat16",
        compile=True,
        batch_size=12,
        gradient_accumulation_steps=8,  # effective batch 96, 49,152 tokens/iter
        # epochs is the real knob: 1.6 passes over pirate_enhanced_full
        # (~4.75B tokens) ≈ 154.6k iters at 49,152 tokens/iter. max_iters is just
        # the safety ceiling; resolve_max_iters() derives the real horizon from
        # the actual train.bin size at train start.
        epochs=1.6,
        max_iters=160000,
        warmup_iters=2000,
        lr_decay_iters=160000,
        eval_interval=1000,
        eval_iters=100,
        wandb_project="pirate-llm",
        **ARCH,
    )


def make_config() -> Config:
    """Variant dispatcher. CONFIG_VARIANT=smoke|gpu (default: smoke)."""
    variant = os.environ.get("CONFIG_VARIANT", "smoke")
    return {
        "smoke": make_config_smoke,
        "gpu": make_config_gpu,
    }[variant]()
