"""nanoBeard Galleon — bigger preset, trained on the pirate_enhanced corpus.

Same Sloop architecture (config-driven GPT), scaled up to exploit the larger
dataset: pirate_enhanced is ~1.55B train tokens / vocab 16384 vs the original
tiny_pirate_stories. `model_name` stays "sloop" — that is the architecture
dispatch key in MODEL_REGISTRY; "galleon" is just a larger configuration of it.

vocab_size is NOT set here: train.resolve_vocab_size() overrides it from the
dataset's tokenizer (16384 for pirate_enhanced), so the recipe is the one knob.

Sizing (galleon-gpu): n_layer=8, n_head=8, n_embd=512, block_size=512
  -> ~42M params (12*n_layer*n_embd^2 transformer + ~17M embeddings @ vocab 16384).
  Chinchilla-optimal tokens ~= 20 * 42M = 840M; we have 1.55B, so the model is
  comfortably (slightly over-) fed — extra data helps a small model.

Token budget: tokens/iter = batch_size * block_size * grad_accum = 32*512*3 = 49152.
  1 epoch (1.55B) ~= 31.5k iters; max_iters=40000 ~= 1.27 epochs.

OOM on a 24GB card? Lower batch_size to 16 and raise gradient_accumulation_steps
to 6 (same effective batch of 96), or drop block_size back to 256.

Variants: CONFIG_VARIANT=smoke|gpu (default: smoke locally, gpu otherwise).
"""

import os

from nanobeard.config import Config

DATA_DIR = "data/datasets/pirate_enhanced"


def make_config_smoke() -> Config:
    """Tiny M1 Mac sanity-check. Bigger arch but few iters — NOT a good model."""
    return Config(
        run_name="galleon-m1-smoke",
        model_name="sloop",
        data_dir=DATA_DIR,
        run_dir="runs/galleon-smoke",
        hf_model_repo="younissk/nanoBeard-galleon-34M",
        block_size=512,
        n_layer=8,
        n_head=8,
        n_embd=512,
        dropout=0.05,
        device="mps",
        dtype="float32",
        compile=False,
        batch_size=2,
        gradient_accumulation_steps=1,
        max_iters=50,
        eval_interval=25,
        eval_iters=5,
        warmup_iters=10,
        lr_decay_iters=50,
    )


def make_config_gpu() -> Config:
    """Full Galleon run for a single RTX 4090 (24GB)."""
    return Config(
        run_name="galleon-gpu",
        model_name="sloop",
        data_dir=DATA_DIR,
        run_dir="runs/galleon",
        hf_model_repo="younissk/nanoBeard-galleon-34M",
        # Own repo — galleon's 512-dim/16384-vocab ckpt is incompatible with the
        # sloop ckpt in pirate-llm-ckpts; sharing would crash resume + clobber it.
        hf_ckpt_repo="younissk/galleon-ckpts",
        # --- Architecture (scaled up from Sloop) ---
        block_size=512,
        n_layer=8,
        n_head=8,
        n_embd=512,
        dropout=0.05,
        # --- System ---
        device="cuda",
        dtype="bfloat16",
        compile=True,
        # --- Training loop ---
        batch_size=32,
        gradient_accumulation_steps=3,  # effective batch 96
        max_iters=40000,
        warmup_iters=400,
        lr_decay_iters=40000,
        eval_interval=500,
        eval_iters=100,
        wandb_project="pirate-llm",
    )


def make_config_sft() -> Config:
    """Chat SFT on top of the pretrained Galleon (RTX 4090).

    block_size MUST equal the pretraining block_size (512) — load_pretrained
    hard-fails otherwise, since the position embedding is fixed. The 512 window
    is also why Galleon is the right base for the chat/memory goal: ~5-6 short
    turns fit, vs ~2-3 at Sloop's 256.
    """
    return Config(
        run_name="galleon-sft",
        model_name="sloop",
        data_dir=DATA_DIR,
        run_dir="runs/galleon-sft",
        hf_model_repo="younissk/nanoBeard-galleon-34M",
        hf_ckpt_repo="younissk/galleon-sft-ckpts",
        # SFT loads the pretraining ckpt.pt from here (not the model repo).
        pretrained_ckpt_repo="younissk/galleon-ckpts",
        block_size=512,  # must match pretraining
        n_layer=8,
        n_head=8,
        n_embd=512,
        dropout=0.0,
        device="cuda",
        dtype="bfloat16",
        compile=False,  # short run; compile's warmup + dynamic shapes aren't worth it
        # SFT optimizer: low LR, no weight decay.
        learning_rate=2e-5,
        min_lr=2e-6,
        weight_decay=0.0,
        warmup_iters=100,
        lr_decay_iters=3000,
        max_iters=3000,
        batch_size=16,
        gradient_accumulation_steps=2,  # effective batch 32
        eval_interval=200,
        eval_iters=50,
        log_interval=20,
        resume=False,
        wandb_project="pirate-llm",
    )


def make_config() -> Config:
    """Variant dispatcher. CONFIG_VARIANT=smoke|gpu|sft (default: smoke)."""
    variant = os.environ.get("CONFIG_VARIANT", "smoke")
    return {
        "smoke": make_config_smoke,
        "gpu": make_config_gpu,
        "sft": make_config_sft,
    }[variant]()
