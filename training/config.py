from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Config:
    # ----- Run identity -----
    run_name: str = "tiny-pirate-v1"
    out_dir: str = "out"  # where checkpoints go

    # ----- Data -----
    train_bin: str = "train.bin"
    val_bin: str = "val.bin"
    # Tokenizer (only needed for sampling, not training)
    tokenizer_path: str = "pirate_bpe.json"

    # ----- Model architecture (the "Tiny" tier from our curriculum) -----
    vocab_size: int = 8192  # must match your trained tokenizer
    block_size: int = 256  # context length in tokens
    n_layer: int = 6  # number of transformer blocks
    n_head: int = 6  # attention heads per block
    n_embd: int = 384  # hidden dimension; must be divisible by n_head
    dropout: float = 0.1  # mild regularization
    bias: bool = False  # no bias in Linears/LayerNorms (small efficiency win)

    # ----- Optimizer (AdamW — Lesson 4 defaults) -----
    learning_rate: float = 3e-4  # peak LR after warmup
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95  # lower than Adam default; transformer standard
    grad_clip: float = 1.0  # clip gradient norm — saves you from blowups

    # ----- LR schedule (warmup + cosine — Lesson 4) -----
    warmup_iters: int = 200  # ~1% of total
    lr_decay_iters: int = 20000  # should roughly equal max_iters
    min_lr: float = 3e-5  # 10% of peak — the cosine floor

    # ----- Training loop -----
    max_iters: int = 20000  # total optimization steps
    batch_size: int = 32  # sequences per batch
    gradient_accumulation_steps: int = 1  # raise this on smaller GPUs
    eval_interval: int = 500  # how often to check val loss
    eval_iters: int = 100  # batches averaged for val loss estimate
    log_interval: int = 10  # print train loss every N iters

    # ----- System -----
    device: str = "cpu"  # 'cpu', 'mps' (M1 Mac), or 'cuda'
    dtype: str = "float32"  # 'float32', 'bfloat16', or 'float16'
    compile: bool = False  # torch.compile (cuda only, not on MPS)
    seed: int = 1337

    # ----- Checkpoint sync (Hugging Face Hub) -----
    hf_repo_id: Optional[str] = None  # e.g. "younissk/pirate-llm-ckpts"
    hf_private: bool = True
    resume: bool = True  # try to pull latest ckpt on start

    # ----- Logging (Weights & Biases) -----
    wandb_project: Optional[str] = None  # e.g. "pirate-llm"
    wandb_entity: Optional[str] = None  # your W&B username/team, optional

    # ----- Convenience factories — call these instead of constructing by hand -----
    @classmethod
    def for_m1_smoke_test(cls) -> "Config":
        """Tiny config for an M1 Mac sanity-check run. Will NOT produce a good model."""
        return cls(
            run_name="m1-smoke",
            device="mps",  # change to 'cpu' if MPS gives you trouble
            dtype="float32",
            compile=False,
            batch_size=4,
            max_iters=50,  # just a few steps to confirm forward+backward work
            eval_interval=25,
            eval_iters=5,
            warmup_iters=10,
            lr_decay_iters=50,
        )

    @classmethod
    def for_gpu_training(cls) -> "Config":
        """Full Tiny config for vast.ai with a real GPU."""
        return cls(
            run_name="tiny-pirate-gpu",
            device="cuda",
            dtype="bfloat16",
            compile=True,
            batch_size=64,
            max_iters=20000,
            hf_repo_id="younissk/pirate-llm-ckpts",
            wandb_project="pirate-llm",
        )
