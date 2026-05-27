import importlib.util
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    # ----- Run identity -----
    run_name: str = "nanobeard-sloop"

    # ----- Model dispatch (key into nanobeard.models.MODEL_REGISTRY) -----
    model_name: str = "sloop"

    # ----- Paths (everything else derives from these two) -----
    data_dir: str = "data/sloop"  # holds train.bin, val.bin, tokenizer
    run_dir: str = "runs/sloop"  # holds ckpt.pt, sft_ckpt.pt

    # ----- Model architecture -----
    vocab_size: int = 8192
    block_size: int = 256
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.1
    bias: bool = False

    # ----- Optimizer -----
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    # ----- LR schedule -----
    warmup_iters: int = 200
    lr_decay_iters: int = 20000
    min_lr: float = 3e-5

    # ----- Training loop -----
    max_iters: int = 20000
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    eval_interval: int = 500
    eval_iters: int = 100
    log_interval: int = 10

    # ----- System -----
    device: str = "cpu"
    dtype: str = "float32"
    compile: bool = False
    seed: int = 1337

    # ----- HF Hub -----
    hf_model_repo: str = "younissk/nanoBeard"  # final published model repo
    hf_ckpt_repo: str | None = None  # rolling ckpt sync repo (private)
    hf_private: bool = True
    resume: bool = True

    # ----- W&B -----
    wandb_project: str | None = None
    wandb_entity: str | None = None

    # ----- Derived paths (do not set directly) -----
    @property
    def train_bin(self) -> str:
        return os.path.join(self.data_dir, "train.bin")

    @property
    def val_bin(self) -> str:
        return os.path.join(self.data_dir, "val.bin")

    @property
    def tokenizer_path(self) -> str:
        return os.path.join(self.data_dir, "pirate_bpe.json")

    @property
    def out_dir(self) -> str:
        """Back-compat alias for run_dir."""
        return self.run_dir

    @property
    def ckpt_path(self) -> str:
        return os.path.join(self.run_dir, "ckpt.pt")

    @property
    def sft_ckpt_path(self) -> str:
        return os.path.join(self.run_dir, "sft_ckpt.pt")


def load_config(path: str) -> Config:
    """Load a Config from a Python file that defines a `config` module-level variable
    or a `make_config()` function. E.g. `configs/sloop.py`."""
    p = Path(path).resolve()
    spec = importlib.util.spec_from_file_location(p.stem, p)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load config module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if hasattr(mod, "make_config"):
        return mod.make_config()
    if hasattr(mod, "config"):
        return mod.config
    raise AttributeError(
        f"{path} must define `config = Config(...)` or `def make_config() -> Config`"
    )
