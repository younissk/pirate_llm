"""Convert out/ckpt.pt to a Hub-ready folder and upload to younissk/nanoBeard."""

import json
import os
import shutil
from dataclasses import asdict
from pathlib import Path

import torch
from safetensors.torch import save_model
from huggingface_hub import HfApi

from training.config import Config
from training.model import GPT

REPO_ID = "younissk/nanoBeard"
CKPT_PATH = Path("out/ckpt.pt")
TOKENIZER_PATH = Path("pirate_bpe.json")
README_PATH = Path("README.md")
BANNER_PATH = Path("banner.png")
STAGE_DIR = Path("hf_release")

ARCH_FIELDS = ("vocab_size", "block_size", "n_layer", "n_head", "n_embd", "dropout", "bias")


def main() -> None:
    ck = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    cfg: Config = ck["config"]

    STAGE_DIR.mkdir(exist_ok=True)

    # Rebuild model, load weights, save as safetensors (handles weight-tying).
    model = GPT(cfg)
    model.load_state_dict(ck["model"])
    model.eval()
    save_model(model, str(STAGE_DIR / "model.safetensors"))

    config_json = {k: getattr(cfg, k) for k in ARCH_FIELDS}
    config_json["model_type"] = "nanobeard-gpt"
    config_json["architectures"] = ["GPT"]
    (STAGE_DIR / "config.json").write_text(json.dumps(config_json, indent=2) + "\n")

    shutil.copy(TOKENIZER_PATH, STAGE_DIR / "pirate_bpe.json")
    shutil.copy(README_PATH, STAGE_DIR / "README.md")
    shutil.copy(BANNER_PATH, STAGE_DIR / "banner.png")

    training_meta = {
        "stage": ck.get("stage"),
        "iter_num": ck.get("iter_num"),
        "val_loss": ck.get("val_loss"),
        "best_val_loss": ck.get("best_val_loss"),
        "full_train_config": asdict(cfg),
    }
    (STAGE_DIR / "training_metadata.json").write_text(
        json.dumps(training_meta, indent=2, default=str) + "\n"
    )

    token = os.environ["HF_TOKEN"]
    api = HfApi(token=token)
    api.upload_folder(
        folder_path=str(STAGE_DIR),
        repo_id=REPO_ID,
        repo_type="model",
        commit_message="Unify README with root; include banner",
    )
    print(f"Uploaded to https://huggingface.co/{REPO_ID}")


if __name__ == "__main__":
    main()
