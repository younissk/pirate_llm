"""Copy training/{config,model}.py into space/ and upload to the HF Space."""

import os
import shutil
from pathlib import Path

from huggingface_hub import HfApi

SPACE_ID = "younissk/nanoBeard-playground"
SPACE_DIR = Path("space")
ASSETS_DIR = SPACE_DIR / "assets"
COPIES = [
    (Path("training/config.py"), SPACE_DIR / "config.py"),
    (Path("training/model.py"), SPACE_DIR / "model.py"),
]
ASSET_COPIES = [
    (Path("nanoBeard.png"), ASSETS_DIR / "nanoBeard.png"),
]


def _copy_with_flat_imports(src: Path, dst: Path) -> None:
    """Copy a module from training/ into space/, rewriting `training.x` imports
    to bare `x` so the file works without the parent package."""
    text = src.read_text()
    text = text.replace("from training.config", "from config")
    text = text.replace("from training.model", "from model")
    dst.write_text(text)


def main() -> None:
    ASSETS_DIR.mkdir(exist_ok=True)
    for src, dst in COPIES:
        _copy_with_flat_imports(src, dst)
        print(f"copied {src} -> {dst}")
    for src, dst in ASSET_COPIES:
        shutil.copy(src, dst)
        print(f"copied {src} -> {dst}")

    token = os.environ["HF_TOKEN"]
    api = HfApi(token=token)
    api.upload_folder(
        folder_path=str(SPACE_DIR),
        repo_id=SPACE_ID,
        repo_type="space",
        commit_message="Update nanoBeard playground",
    )
    print(f"Uploaded to https://huggingface.co/spaces/{SPACE_ID}")


if __name__ == "__main__":
    main()
