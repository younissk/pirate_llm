"""Model registry. Add new versions by editing MODEL_REGISTRY below."""

from dataclasses import dataclass

import torch.nn as nn

from nanobeard.config import Config

from .sloop import ARCH_FIELDS as SLOOP_ARCH_FIELDS
from .sloop import GPT as GPTSloop


@dataclass(frozen=True)
class ModelSpec:
    dispatch_key: str  # Config.model_name, ckpt key
    codename: str  # display name
    hf_repo: str  # final published HF model repo
    cls: type[nn.Module]  # model class
    arch_fields: tuple[str, ...]  # fields to persist into config.json


# ---- Single source of truth. Change codename / hf_repo here. ----
MODEL_REGISTRY: dict[str, ModelSpec] = {
    "sloop": ModelSpec(
        dispatch_key="sloop",
        codename="Sloop",
        hf_repo="younissk/nanoBeard",
        cls=GPTSloop,
        arch_fields=SLOOP_ARCH_FIELDS,
    ),
    # When v2 is ready: uncomment, import GPT as GPTBrig from .brig, rename
    # codename + hf_repo below. Nothing else in the codebase needs to change.
    #
    # "brig": ModelSpec(
    #     dispatch_key="brig",
    #     codename="Brig",
    #     hf_repo="younissk/nanoBeard-Brig",
    #     cls=GPTBrig,
    #     arch_fields=BRIG_ARCH_FIELDS,
    # ),
}


def build_model(cfg: Config) -> nn.Module:
    return MODEL_REGISTRY[cfg.model_name].cls(cfg)


def spec_for(cfg: Config) -> ModelSpec:
    return MODEL_REGISTRY[cfg.model_name]
