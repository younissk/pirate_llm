"""Player models — anything with `.name` and `.generate(prompt) -> str`.

A "model" here is a *configured* generator: the same backing LLM with two
different temperatures is two different players. Players are dataclasses
with frozen fields so the registered config can't drift mid-run. Pick a
`name` that captures the full parameter signature — e.g. `gpt-4o-t0.7`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class Model(Protocol):
    name: str

    def generate(self, prompt: str) -> str: ...


@dataclass(frozen=True)
class EchoModel:
    """Placeholder model: returns the prompt unchanged. Useful for smoke tests."""

    name: str = "echo"

    def generate(self, prompt: str) -> str:
        return prompt


# Pinned OpenAI snapshot — `gpt-4o` (no date) silently rolls forward.
GPT_4O_PIN = "gpt-4o-2024-11-20"


@dataclass(frozen=True)
class OpenAIModel:
    """OpenAI chat-completions player. Needs `OPENAI_API_KEY` in the env.

    Parameters are frozen after construction — a different temperature or
    system prompt is a different player. Register each variant under a name
    that encodes the parameters, e.g. `gpt-4o-t0.7`.
    """

    name: str
    model: str = GPT_4O_PIN
    temperature: float = 0.7
    max_tokens: int = 256
    system: str | None = None

    def generate(self, prompt: str) -> str:
        from openai import OpenAI

        client = OpenAI()
        messages: list[dict[str, str]] = []
        if self.system:
            messages.append({"role": "system", "content": self.system})
        messages.append({"role": "user", "content": prompt})
        resp = client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=messages,
        )
        return resp.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# nanoBeard — local PyTorch checkpoints from this repo
# ---------------------------------------------------------------------------

# Module-level cache so each (model, tokenizer) is loaded once even if many
# NanoBeardModel instances share the same checkpoint. Per-entry lock because
# local PyTorch inference is not safe to call concurrently from threads.
_NANOBEARD_CACHE: dict[str, tuple] = {}


@dataclass(frozen=True)
class NanoBeardModel:
    """A nanoBeard checkpoint as an arena player.

    Generation params (`temperature`, `top_k`, `max_new_tokens`) are part of
    the player identity — same checkpoint at temp=0.2 and temp=0.9 are two
    different players. Encode the params in `name`.
    """

    name: str
    ckpt_path: str
    tokenizer_path: str
    device: str = "cpu"
    max_new_tokens: int = 128
    temperature: float = 0.8
    top_k: int | None = 40

    def _load(self):
        import threading

        if self.name not in _NANOBEARD_CACHE:
            import torch  # noqa: F401  (warm import side-effects)
            from tokenizers import Tokenizer

            from nanobeard.sample import load_checkpoint

            model = load_checkpoint(self.ckpt_path, self.device, tokenizer_path=self.tokenizer_path)
            tokenizer = Tokenizer.from_file(self.tokenizer_path)
            _NANOBEARD_CACHE[self.name] = (model, tokenizer, threading.Lock())
        return _NANOBEARD_CACHE[self.name]

    def generate(self, prompt: str) -> str:
        import torch

        from nanobeard.sample import generate as _gen

        model, tokenizer, lock = self._load()
        prompt_ids = tokenizer.encode(prompt).ids
        idx = torch.tensor(prompt_ids, dtype=torch.long, device=self.device).unsqueeze(0)
        with lock:
            out = _gen(
                model,
                idx,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_k=self.top_k,
            )
        new_ids = out[0].cpu().tolist()[len(prompt_ids) :]
        return tokenizer.decode(new_ids)


def nanobeard_from_config(
    name: str,
    config_path: str,
    ckpt_path: str | None = None,
    device: str | None = None,
    **gen_overrides,
) -> NanoBeardModel:
    """Build a NanoBeardModel from a `configs/*.py` file.

    `gen_overrides` is forwarded to `NanoBeardModel` — pass `temperature`,
    `top_k`, `max_new_tokens` here to register parameter-varied players from
    the same checkpoint.
    """
    from nanobeard.config import load_config

    cfg = load_config(config_path)
    return NanoBeardModel(
        name=name,
        ckpt_path=ckpt_path or cfg.ckpt_path,
        tokenizer_path=cfg.tokenizer_path,
        device=device or cfg.device,
        **gen_overrides,
    )


REGISTRY: dict[str, Model] = {}


def register(model: Model) -> None:
    if model.name in REGISTRY and REGISTRY[model.name] != model:
        raise ValueError(
            f"Model name '{model.name}' already registered with different config. "
            f"Use a name that encodes the params (e.g. include temperature)."
        )
    REGISTRY[model.name] = model


register(EchoModel())
register(OpenAIModel(name="gpt-4o-t0.0", temperature=0.0))
register(OpenAIModel(name="gpt-4o-t0.7", temperature=0.7))

# ---------------------------------------------------------------------------
# Register your nanoBeard checkpoints here. Examples:
#
register(
    nanobeard_from_config(
        name="sloop-t0.8",
        config_path="configs/sloop.py",
        temperature=0.8,
    )
)
register(
    nanobeard_from_config(
        name="sloop-t0.3",
        config_path="configs/sloop.py",
        temperature=0.3,
    )
)
# ---------------------------------------------------------------------------
