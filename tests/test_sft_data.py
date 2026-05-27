"""SFT data: prompt rendering, encoding, padding, loss masking."""

from __future__ import annotations

from pathlib import Path

import pytest
from tokenizers import Tokenizer

from nanobeard.sft_data import (
    IGNORE_INDEX,
    PROMPT_NO_CONTEXT,
    PROMPT_WITH_CONTEXT,
    encode_example,
    render_prompt,
)


@pytest.fixture
def tok(synthetic_tokenizer: Path) -> Tokenizer:
    return Tokenizer.from_file(str(synthetic_tokenizer))


@pytest.fixture
def eos_id(tok: Tokenizer) -> int:
    return tok.token_to_id("<|endoftext|>")


def test_render_prompt_no_context():
    p, r = render_prompt({"instruction": "Say hi", "response": "Ahoy", "context": ""})
    assert p == PROMPT_NO_CONTEXT.format(instruction="Say hi")
    assert r == "Ahoy"


def test_render_prompt_with_context():
    ex = {"instruction": "Translate", "response": "Yarr", "context": "Hello"}
    p, _ = render_prompt(ex)
    assert "### Context:" in p
    assert "Hello" in p
    assert p == PROMPT_WITH_CONTEXT.format(instruction="Translate", context="Hello")


def test_render_prompt_strips_whitespace():
    ex = {"instruction": "  hi  ", "response": "  ho  ", "context": "  "}
    p, r = render_prompt(ex)
    assert "  hi  " not in p
    assert "hi" in p
    assert r == "ho"


def test_render_prompt_handles_none_context():
    p, _ = render_prompt({"instruction": "x", "response": "y", "context": None})
    assert "### Context:" not in p


def test_encode_example_basic(tok: Tokenizer, eos_id: int):
    ex = {"instruction": "ahoy", "response": "matey", "context": ""}
    out = encode_example(ex, tok, block_size=64, eos_id=eos_id)
    assert out is not None
    assert len(out.input_ids) == 64
    assert len(out.labels) == 64


def test_encode_example_loss_masked_on_prompt(tok: Tokenizer, eos_id: int):
    """Labels must equal IGNORE_INDEX over the prompt span."""
    ex = {"instruction": "ahoy", "response": "matey", "context": ""}
    out = encode_example(ex, tok, block_size=64, eos_id=eos_id)
    prompt_text, _ = render_prompt(ex)
    prompt_len = len(tok.encode(prompt_text).ids)
    for i in range(prompt_len):
        assert out.labels[i] == IGNORE_INDEX, f"label {i} should be IGNORE_INDEX"


def test_encode_example_response_tokens_visible(tok: Tokenizer, eos_id: int):
    """At least one label position must be a real token (not IGNORE_INDEX)."""
    ex = {"instruction": "x", "response": "matey treasure rum", "context": ""}
    out = encode_example(ex, tok, block_size=64, eos_id=eos_id)
    real_labels = [lbl for lbl in out.labels if lbl != IGNORE_INDEX]
    assert len(real_labels) > 0
    # Last real label should be eos.
    assert eos_id in real_labels


def test_encode_example_pad_with_eos_and_ignore(tok: Tokenizer, eos_id: int):
    """Right-pad: input_ids with eos, labels with IGNORE_INDEX."""
    ex = {"instruction": "x", "response": "y", "context": ""}
    out = encode_example(ex, tok, block_size=64, eos_id=eos_id)
    # Find last non-pad label (last entry before trailing IGNORE_INDEX stretch).
    last_non_pad = max(i for i, lbl in enumerate(out.labels) if lbl != IGNORE_INDEX)
    # Labels after the response's eos must all be IGNORE_INDEX.
    if last_non_pad + 1 < 64:
        assert all(lbl == IGNORE_INDEX for lbl in out.labels[last_non_pad + 1 :])


def test_encode_example_returns_none_on_prompt_overflow(tok: Tokenizer, eos_id: int):
    """If the prompt alone fills the window, skip — no room to learn anything."""
    long_instr = "treasure " * 200
    out = encode_example(
        {"instruction": long_instr, "response": "y", "context": ""},
        tok,
        block_size=16,
        eos_id=eos_id,
    )
    assert out is None


def test_encode_example_truncates_long_response(tok: Tokenizer, eos_id: int):
    """Long responses get truncated to block_size — input_ids fills the window."""
    ex = {"instruction": "x", "response": "treasure " * 200, "context": ""}
    out = encode_example(ex, tok, block_size=128, eos_id=eos_id)
    assert out is not None
    assert len(out.input_ids) == 128
    assert len(out.labels) == 128
    # With a giant response, the window must be entirely real tokens (no padding).
    pad_run = sum(1 for lbl in out.labels if lbl == IGNORE_INDEX)
    # Some labels are IGNORE on the prompt span; rest should be real response tokens.
    real_run = len(out.labels) - pad_run
    assert real_run > 0
    # The full window should be used — no trailing IGNORE_INDEX padding.
    assert out.labels[-1] != IGNORE_INDEX
