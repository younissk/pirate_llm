"""Eval harness: perplexity math + gallery rendering."""

from __future__ import annotations

import math
from pathlib import Path

from tokenizers import Tokenizer

from nanobeard.config import Config
from nanobeard.eval.gallery import GallerySample, load_prompts, render_markdown, run_gallery
from nanobeard.eval.perplexity import compute_perplexity
from nanobeard.models import build_model


def test_perplexity_returns_finite_metrics(synthetic_bins: Config):
    model = build_model(synthetic_bins)
    res = compute_perplexity(model, synthetic_bins, n_batches=3)
    assert math.isfinite(res.loss)
    assert res.perplexity == math.exp(res.loss)
    assert res.n_tokens > 0
    assert res.n_batches == 3


def test_perplexity_uses_specified_bin(synthetic_bins: Config):
    """If bin_path is provided, should override config.val_bin."""
    model = build_model(synthetic_bins)
    res = compute_perplexity(model, synthetic_bins, bin_path=synthetic_bins.train_bin, n_batches=2)
    assert res.n_batches == 2


def test_load_prompts_skips_blank_and_comments(tmp_path: Path):
    f = tmp_path / "p.jsonl"
    f.write_text('{"prompt": "first"}\n\n# this is a comment\n{"prompt": "second"}\n')
    prompts = load_prompts(f)
    assert len(prompts) == 2
    assert prompts[0]["prompt"] == "first"
    assert prompts[1]["prompt"] == "second"


def test_load_default_prompts():
    """Project's bundled prompts file must be valid JSONL."""
    prompts = load_prompts("evals/prompts.jsonl")
    assert len(prompts) > 0
    for p in prompts:
        assert "prompt" in p


def test_run_gallery_produces_samples(tokenized_cfg: Config):
    model = build_model(tokenized_cfg).eval()
    tok = Tokenizer.from_file(tokenized_cfg.tokenizer_path)
    prompts = [{"prompt": "ahoy"}, {"prompt": "matey"}]
    samples = run_gallery(model, tok, prompts, device="cpu", max_new_tokens=3, top_k=2)
    assert len(samples) == 2
    for s in samples:
        assert isinstance(s, GallerySample)
        assert s.n_output_tokens > 0


def test_render_markdown_includes_all_prompts():
    samples = [
        GallerySample(prompt="ahoy", completion="ahoy matey", n_input_tokens=1, n_output_tokens=2),
        GallerySample(prompt="yarr", completion="yarr there", n_input_tokens=1, n_output_tokens=2),
    ]
    md = render_markdown("nanoBeard Test (1M params)", samples)
    assert "nanoBeard Test" in md
    assert "ahoy" in md
    assert "yarr" in md
    assert "input tokens: 1" in md
