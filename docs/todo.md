# Setup TODOs

Project state as of 2026-05-27 — captured here so future-you (and Claude in
follow-up sessions) can pick this up without re-deriving context.

## Deferred — wired locally, not yet on CI

- [ ] **GitHub Actions CI.** `pyproject.toml` has full `[tool.pytest.ini_options]`
      with markers and the Makefile exposes `test`, `test-slow`, `test-all`.
      The only missing piece is `.github/workflows/test.yml` running
      `uv sync --dev && uv run pytest` plus a step for `pre-commit run -a` and
      `mkdocs build --strict`. User explicitly chose to defer.
- [ ] Required check on `main` once CI is green.
- [ ] Dependabot or Renovate for `uv.lock` upkeep.

## Half-built — scaffolded, needs flesh

- [ ] **Eval harness** under `nanobeard/eval/`. Perplexity + sample gallery
      runners exist as runnable stubs. Need: a real held-out corpus per model
      version (not just `val.bin`), richer prompt set, and an aggregate report
      comparing v1 vs v2.
- [ ] **mkdocs site deployment.** Local build works (`make docs-serve`). Hosting
      not configured — pick GitHub Pages or HF Spaces docs.

## Nice-to-have, when scale demands

- [ ] Ckpt migration script (drop the `training.config` shim long-term).
- [ ] Sample regression test — store golden samples per release tag.
- [ ] Docker image pinned for Vast.ai (CUDA + torch versions).
- [ ] DVC or hash-based data versioning for the bins.
- [ ] HF Hub model-card auto-gen from `training_metadata.json`.
