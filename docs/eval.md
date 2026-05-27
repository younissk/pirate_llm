# Eval harness

Two complementary signals.

## Perplexity (quantitative)

`compute_perplexity(model, config, n_batches=200)` averages cross-entropy
over `n_batches` random crops of `config.val_bin`. Same sampling strategy as
`get_batch`, so the metric is comparable to the `val/loss` the training loop
reports.

```python
from nanobeard.eval.perplexity import compute_perplexity
res = compute_perplexity(model, cfg)
print(res.loss, res.perplexity, res.n_tokens)
```

PPL is comparable **only within the same tokenizer + same eval corpus.**
Treat cross-tokenizer comparisons as suspect — a model with a smaller vocab
sees fewer choices per step and can artificially win on PPL.

## Sample gallery (qualitative)

`run_gallery(model, tokenizer, prompts)` generates completions for a fixed
prompt list and renders markdown that diffs cleanly between model versions.

`evals/prompts.jsonl` ships a starter set covering:

- raw story continuations
- dialogue prompts
- SFT-format instructions
- fantasy beats

Add prompts you care about as JSONL lines: `{"prompt": "...", "category": "..."}`.

## Running

```bash
make eval CONFIG=sloop         # perplexity + gallery -> evals/results/<date>/sloop/
make eval-quick CONFIG=sloop   # faster pass for in-progress runs
```

Outputs per model:

- `gallery.md` — one section per prompt, completion in a fenced block
- `metrics.json` — `{display_name, model_name, ckpt, perplexity, ...}`

## Comparing versions

The eval harness writes one directory per model per date.
Stage 1 — eyeball:

```bash
diff -r evals/results/2026-05-27/sloop/ evals/results/2026-05-27/brig/
```

Stage 2 (TODO) — aggregated report. See `docs/todo.md`.
