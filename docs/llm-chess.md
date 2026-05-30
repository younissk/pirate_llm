# LLM-chess arena

A round-robin-ish tournament where models answer the same prompts and a
judge model picks the winner of each pair. Wins/losses/ties feed a
chess-style Elo so many noisy pairwise verdicts collapse into a single
comparable rating per model — with bootstrap confidence intervals so the
number isn't read as more precise than it is.

Lives in `evals/llm-chess/`. Separate from perplexity / sample-gallery —
those measure a model against a reference; this measures models against
**each other**.

## Why Elo

Pairwise preference is the cheapest signal a judge can give: "A or B?"
Elo turns that stream of binary outcomes into a rating that is

- **transitive-ish** — beating strong opponents matters more than beating weak ones,
- **incremental** — new models slot in without re-judging the old matches,
- **interpretable** — a 100-point gap ≈ 64% expected win rate.

K = 8 (not the chess default of 32). LLM arenas play far fewer games per
player than chess players do; K=32 would mean ±300 points of swing from
variance alone after ~60 matches. K=8 stabilises ratings on the same match
budget. Base rating = 1500.

## Why a `random` judge ships first

The random judge picks A or B by coin flip and returns an empty reason.
Combined with the A/B swap (see below), ~50% of random verdicts collapse
to ties, so random produces a slow random walk in Elo rather than a fast
one. It is the **baseline a real judge must beat**: if a "smart" judge can't
separate models better than random, the judge is broken.

## Judge rubric

Every non-random judge is sent the same rubric (`judges.JUDGE_RUBRIC`).
The instruction is explicit:

> Ignore length. Ignore formatting. Judge correctness and helpfulness only.

Verbosity bias is the most common failure mode of LLM-as-judge — long,
confident, wrong answers beat short correct ones. Telling the judge to
ignore length and formatting up front cuts that bias.

Four criteria, applied in order, then one overall winner (or `tie` if
equivalent in substance):

| # | Criterion | What it catches |
|---|-----------|-----------------|
| 1 | Correctness | Factually wrong claims |
| 2 | Relevance | Off-topic answers |
| 3 | Completeness | Missing key points |
| 4 | Hallucination | Invented facts, fake citations, made-up names |

Pirate-style voice is no longer in the rubric — for reasoning/factual
prompts a pirate accent actively obscures the substance, and the model
under test is the one we want pirate-styled, not the judges.

Output is strict JSON: `{"winner": "A" | "B" | "tie", "reason": "<one sentence>"}`.

## A/B-swap to kill position bias

LLM judges have a slot bias — they prefer "A" or "B" position regardless
of content. The rubric does not fix this; the judge code does.

For every pair `(model_a, model_b)` and every judge:

1. Forward call: judge sees `response_a` as slot A, `response_b` as slot B.
2. Reverse call: judge sees `response_b` as slot A, `response_a` as slot B.
3. Consolidate:
   - Both calls pick the same *model* → that model wins.
   - Calls pick different models, or either call says `"tie"` → **tie**.

Costs 2x judge calls, kills the dominant noise source. The judge cache
keys forward and reverse separately, so both calls dedupe across reruns.

Side effect: random judge produces ~50% ties through this mechanism, which
correctly identifies it as carrying no information.

## Response cache

Stochastic models (temperature > 0) produce a different sample on every
call. Re-running the tournament without dedup would re-roll every
response, burning OpenAI tokens and local compute and changing the
leaderboard purely from sampling noise.

`ResponseCache` is an append-only JSONL keyed by

```
sha256(model_name + prompt)
```

The contract is enforced by the registry: same `name` = same frozen
config = same player. The cached response is therefore a valid replay of
"what this player said on this prompt." This freezes one sample per
`(model, prompt)`; reruns reuse it.

Default path: `evals/llm-chess/results/response_cache.jsonl`.
Disable with `--no-response-cache` to force fresh generations.

To force a single model to re-roll, either rename it (encode a version
suffix in the name) or delete the cache file.

## Judge call cache

`JudgeCache` is an append-only JSONL keyed by

```
sha256(judge_name + judge_version + prompt + response_a + response_b)
```

Judges at temperature 0 are effectively deterministic, so reruns must not
re-bill old verdicts. Default path:
`evals/llm-chess/results/judge_cache.jsonl`. Disable with `--no-cache`.

`judge_version` is in the cache key on purpose: a new judge release misses
every prior entry and forces fresh verdicts. By design.

## Pinning judges and not mixing versions

OpenAI's `gpt-4o` (no date) is a moving alias — the model behind it
changes over time, silently. `OpenAIJudge` and `OpenAIModel` both pin to a
dated snapshot:

```python
GPT_4O_PIN = "gpt-4o-2024-11-20"
```

Every `Match` row stores `judge` + `judge_version`. `compute_ratings()`
scans the log on read and **warns** if any single judge name appears with
more than one version. Mixing two judge releases gives you a number with
no meaning.

**Policy: new judge release = new eval run, not mixed.**

## Models as parameter-locked players

A "model" in this arena is a *configured generator* — the same backing
LLM at two temperatures is two different players. `Model` dataclasses are
`frozen=True`, the registry rejects re-registering a name with a
different config, and the convention is to put parameter info in the
name itself:

```python
register(OpenAIModel(name="gpt-4o-t0.0", model=GPT_4O_PIN, temperature=0.0))
register(OpenAIModel(name="gpt-4o-t0.7", model=GPT_4O_PIN, temperature=0.7))
```

Now you can run the same backing model against itself at different
temperatures and the leaderboard treats them as separate players —
exactly what you want for "does temp=0.2 beat temp=0.7 on creative
prompts?".

There is no default pirate system prompt on `OpenAIModel`. If you want a
pirate-styled baseline, register one explicitly:

```python
register(OpenAIModel(
    name="gpt-4o-t0.7-pirate",
    temperature=0.7,
    system="You are a pirate. Reply in pirate voice.",
))
```

## Prompt set

`prompts.jsonl` ships 45 prompts across three categories:

| Category | Count | What it tests |
|----------|-------|---------------|
| `factual` | 5 | Easy nautical / pirate-era facts, all with gold. |
| `creative` | 20 | Shanties, dialogue, scenes, story continuations, letters, curses, epitaphs. |
| `instruction_following` | 20 | Strict formats — exact strings, JSON, YAML, markdown, exact word counts, regex-like constraints. |

Reasoning prompts were dropped. The goal of these LLM families is
creative voice and instruction compliance, not arithmetic.

Each record:

```json
{"id": "F1", "category": "factual", "prompt": "...", "gold": "..." | null}
```

### Gold subset (calibration)

12 of the 45 prompts carry a `gold` reference answer:

- All 5 factual prompts (F1-F5).
- 7 instruction-following prompts with verifiable formats (I1, I2, I4, I5, I11, I15, I18).

Creative prompts have no gold — taste is not gradable.

Gold is consumed by `calibrate.py`:

```bash
uv run python evals/llm-chess/calibrate.py \
    --models gpt-4o-t0.0 \
    --judges gpt-4o-judge
```

For each gold prompt × each model × each judge, calibrate plays a single
A/B-swapped match of `(gold, model_response)`. A judge that prefers the
model output over the gold answer counts as a **miss**. Per-category miss
rate over 30% flags the judge.

> If miss rate is > 30% on the gold subset, fix the judge prompt or swap
> the judge model before reading anything into the leaderboard.

## Pieces

| File | Job |
|------|-----|
| `elo.py` | `expected()` + `update()`. K=8, base=1500. |
| `models.py` | `Model` protocol, `EchoModel`, `OpenAIModel`, `NanoBeardModel` (frozen), `REGISTRY`. |
| `judges.py` | `Judge` protocol, `Verdict`, `JUDGE_RUBRIC` (4 criteria + tie), `RandomJudge`, `OpenAIJudge`. |
| `arena.py` | `Match` (incl. `verdict_forward`/`verdict_reverse`), `JudgeCache`, `consolidate_swap()`, `run_swiss()`. |
| `leaderboard.py` | `compute_ratings()` with bootstrap 95% CIs and mixed-version filter. |
| `calibrate.py` | Judge ↔ gold miss-rate per category. |
| `export.py` | JSONL → CSV (`matches`, `leaderboard`, `head_to_head`). |
| `run.py` | CLI: `--prompts --models --judges --rounds --workers --no-cache --judge-versions --n-boot --csv-dir --no-csv`. |
| `prompts.jsonl` | 45 prompts; `factual`, `creative`, `instruction_following`. |
| `results/matches.jsonl` | Append-only match log. |
| `results/judge_cache.jsonl` | Append-only judge verdict cache. |
| `results/response_cache.jsonl` | Append-only `(model, prompt) → response` cache. |
| `results/calibration.jsonl` + `.csv` | Output of the calibrate command. |
| `results/matches.csv` | Flat match rows (auto-written by `run.py`). |
| `results/leaderboard.csv` | Per-model Elo + 95% CI + W/L/T/N. |
| `results/head_to_head.csv` | Square pairwise `wins-losses-ties/n` matrix. |

## Data flow

```
prompts.jsonl
     │
     ▼
 for each prompt:
   model.generate(prompt)            ← parallel, once per model per prompt
     │
     ▼
   for K Swiss rounds:
     sort models by current rating, pair nearest unplayed opponents
     for each pair (A, B), for each judge:
       forward = judge(prompt, resp_A, resp_B)   ← cache by sha256(...)
       reverse = judge(prompt, resp_B, resp_A)   ← cache, separate key
       winner  = consolidate_swap(forward, reverse)   ← A / B / tie
       update Elo immediately so next round pairs better
       append Match{..., judge_version, verdict_forward, verdict_reverse}
                 to matches.jsonl

leaderboard.compute_ratings(matches.jsonl, n_boot=200)
   → warns on mixed judge versions, replays, returns rows with 95% CI

calibrate.calibrate(prompts.jsonl, ...)
   → judge gold vs each model_response with the same A/B swap
   → report agree / tie / miss per (judge, category, model)
```

## Running

```bash
# random judge — sanity check
uv run python evals/llm-chess/run.py \
    --models echo gpt-4o-t0.0 \
    --judges random \
    --rounds 3

# OpenAI judge, pinned snapshot, dedicated log
uv run python evals/llm-chess/run.py \
    --models gpt-4o-t0.0 gpt-4o-t0.7 \
    --judges gpt-4o-judge \
    --rounds 3 \
    --workers 8 \
    --out evals/llm-chess/results/matches.gpt-4o-2024-11-20.jsonl

# calibrate the judge against gold
uv run python evals/llm-chess/calibrate.py \
    --models gpt-4o-t0.0 \
    --judges gpt-4o-judge

# replay a log filtered to one judge version
uv run python evals/llm-chess/run.py \
    --models gpt-4o-t0.0 gpt-4o-t0.7 --judges gpt-4o-judge \
    --judge-versions gpt-4o-judge=gpt-4o-2024-11-20
```

Flags: `--rounds`, `--workers`, `--cache` / `--no-cache`,
`--judge-versions JUDGE=VERSION`, `--n-boot`, `--seed`, `--csv-dir`,
`--no-csv`.

## CSV exports

By default `run.py` writes three CSVs next to `matches.jsonl`:

| File | Shape |
|------|-------|
| `matches.csv` | One row per match; flat columns including `judge_version`, `verdict_forward`, `verdict_reverse`, `reason`. |
| `leaderboard.csv` | One row per model: `rating`, `ci_low`, `ci_high`, `wins`, `losses`, `draws`, `games`. |
| `head_to_head.csv` | Square matrix indexed by `row_model`; each cell is `wins-losses-ties/n` from that row's perspective. |

`calibrate.py` writes `calibration.csv` alongside its JSONL.

For replay or after-the-fact exports against any existing log:

```bash
uv run python evals/llm-chess/export.py \
    --matches evals/llm-chess/results/matches.sloop.jsonl \
    --out-dir evals/llm-chess/results/sloop/
```

Disable auto-CSV with `--no-csv` on `run.py` if you only want JSONL.

## Benchmarking a nanoBeard checkpoint

`NanoBeardModel` wraps a local nanoBeard checkpoint as an arena player.
Each `(ckpt × temperature × top_k)` is a separate player — encode the
params in the name.

Edit the bottom of `evals/llm-chess/models.py`:

```python
register(nanobeard_from_config(
    name="sloop-t0.8",
    config_path="configs/sloop.py",
    temperature=0.8,
    top_k=40,
    max_new_tokens=128,
))
register(nanobeard_from_config(
    name="sloop-t0.3",
    config_path="configs/sloop.py",
    temperature=0.3,
))
# Override ckpt + device if needed:
# register(nanobeard_from_config(
#     name="sloop-best-t0.8",
#     config_path="configs/sloop.py",
#     ckpt_path="runs/sloop/ckpt-best.pt",
#     device="mps",
#     temperature=0.8,
# ))
```

Then run:

```bash
# 1. Calibrate the judge on the gold subset first
uv run python evals/llm-chess/calibrate.py \
    --models sloop-t0.8 \
    --judges gpt-4o-judge

# 2. If miss-rate < 30%, run the tournament
uv run python evals/llm-chess/run.py \
    --models sloop-t0.8 sloop-t0.3 gpt-4o-t0.7 \
    --judges gpt-4o-judge \
    --rounds 3 \
    --out evals/llm-chess/results/matches.sloop.jsonl
```

Notes:

- Local PyTorch inference is serialised per-checkpoint via a lock — it is
  not safe to call from threads. OpenAI calls still run in parallel.
- The checkpoint is loaded once and reused across all matches in a run,
  regardless of how many `NanoBeardModel` instances share it.
- Generated text excludes the prompt — only the model's new tokens are
  returned, so the judge compares completions, not echoes.

## Adding a model

1. In `models.py`, define a frozen dataclass with `name: str` and
   `generate(prompt) -> str`.
2. Encode params in the name (e.g. `mistral-large-t0.3`).
3. `register(YourModel(...))` at module bottom.

The registry refuses to overwrite a name with a different config —
parameter drift between runs is treated as a bug.

## Adding a judge

1. In `judges.py`, define a dataclass with `name`, `version`, and
   `judge(prompt, response_a, response_b) -> Verdict`.
2. Use `render_judge_prompt(prompt, a, b)` so the rubric stays consistent.
3. `register(YourJudge(...))`.

## Caveats

- **Self-preference**: an OpenAI judge grading OpenAI players can bias
  toward its own family. Use a different family for the judge when
  comparing OpenAI models seriously.
- **Sample size = 1 per (model, prompt)**: stochastic models produce
  different responses each run. Treat the leaderboard as one sample
  point, not the truth. CIs help.
- **Cost**: with Swiss + swap the per-prompt call count is
  `2 · min(rounds · M/2, C(M,2)) · |judges|`, bounded by 2x round-robin.
  Cache cuts reruns to zero new calls.
- **One log, one judge version**: don't append matches from two judge
  releases to the same file. Use a per-version path and rely on the
  mixed-version warning.
