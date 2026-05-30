"""Replay matches.jsonl → Elo ratings, W/L/T counts, bootstrap CIs.

Detects mixed judge versions in the log and refuses to silently average
them — pass `judge_versions={...}` to filter, or rerun on a clean log.

Bootstrap CIs are reported because a single Elo number with K=8 and ~60
matches per model still has ±50 points of variance. A 30-point gap with
overlapping 95% CIs is not a real gap.
"""

from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from elo import DEFAULT_K, DEFAULT_RATING, update


@dataclass
class Row:
    model: str
    rating: float
    ci_low: float
    ci_high: float
    wins: int
    losses: int
    draws: int
    games: int


def _scan_versions(matches_path: Path) -> dict[str, set[str]]:
    versions: dict[str, set[str]] = defaultdict(set)
    for line in matches_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        m = json.loads(line)
        versions[m["judge"]].add(m.get("judge_version", ""))
    return versions


def _percentile(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    xs_s = sorted(xs)
    k = (len(xs_s) - 1) * p
    f = int(k)
    c = min(f + 1, len(xs_s) - 1)
    if f == c:
        return xs_s[f]
    return xs_s[f] + (k - f) * (xs_s[c] - xs_s[f])


def _score_a(match: dict, a: str, b: str) -> float | None:
    w = match["winner"]
    if w == a:
        return 1.0
    if w == b:
        return 0.0
    if w == "tie":
        return 0.5
    return None


def _replay(matches: list[dict], k: float, base: float) -> dict[str, float]:
    ratings: dict[str, float] = defaultdict(lambda: base)
    for m in matches:
        a, b = m["model_a"], m["model_b"]
        s = _score_a(m, a, b)
        if s is None:
            continue
        ratings[a], ratings[b] = update(ratings[a], ratings[b], s, k=k)
    return ratings


def compute_ratings(
    matches_path: str | Path,
    k: float = DEFAULT_K,
    base: float = DEFAULT_RATING,
    judge_versions: dict[str, str] | None = None,
    n_boot: int = 200,
    boot_seed: int = 0,
) -> list[Row]:
    """Replay matches, return sorted Row list with bootstrap 95% CIs.

    judge_versions: optional `{judge_name: version}` filter. When `None`,
    the log is scanned for mixed versions and a warning is printed to
    stderr — mixing two judge releases is a footgun.

    n_boot: bootstrap resamples for the CI. 0 disables (ci_low/ci_high = rating).
    """
    matches_path = Path(matches_path)

    if judge_versions is None:
        seen = _scan_versions(matches_path)
        mixed = {name: vs for name, vs in seen.items() if len(vs) > 1}
        if mixed:
            print(
                f"WARN: {matches_path} mixes judge versions: {mixed}. "
                f"Pass judge_versions={{...}} to filter, or start a fresh log.",
                file=sys.stderr,
            )

    matches: list[dict] = []
    for line in matches_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        m = json.loads(line)
        if judge_versions is not None:
            wanted = judge_versions.get(m["judge"])
            if wanted is not None and m.get("judge_version", "") != wanted:
                continue
        matches.append(m)

    wins: dict[str, int] = defaultdict(int)
    losses: dict[str, int] = defaultdict(int)
    draws: dict[str, int] = defaultdict(int)
    games: dict[str, int] = defaultdict(int)
    for m in matches:
        a, b, w = m["model_a"], m["model_b"], m["winner"]
        if w == a:
            wins[a] += 1
            losses[b] += 1
        elif w == b:
            wins[b] += 1
            losses[a] += 1
        elif w == "tie":
            draws[a] += 1
            draws[b] += 1
        games[a] += 1
        games[b] += 1

    point_ratings = _replay(matches, k=k, base=base)

    if n_boot > 0 and matches:
        rng = random.Random(boot_seed)
        boot_samples: dict[str, list[float]] = defaultdict(list)
        for _ in range(n_boot):
            sample = [matches[rng.randrange(len(matches))] for _ in matches]
            sample_ratings = _replay(sample, k=k, base=base)
            for name, r in sample_ratings.items():
                boot_samples[name].append(r)
        ci_low = {n: _percentile(s, 0.025) for n, s in boot_samples.items()}
        ci_high = {n: _percentile(s, 0.975) for n, s in boot_samples.items()}
    else:
        ci_low = dict(point_ratings)
        ci_high = dict(point_ratings)

    rows = [
        Row(
            model=name,
            rating=round(r, 1),
            ci_low=round(ci_low.get(name, r), 1),
            ci_high=round(ci_high.get(name, r), 1),
            wins=wins[name],
            losses=losses[name],
            draws=draws[name],
            games=games[name],
        )
        for name, r in point_ratings.items()
    ]
    rows.sort(key=lambda row: row.rating, reverse=True)
    return rows
