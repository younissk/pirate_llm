"""Export matches.jsonl into CSV views.

Writes three side-by-side files for spreadsheet / pandas consumption:

- `matches.csv`       — one row per match, all columns flat.
- `leaderboard.csv`   — one row per model: elo, 95% CI, W/L/T/N.
- `head_to_head.csv`  — square matrix `wins-losses-ties/n` per pair.

Standalone: `uv run python evals/llm-chess/export.py --matches ...`. Also
called by `run.py` after a tournament so CSVs land next to the JSONL.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

from rich.console import Console

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from leaderboard import compute_ratings  # noqa: E402

_console = Console(stderr=True)

MATCH_COLS = [
    "match_id",
    "ts",
    "prompt",
    "model_a",
    "model_b",
    "response_a",
    "response_b",
    "judge",
    "judge_version",
    "winner",
    "verdict_forward",
    "verdict_reverse",
    "reason",
]


def _load_matches(matches_path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in matches_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def export_matches(matches_path: Path, out: Path) -> int:
    rows = _load_matches(matches_path)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MATCH_COLS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return len(rows)


def export_leaderboard(
    matches_path: Path,
    out: Path,
    n_boot: int = 200,
    judge_versions: dict[str, str] | None = None,
) -> int:
    rows = compute_ratings(matches_path, n_boot=n_boot, judge_versions=judge_versions)
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["model", "rating", "ci_low", "ci_high", "wins", "losses", "draws", "games"]
        )
        for r in rows:
            w.writerow(
                [r.model, r.rating, r.ci_low, r.ci_high, r.wins, r.losses, r.draws, r.games]
            )
    return len(rows)


def export_head_to_head(matches_path: Path, out: Path) -> int:
    """Square `wins-losses-ties/n` matrix from the row model's perspective."""
    counts: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: {"wins": 0, "losses": 0, "ties": 0, "n": 0}
    )
    models: set[str] = set()
    for m in _load_matches(matches_path):
        a, b, w = m["model_a"], m["model_b"], m["winner"]
        models.add(a)
        models.add(b)
        if w == a:
            counts[(a, b)]["wins"] += 1
            counts[(b, a)]["losses"] += 1
        elif w == b:
            counts[(a, b)]["losses"] += 1
            counts[(b, a)]["wins"] += 1
        else:
            counts[(a, b)]["ties"] += 1
            counts[(b, a)]["ties"] += 1
        counts[(a, b)]["n"] += 1
        counts[(b, a)]["n"] += 1
    sorted_models = sorted(models)
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row_model"] + sorted_models)
        for row_m in sorted_models:
            cells: list[str] = []
            for col_m in sorted_models:
                if row_m == col_m:
                    cells.append("-")
                else:
                    c = counts[(row_m, col_m)]
                    cells.append(f"{c['wins']}-{c['losses']}-{c['ties']}/{c['n']}")
            w.writerow([row_m] + cells)
    return len(sorted_models)


def export_all(
    matches_path: Path,
    out_dir: Path,
    n_boot: int = 200,
    judge_versions: dict[str, str] | None = None,
    quiet: bool = False,
) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "matches": out_dir / "matches.csv",
        "leaderboard": out_dir / "leaderboard.csv",
        "head_to_head": out_dir / "head_to_head.csv",
    }
    n_matches = export_matches(matches_path, paths["matches"])
    n_models = export_leaderboard(
        matches_path, paths["leaderboard"], n_boot=n_boot, judge_versions=judge_versions
    )
    n_h2h = export_head_to_head(matches_path, paths["head_to_head"])
    if not quiet:
        _console.log(
            f"[dim]CSV →[/] {paths['matches']} "
            f"({n_matches} matches, {n_models} models, {n_h2h}×{n_h2h} h2h)"
        )
    return paths


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--matches", type=Path, default=HERE / "results" / "matches.jsonl")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Default: same dir as --matches.",
    )
    p.add_argument("--n-boot", type=int, default=200)
    args = p.parse_args()

    out_dir = args.out_dir or args.matches.parent
    export_all(args.matches, out_dir, n_boot=args.n_boot)


if __name__ == "__main__":
    main()
