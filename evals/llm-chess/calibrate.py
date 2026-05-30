"""Judge calibration against the gold subset.

For each prompt that carries a gold answer, generate a response from each
model, then ask each judge to compare gold vs model_response using the
same A/B swap as the arena. The judge "agrees with gold" when it prefers
the gold answer; "ties" when the swap disagrees or the judge says tie;
"misses" when it prefers the model output.

A judge that loses to gold often is either (a) wrong about the rubric or
(b) being beaten by the model under test. Either way, ranking based on
that judge is suspect.

Rule of thumb: > 30% miss rate on the gold subset → fix the judge before
trusting the leaderboard.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

load_dotenv()

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from arena import (  # noqa: E402
    JudgeCache,
    ResponseCache,
    _generate_with_cache,
    _judge_with_cache,
    consolidate_swap,
)
from judges import REGISTRY as JUDGES  # noqa: E402
from models import REGISTRY as MODELS  # noqa: E402

_console = Console(stderr=True)


@dataclass
class CalibrationRow:
    category: str
    judge: str
    judge_version: str
    model: str
    n: int
    agree: int  # judge preferred gold
    tie: int  # swap disagreed or judge said tie
    miss: int  # judge preferred model output over gold
    miss_rate: float


def load_gold_prompts(path: Path) -> list[dict]:
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        rec = json.loads(line)
        if rec.get("gold"):
            out.append(rec)
    return out


def calibrate(
    prompts_path: Path,
    model_names: list[str],
    judge_names: list[str],
    cache_path: Path | None = None,
    response_cache_path: Path | None = None,
    quiet: bool = False,
) -> list[CalibrationRow]:
    prompts = load_gold_prompts(prompts_path)
    models = [MODELS[n] for n in model_names]
    judges = [JUDGES[n] for n in judge_names]
    cache = JudgeCache(cache_path) if cache_path else None
    rcache = ResponseCache(response_cache_path) if response_cache_path else None

    counts: dict[tuple[str, str, str, str], dict[str, int]] = defaultdict(
        lambda: {"agree": 0, "tie": 0, "miss": 0, "n": 0}
    )

    total_units = len(prompts) * len(models) * len(judges)

    if not quiet:
        _console.rule(
            f"[bold cyan]calibrate[/]  ·  {len(prompts)} gold prompts  ·  "
            f"{len(models)} models  ·  {len(judges)} judges"
        )
        _console.log(f"[dim]models:[/] {', '.join(m.name for m in models)}")
        _console.log(
            "[dim]judges:[/] "
            + ", ".join(f"{j.name}@{j.version}" for j in judges)
        )
        if cache is not None:
            _console.log(f"[dim]judge cache:[/] {cache.path} (prior entries: {len(cache.mem)})")
        if rcache is not None:
            _console.log(
                f"[dim]response cache:[/] {rcache.path} (prior entries: {len(rcache.mem)})"
            )

    progress = None
    task = None
    if not quiet:
        progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn(
                "[dim]· agree {task.fields[agree]} miss {task.fields[miss]} "
                "tie {task.fields[tie]}[/]"
            ),
            TimeElapsedColumn(),
            console=_console,
        )
        progress.start()
        task = progress.add_task(
            "judging gold", total=total_units, agree=0, miss=0, tie=0
        )

    agg = {"agree": 0, "miss": 0, "tie": 0}
    try:
        for p in prompts:
            cat = p["category"]
            gold = p["gold"]
            for m in models:
                resp = _generate_with_cache(m, p["prompt"], rcache)
                for j in judges:
                    fwd = _judge_with_cache(j, p["prompt"], gold, resp, cache)
                    rev = _judge_with_cache(j, p["prompt"], resp, gold, cache)
                    winner_slot, _ = consolidate_swap(fwd, rev)
                    key = (cat, j.name, j.version, m.name)
                    c = counts[key]
                    c["n"] += 1
                    if winner_slot == "a":  # forward A is gold
                        c["agree"] += 1
                        agg["agree"] += 1
                    elif winner_slot == "b":
                        c["miss"] += 1
                        agg["miss"] += 1
                    else:
                        c["tie"] += 1
                        agg["tie"] += 1
                    if progress is not None and task is not None:
                        progress.update(
                            task,
                            advance=1,
                            description=f"[dim]{p['id']}[/] {j.name} vs {m.name}",
                            agree=agg["agree"],
                            miss=agg["miss"],
                            tie=agg["tie"],
                        )
    finally:
        if progress is not None:
            progress.stop()
        if cache is not None:
            cache.close()
        if rcache is not None:
            rcache.close()

    rows = []
    for (cat, jname, jver, mname), c in counts.items():
        rows.append(
            CalibrationRow(
                category=cat,
                judge=jname,
                judge_version=jver,
                model=mname,
                n=c["n"],
                agree=c["agree"],
                tie=c["tie"],
                miss=c["miss"],
                miss_rate=round(c["miss"] / c["n"], 3) if c["n"] else 0.0,
            )
        )
    rows.sort(key=lambda r: (r.judge, r.category, r.model))
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prompts", type=Path, default=HERE / "prompts.jsonl")
    p.add_argument(
        "--models",
        nargs="+",
        required=True,
        help=f"Model names. Available: {sorted(MODELS)}",
    )
    p.add_argument(
        "--judges",
        nargs="+",
        required=True,
        help=f"Judge names. Available: {sorted(JUDGES)}",
    )
    p.add_argument("--cache", type=Path, default=HERE / "results" / "judge_cache.jsonl")
    p.add_argument("--no-cache", action="store_true")
    p.add_argument(
        "--response-cache",
        type=Path,
        default=HERE / "results" / "response_cache.jsonl",
    )
    p.add_argument("--no-response-cache", action="store_true")
    p.add_argument("--out", type=Path, default=HERE / "results" / "calibration.jsonl")
    args = p.parse_args()

    rows = calibrate(
        prompts_path=args.prompts,
        model_names=args.models,
        judge_names=args.judges,
        cache_path=None if args.no_cache else args.cache,
        response_cache_path=None if args.no_response_cache else args.response_cache,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for r in rows:
            f.write(json.dumps(asdict(r)) + "\n")

    csv_out = args.out.with_suffix(".csv")
    col_names = [f.name for f in fields(CalibrationRow)]
    with csv_out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(col_names)
        for r in rows:
            w.writerow([getattr(r, c) for c in col_names])

    table = Table(title="Judge calibration vs gold", title_style="bold cyan")
    table.add_column("judge", style="cyan")
    table.add_column("version", style="dim")
    table.add_column("category")
    table.add_column("model", style="magenta")
    table.add_column("n", justify="right")
    table.add_column("agree", justify="right", style="green")
    table.add_column("tie", justify="right", style="yellow")
    table.add_column("miss", justify="right", style="red")
    table.add_column("miss%", justify="right")
    for r in rows:
        pct = f"{r.miss_rate * 100:.1f}%"
        pct_styled = f"[bold red]{pct} !!![/]" if r.miss_rate > 0.30 else pct
        table.add_row(
            r.judge,
            r.judge_version,
            r.category,
            r.model,
            str(r.n),
            str(r.agree),
            str(r.tie),
            str(r.miss),
            pct_styled,
        )
    _console.print(table)
    _console.log(f"[dim]full rows →[/] {args.out}")
    _console.log(f"[dim]csv      →[/] {csv_out}")
    _console.log("[dim]rows flagged [bold red]!!![/] when miss-rate > 30% — judge is the bottleneck[/]")


if __name__ == "__main__":
    main()
