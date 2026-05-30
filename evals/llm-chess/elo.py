"""Chess-style Elo rating updates.

K is lower than chess defaults (32) because LLM arenas play far fewer games
per player than chess players do. With M=8 models and 40 prompts at 3 Swiss
rounds, each model plays ~60 matches — K=32 means ~600 points of swing from
variance alone. K=8 gives stable ratings with the same match budget.
"""

from __future__ import annotations

DEFAULT_RATING = 1500.0
DEFAULT_K = 8.0


def expected(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400))


def update(
    rating_a: float,
    rating_b: float,
    score_a: float,
    k: float = DEFAULT_K,
) -> tuple[float, float]:
    """Return new (rating_a, rating_b). score_a: 1=A win, 0=B win, 0.5=draw."""
    ea = expected(rating_a, rating_b)
    new_a = rating_a + k * (score_a - ea)
    new_b = rating_b + k * ((1.0 - score_a) - (1.0 - ea))
    return new_a, new_b
