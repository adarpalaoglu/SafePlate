"""Production trigger detection: Haldane-corrected odds ratios.

This is the detection path the web app runs (``/run_detection``). It was
moved out of ``app.py`` so it can be tested on its own. The empirical Bayes
shrinkage model in ``group_aware_trigger_detector.py`` is kept for reference
but is not used here (see the README for why).
"""

from __future__ import annotations

import math
from datetime import datetime

HIGH_MIN_MEALS = 20
MEDIUM_MIN_MEALS = 10


def exposure_counts(
    meals: list[dict],
    symptoms: list[dict],
    hazard_window_hours: float = 6,
    min_lag_hours: float = 1,
) -> dict[str, tuple[int, int]]:
    """For each ingredient, (U, V).

    U is the number of meals containing the ingredient that were followed by
    at least one symptom between ``min_lag_hours`` and ``hazard_window_hours``.
    V is the number of meals containing it that were not.
    """
    symptom_times = [datetime.fromisoformat(s["timestamp"]) for s in symptoms]
    counts: dict[str, list[int]] = {}
    for meal in meals:
        meal_time = datetime.fromisoformat(meal["timestamp"])
        followed = any(
            min_lag_hours <= (t - meal_time).total_seconds() / 3600 <= hazard_window_hours
            for t in symptom_times
        )
        for ingredient in set(meal["ingredients"]):
            u_v = counts.setdefault(ingredient, [0, 0])
            u_v[0 if followed else 1] += 1
    return {ing: (u, v) for ing, (u, v) in counts.items()}


def haldane_odds_ratio(u: int, v: int) -> float:
    """(U + 0.5) / (V + 0.5): the +0.5 keeps sparse counts finite."""
    return (u + 0.5) / (v + 0.5)


def confidence_level(u: int, v: int) -> str:
    """HIGH needs 20+ meals and a positive signal, MEDIUM 10+ meals."""
    n = u + v
    if n >= HIGH_MIN_MEALS and haldane_odds_ratio(u, v) > 1.0:
        return "HIGH"
    if n >= MEDIUM_MIN_MEALS:
        return "MEDIUM"
    return "LOW"


def score_ingredients(
    meals: list[dict],
    symptoms: list[dict],
    hazard_window_hours: float = 6,
    min_lag_hours: float = 1,
) -> list[dict]:
    """Every ingredient's U, V, odds ratio, log odds ratio and confidence."""
    rows = []
    for ingredient, (u, v) in exposure_counts(
        meals, symptoms, hazard_window_hours, min_lag_hours
    ).items():
        odds_ratio = haldane_odds_ratio(u, v)
        rows.append({
            "ingredient": ingredient,
            "U": u,
            "V": v,
            "odds_ratio": odds_ratio,
            "log_odds_ratio": math.log(odds_ratio),
            "confidence": confidence_level(u, v),
        })
    return rows
