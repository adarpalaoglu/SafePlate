"""Tests for the production detection path (backend/safeplate/detection.py)."""

import importlib.util
import math
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parent.parent / "backend" / "safeplate" / "detection.py"
_spec = importlib.util.spec_from_file_location("detection", _PATH)
detection = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(detection)


def meal(hour, *ingredients):
    return {"timestamp": f"2026-01-01T{hour:02d}:00:00", "ingredients": list(ingredients)}


def symptom(hour):
    return {"timestamp": f"2026-01-01T{hour:02d}:00:00"}


def test_a_meal_followed_by_a_symptom_in_the_window_counts_as_u():
    counts = detection.exposure_counts([meal(8, "milk")], [symptom(10)])
    assert counts["milk"] == (1, 0)


def test_symptoms_outside_the_window_count_as_v():
    too_early = detection.exposure_counts([meal(8, "milk")], [symptom(8)], min_lag_hours=1)
    too_late = detection.exposure_counts([meal(8, "milk")], [symptom(16)], hazard_window_hours=6)
    before = detection.exposure_counts([meal(8, "milk")], [symptom(7)])
    assert too_early["milk"] == too_late["milk"] == before["milk"] == (0, 1)


def test_two_symptoms_after_one_meal_count_that_meal_once():
    counts = detection.exposure_counts([meal(8, "milk")], [symptom(10), symptom(11)])
    assert counts["milk"] == (1, 0)
    u, v = counts["milk"]
    assert u + v == 1  # U + V is always the number of meals with the ingredient


def test_an_ingredient_listed_twice_in_a_meal_counts_once():
    counts = detection.exposure_counts([meal(8, "milk", "milk")], [])
    assert counts["milk"] == (0, 1)


def test_haldane_correction_keeps_zero_counts_finite():
    assert detection.haldane_odds_ratio(0, 0) == 1.0
    assert detection.haldane_odds_ratio(3, 0) == 7.0
    assert detection.haldane_odds_ratio(0, 3) == pytest.approx(1 / 7)


@pytest.mark.parametrize(
    "u, v, expected",
    [(15, 5, "HIGH"), (5, 15, "MEDIUM"), (10, 10, "MEDIUM"), (5, 4, "LOW"), (0, 0, "LOW")],
)
def test_confidence_levels(u, v, expected):
    assert detection.confidence_level(u, v) == expected


def test_a_planted_trigger_ranks_first():
    meals, symptoms = [], []
    for day in range(1, 29):
        date = f"2026-02-{day:02d}"
        meals.append({"timestamp": f"{date}T08:00:00", "ingredients": ["bread", "butter"]})
        meals.append({"timestamp": f"{date}T13:00:00", "ingredients": ["bread", "cheddar cheese"]})
        if day % 4:  # 3 days in 4, a symptom two hours after the cheese
            symptoms.append({"timestamp": f"{date}T15:00:00"})
    rows = detection.score_ingredients(meals, symptoms)
    ranked = sorted(rows, key=lambda r: r["odds_ratio"], reverse=True)
    assert ranked[0]["ingredient"] == "cheddar cheese"
    assert ranked[0]["confidence"] == "HIGH"
    assert ranked[0]["log_odds_ratio"] == pytest.approx(math.log(ranked[0]["odds_ratio"]))
    butter = next(r for r in rows if r["ingredient"] == "butter")
    assert butter["U"] == 0
