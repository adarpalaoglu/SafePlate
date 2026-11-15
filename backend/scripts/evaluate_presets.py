"""How often does SafePlate find the hidden trigger? One table per preset.

For each preset in the web app (Easy, Balanced, Difficult), this generates
synthetic diaries from real Food.com recipes for several trigger ingredients
and random seeds, runs the production detection (safeplate/detection.py) and
ranks ingredients exactly as the web app does. It reports how often the true
trigger ranks first, how often it is in the top 3, and the median rank.

Run from the project root after prepare_data.py:

    python backend/scripts/evaluate_presets.py --seeds 10
"""

from __future__ import annotations

import argparse
import ast
import math
import statistics
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from safeplate.detection import score_ingredients  # noqa: E402
from safeplate.diary_generator import DiaryGenerator, DiaryGeneratorConfig  # noqa: E402

# Same values as PRESET_SCENARIOS in backend/static/js/main.js.
PRESETS = {
    "Easy": dict(n_days=90, effect=0.70, baseline=0.01, exposure=0.50),
    "Balanced": dict(n_days=60, effect=0.70, baseline=0.04, exposure=0.35),
    "Difficult": dict(n_days=45, effect=0.40, baseline=0.08, exposure=0.25),
}
# Ten common ingredients from the cleaned vocabulary, of different kinds.
TRIGGERS = ["egg", "milk", "cheddar cheese", "parmesan cheese", "tomato",
            "chicken", "garlic", "mushroom", "soy sauce", "cinnamon"]
# Only used by the generator's input checks (real recipes supply the meals).
POOL = TRIGGERS
CONFIDENCE_WEIGHT = {"HIGH": 1.0, "MEDIUM": 0.5, "LOW": 0.1}


def load_recipes(path: Path) -> list[tuple[str, list[str]]]:
    df = pd.read_csv(path, usecols=["name", "ingredients"])
    recipes = []
    for name, raw in zip(df["name"], df["ingredients"]):
        try:
            ingredients = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            continue
        if isinstance(ingredients, list) and len(ingredients) >= 2:
            recipes.append((str(name), ingredients))
    return recipes


def app_ranking(rows: list[dict], meals: list[dict]) -> list[str]:
    """The web app's ranking: OR x confidence weight x log(U + 1), with its
    co-occurrence penalty for non-HIGH ingredients that mostly appear
    alongside a HIGH-confidence suspect."""
    high = {r["ingredient"] for r in rows if r["confidence"] == "HIGH" and r["odds_ratio"] > 1}
    counts, with_high = {}, {}
    for meal in meals:
        has_high = any(i in high for i in meal["ingredients"])
        for i in meal["ingredients"]:
            counts[i] = counts.get(i, 0) + 1
            if has_high:
                with_high[i] = with_high.get(i, 0) + 1

    def score(r):
        base = r["odds_ratio"] * CONFIDENCE_WEIGHT[r["confidence"]] * math.log(r["U"] + 1)
        ratio = with_high.get(r["ingredient"], 0) / counts[r["ingredient"]] if high else 0
        return base * (1 - ratio) if ratio > 0.8 and r["confidence"] != "HIGH" else base

    return [r["ingredient"] for r in sorted(rows, key=score, reverse=True)]


def run(seeds: int) -> pd.DataFrame:
    recipes = load_recipes(ROOT / "data" / "processed" / "recipes_cleaned.csv")
    results = []
    for preset, p in PRESETS.items():
        for trigger in TRIGGERS:
            with_t = [r for r in recipes if trigger in r[1]]
            without_t = [r for r in recipes if trigger not in r[1]]
            for seed in range(seeds):
                config = DiaryGeneratorConfig(
                    n_days=p["n_days"], meals_per_day_mean=3,
                    ingredients_per_meal_min=3, ingredients_per_meal_max=8,
                    ingredient_pool=POOL,
                    true_triggers=[trigger], negative_control="olive oil",
                    trigger_effect_strength=p["effect"],
                    trigger_exposure_rate=p["exposure"],
                    baseline_symptom_rate=p["baseline"],
                    hazard_window_hours=6, min_lag_hours=1,
                    use_real_recipes=False,
                )
                gen = DiaryGenerator(config, seed=seed)
                # Reuse the recipes parsed once above instead of re-reading the CSV.
                gen._recipes_with_trigger = with_t
                gen._recipes_without_trigger = without_t
                gen._using_real_recipes = True
                meals = [
                    {"timestamp": m["timestamp"].isoformat(), "ingredients": m["ingredients"]}
                    for m in gen.generate_meals()
                ]
                symptoms = [{"timestamp": s["timestamp"].isoformat()} for s in gen.generate_symptoms()]
                ranking = app_ranking(score_ingredients(meals, symptoms), meals)
                rank = ranking.index(trigger) + 1 if trigger in ranking else len(ranking) + 1
                results.append({"preset": preset, "trigger": trigger, "seed": seed, "rank": rank})
    return pd.DataFrame(results)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10)
    args = parser.parse_args()
    df = run(args.seeds)
    out = ROOT / "results" / "evaluation_presets.csv"
    out.parent.mkdir(exist_ok=True)
    df.to_csv(out, index=False)
    print(f"{len(TRIGGERS)} triggers x {args.seeds} seeds per preset, raw runs in {out.name}\n")
    print("| Preset | Runs | Trigger ranked 1st | In top 3 | Median rank |")
    print("|---|---|---|---|---|")
    for preset in PRESETS:
        ranks = df[df.preset == preset]["rank"]
        print(f"| {preset} | {len(ranks)} | {(ranks == 1).mean():.0%} | "
              f"{(ranks <= 3).mean():.0%} | {statistics.median(ranks):g} |")


if __name__ == "__main__":
    main()
