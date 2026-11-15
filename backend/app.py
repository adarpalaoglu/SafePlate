"""SafePlate Test Lab - Flask entry point for trigger detection evaluation."""

import os
import re
import uuid
from flask import Flask, render_template, request, jsonify
import pandas as pd
from safeplate.diary_generator import DiaryGenerator, DiaryGeneratorConfig
from safeplate.integrated_pipeline import SafePlatePipeline, PipelineConfig
from safeplate.embedding_loader import EmbeddingLoader
from safeplate.ingredient_groups import IngredientGroupManager
from safeplate.recommendation_engine import RecommendationEngine
from safeplate.preference_aware_recommendation import PreferenceAwareRecommendationEngine
from safeplate.user_preferences import UserPreferenceStore
from safeplate.group_aware_trigger_detector import (
    ShrinkageResult,
    FullShrinkageOutput,
)
from safeplate.recommendation_engine import IngredientNotFoundError
from safeplate.detection import score_ingredients

app = Flask(__name__)

# Unique ID generated once per server run. Used by the frontend to detect
# whether the server has been restarted and re-show the guided tour.
SERVER_RUN_ID = uuid.uuid4().hex

# Fallback ingredient pool for random-mode diary generation
INGREDIENT_POOL = [
    "milk", "cheddar cheese", "mozzarella", "parmesan", "butter",
    "egg", "chicken", "beef", "pork", "salmon",
    "tomato", "onion", "garlic", "potato", "carrot",
    "rice", "pasta", "bread", "flour", "olive oil",
    "salt", "pepper", "basil", "oregano", "cumin",
]

DEFAULT_TRIGGER = "cheddar cheese"
NEGATIVE_CONTROL = "olive oil"  # known non-trigger, used as baseline in generation

# Initialize pipeline components once at startup
EMBEDDING_LOADER = EmbeddingLoader()
# Full ingredient vocabulary from embeddings (available as triggers)
ALL_INGREDIENTS = sorted(EMBEDDING_LOADER.get_all_ingredients())
GROUP_MANAGER = IngredientGroupManager()
PIPELINE_CONFIG = PipelineConfig(
    trigger_or_threshold=1.5,
    trigger_min_confidence="HIGH",
    propagate_to_untested=True,
    recommendation_top_n=10,
)
PIPELINE = SafePlatePipeline(
    embedding_loader=EMBEDDING_LOADER,
    group_manager=GROUP_MANAGER,
    config=PIPELINE_CONFIG,
)
RECOMMENDATION_ENGINE = RecommendationEngine(embedding_loader=EMBEDDING_LOADER)

# Initialize preference-aware recommendation engine
PREFERENCE_STORE = UserPreferenceStore("data/user_preferences.json")
PREFERENCE_ENGINE = PreferenceAwareRecommendationEngine(
    pipeline=PIPELINE,
    group_manager=GROUP_MANAGER,
    embedding_loader=EMBEDDING_LOADER,
    user_pref_store=PREFERENCE_STORE,
)

# Module-level state for the current experiment. This is a single-user test
# lab so concurrent requests aren't a concern. Flask session cookies cap at
# ~4KB which real-recipe meals easily exceed, so we store state here instead.
_diary_meals = None
_diary_symptoms = None
_diary_config = None
_selected_trigger = None
_shrinkage_result = None
_last_shrinkage_result = None


@app.route("/")
def index():
    return render_template("index.html", server_run_id=SERVER_RUN_ID)


@app.route("/get_ingredient_pool")
def get_ingredient_pool():
    return jsonify({
        "status": "success",
        "ingredients": sorted(INGREDIENT_POOL),
        "default_trigger": DEFAULT_TRIGGER,
    })


@app.route("/get_all_ingredients")
def get_all_ingredients():
    return jsonify({
        "status": "success",
        "ingredients": ALL_INGREDIENTS,
        "default_trigger": DEFAULT_TRIGGER,
    })


@app.route("/generate_diary", methods=["POST"])
def generate_diary():
    """Build a synthetic diary from the posted config and store it in memory."""
    try:
        data = request.get_json()

        selected_trigger = data.get("selected_trigger", DEFAULT_TRIGGER)
        if selected_trigger not in ALL_INGREDIENTS:
            return jsonify({"status": "error", "message": f"Invalid trigger ingredient: {selected_trigger}"}), 400

        config = DiaryGeneratorConfig(
            n_days=int(data["n_days"]),
            meals_per_day_mean=float(data["meals_per_day_mean"]),
            ingredients_per_meal_min=int(data["ingredients_per_meal_min"]),
            ingredients_per_meal_max=int(data["ingredients_per_meal_max"]),
            ingredient_pool=INGREDIENT_POOL,
            true_triggers=[selected_trigger],
            negative_control=NEGATIVE_CONTROL,
            trigger_effect_strength=float(data["trigger_effect_strength"]),
            trigger_exposure_rate=float(data["trigger_exposure_rate"]),
            baseline_symptom_rate=float(data["baseline_symptom_rate"]),
            hazard_window_hours=int(data["hazard_window_hours"]),
            min_lag_hours=int(data["min_lag_hours"]),
            use_real_recipes=True,
            recipe_dataset_path="data/processed/recipes_cleaned.csv",
        )

        generator = DiaryGenerator(config, seed=42)
        meals = generator.generate_meals()
        symptoms = generator.generate_symptoms()

        # datetime objects aren't JSON-serialisable, so convert to ISO strings
        serialisable_meals = []
        for meal in meals:
            serialisable_meals.append({
                "timestamp": meal["timestamp"].isoformat(),
                "meal_id": meal["meal_id"],
                "ingredients": meal["ingredients"],
                "recipe_name": meal.get("recipe_name"),
            })

        serialisable_symptoms = []
        for symptom in symptoms:
            serialisable_symptoms.append({
                "timestamp": symptom["timestamp"].isoformat(),
                "symptom_id": symptom["symptom_id"],
                "episode_id": symptom["episode_id"],
                "triggering_meal_id": symptom.get("triggering_meal_id"),
                "is_trigger_caused": symptom.get("is_trigger_caused", False),
            })

        global _diary_meals, _diary_symptoms, _diary_config, _selected_trigger
        _diary_config = data
        _diary_meals = serialisable_meals
        _diary_symptoms = serialisable_symptoms
        _selected_trigger = selected_trigger

        return jsonify({
            "status": "success",
            "n_meals": len(meals),
            "n_symptoms": len(symptoms),
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


@app.route("/run_detection", methods=["POST"])
def run_detection():
    meals = _diary_meals
    symptoms = _diary_symptoms

    if not meals or symptoms is None:
        return jsonify({"status": "error", "message": "No diary found. Generate a diary first."}), 400

    try:
        # Production detection: raw Haldane-corrected odds ratios, no shrinkage.
        # Shrinkage (group_aware_trigger_detector.py) was removed here because
        # recipe ingredient clustering violates its group homogeneity assumption.
        # ShrinkageResult/FullShrinkageOutput are reused as data containers with
        # shrinkage-specific fields (weight, mu_g, tau_sq) set to dummy values.

        config = _diary_config or {}
        rows = score_ingredients(
            meals,
            symptoms,
            hazard_window_hours=config.get("hazard_window_hours", 6),
            min_lag_hours=config.get("min_lag_hours", 1),
        )

        results = []
        for row in rows:
            n = row["U"] + row["V"]
            results.append(ShrinkageResult(
                ingredient=row["ingredient"],
                U=row["U"], V=row["V"],
                raw_log_OR=row["log_odds_ratio"], raw_OR=row["odds_ratio"],
                sigma_sq=1.0 / max(n, 1),
                group=GROUP_MANAGER.get_group(row["ingredient"]),
                final_log_OR=row["log_odds_ratio"], final_OR=row["odds_ratio"],
                weight=0.0, mu_g=0.0, tau_sq=0.0,
                confidence=row["confidence"],
                evidence_source="raw_case_crossover",
                protected=False,
            ))

        global _last_shrinkage_result, _shrinkage_result
        raw_result = FullShrinkageOutput(ingredients=results, groups=[])
        _last_shrinkage_result = raw_result
        summary = PIPELINE.get_pipeline_summary(raw_result)

        _shrinkage_result = {
            "ingredients": [
                {
                    "ingredient": r.ingredient,
                    "final_OR": r.final_OR,
                    "confidence": r.confidence,
                    "U": r.U,
                    "V": r.V,
                    "protected": r.protected,
                }
                for r in raw_result.ingredients
                if r.U + r.V > 0
            ],
            "summary": summary,
        }

        return jsonify({
            "status": "success",
            "n_triggers": summary["n_triggers"],
            "n_tested": summary["n_tested"],
            "n_high_confidence": summary["n_high_confidence"],
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


@app.route("/get_detection_results")
def get_detection_results():
    """Return the full ranked ingredient list and summary from the last run."""
    if not _shrinkage_result:
        return jsonify({"status": "no_data"})
    return jsonify({
        "status": "success",
        "ingredients": _shrinkage_result["ingredients"],
        "summary": _shrinkage_result["summary"],
        "ground_truth_trigger": _selected_trigger or "Unknown",
    })


@app.route("/get_diary_ingredients")
def get_diary_ingredients():
    """Return sorted list of unique ingredients in the current diary."""
    if not _diary_meals:
        return jsonify({"status": "no_data"})
    unique = set()
    for meal in _diary_meals:
        unique.update(meal["ingredients"])
    return jsonify({"status": "success", "ingredients": sorted(unique)})


@app.route("/get_recommendations", methods=["POST"])
def get_recommendations():
    """Return substitute ingredients for a query, filtered by safety and group."""
    if not _shrinkage_result:
        return jsonify({"status": "error", "message": "No trigger detection results found. Run detection first."}), 400
    result = _shrinkage_result

    data = request.get_json()
    query = data.get("query_ingredient", "").strip()
    hide_same_group = data.get("hide_same_group", False)
    if not query:
        return jsonify({"status": "error", "message": "No ingredient specified."}), 400

    try:
        trigger_list = result["summary"].get("trigger_list", [])
        # Over-fetch when filtering by group since some results will be dropped
        fetch_count = 30 if hide_same_group else 10

        # Prefer the preference-aware engine; fall back to basic if no results
        try:
            if _last_shrinkage_result is not None:
                all_recommendations = PREFERENCE_ENGINE.get_substitutes_with_preferences(
                    query_ingredient=query,
                    shrinkage_result=_last_shrinkage_result,
                    top_n=fetch_count,
                    include_explanations=True,
                )
            else:
                # Fallback to basic engine if no shrinkage result in memory
                basic_recs = RECOMMENDATION_ENGINE.find_substitutes(
                    ingredient=query,
                    triggers=trigger_list,
                    top_n=fetch_count,
                )
                from safeplate.preference_aware_recommendation import Recommendation
                all_recommendations = [
                    Recommendation(ingredient=name, similarity_score=score, is_safe=True, user_rank=None, explanation=None)
                    for name, score in basic_recs
                ]
        except IngredientNotFoundError:
            return jsonify({
                "status": "success",
                "query_ingredient": query,
                "recommendations": [],
                "excluded_triggers": trigger_list,
                "n_recommendations": 0,
                "hide_same_group": hide_same_group,
                "n_same_group_hidden": 0,
                "no_results_reason": "not_in_vocabulary",
            })

        # Flag recs in the same food group as a trigger (they may share
        # problematic properties) and optionally filter them out
        enhanced = []
        n_same_group_skipped = 0
        for rec in all_recommendations:
            # Check group membership
            is_same_group = False
            same_group_trigger = None
            rec_group = GROUP_MANAGER.get_group(rec.ingredient)
            if rec_group:
                for trigger in trigger_list:
                    trigger_group = GROUP_MANAGER.get_group(trigger)
                    if trigger_group and rec_group == trigger_group:
                        is_same_group = True
                        same_group_trigger = trigger
                        break

            # Skip same-group items when filtering is active
            if hide_same_group and is_same_group:
                n_same_group_skipped += 1
                continue

            explanation_text = None
            if rec.explanation:
                explanation_text = rec.explanation.to_text()
                # Convert markdown **bold** to HTML <strong>
                explanation_text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', explanation_text)

            # Look up structured OR/confidence data for frontend text generation
            or_value = None
            confidence_val = None
            is_tested = False
            if _last_shrinkage_result is not None:
                for r in _last_shrinkage_result.ingredients:
                    if r.ingredient == rec.ingredient:
                        or_value = round(float(r.final_OR), 4)
                        confidence_val = r.confidence
                        is_tested = True
                        break

            enhanced.append({
                "ingredient": rec.ingredient,
                "similarity": float(rec.similarity_score),
                "is_same_group": is_same_group,
                "same_group_as_trigger": same_group_trigger,
                "explanation": explanation_text,
                "or_value": or_value,
                "confidence": confidence_val,
                "is_tested": is_tested,
                "user_rank": rec.user_rank,
            })

            # Stop once we have 10
            if len(enhanced) >= 10:
                break

        no_results_reason = "all_excluded" if len(enhanced) == 0 else None
        return jsonify({
            "status": "success",
            "query_ingredient": query,
            "recommendations": enhanced,
            "excluded_triggers": trigger_list,
            "n_recommendations": len(enhanced),
            "hide_same_group": hide_same_group,
            "n_same_group_hidden": n_same_group_skipped,
            "no_results_reason": no_results_reason,
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/get_diary_data")
def get_diary_data():
    """Return full diary (meals + symptoms) sorted chronologically."""
    if not _diary_meals or not _diary_symptoms:
        return jsonify({"status": "no_data"})

    sorted_meals = sorted(_diary_meals, key=lambda m: m["timestamp"])
    sorted_symptoms = sorted(_diary_symptoms, key=lambda s: s["timestamp"])

    # Compute date range from first and last meal timestamps
    first_date = sorted_meals[0]["timestamp"][:10]
    last_date = sorted_meals[-1]["timestamp"][:10]

    return jsonify({
        "status": "success",
        "meals": sorted_meals,
        "symptoms": sorted_symptoms,
        "summary": {
            "n_meals": len(_diary_meals),
            "n_symptoms": len(_diary_symptoms),
            "n_days": _diary_config["n_days"],
            "date_range": f"{first_date} to {last_date}",
        },
    })


@app.route("/save_preferences", methods=["POST"])
def save_preferences():
    """Persist a user's ranked substitute list for a given ingredient."""
    data = request.get_json()
    query = data.get("query_ingredient", "").strip()
    ranked = data.get("ranked_ingredients", [])
    if not query:
        return jsonify({"status": "error", "message": "No ingredient specified."}), 400
    if not ranked or not isinstance(ranked, list):
        return jsonify({"status": "error", "message": "No rankings provided."}), 400
    try:
        PREFERENCE_STORE.save_ranking(query, ranked)
        return jsonify({"status": "success", "query_ingredient": query, "n_ranked": len(ranked)})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/get_preferences/<ingredient>")
def get_preferences(ingredient):
    """Load saved preference ranking for a specific ingredient."""
    try:
        ranking = PREFERENCE_STORE.get_ranking(ingredient)
        if ranking is None:
            return jsonify({"status": "no_data", "message": "No preferences saved for this ingredient."})
        return jsonify({
            "status": "success",
            "query_ingredient": ranking.query_ingredient,
            "ranked_ingredients": ranking.ranked_ingredients,
            "timestamp": ranking.timestamp.isoformat(),
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/delete_preferences/<ingredient>", methods=["DELETE"])
def delete_preferences(ingredient):
    """Remove saved ranking for an ingredient."""
    try:
        deleted = PREFERENCE_STORE.delete_ranking(ingredient)
        if deleted:
            return jsonify({"status": "success", "message": "Ranking deleted."})
        return jsonify({"status": "no_data", "message": "No ranking found to delete."})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/session/clear")
def clear_session():
    """Reset all module-level state so the user can start a fresh experiment."""
    global _diary_meals, _diary_symptoms, _diary_config, _selected_trigger
    global _shrinkage_result, _last_shrinkage_result
    _diary_meals = None
    _diary_symptoms = None
    _diary_config = None
    _selected_trigger = None
    _shrinkage_result = None
    _last_shrinkage_result = None
    return jsonify({"status": "cleared"})


if __name__ == "__main__":
    # Port 5001 to avoid macOS AirPlay Receiver conflict on 5000
    # Debug mode only when asked for: FLASK_DEBUG=1 python backend/app.py
    app.run(debug=os.environ.get("FLASK_DEBUG") == "1", port=5001)
