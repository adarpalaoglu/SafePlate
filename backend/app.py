"""SafePlate Test Lab - Flask web interface for trigger detection evaluation."""

import os
import re
from datetime import datetime
from flask import Flask, render_template, request, session, jsonify
import pandas as pd
from safeplate.diary_generator import DiaryGenerator, DiaryGeneratorConfig
from safeplate.integrated_pipeline import SafePlatePipeline, PipelineConfig
from safeplate.embedding_loader import EmbeddingLoader
from safeplate.ingredient_groups import IngredientGroupManager
from safeplate.recommendation_engine import RecommendationEngine
from safeplate.preference_aware_recommendation import PreferenceAwareRecommendationEngine
from safeplate.user_preferences import UserPreferenceStore

app = Flask(__name__)
app.secret_key = "safeplate-test-lab-secret-key-cs310"
app.config["SESSION_PERMANENT"] = True

# Hardcoded ingredient pool and triggers for the test lab
INGREDIENT_POOL = [
    "milk", "cheddar cheese", "mozzarella", "parmesan", "butter",
    "egg", "chicken", "beef", "pork", "salmon",
    "tomato", "onion", "garlic", "potato", "carrot",
    "rice", "pasta", "bread", "flour", "olive oil",
    "salt", "pepper", "basil", "oregano", "cumin",
]

DEFAULT_TRIGGER = "cheddar cheese"
NEGATIVE_CONTROL = "olive oil"

# Initialize pipeline components once at startup
EMBEDDING_LOADER = EmbeddingLoader()
GROUP_MANAGER = IngredientGroupManager()
PIPELINE_CONFIG = PipelineConfig(
    trigger_or_threshold=1.0,
    trigger_min_confidence="MEDIUM",
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

# Store last shrinkage result for recommendation explanations (single-user test lab)
_last_shrinkage_result = None


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/get_ingredient_pool")
def get_ingredient_pool():
    return jsonify({
        "status": "success",
        "ingredients": sorted(INGREDIENT_POOL),
        "default_trigger": DEFAULT_TRIGGER,
    })


@app.route("/generate_diary", methods=["POST"])
def generate_diary():
    try:
        data = request.get_json()

        selected_trigger = data.get("selected_trigger", DEFAULT_TRIGGER)
        if selected_trigger not in INGREDIENT_POOL:
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
        )

        generator = DiaryGenerator(config, seed=42)
        meals = generator.generate_meals()
        symptoms = generator.generate_symptoms()

        # Convert datetimes to ISO strings for JSON serialisation
        serialisable_meals = []
        for meal in meals:
            serialisable_meals.append({
                "timestamp": meal["timestamp"].isoformat(),
                "meal_id": meal["meal_id"],
                "ingredients": meal["ingredients"],
            })

        serialisable_symptoms = []
        for symptom in symptoms:
            serialisable_symptoms.append({
                "timestamp": symptom["timestamp"].isoformat(),
                "symptom_id": symptom["symptom_id"],
                "episode_id": symptom["episode_id"],
            })

        session["diary_config"] = data
        session["meals"] = serialisable_meals
        session["symptoms"] = serialisable_symptoms
        session["selected_trigger"] = selected_trigger

        return jsonify({
            "status": "success",
            "n_meals": len(meals),
            "n_symptoms": len(symptoms),
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


@app.route("/run_detection", methods=["POST"])
def run_detection():
    meals = session.get("meals")
    symptoms = session.get("symptoms")

    if not meals or symptoms is None:
        return jsonify({"status": "error", "message": "No diary found. Generate a diary first."}), 400

    try:
        # Collect unique ingredients
        all_ingredients = set()
        for meal in meals:
            all_ingredients.update(meal["ingredients"])

        # Simplified case-crossover: count symptom co-occurrences per ingredient
        config = session.get("diary_config", {})
        hazard_window = config.get("hazard_window_hours", 6)
        min_lag = config.get("min_lag_hours", 1)

        stats = []
        for ingredient in all_ingredients:
            meals_with_ing = [m for m in meals if ingredient in m["ingredients"]]
            symptom_count = 0
            for meal in meals_with_ing:
                meal_time = datetime.fromisoformat(meal["timestamp"])
                for symptom in symptoms:
                    symptom_time = datetime.fromisoformat(symptom["timestamp"])
                    diff_hours = (symptom_time - meal_time).total_seconds() / 3600
                    if min_lag <= diff_hours <= hazard_window:
                        symptom_count += 1

            U = symptom_count
            V = max(len(meals_with_ing) - symptom_count, 0)
            if U + V > 0:
                stats.append({
                    "ingredient": ingredient,
                    "U_case_not_ctrl": U,
                    "V_ctrl_not_case": V,
                })

        global _last_shrinkage_result
        ingredient_stats = pd.DataFrame(stats)
        shrinkage_result = PIPELINE.detect_triggers_from_stats(ingredient_stats)
        _last_shrinkage_result = shrinkage_result
        summary = PIPELINE.get_pipeline_summary(shrinkage_result)

        # Store results in session (only tested ingredients to keep cookie small)
        session["shrinkage_result"] = {
            "ingredients": [
                {
                    "ingredient": r.ingredient,
                    "final_OR": r.final_OR,
                    "confidence": r.confidence,
                    "U": r.U,
                    "V": r.V,
                    "protected": r.protected,
                }
                for r in shrinkage_result.ingredients
                if r.U + r.V > 0 or r.protected
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
    result = session.get("shrinkage_result")
    if not result:
        return jsonify({"status": "no_data"})
    return jsonify({
        "status": "success",
        "ingredients": result["ingredients"],
        "summary": result["summary"],
        "ground_truth_trigger": session.get("selected_trigger", "Unknown"),
    })


@app.route("/get_diary_ingredients")
def get_diary_ingredients():
    if "meals" not in session:
        return jsonify({"status": "no_data"})
    unique = set()
    for meal in session["meals"]:
        unique.update(meal["ingredients"])
    return jsonify({"status": "success", "ingredients": sorted(unique)})


@app.route("/get_recommendations", methods=["POST"])
def get_recommendations():
    result = session.get("shrinkage_result")
    if not result:
        return jsonify({"status": "error", "message": "No trigger detection results found. Run detection first."}), 400

    data = request.get_json()
    query = data.get("query_ingredient", "").strip()
    hide_same_group = data.get("hide_same_group", False)
    if not query:
        return jsonify({"status": "error", "message": "No ingredient specified."}), 400

    try:
        trigger_list = result["summary"].get("trigger_list", [])
        # Fetch more when filtering same-group to guarantee 10 results
        fetch_count = 30 if hide_same_group else 10

        # Use preference engine with explanations if shrinkage result available
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

        # Check group membership and filter if needed
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

            enhanced.append({
                "ingredient": rec.ingredient,
                "similarity": float(rec.similarity_score),
                "is_same_group": is_same_group,
                "same_group_as_trigger": same_group_trigger,
                "explanation": explanation_text,
            })

            # Stop once we have 10
            if len(enhanced) >= 10:
                break

        return jsonify({
            "status": "success",
            "query_ingredient": query,
            "recommendations": enhanced,
            "excluded_triggers": trigger_list,
            "n_recommendations": len(enhanced),
            "hide_same_group": hide_same_group,
            "n_same_group_hidden": n_same_group_skipped,
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/get_diary_data")
def get_diary_data():
    meals = session.get("meals")
    symptoms = session.get("symptoms")
    diary_config = session.get("diary_config")

    if not meals or not symptoms:
        return jsonify({"status": "no_data"})

    # Compute date range from first and last meal timestamps
    timestamps = [m["timestamp"] for m in meals]
    first_date = timestamps[0][:10]
    last_date = timestamps[-1][:10]

    return jsonify({
        "status": "success",
        "meals": meals,
        "symptoms": symptoms,
        "summary": {
            "n_meals": len(meals),
            "n_symptoms": len(symptoms),
            "n_days": diary_config["n_days"],
            "date_range": f"{first_date} to {last_date}",
        },
    })


@app.route("/save_preferences", methods=["POST"])
def save_preferences():
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
    try:
        deleted = PREFERENCE_STORE.delete_ranking(ingredient)
        if deleted:
            return jsonify({"status": "success", "message": "Ranking deleted."})
        return jsonify({"status": "no_data", "message": "No ranking found to delete."})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/session/clear")
def clear_session():
    global _last_shrinkage_result
    session.clear()
    _last_shrinkage_result = None
    return jsonify({"status": "cleared"})


if __name__ == "__main__":
    app.run(debug=True)
