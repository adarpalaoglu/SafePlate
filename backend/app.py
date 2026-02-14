"""SafePlate Test Lab - Flask web interface for trigger detection evaluation."""

import os
from datetime import datetime
from flask import Flask, render_template, request, session, jsonify
import pandas as pd
from safeplate.diary_generator import DiaryGenerator, DiaryGeneratorConfig
from safeplate.integrated_pipeline import SafePlatePipeline, PipelineConfig
from safeplate.embedding_loader import EmbeddingLoader
from safeplate.ingredient_groups import IngredientGroupManager
from safeplate.recommendation_engine import RecommendationEngine

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

        ingredient_stats = pd.DataFrame(stats)
        shrinkage_result = PIPELINE.detect_triggers_from_stats(ingredient_stats)
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
    if not query:
        return jsonify({"status": "error", "message": "No ingredient specified."}), 400

    try:
        trigger_list = result["summary"].get("trigger_list", [])
        recommendations = RECOMMENDATION_ENGINE.find_substitutes(
            ingredient=query,
            triggers=trigger_list,
            top_n=PIPELINE_CONFIG.recommendation_top_n,
        )
        # Check group membership for each recommendation
        enhanced = []
        for name, score in recommendations:
            rec = {"ingredient": name, "similarity": float(score), "is_same_group": False, "same_group_as_trigger": None}
            rec_group = GROUP_MANAGER.get_group(name)
            if rec_group:
                for trigger in trigger_list:
                    trigger_group = GROUP_MANAGER.get_group(trigger)
                    if trigger_group and rec_group == trigger_group:
                        rec["is_same_group"] = True
                        rec["same_group_as_trigger"] = trigger
                        break
            enhanced.append(rec)

        return jsonify({
            "status": "success",
            "query_ingredient": query,
            "recommendations": enhanced,
            "excluded_triggers": trigger_list,
            "n_recommendations": len(recommendations),
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


@app.route("/session/clear")
def clear_session():
    session.clear()
    return jsonify({"status": "cleared"})


if __name__ == "__main__":
    app.run(debug=True)
