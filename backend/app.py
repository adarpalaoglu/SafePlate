"""SafePlate Test Lab - Flask web interface for trigger detection evaluation."""

import os
from flask import Flask, render_template, request, session, jsonify
from safeplate.diary_generator import DiaryGenerator, DiaryGeneratorConfig

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

TRUE_TRIGGERS = ["cheddar cheese"]
NEGATIVE_CONTROL = "olive oil"


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/generate_diary", methods=["POST"])
def generate_diary():
    try:
        data = request.get_json()

        config = DiaryGeneratorConfig(
            n_days=int(data["n_days"]),
            meals_per_day_mean=float(data["meals_per_day_mean"]),
            ingredients_per_meal_min=int(data["ingredients_per_meal_min"]),
            ingredients_per_meal_max=int(data["ingredients_per_meal_max"]),
            ingredient_pool=INGREDIENT_POOL,
            true_triggers=TRUE_TRIGGERS,
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

        return jsonify({
            "status": "success",
            "n_meals": len(meals),
            "n_symptoms": len(symptoms),
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


@app.route("/session/clear")
def clear_session():
    session.clear()
    return jsonify({"status": "cleared"})


if __name__ == "__main__":
    app.run(debug=True)
