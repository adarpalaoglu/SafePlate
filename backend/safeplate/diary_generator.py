"""Synthetic diary generator for SafePlate test lab.

Generates synthetic meal and symptom logs with known ground truth triggers
for validating the trigger detection pipeline.

The generator uses an exposure-driven causal model:
- Meals containing true triggers have elevated probability of causing symptoms
- Baseline symptoms occur at a lower rate independent of trigger exposure
- Negative control ingredients are included for false positive validation
"""

import json
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class DiaryGeneratorConfig:
    """Configuration for synthetic diary generation.

    Attributes:
        n_days: Length of diary in days.
        meals_per_day_mean: Poisson rate for number of meals per day.
        ingredients_per_meal_min: Minimum ingredients per meal.
        ingredients_per_meal_max: Maximum ingredients per meal.
        ingredient_pool: List of ingredients to sample from.
        true_triggers: Ground truth trigger ingredients.
        negative_control: Known safe ingredient for false positive checks.
        trigger_effect_strength: P(symptom | trigger in meal), range [0, 1].
        trigger_exposure_rate: P(trigger appears in meal), range [0, 1].
        baseline_symptom_rate: P(symptom | no trigger), range [0, 1].
        hazard_window_hours: Time range after meal where symptom can occur.
        min_lag_hours: Minimum delay before symptom can occur after meal.
    """

    n_days: int = 30
    meals_per_day_mean: float = 3.0
    ingredients_per_meal_min: int = 3
    ingredients_per_meal_max: int = 8
    ingredient_pool: list[str] = field(default_factory=list)
    true_triggers: list[str] = field(default_factory=list)
    negative_control: str = "olive oil"
    trigger_effect_strength: float = 0.5
    trigger_exposure_rate: float = 0.3
    baseline_symptom_rate: float = 0.05
    hazard_window_hours: int = 6
    min_lag_hours: int = 1

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.n_days <= 0:
            raise ValueError("n_days must be positive")
        if self.meals_per_day_mean <= 0:
            raise ValueError("meals_per_day_mean must be positive")
        if self.ingredients_per_meal_min <= 0:
            raise ValueError("ingredients_per_meal_min must be positive")
        if self.ingredients_per_meal_max < self.ingredients_per_meal_min:
            raise ValueError(
                "ingredients_per_meal_max must be >= ingredients_per_meal_min"
            )
        if not 0 <= self.trigger_effect_strength <= 1:
            raise ValueError("trigger_effect_strength must be in [0, 1]")
        if not 0 <= self.trigger_exposure_rate <= 1:
            raise ValueError("trigger_exposure_rate must be in [0, 1]")
        if not 0 <= self.baseline_symptom_rate <= 1:
            raise ValueError("baseline_symptom_rate must be in [0, 1]")
        if self.hazard_window_hours <= self.min_lag_hours:
            raise ValueError("hazard_window_hours must be > min_lag_hours")


class DiaryGenerator:
    """Generates synthetic meal and symptom diaries with known ground truth.

    This class creates synthetic food diary data suitable for validating
    the SafePlate trigger detection pipeline. The generated data follows
    an exposure-driven causal model where meals containing specified
    trigger ingredients have an elevated probability of causing symptoms.

    Example:
        >>> config = DiaryGeneratorConfig(
        ...     n_days=30,
        ...     true_triggers=["cheddar cheese"],
        ...     ingredient_pool=["milk", "egg", "flour", "cheddar cheese", "olive oil"],
        ... )
        >>> generator = DiaryGenerator(config, seed=42)
        >>> meals = generator.generate_meals()
        >>> symptoms = generator.generate_symptoms(meals)
        >>> df = generator.to_dataframe()
    """

    def __init__(
        self,
        config: DiaryGeneratorConfig,
        seed: int = 42,
    ) -> None:
        """Initialize the diary generator.

        Args:
            config: Configuration parameters for diary generation.
            seed: Random seed for reproducibility.
        """
        self._config = config
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        random.seed(seed)
        np.random.seed(seed)

        # Generated data (populated by generate methods)
        self._meals: list[dict] = []
        self._symptoms: list[dict] = []
        self._generation_timestamp: Optional[datetime] = None

        # Ensure ingredient pool has enough variety
        if len(config.ingredient_pool) < config.ingredients_per_meal_max:
            raise ValueError(
                f"ingredient_pool ({len(config.ingredient_pool)}) must have "
                f"at least ingredients_per_meal_max ({config.ingredients_per_meal_max}) items"
            )

    def generate_meals(self) -> list[dict]:
        """Generate synthetic meal records.

        Returns:
            List of meal dictionaries with keys:
            - timestamp: datetime
            - meal_id: int
            - ingredients: list[str]
        """
        self._meals = []
        meal_id = 0

        # Base date for diary
        base_date = datetime(2026, 1, 1)

        for day in range(self._config.n_days):
            # Sample number of meals for this day
            n_meals = self._rng.poisson(self._config.meals_per_day_mean)
            n_meals = max(1, n_meals)  # At least one meal per day

            for _ in range(n_meals):
                # Sample time of day (minutes from midnight)
                minutes = self._rng.integers(0, 24 * 60)
                timestamp = base_date + timedelta(days=day, minutes=int(minutes))

                # Sample number of ingredients
                n_ingredients = self._rng.integers(
                    self._config.ingredients_per_meal_min,
                    self._config.ingredients_per_meal_max + 1,
                )

                ingredients = self._sample_meal_ingredients(n_ingredients)

                self._meals.append({
                    "timestamp": timestamp,
                    "meal_id": meal_id,
                    "ingredients": ingredients,
                })
                meal_id += 1

        # Sort by timestamp
        self._meals.sort(key=lambda m: m["timestamp"])
        return self._meals

    def _sample_meal_ingredients(self, n_ingredients: int) -> list[str]:
        """Sample ingredients for a single meal.

        Args:
            n_ingredients: Number of ingredients to sample.

        Returns:
            List of ingredient names (no duplicates).
        """
        ingredients: set[str] = set()

        # Pool of non-trigger, non-control ingredients
        regular_pool = [
            ing for ing in self._config.ingredient_pool
            if ing not in self._config.true_triggers
            and ing != self._config.negative_control
        ]

        # Decide if this meal contains a trigger
        include_trigger = self._rng.random() < self._config.trigger_exposure_rate
        if include_trigger and self._config.true_triggers:
            # Add one random trigger
            trigger = self._rng.choice(self._config.true_triggers)
            ingredients.add(trigger)

        # Decide if this meal contains negative control (10% of non-trigger meals)
        # Don't put negative control in trigger meals to keep validation clean
        include_control = (
            not include_trigger
            and self._rng.random() < 0.1
            and self._config.negative_control in self._config.ingredient_pool
        )
        if include_control:
            ingredients.add(self._config.negative_control)

        # Fill remaining slots with regular ingredients
        while len(ingredients) < n_ingredients:
            if not regular_pool:
                break
            candidate = self._rng.choice(regular_pool)
            ingredients.add(candidate)

        return list(ingredients)

    def generate_symptoms(self, meals: Optional[list[dict]] = None) -> list[dict]:
        """Generate synthetic symptom records based on meal data.

        Uses an exposure-driven causal model:
        - If meal contains a trigger: P(symptom) = trigger_effect_strength
        - If meal has no trigger: P(symptom) = baseline_symptom_rate

        Args:
            meals: List of meal dictionaries. If None, uses self._meals.

        Returns:
            List of symptom dictionaries with keys:
            - timestamp: datetime
            - symptom_id: int
            - episode_id: int
        """
        if meals is None:
            meals = self._meals

        if not meals:
            raise ValueError("No meals to generate symptoms from. Call generate_meals() first.")

        self._symptoms = []
        symptom_id = 0

        trigger_set = set(self._config.true_triggers)

        for meal in meals:
            meal_ingredients = set(meal["ingredients"])
            has_trigger = bool(meal_ingredients & trigger_set)

            # Determine if symptom occurs
            if has_trigger:
                symptom_prob = self._config.trigger_effect_strength
            else:
                symptom_prob = self._config.baseline_symptom_rate

            if self._rng.random() < symptom_prob:
                # Generate symptom with lag
                lag_minutes = self._rng.integers(
                    self._config.min_lag_hours * 60,
                    self._config.hazard_window_hours * 60 + 1,
                )
                symptom_time = meal["timestamp"] + timedelta(minutes=int(lag_minutes))

                self._symptoms.append({
                    "timestamp": symptom_time,
                    "symptom_id": symptom_id,
                    "episode_id": symptom_id,  # Each symptom is its own episode
                })
                symptom_id += 1

        # Sort by timestamp
        self._symptoms.sort(key=lambda s: s["timestamp"])
        self._generation_timestamp = datetime.now()
        return self._symptoms

    def to_dataframe(self) -> pd.DataFrame:
        """Combine meals and symptoms into a single DataFrame.

        Returns:
            DataFrame with columns:
            - timestamp: datetime
            - event_type: str ("meal" or "symptom")
            - ingredients: str (JSON-encoded list for meals, None for symptoms)
            - meal_id: int | None
            - symptom_id: int | None
            - episode_id: int | None
        """
        if not self._meals:
            raise ValueError("No meals generated. Call generate_meals() first.")

        rows = []

        # Add meal events
        for meal in self._meals:
            rows.append({
                "timestamp": meal["timestamp"],
                "event_type": "meal",
                "ingredients": json.dumps(meal["ingredients"]),
                "meal_id": meal["meal_id"],
                "symptom_id": None,
                "episode_id": None,
            })

        # Add symptom events
        for symptom in self._symptoms:
            rows.append({
                "timestamp": symptom["timestamp"],
                "event_type": "symptom",
                "ingredients": None,
                "meal_id": None,
                "symptom_id": symptom["symptom_id"],
                "episode_id": symptom["episode_id"],
            })

        df = pd.DataFrame(rows)
        df = df.sort_values("timestamp").reset_index(drop=True)
        return df

    def get_metadata(self) -> dict:
        """Get metadata about the generated diary.

        Returns:
            Dictionary with:
            - true_triggers: list[str]
            - negative_control: str
            - scenario_params: dict (all config params)
            - seed: int
            - generation_timestamp: datetime
            - summary: dict (counts and statistics)
        """
        # Count trigger exposures
        trigger_exposure_count = 0
        trigger_set = set(self._config.true_triggers)
        for meal in self._meals:
            if set(meal["ingredients"]) & trigger_set:
                trigger_exposure_count += 1

        return {
            "true_triggers": list(self._config.true_triggers),
            "negative_control": self._config.negative_control,
            "scenario_params": {
                "n_days": self._config.n_days,
                "meals_per_day_mean": self._config.meals_per_day_mean,
                "ingredients_per_meal_min": self._config.ingredients_per_meal_min,
                "ingredients_per_meal_max": self._config.ingredients_per_meal_max,
                "trigger_effect_strength": self._config.trigger_effect_strength,
                "trigger_exposure_rate": self._config.trigger_exposure_rate,
                "baseline_symptom_rate": self._config.baseline_symptom_rate,
                "hazard_window_hours": self._config.hazard_window_hours,
                "min_lag_hours": self._config.min_lag_hours,
                "ingredient_pool_size": len(self._config.ingredient_pool),
            },
            "seed": self._seed,
            "generation_timestamp": (
                self._generation_timestamp.isoformat()
                if self._generation_timestamp
                else None
            ),
            "summary": {
                "n_meals": len(self._meals),
                "n_symptoms": len(self._symptoms),
                "trigger_exposure_count": trigger_exposure_count,
                "trigger_exposure_rate_actual": (
                    trigger_exposure_count / len(self._meals)
                    if self._meals
                    else 0.0
                ),
            },
        }

    @property
    def meals(self) -> list[dict]:
        """Get generated meals."""
        return self._meals

    @property
    def symptoms(self) -> list[dict]:
        """Get generated symptoms."""
        return self._symptoms
