"""Case-crossover simulator for SafePlate test lab.

Converts synthetic diary data into (U, V) discordant pair statistics
suitable for input to the trigger detection pipeline.

The case-crossover design compares ingredient exposure in hazard windows
(preceding symptoms) versus control windows (matched time periods without
symptoms) to identify ingredients associated with symptom occurrence.
"""

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd


@dataclass
class CaseCrossoverConfig:
    """Configuration for case-crossover simulation.

    Attributes:
        hazard_window_hours: Hours before symptom to check for exposure.
        min_lag_hours: Minimum hours between meal and symptom for exposure.
        control_windows_per_episode: Number of control windows per symptom episode.
        match_by_day_of_week: Whether to match controls by day of week.
        match_by_time_of_day: Whether to match controls by time of day.
        time_of_day_tolerance_hours: Tolerance for time-of-day matching.
        episode_gap_hours: Minimum hours between symptoms to be separate episodes.
    """

    hazard_window_hours: int = 6
    min_lag_hours: int = 1
    control_windows_per_episode: int = 4
    match_by_day_of_week: bool = True
    match_by_time_of_day: bool = True
    time_of_day_tolerance_hours: int = 2
    episode_gap_hours: int = 6


class CaseCrossoverSimulator:
    """Simulates case-crossover analysis on diary data.

    Converts a diary DataFrame containing meals and symptoms into
    (U, V) discordant pair statistics for each ingredient. These
    statistics can be fed directly into the SafePlate trigger
    detection pipeline.

    Example:
        >>> config = CaseCrossoverConfig(
        ...     hazard_window_hours=6,
        ...     control_windows_per_episode=4,
        ... )
        >>> simulator = CaseCrossoverSimulator(config)
        >>> stats = simulator.run(diary_df)
        >>> print(stats[["ingredient", "U_case_not_ctrl", "V_ctrl_not_case"]])
    """

    def __init__(self, config: Optional[CaseCrossoverConfig] = None) -> None:
        """Initialize the case-crossover simulator.

        Args:
            config: Configuration parameters. Uses defaults if None.
        """
        self._config = config or CaseCrossoverConfig()

    def run(self, diary: pd.DataFrame) -> pd.DataFrame:
        """Run case-crossover analysis on diary data.

        Args:
            diary: DataFrame with columns:
                - timestamp: datetime
                - event_type: str ("meal" or "symptom")
                - ingredients: str (JSON-encoded list for meals)
                - meal_id: int | None
                - symptom_id: int | None
                - episode_id: int | None

        Returns:
            DataFrame with columns:
                - ingredient: str
                - U_case_not_ctrl: int (meals in hazard but not control)
                - V_ctrl_not_case: int (meals in control but not hazard)

        Raises:
            ValueError: If diary has no symptom episodes.
        """
        # Ensure timestamp is datetime
        diary = diary.copy()
        diary["timestamp"] = pd.to_datetime(diary["timestamp"])

        # Separate meals and symptoms
        meals_df = diary[diary["event_type"] == "meal"].copy()
        symptoms_df = diary[diary["event_type"] == "symptom"].copy()

        if symptoms_df.empty:
            raise ValueError(
                "Diary has no symptom episodes. Cannot run case-crossover analysis."
            )

        # Parse meal ingredients
        meals_df["ingredients_list"] = meals_df["ingredients"].apply(
            lambda x: json.loads(x) if pd.notna(x) else []
        )

        # Cluster symptoms into episodes
        episodes = self._cluster_symptoms_into_episodes(symptoms_df)

        # Get all unique ingredients
        all_ingredients: set[str] = set()
        for ing_list in meals_df["ingredients_list"]:
            all_ingredients.update(ing_list)

        # Initialize counts
        U_counts: dict[str, int] = {ing: 0 for ing in all_ingredients}
        V_counts: dict[str, int] = {ing: 0 for ing in all_ingredients}

        # Process each episode
        for episode in episodes:
            episode_time = episode["timestamp"]

            # Define hazard window
            hazard_start = episode_time - timedelta(
                hours=self._config.hazard_window_hours
            )
            hazard_end = episode_time - timedelta(hours=self._config.min_lag_hours)

            # Get meals in hazard window
            hazard_meals = meals_df[
                (meals_df["timestamp"] >= hazard_start)
                & (meals_df["timestamp"] <= hazard_end)
            ]

            # Get control windows
            control_windows = self._sample_control_windows(
                episode_time,
                meals_df,
                symptoms_df,
                episodes,
            )

            # Get ingredients in hazard window
            hazard_ingredients: set[str] = set()
            for _, meal in hazard_meals.iterrows():
                hazard_ingredients.update(meal["ingredients_list"])

            # Get ingredients in control windows
            control_ingredients: set[str] = set()
            for ctrl_start, ctrl_end in control_windows:
                ctrl_meals = meals_df[
                    (meals_df["timestamp"] >= ctrl_start)
                    & (meals_df["timestamp"] <= ctrl_end)
                ]
                for _, meal in ctrl_meals.iterrows():
                    control_ingredients.update(meal["ingredients_list"])

            # Count discordant pairs
            for ing in all_ingredients:
                in_hazard = ing in hazard_ingredients
                in_control = ing in control_ingredients

                if in_hazard and not in_control:
                    U_counts[ing] += 1
                elif in_control and not in_hazard:
                    V_counts[ing] += 1

        # Build result DataFrame
        rows = []
        for ing in sorted(all_ingredients):
            rows.append({
                "ingredient": ing,
                "U_case_not_ctrl": U_counts[ing],
                "V_ctrl_not_case": V_counts[ing],
            })

        return pd.DataFrame(rows)

    def _cluster_symptoms_into_episodes(
        self, symptoms_df: pd.DataFrame
    ) -> list[dict]:
        """Cluster symptoms into episodes based on time gaps.

        Symptoms within episode_gap_hours of each other are grouped
        into the same episode. For simplicity, we use the first
        symptom in each cluster as the episode representative.

        Args:
            symptoms_df: DataFrame of symptom events.

        Returns:
            List of episode dictionaries with timestamp.
        """
        if symptoms_df.empty:
            return []

        symptoms_sorted = symptoms_df.sort_values("timestamp").reset_index(drop=True)
        episodes = []
        current_episode_start = None

        gap_threshold = timedelta(hours=self._config.episode_gap_hours)

        for _, row in symptoms_sorted.iterrows():
            ts = row["timestamp"]

            if current_episode_start is None:
                # Start new episode
                current_episode_start = ts
                episodes.append({"timestamp": ts, "symptom_ids": [row["symptom_id"]]})
            elif ts - episodes[-1]["timestamp"] > gap_threshold:
                # Start new episode (gap too large)
                current_episode_start = ts
                episodes.append({"timestamp": ts, "symptom_ids": [row["symptom_id"]]})
            else:
                # Add to current episode
                episodes[-1]["symptom_ids"].append(row["symptom_id"])

        return episodes

    def _sample_control_windows(
        self,
        episode_time: datetime,
        meals_df: pd.DataFrame,
        symptoms_df: pd.DataFrame,
        all_episodes: list[dict],
    ) -> list[tuple[datetime, datetime]]:
        """Sample control windows for a given episode.

        Control windows are matched by:
        - Day of week (if enabled)
        - Time of day (if enabled)
        - Not overlapping with any symptom or hazard window

        Args:
            episode_time: Timestamp of the episode.
            meals_df: All meals in the diary.
            symptoms_df: All symptoms in the diary.
            all_episodes: All episode timestamps (for exclusion).

        Returns:
            List of (start, end) tuples for control windows.
        """
        window_duration = timedelta(
            hours=self._config.hazard_window_hours - self._config.min_lag_hours
        )
        min_lag = timedelta(hours=self._config.min_lag_hours)

        # Get diary time range
        all_times = meals_df["timestamp"]
        diary_start = all_times.min()
        diary_end = all_times.max()

        # Episode times to exclude
        episode_times = {ep["timestamp"] for ep in all_episodes}

        # Candidate days (before the episode day)
        episode_day = episode_time.date()
        candidate_days = []

        current_day = diary_start.date()
        while current_day < episode_day:
            # Check day of week matching
            if self._config.match_by_day_of_week:
                if current_day.weekday() == episode_day.weekday():
                    candidate_days.append(current_day)
            else:
                candidate_days.append(current_day)
            current_day += timedelta(days=1)

        # Sample control windows
        control_windows = []
        time_tolerance = timedelta(hours=self._config.time_of_day_tolerance_hours)

        for day in candidate_days:
            if len(control_windows) >= self._config.control_windows_per_episode:
                break

            # Calculate control window time on this day
            episode_time_of_day = episode_time.time()
            ctrl_end = datetime.combine(day, episode_time_of_day) - min_lag
            ctrl_start = ctrl_end - window_duration

            # Check time-of-day matching
            if self._config.match_by_time_of_day:
                # Control window should be at similar time of day
                ctrl_hour = ctrl_end.hour
                episode_hour = episode_time.hour
                if abs(ctrl_hour - episode_hour) > self._config.time_of_day_tolerance_hours:
                    continue

            # Check for overlap with symptoms
            overlaps_symptom = False
            for _, sym in symptoms_df.iterrows():
                sym_time = sym["timestamp"]
                # Exclude if control window is within hazard range of any symptom
                sym_hazard_start = sym_time - timedelta(
                    hours=self._config.hazard_window_hours
                )
                sym_hazard_end = sym_time

                if not (ctrl_end < sym_hazard_start or ctrl_start > sym_hazard_end):
                    overlaps_symptom = True
                    break

            if not overlaps_symptom:
                # Check for overlap with other hazard windows
                overlaps_hazard = False
                for ep_time in episode_times:
                    if ep_time == episode_time:
                        continue
                    ep_hazard_start = ep_time - timedelta(
                        hours=self._config.hazard_window_hours
                    )
                    ep_hazard_end = ep_time - min_lag

                    if not (ctrl_end < ep_hazard_start or ctrl_start > ep_hazard_end):
                        overlaps_hazard = True
                        break

                if not overlaps_hazard:
                    control_windows.append((ctrl_start, ctrl_end))

        return control_windows[:self._config.control_windows_per_episode]
