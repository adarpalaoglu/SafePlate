"""User preference storage for SafePlate test lab.

Stores and retrieves user rankings for ingredient substitutions.
Rankings are persisted to a JSON file for use across sessions.
"""

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass
class UserRanking:
    """A user's ranked list of substitute ingredients.

    Attributes:
        query_ingredient: The ingredient being substituted.
        ranked_ingredients: Ordered list of substitutes (rank 1 to N).
        timestamp: When the ranking was created/updated.
    """

    query_ingredient: str
    ranked_ingredients: list[str]
    timestamp: datetime

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "query_ingredient": self.query_ingredient,
            "ranked_ingredients": self.ranked_ingredients,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, query: str, data: dict) -> "UserRanking":
        """Create from dictionary loaded from JSON."""
        return cls(
            query_ingredient=query,
            ranked_ingredients=data["ranked_ingredients"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
        )


class UserPreferenceStore:
    """Persistent storage for user ingredient preference rankings.

    Rankings are stored in a JSON file and can be retrieved by
    query ingredient. Each query can have one ranking (updated
    on subsequent saves).

    Example:
        >>> store = UserPreferenceStore("data/user_preferences.json")
        >>> store.save_ranking("milk", ["oat milk", "almond milk", "soy milk"])
        >>> ranking = store.get_ranking("milk")
        >>> print(ranking.ranked_ingredients)
        ['oat milk', 'almond milk', 'soy milk']
    """

    def __init__(self, storage_path: str = "data/user_preferences.json") -> None:
        """Initialize the preference store.

        Args:
            storage_path: Path to JSON file for persistence.
        """
        self._storage_path = Path(storage_path)
        self._cache: dict[str, UserRanking] = {}
        self._loaded = False

    def _ensure_loaded(self) -> None:
        """Load data from file if not already loaded."""
        if self._loaded:
            return

        if self._storage_path.exists():
            try:
                with open(self._storage_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for query, ranking_data in data.items():
                        self._cache[query] = UserRanking.from_dict(query, ranking_data)
            except (json.JSONDecodeError, KeyError) as e:
                # Handle corrupted file gracefully
                print(f"Warning: Could not load preferences from {self._storage_path}: {e}")
                self._cache = {}

        self._loaded = True

    def _save_to_file(self) -> None:
        """Save current cache to file."""
        # Ensure directory exists
        self._storage_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            query: ranking.to_dict()
            for query, ranking in self._cache.items()
        }

        with open(self._storage_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def save_ranking(self, query: str, ranked_ingredients: list[str]) -> None:
        """Save a ranking for a query ingredient.

        If a ranking already exists for the query, it is replaced.

        Args:
            query: The ingredient being substituted.
            ranked_ingredients: Ordered list of substitute preferences.
        """
        self._ensure_loaded()

        ranking = UserRanking(
            query_ingredient=query,
            ranked_ingredients=ranked_ingredients,
            timestamp=datetime.now(),
        )

        self._cache[query] = ranking
        self._save_to_file()

    def get_ranking(self, query: str) -> Optional[UserRanking]:
        """Get the stored ranking for a query ingredient.

        Args:
            query: The ingredient to look up.

        Returns:
            UserRanking if exists, None otherwise.
        """
        self._ensure_loaded()
        return self._cache.get(query)

    def has_ranking(self, query: str) -> bool:
        """Check if a ranking exists for a query ingredient.

        Args:
            query: The ingredient to check.

        Returns:
            True if ranking exists, False otherwise.
        """
        self._ensure_loaded()
        return query in self._cache

    def load_all(self) -> dict[str, UserRanking]:
        """Get all stored rankings.

        Returns:
            Dictionary mapping query ingredients to their rankings.
        """
        self._ensure_loaded()
        return dict(self._cache)

    def delete_ranking(self, query: str) -> bool:
        """Delete the ranking for a query ingredient.

        Args:
            query: The ingredient to delete ranking for.

        Returns:
            True if ranking was deleted, False if it didn't exist.
        """
        self._ensure_loaded()

        if query in self._cache:
            del self._cache[query]
            self._save_to_file()
            return True
        return False

    def clear_all(self) -> None:
        """Delete all stored rankings."""
        self._cache = {}
        self._loaded = True
        self._save_to_file()
