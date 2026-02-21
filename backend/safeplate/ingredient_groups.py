"""Ingredient grouping logic for trigger detection.

Groups allow us to infer triggers for related ingredients based on shared
chemical properties. For example, if "cheddar cheese" is a trigger, other
cheese group members (mozzarella, parmesan) are likely triggers too.
"""

from pathlib import Path
from typing import Optional

import pandas as pd

from .utils import normalize_ingredient_name


class IngredientGroupManager:
    """
    Manage ingredient-to-group mappings for group-aware trigger detection.

    Provides O(1) lookups for:
    - Getting the group of an ingredient
    - Getting all members of a group
    - Checking if two ingredients are in the same group
    """

    DEFAULT_GROUPS_PATH = "data/processed/ingredient_groups.csv"

    def __init__(self, groups_path: Optional[str] = None):
        """
        Initialize the ingredient group manager.

        Args:
            groups_path: Path to ingredient_groups.csv.
                        If None, uses default path relative to project root.
        """
        self._groups_path = groups_path
        self._ingredient_to_group: Optional[dict[str, str]] = None
        self._group_to_members: Optional[dict[str, list[str]]] = None
        self._loaded = False

    def _resolve_path(self) -> Path:
        """Resolve the groups file path."""
        if self._groups_path:
            return Path(self._groups_path)

        # Try to find project root
        current = Path.cwd()
        for parent in [current] + list(current.parents):
            candidate = parent / self.DEFAULT_GROUPS_PATH
            if candidate.exists():
                return candidate
            if (parent / "backend").exists() and (parent / "data").exists():
                return parent / self.DEFAULT_GROUPS_PATH

        return Path(self.DEFAULT_GROUPS_PATH)

    def load(self) -> None:
        """Load ingredient groups from CSV file."""
        if self._loaded:
            return

        path = self._resolve_path()
        if not path.exists():
            raise FileNotFoundError(f"Ingredient groups file not found: {path}")

        df = pd.read_csv(path)

        # Build ingredient -> group mapping (O(1) lookup)
        self._ingredient_to_group = {}
        for _, row in df.iterrows():
            ingredient = normalize_ingredient_name(row["ingredient"])
            group = normalize_ingredient_name(row["group_head"])
            self._ingredient_to_group[ingredient] = group

        # Build group -> members mapping (O(1) lookup)
        self._group_to_members = {}
        for ingredient, group in self._ingredient_to_group.items():
            if group not in self._group_to_members:
                self._group_to_members[group] = []
            self._group_to_members[group].append(ingredient)

        self._loaded = True

    def get_group(self, ingredient: str) -> Optional[str]:
        """
        Get the group head for an ingredient.

        Args:
            ingredient: Ingredient name

        Returns:
            Group head name, or None if ingredient not in any group
        """
        self.load()
        normalized = normalize_ingredient_name(ingredient)
        return self._ingredient_to_group.get(normalized)

    def get_group_members(self, group_head: str) -> list[str]:
        """
        Get all ingredients belonging to a group.

        Args:
            group_head: Group head name (e.g., "cheese")

        Returns:
            List of ingredient names in the group, or empty list if group not found
        """
        self.load()
        normalized = normalize_ingredient_name(group_head)
        return self._group_to_members.get(normalized, [])

    def are_in_same_group(self, ingredient1: str, ingredient2: str) -> bool:
        """
        Check if two ingredients belong to the same group.

        Args:
            ingredient1: First ingredient name
            ingredient2: Second ingredient name

        Returns:
            True if both ingredients are in the same group, False otherwise
        """
        group1 = self.get_group(ingredient1)
        group2 = self.get_group(ingredient2)

        if group1 is None or group2 is None:
            return False

        return group1 == group2

    def get_all_groups(self) -> dict[str, list[str]]:
        """
        Get all groups and their members.

        Returns:
            Dictionary mapping group_head -> list of member ingredients
        """
        self.load()
        return dict(self._group_to_members)

    def get_groups_with_multiple_members(self) -> dict[str, list[str]]:
        """
        Get only groups that have more than one member.

        These are the groups useful for trigger propagation.

        Returns:
            Dictionary of groups with 2+ members
        """
        self.load()
        return {
            group: members
            for group, members in self._group_to_members.items()
            if len(members) > 1
        }

    def expand_to_group_members(self, ingredients: list[str]) -> set[str]:
        """
        Expand a list of ingredients to include all group members.

        Useful for expanding trigger lists to include related ingredients.

        Args:
            ingredients: List of ingredient names

        Returns:
            Set of all ingredients including group members
        """
        self.load()
        expanded = set()

        for ingredient in ingredients:
            normalized = normalize_ingredient_name(ingredient)
            expanded.add(normalized)

            # Get group and add all members
            group = self.get_group(normalized)
            if group:
                members = self.get_group_members(group)
                expanded.update(members)

        return expanded

    def get_group_statistics(self) -> dict:
        """
        Get statistics about the ingredient groups.

        Returns:
            Dictionary with group statistics
        """
        self.load()
        group_sizes = [len(members) for members in self._group_to_members.values()]

        return {
            "total_ingredients": len(self._ingredient_to_group),
            "total_groups": len(self._group_to_members),
            "multi_member_groups": sum(1 for s in group_sizes if s > 1),
            "single_member_groups": sum(1 for s in group_sizes if s == 1),
            "largest_group_size": max(group_sizes) if group_sizes else 0,
            "average_group_size": sum(group_sizes) / len(group_sizes) if group_sizes else 0,
        }

    @property
    def num_ingredients(self) -> int:
        """Get total number of ingredients."""
        self.load()
        return len(self._ingredient_to_group)

    @property
    def num_groups(self) -> int:
        """Get total number of unique groups."""
        self.load()
        return len(self._group_to_members)


# Global singleton for caching
_default_manager: Optional[IngredientGroupManager] = None


def get_group_manager(groups_path: Optional[str] = None) -> IngredientGroupManager:
    """
    Get the ingredient group manager (singleton pattern for default path).

    Args:
        groups_path: Custom path to groups file.
                    If None, returns cached default manager.

    Returns:
        IngredientGroupManager instance
    """
    global _default_manager

    if groups_path is not None:
        return IngredientGroupManager(groups_path)

    if _default_manager is None:
        _default_manager = IngredientGroupManager()

    return _default_manager
