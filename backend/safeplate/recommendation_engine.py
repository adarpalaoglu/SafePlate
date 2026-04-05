"""Recommendation engine for finding similar ingredients using Node2Vec embeddings."""

from typing import Optional

import numpy as np

from .embedding_loader import EmbeddingLoader, get_embedding_loader
from .ingredient_groups import IngredientGroupManager, get_group_manager
from .utils import cosine_similarity_batch, normalize_ingredient_name


class IngredientNotFoundError(Exception):
    """Raised when a queried ingredient is not in the embeddings."""

    pass


class RecommendationEngine:
    """
    Find similar ingredients using Node2Vec embedding similarity.

    Uses cosine similarity between embedding vectors to find
    ingredients that appear in similar recipe contexts.
    """

    def __init__(self, embedding_loader: Optional[EmbeddingLoader] = None):
        """
        Initialize the recommendation engine.

        Args:
            embedding_loader: Custom embedding loader.
                            If None, uses the default singleton loader.
        """
        self._loader = embedding_loader or get_embedding_loader()

    def get_similar_ingredients(
        self,
        query_ingredient: str,
        top_n: int = 10,
        exclude_list: Optional[list[str]] = None
    ) -> list[tuple[str, float]]:
        """
        Find ingredients most similar to the query ingredient.

        Args:
            query_ingredient: The ingredient to find similar items for
            top_n: Number of similar ingredients to return
            exclude_list: List of ingredients to exclude from results
                         (useful for filtering out known triggers)

        Returns:
            List of tuples (ingredient_name, similarity_score)
            sorted by similarity (highest first)

        Raises:
            IngredientNotFoundError: If query_ingredient not in embeddings
        """
        # Normalize inputs
        normalized_query = normalize_ingredient_name(query_ingredient)
        exclude_set = set()
        if exclude_list:
            exclude_set = {normalize_ingredient_name(x) for x in exclude_list}

        # Always exclude the query ingredient itself
        exclude_set.add(normalized_query)

        # Get query embedding
        query_embedding = self._loader.get_embedding(normalized_query)
        if query_embedding is None:
            available = self._find_closest_match(normalized_query)
            suggestion = f" Did you mean '{available}'?" if available else ""
            raise IngredientNotFoundError(
                f"Ingredient '{query_ingredient}' not found in embeddings.{suggestion}"
            )

        # Compute similarities to all ingredients
        embedding_matrix = self._loader.get_embedding_matrix()
        similarities = cosine_similarity_batch(query_embedding, embedding_matrix)

        # Build results, filtering excluded ingredients
        all_ingredients = self._loader.get_all_ingredients()
        results: list[tuple[str, float]] = []

        for idx, (ingredient, sim) in enumerate(zip(all_ingredients, similarities)):
            if ingredient not in exclude_set:
                results.append((ingredient, float(sim)))

        # Sort by similarity (descending) and take top_n
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_n]

    def get_similarity_score(
        self,
        ingredient_a: str,
        ingredient_b: str
    ) -> float:
        """
        Get similarity score between two specific ingredients.

        Args:
            ingredient_a: First ingredient
            ingredient_b: Second ingredient

        Returns:
            Cosine similarity score between the two ingredients

        Raises:
            IngredientNotFoundError: If either ingredient not found
        """
        emb_a = self._loader.get_embedding(ingredient_a)
        emb_b = self._loader.get_embedding(ingredient_b)

        if emb_a is None:
            raise IngredientNotFoundError(f"Ingredient '{ingredient_a}' not found")
        if emb_b is None:
            raise IngredientNotFoundError(f"Ingredient '{ingredient_b}' not found")

        # Compute cosine similarity
        dot_product = np.dot(emb_a, emb_b)
        norm_a = np.linalg.norm(emb_a)
        norm_b = np.linalg.norm(emb_b)

        return float(dot_product / (norm_a * norm_b))

    def find_substitutes(
        self,
        ingredient: str,
        triggers: list[str],
        top_n: int = 5
    ) -> list[tuple[str, float]]:
        """
        Find substitute ingredients that avoid specified triggers.

        This is the core use case for dietary trigger avoidance:
        find similar ingredients while excluding known problematic ones.

        Args:
            ingredient: The ingredient to find substitutes for
            triggers: List of ingredients/categories to avoid
            top_n: Number of substitutes to return

        Returns:
            List of (ingredient, similarity_score) tuples
        """
        return self.get_similar_ingredients(
            query_ingredient=ingredient,
            top_n=top_n,
            exclude_list=triggers
        )

    def batch_similar_ingredients(
        self,
        ingredients: list[str],
        top_n: int = 5,
        exclude_list: Optional[list[str]] = None
    ) -> dict[str, list[tuple[str, float]]]:
        """
        Get similar ingredients for multiple query ingredients at once.

        Args:
            ingredients: List of ingredients to query
            top_n: Number of similar ingredients per query
            exclude_list: Global exclusion list

        Returns:
            Dictionary mapping each ingredient to its similar ingredients
        """
        results = {}
        for ingredient in ingredients:
            try:
                results[ingredient] = self.get_similar_ingredients(
                    query_ingredient=ingredient,
                    top_n=top_n,
                    exclude_list=exclude_list
                )
            except IngredientNotFoundError:
                results[ingredient] = []  # Empty list for not found
        return results

    def _find_closest_match(self, query: str, max_distance: int = 3) -> Optional[str]:
        """
        Find the closest matching ingredient name (for suggestions).

        Uses simple substring and prefix matching.
        """
        all_ingredients = self._loader.get_all_ingredients()

        # First, try substring match
        for ingredient in all_ingredients:
            if query in ingredient or ingredient in query:
                return ingredient

        # Try prefix match
        for ingredient in all_ingredients:
            if ingredient.startswith(query[:3]) or query.startswith(ingredient[:3]):
                return ingredient

        return None

    def list_all_ingredients(self) -> list[str]:
        """Get list of all available ingredients."""
        return self._loader.get_all_ingredients()

    def ingredient_exists(self, ingredient: str) -> bool:
        """Check if an ingredient exists in the embeddings."""
        return self._loader.has_ingredient(ingredient)

    @property
    def num_ingredients(self) -> int:
        """Get total number of ingredients."""
        return self._loader.num_ingredients

    @property
    def embedding_dim(self) -> int:
        """Get embedding dimension."""
        return self._loader.embedding_dim

    # Group-aware filtering methods

    def find_substitutes_group_aware(
        self,
        query_ingredient: str,
        trigger_ingredients: Optional[list[str]] = None,
        trigger_groups: Optional[list[str]] = None,
        top_n: int = 10,
        group_manager: Optional[IngredientGroupManager] = None
    ) -> list[tuple[str, float]]:
        """
        Find substitutes with group-aware filtering.

        This method excludes not only individual trigger ingredients but also
        all members of specified trigger groups. This is critical for dietary
        conditions where related ingredients share problematic properties.

        Example: If trigger_groups=["cheese"], excludes cheddar, mozzarella,
        parmesan, and all other cheese group members.

        Args:
            query_ingredient: The ingredient to find substitutes for
            trigger_ingredients: Individual ingredients to exclude
            trigger_groups: Group heads whose members should all be excluded
            top_n: Number of substitutes to return
            group_manager: Custom group manager (uses default if None)

        Returns:
            List of (ingredient, similarity_score) tuples
        """
        gm = group_manager or get_group_manager()

        # Start with individual trigger ingredients
        exclude_set: set[str] = set()
        if trigger_ingredients:
            exclude_set.update(
                normalize_ingredient_name(t) for t in trigger_ingredients
            )

        # Expand trigger groups to include all their members
        if trigger_groups:
            for group_head in trigger_groups:
                members = gm.get_group_members(group_head)
                exclude_set.update(members)

        return self.get_similar_ingredients(
            query_ingredient=query_ingredient,
            top_n=top_n,
            exclude_list=list(exclude_set)
        )

    def find_substitutes_expand_triggers(
        self,
        query_ingredient: str,
        trigger_ingredients: list[str],
        expand_to_groups: bool = True,
        top_n: int = 10,
        group_manager: Optional[IngredientGroupManager] = None
    ) -> tuple[list[tuple[str, float]], set[str]]:
        """
        Find substitutes, optionally expanding triggers to their group members.

        This is useful when you want to automatically expand individual triggers
        to include related ingredients. For example, if "cheddar cheese" is a
        trigger and expand_to_groups=True, all cheese group members are excluded.

        Args:
            query_ingredient: The ingredient to find substitutes for
            trigger_ingredients: Known trigger ingredients
            expand_to_groups: If True, expand triggers to include group members
            top_n: Number of substitutes to return
            group_manager: Custom group manager (uses default if None)

        Returns:
            Tuple of:
            - List of (ingredient, similarity_score) tuples
            - Set of all excluded ingredients (for transparency)
        """
        gm = group_manager or get_group_manager()

        if expand_to_groups:
            exclude_set = gm.expand_to_group_members(trigger_ingredients)
        else:
            exclude_set = {
                normalize_ingredient_name(t) for t in trigger_ingredients
            }

        results = self.get_similar_ingredients(
            query_ingredient=query_ingredient,
            top_n=top_n,
            exclude_list=list(exclude_set)
        )

        return results, exclude_set

    def get_trigger_group_info(
        self,
        trigger_ingredients: list[str],
        group_manager: Optional[IngredientGroupManager] = None
    ) -> dict[str, dict]:
        """
        Get group information for a list of trigger ingredients.

        Useful for showing users which groups their triggers belong to
        and what related ingredients might also be problematic.

        Args:
            trigger_ingredients: List of trigger ingredient names
            group_manager: Custom group manager (uses default if None)

        Returns:
            Dictionary mapping each trigger to its group info:
            {
                "cheddar cheese": {
                    "group": "cheese",
                    "group_members": ["mozzarella", "parmesan", ...],
                    "group_size": 105
                },
                ...
            }
        """
        gm = group_manager or get_group_manager()
        result = {}

        for trigger in trigger_ingredients:
            normalized = normalize_ingredient_name(trigger)
            group = gm.get_group(normalized)

            if group:
                members = gm.get_group_members(group)
                # Exclude the trigger itself from the members list
                other_members = [m for m in members if m != normalized]
                result[trigger] = {
                    "group": group,
                    "group_members": other_members,
                    "group_size": len(members)
                }
            else:
                result[trigger] = {
                    "group": None,
                    "group_members": [],
                    "group_size": 0
                }

        return result
