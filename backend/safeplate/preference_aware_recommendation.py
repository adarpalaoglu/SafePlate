"""Preference-aware recommendation engine for SafePlate test lab.

Generates ingredient recommendations with:
- Dynamic trigger filtering based on shrinkage results
- User preference integration (respects stored rankings)
- Backfilling when ranked items become unsafe

This module calls existing pipeline methods and does NOT reimplement
any shrinkage or similarity logic.
"""

import logging
from dataclasses import dataclass
from typing import Optional

from .embedding_loader import EmbeddingLoader
from .group_aware_trigger_detector import FullShrinkageOutput
from .ingredient_groups import IngredientGroupManager
from .integrated_pipeline import SafePlatePipeline
from .recommendation_engine import IngredientNotFoundError
from .user_preferences import UserPreferenceStore

logger = logging.getLogger(__name__)


@dataclass
class RecommendationExplanation:
    """Explanation for why an ingredient was recommended.

    This is a SKELETON implementation for Phase 1.
    Full explanation generation will be added in Phase 3.
    """

    safety_reason: str
    similarity_reason: str
    preference_reason: Optional[str]

    def to_text(self) -> str:
        """Generate human-readable explanation text.

        Returns:
            Explanation text (placeholder in Phase 1).
        """
        return "Explanation coming in Phase 3"


@dataclass
class Recommendation:
    """A single ingredient recommendation.

    Attributes:
        ingredient: The recommended ingredient name.
        similarity_score: Node2Vec cosine similarity to query (0-1).
        is_safe: Whether the ingredient is safe (not a trigger).
        user_rank: Original position in user's ranking (1-indexed),
                  or None if backfilled from similarity search.
        explanation: Detailed explanation (None in Phase 1).
    """

    ingredient: str
    similarity_score: float
    is_safe: bool
    user_rank: Optional[int]
    explanation: Optional[RecommendationExplanation]


class PreferenceAwareRecommendationEngine:
    """Recommendation engine with trigger filtering and user preferences.

    This engine combines three sources of information:
    1. Safety: Excludes ingredients identified as triggers
    2. Similarity: Uses Node2Vec embeddings for ranking
    3. Preference: Respects user's stored ingredient rankings

    The algorithm:
    1. Get trigger list from shrinkage results
    2. If user has stored ranking: filter triggers, backfill if needed
    3. If no ranking: use similarity-based recommendations
    4. Return Recommendation objects with metadata

    Example:
        >>> engine = PreferenceAwareRecommendationEngine(
        ...     pipeline=pipeline,
        ...     group_manager=gm,
        ...     embedding_loader=loader,
        ...     user_pref_store=pref_store,
        ... )
        >>> recs = engine.get_substitutes_with_preferences(
        ...     "milk", shrinkage_result, top_n=10
        ... )
        >>> for rec in recs:
        ...     print(f"{rec.ingredient}: rank={rec.user_rank}")
    """

    def __init__(
        self,
        pipeline: SafePlatePipeline,
        group_manager: IngredientGroupManager,
        embedding_loader: EmbeddingLoader,
        user_pref_store: UserPreferenceStore,
    ) -> None:
        """Initialize the preference-aware recommendation engine.

        Args:
            pipeline: SafePlatePipeline for trigger detection.
            group_manager: IngredientGroupManager for group lookups.
            embedding_loader: EmbeddingLoader for similarity searches.
            user_pref_store: UserPreferenceStore for ranking persistence.
        """
        self._pipeline = pipeline
        self._group_manager = group_manager
        self._embedding_loader = embedding_loader
        self._user_pref_store = user_pref_store

    def get_substitutes_with_preferences(
        self,
        query_ingredient: str,
        shrinkage_result: FullShrinkageOutput,
        top_n: int = 10,
    ) -> list[Recommendation]:
        """Get substitute recommendations with preference integration.

        Algorithm:
        1. Get triggers from shrinkage result (NO group expansion)
        2. Check for stored user ranking
        3. If ranking exists:
           - Filter out triggers from ranking
           - Take top_n safe items
           - Backfill from similarity if insufficient
        4. If no ranking:
           - Get similar ingredients via embeddings
           - Filter out triggers
           - Take top_n

        Args:
            query_ingredient: The ingredient to find substitutes for.
            shrinkage_result: Output from pipeline.detect_triggers_from_stats().
            top_n: Number of recommendations to return.

        Returns:
            List of Recommendation objects.

        Raises:
            IngredientNotFoundError: If query_ingredient not in embeddings.
        """
        # Step 1: Get triggers (NO group expansion per spec)
        triggers = self._pipeline.get_trigger_ingredients(shrinkage_result)
        trigger_set = set(triggers)

        # Step 2: Check for stored ranking
        user_ranking = self._user_pref_store.get_ranking(query_ingredient)

        if user_ranking:
            # Step 3: Use user ranking with trigger filtering
            return self._recommendations_from_ranking(
                query_ingredient=query_ingredient,
                user_ranking=user_ranking.ranked_ingredients,
                trigger_set=trigger_set,
                top_n=top_n,
            )
        else:
            # Step 4: Use similarity-based recommendations
            return self._recommendations_from_similarity(
                query_ingredient=query_ingredient,
                trigger_set=trigger_set,
                top_n=top_n,
            )

    def _recommendations_from_ranking(
        self,
        query_ingredient: str,
        user_ranking: list[str],
        trigger_set: set[str],
        top_n: int,
    ) -> list[Recommendation]:
        """Generate recommendations from user ranking with trigger filtering.

        Args:
            query_ingredient: The ingredient being substituted.
            user_ranking: User's ordered preference list.
            trigger_set: Set of trigger ingredients to exclude.
            top_n: Number of recommendations to return.

        Returns:
            List of Recommendation objects.
        """
        recommendations: list[Recommendation] = []

        # Filter ranking to safe items, preserving order
        original_positions: dict[str, int] = {
            ing: idx + 1 for idx, ing in enumerate(user_ranking)
        }
        safe_ranking = [ing for ing in user_ranking if ing not in trigger_set]

        # Get similarity scores for all items (for metadata)
        similarity_scores = self._get_similarity_scores(query_ingredient, safe_ranking)

        # Add safe ranked items
        for ing in safe_ranking[:top_n]:
            recommendations.append(
                Recommendation(
                    ingredient=ing,
                    similarity_score=similarity_scores.get(ing, 0.0),
                    is_safe=True,
                    user_rank=original_positions[ing],
                    explanation=None,
                )
            )

        # Backfill if needed
        if len(recommendations) < top_n:
            backfill_count = top_n - len(recommendations)
            backfilled = self._backfill_recommendations(
                query_ingredient=query_ingredient,
                trigger_set=trigger_set,
                exclude_set=set(user_ranking),  # Exclude all ranked items
                count=backfill_count,
            )
            recommendations.extend(backfilled)

        return recommendations

    def _recommendations_from_similarity(
        self,
        query_ingredient: str,
        trigger_set: set[str],
        top_n: int,
    ) -> list[Recommendation]:
        """Generate recommendations from similarity search.

        Args:
            query_ingredient: The ingredient being substituted.
            trigger_set: Set of trigger ingredients to exclude.
            top_n: Number of recommendations to return.

        Returns:
            List of Recommendation objects.
        """
        # Get similar ingredients
        try:
            similar = self._embedding_loader.get_all_ingredients()
            # We need to use the recommendation engine's method
            # which handles similarity computation properly
            from .recommendation_engine import RecommendationEngine

            temp_engine = RecommendationEngine(self._embedding_loader)
            similar_with_scores = temp_engine.get_similar_ingredients(
                query_ingredient, top_n=50, exclude_list=list(trigger_set)
            )
        except IngredientNotFoundError:
            raise

        recommendations: list[Recommendation] = []
        for ing, score in similar_with_scores[:top_n]:
            if ing not in trigger_set:
                recommendations.append(
                    Recommendation(
                        ingredient=ing,
                        similarity_score=score,
                        is_safe=True,
                        user_rank=None,
                        explanation=None,
                    )
                )

        return recommendations

    def _backfill_recommendations(
        self,
        query_ingredient: str,
        trigger_set: set[str],
        exclude_set: set[str],
        count: int,
    ) -> list[Recommendation]:
        """Backfill recommendations from similarity when ranking is insufficient.

        Args:
            query_ingredient: The ingredient being substituted.
            trigger_set: Set of trigger ingredients to exclude.
            exclude_set: Additional items to exclude (already in results).
            count: Number of items to backfill.

        Returns:
            List of backfilled Recommendation objects (user_rank=None).
        """
        if count <= 0:
            return []

        # Combined exclusion set
        full_exclude = trigger_set | exclude_set

        try:
            from .recommendation_engine import RecommendationEngine

            temp_engine = RecommendationEngine(self._embedding_loader)
            similar_with_scores = temp_engine.get_similar_ingredients(
                query_ingredient, top_n=50 + len(full_exclude), exclude_list=list(full_exclude)
            )
        except IngredientNotFoundError:
            logger.warning(
                f"Could not backfill recommendations: {query_ingredient} not in embeddings"
            )
            return []

        backfilled: list[Recommendation] = []
        for ing, score in similar_with_scores:
            if ing not in full_exclude:
                backfilled.append(
                    Recommendation(
                        ingredient=ing,
                        similarity_score=score,
                        is_safe=True,
                        user_rank=None,  # Backfilled, not from ranking
                        explanation=None,
                    )
                )
                if len(backfilled) >= count:
                    break

        if len(backfilled) < count:
            logger.warning(
                f"Insufficient candidates for backfilling: requested {count}, got {len(backfilled)}"
            )

        return backfilled

    def _get_similarity_scores(
        self, query_ingredient: str, ingredients: list[str]
    ) -> dict[str, float]:
        """Get similarity scores for a list of ingredients.

        Args:
            query_ingredient: The query ingredient.
            ingredients: List of ingredients to score.

        Returns:
            Dictionary mapping ingredient to similarity score.
        """
        scores: dict[str, float] = {}

        try:
            from .recommendation_engine import RecommendationEngine

            temp_engine = RecommendationEngine(self._embedding_loader)

            for ing in ingredients:
                try:
                    score = temp_engine.get_similarity_score(query_ingredient, ing)
                    scores[ing] = score
                except IngredientNotFoundError:
                    scores[ing] = 0.0
        except IngredientNotFoundError:
            # Query ingredient not found
            for ing in ingredients:
                scores[ing] = 0.0

        return scores

    def generate_explanation(
        self,
        rec: Recommendation,
        query: str,
        shrinkage_result: FullShrinkageOutput,
    ) -> RecommendationExplanation:
        """Generate explanation for a recommendation.

        PLACEHOLDER implementation for Phase 1.
        Full explanation generation will be added in Phase 3.

        Args:
            rec: The Recommendation to explain.
            query: The query ingredient.
            shrinkage_result: The shrinkage analysis result.

        Returns:
            RecommendationExplanation (placeholder content).
        """
        return RecommendationExplanation(
            safety_reason="Placeholder - Phase 3",
            similarity_reason="Placeholder - Phase 3",
            preference_reason="Placeholder - Phase 3" if rec.user_rank else None,
        )
