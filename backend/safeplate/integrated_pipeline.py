"""Integrated pipeline connecting case-crossover results to shrinkage and recommendations.

This module provides the ``SafePlatePipeline`` class, which is the primary
integration layer for the SafePlate dietary trigger detection system.  It
connects three independently validated components:

1. **Case-crossover output** (produced by notebook 04) — per-ingredient
   discordant counts (U, V) from the N-of-1 case-crossover analysis.
2. **Empirical Bayes shrinkage** (``GroupAwareTriggerDetector``) — the
   DerSimonian & Laird (1986) hierarchical model that borrows strength
   across ingredient groups to stabilise sparse trigger estimates.
3. **Recommendation engine** (``RecommendationEngine``) — Node2Vec
   embedding-based ingredient similarity for finding safe substitutes.

The pipeline accepts a pandas DataFrame of case-crossover statistics,
applies group-aware shrinkage to produce calibrated trigger risk scores,
and uses those scores to filter substitution recommendations so that
trigger ingredients (and their group relatives) are excluded.

Usage::

    from safeplate.embedding_loader import EmbeddingLoader
    from safeplate.ingredient_groups import IngredientGroupManager
    from safeplate.integrated_pipeline import SafePlatePipeline, PipelineConfig

    loader = EmbeddingLoader("data/processed/embeddings/...")
    gm = IngredientGroupManager("data/processed/ingredient_groups.csv")

    pipeline = SafePlatePipeline(
        embedding_loader=loader,
        group_manager=gm,
        config=PipelineConfig(trigger_or_threshold=1.5),
    )

    import pandas as pd
    stats = pd.read_csv("data/processed/trigger_detection/ingredient_stats_all.csv")

    shrinkage_result = pipeline.detect_triggers_from_stats(stats)
    triggers = pipeline.get_trigger_ingredients(shrinkage_result)
    substitutes = pipeline.get_safe_substitutes("milk", shrinkage_result)

Reference:
    DerSimonian, R. & Laird, N. (1986). Meta-analysis in clinical trials.
    Controlled Clinical Trials, 7(3), 177-188.
"""

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from .embedding_loader import EmbeddingLoader
from .group_aware_trigger_detector import (
    FullShrinkageOutput,
    GroupAwareTriggerDetector,
    ShrinkageResult,
)
from .ingredient_groups import IngredientGroupManager
from .recommendation_engine import IngredientNotFoundError, RecommendationEngine

# Ordered ranking for confidence filtering
_CONFIDENCE_RANK = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}

# Columns that notebook 04 must have produced
_REQUIRED_COLUMNS = {"ingredient", "U_case_not_ctrl", "V_ctrl_not_case"}


@dataclass
class PipelineConfig:
    """Configuration for the SafePlate integrated pipeline.

    Attributes:
        trigger_or_threshold: Minimum final OR to consider an ingredient
            a trigger.  Default 1.0 (any OR above parity).
        trigger_min_confidence: Minimum confidence level for trigger
            inclusion.  Must be ``"HIGH"``, ``"MEDIUM"``, or ``"LOW"``.
            Default ``"MEDIUM"`` excludes only untested group-mean
            inheritors (``"LOW"``).
        propagate_to_untested: If True, untested group members inherit
            the group mean from the shrinkage model.  Default True.
        recommendation_top_n: Default number of substitute ingredients
            to return.  Default 10.
    """

    trigger_or_threshold: float = 1.0
    trigger_min_confidence: str = "MEDIUM"
    propagate_to_untested: bool = True
    recommendation_top_n: int = 10

    def __post_init__(self) -> None:
        if self.trigger_min_confidence not in _CONFIDENCE_RANK:
            raise ValueError(
                f"trigger_min_confidence must be one of "
                f"{list(_CONFIDENCE_RANK.keys())}, "
                f"got '{self.trigger_min_confidence}'"
            )


class SafePlatePipeline:
    """End-to-end pipeline: case-crossover stats -> shrinkage -> recommendations.

    This class orchestrates the three core SafePlate components so that
    downstream consumers (notebooks, Flask UI, test harnesses) can call
    a single coherent API rather than wiring the components together
    manually each time.
    """

    def __init__(
        self,
        embedding_loader: EmbeddingLoader,
        group_manager: IngredientGroupManager,
        config: Optional[PipelineConfig] = None,
    ) -> None:
        """Initialise the pipeline with its component dependencies.

        Args:
            embedding_loader: A configured ``EmbeddingLoader`` instance
                pointing at the Node2Vec embeddings file.
            group_manager: A configured ``IngredientGroupManager``
                instance pointing at ``ingredient_groups.csv``.
            config: Pipeline configuration.  If ``None``, uses
                ``PipelineConfig()`` defaults.
        """
        self._config = config or PipelineConfig()
        self._group_manager = group_manager
        self._detector = GroupAwareTriggerDetector(group_manager=group_manager)
        self._engine = RecommendationEngine(embedding_loader=embedding_loader)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_triggers_from_stats(
        self,
        ingredient_stats: pd.DataFrame,
    ) -> FullShrinkageOutput:
        """Run the empirical Bayes shrinkage model on case-crossover output.

        Validates the input DataFrame, converts it to the row-dict format
        expected by ``GroupAwareTriggerDetector.run_shrinkage_from_dataframe_rows``,
        and returns the full shrinkage output.

        Args:
            ingredient_stats: DataFrame with at least the columns
                ``ingredient``, ``U_case_not_ctrl``, and
                ``V_ctrl_not_case``.  Additional columns are ignored.

        Returns:
            ``FullShrinkageOutput`` containing per-ingredient and
            per-group shrinkage results.

        Raises:
            ValueError: If required columns are missing from the
                DataFrame, or if the DataFrame is empty.
        """
        if ingredient_stats.empty:
            raise ValueError(
                "ingredient_stats DataFrame is empty.  "
                "Run notebook 04 first to produce case-crossover results."
            )

        actual_cols = set(ingredient_stats.columns)
        missing = _REQUIRED_COLUMNS - actual_cols
        if missing:
            raise ValueError(
                f"ingredient_stats is missing required columns: "
                f"{sorted(missing)}.  "
                f"Expected at least: {sorted(_REQUIRED_COLUMNS)}.  "
                f"Got: {sorted(actual_cols)}."
            )

        rows = ingredient_stats.to_dict(orient="records")

        return self._detector.run_shrinkage_from_dataframe_rows(
            rows=rows,
            ingredient_col="ingredient",
            u_col="U_case_not_ctrl",
            v_col="V_ctrl_not_case",
            propagate_to_groups=self._config.propagate_to_untested,
        )

    def get_trigger_ingredients(
        self,
        shrinkage_result: FullShrinkageOutput,
    ) -> list[str]:
        """Extract ingredient names that qualify as triggers.

        An ingredient is considered a trigger if:
        - Its ``final_OR`` >= ``config.trigger_or_threshold``, AND
        - Its ``confidence`` meets or exceeds
          ``config.trigger_min_confidence``.

        Args:
            shrinkage_result: Output from ``detect_triggers_from_stats``.

        Returns:
            List of trigger ingredient names, sorted alphabetically.

        Notes:
            Protected-safe ingredients are never included regardless
            of their OR, because ``protected == True`` implies
            ``evidence_source == "protected_safe"`` and the raw OR
            is below 1.0 by construction.
        """
        min_rank = _CONFIDENCE_RANK[self._config.trigger_min_confidence]
        threshold = self._config.trigger_or_threshold

        triggers = [
            r.ingredient
            for r in shrinkage_result.ingredients
            if (
                r.final_OR >= threshold
                and _CONFIDENCE_RANK.get(r.confidence, 0) >= min_rank
                and not r.protected
            )
        ]
        return sorted(triggers)

    def get_safe_substitutes(
        self,
        query_ingredient: str,
        shrinkage_result: FullShrinkageOutput,
        top_n: Optional[int] = None,
    ) -> list[tuple[str, float]]:
        """Find substitute ingredients that avoid detected triggers.

        Uses the trigger list from the shrinkage result as the exclusion
        set for the recommendation engine.

        Args:
            query_ingredient: The ingredient to find substitutes for.
            shrinkage_result: Output from ``detect_triggers_from_stats``.
            top_n: Number of substitutes to return.  If ``None``, uses
                ``config.recommendation_top_n``.

        Returns:
            List of ``(ingredient_name, similarity_score)`` tuples,
            sorted by descending similarity.

        Raises:
            IngredientNotFoundError: If ``query_ingredient`` is not in
                the Node2Vec embeddings.
        """
        n = top_n if top_n is not None else self._config.recommendation_top_n
        trigger_list = self.get_trigger_ingredients(shrinkage_result)

        return self._engine.find_substitutes(
            ingredient=query_ingredient,
            triggers=trigger_list,
            top_n=n,
        )

    def get_pipeline_summary(
        self,
        shrinkage_result: FullShrinkageOutput,
    ) -> dict:
        """Produce a high-level summary of the shrinkage analysis.

        Args:
            shrinkage_result: Output from ``detect_triggers_from_stats``.

        Returns:
            Dictionary with keys:
            - ``n_ingredients_total``: Total ingredients in output.
            - ``n_tested``: Ingredients with U + V > 0.
            - ``n_untested``: Ingredients with U = V = 0.
            - ``n_protected_safe``: Ingredients protected by Rule 1.
            - ``n_triggers``: Ingredients qualifying as triggers under
              current config.
            - ``n_high_confidence``: Count with confidence ``"HIGH"``.
            - ``n_medium_confidence``: Count with confidence ``"MEDIUM"``.
            - ``n_low_confidence``: Count with confidence ``"LOW"``.
            - ``trigger_list``: Sorted list of trigger ingredient names.
            - ``protected_safe_list``: Sorted list of protected-safe
              ingredient names.
        """
        all_results = shrinkage_result.ingredients
        trigger_list = self.get_trigger_ingredients(shrinkage_result)
        protected = [r.ingredient for r in all_results if r.protected]

        return {
            "n_ingredients_total": len(all_results),
            "n_tested": sum(1 for r in all_results if r.U + r.V > 0),
            "n_untested": sum(1 for r in all_results if r.U == 0 and r.V == 0),
            "n_protected_safe": len(protected),
            "n_triggers": len(trigger_list),
            "n_high_confidence": sum(
                1 for r in all_results if r.confidence == "HIGH"
            ),
            "n_medium_confidence": sum(
                1 for r in all_results if r.confidence == "MEDIUM"
            ),
            "n_low_confidence": sum(
                1 for r in all_results if r.confidence == "LOW"
            ),
            "trigger_list": trigger_list,
            "protected_safe_list": sorted(protected),
        }
