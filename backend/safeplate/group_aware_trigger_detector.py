"""Group-aware trigger detection via empirical Bayes hierarchical shrinkage.

.. note:: Production Status

   This module implements the empirical Bayes hierarchical shrinkage model
   (DerSimonian & Laird, 1986).  It was implemented, tested (see
   ``tests/test_shrinkage.py``), and then deliberately removed from the
   production detection path.  When meals are sampled from real recipes,
   ingredient clustering within recipes violates the group homogeneity
   assumption that shrinkage relies on, causing genuine trigger signals to
   be pulled toward the group mean and inflating confounded co-occurring
   ingredients.  The full rationale for the removal is discussed in the
   dissertation.

   The ``ShrinkageResult`` and ``FullShrinkageOutput`` data classes defined
   here ARE still used in production (in ``app.py``) as data containers for
   raw Haldane-corrected odds ratio results.  The shrinkage-specific fields
   on these classes (``weight``, ``mu_g``, ``tau_sq``) are set to dummy
   values (0.0) in the production path.

This module implements a two-level empirical Bayes model for dietary trigger
detection that borrows strength across ingredients within the same food group.
The group-level hyperparameters (mean log-OR μ_g and between-ingredient
variance τ²) are estimated via the DerSimonian & Laird (1986) method-of-moments
estimator, and each ingredient's raw log-OR is then shrunk toward the group
mean in proportion to its own uncertainty.

Two guard rules are applied before shrinkage:
    Rule 1 (protect safe): ingredients with statistically significant
        evidence in the safety direction are excluded from the shrinkage
        pool so they cannot be overridden by a trigger-heavy group.
    Rule 2 (shrink everything else): all remaining tested ingredients
        contribute to and receive shrinkage from the group mean. Untested
        group members inherit the group mean directly.

Reference:
    DerSimonian, R. & Laird, N. (1986). Meta-analysis in clinical trials.
    Controlled Clinical Trials, 7(3), 177-188.
"""

from dataclasses import dataclass, field
from math import comb
from typing import Optional

import numpy as np

from .ingredient_groups import IngredientGroupManager, get_group_manager
from .utils import normalize_ingredient_name



def _binomial_survival(v: int, n: int) -> float:
    """Compute P(X >= v | X ~ Binomial(n, 0.5)) exactly.

    Uses the combinatorial identity:

        P(X >= v) = sum_{x=v}^{n} C(n, x) * 0.5^n

    computed via Python arbitrary-precision integers (math.comb) divided
    by 2^n.  No normal approximation is used.  For the sample sizes that
    arise in food-diary case-crossover studies (n typically < 100), this
    is both fast and exact to floating-point precision.

    Args:
        v: Threshold (non-negative integer).
        n: Number of trials (non-negative integer).

    Returns:
        Exact tail probability.
    """
    if v <= 0:
        return 1.0
    if v > n:
        return 0.0
    total = sum(comb(n, x) for x in range(v, n + 1))
    return total / (2 ** n)



def _raw_estimates(U: int, V: int) -> tuple[float, float, float]:
    """Compute Haldane-corrected log-OR, its variance, and the raw OR.

    The 0.5 Haldane correction is applied consistently to both the point
    estimate and the variance so that zero cells do not produce infinities.

        θ̂  = log( (U + 0.5) / (V + 0.5) )
        σ² = 1/(U + 0.5)  +  1/(V + 0.5)

    Args:
        U: Discordant strata where ingredient appears only in hazard window.
        V: Discordant strata where ingredient appears only in control window.

    Returns:
        (theta_hat, sigma_sq, raw_OR)
    """
    u_corr = U + 0.5
    v_corr = V + 0.5
    theta_hat = float(np.log(u_corr / v_corr))
    sigma_sq = 1.0 / u_corr + 1.0 / v_corr
    raw_OR = float(np.exp(theta_hat))
    return theta_hat, sigma_sq, raw_OR


# Rule 1: protect safe

def _is_protected_safe(U: int, V: int, raw_OR: float) -> bool:
    """Test whether an ingredient qualifies for Rule 1 protection.

    The test is one-sided in the SAFETY direction: we ask whether the
    observed number of control-only discordant strata (V) is improbably
    large under the null hypothesis of no association (p = 0.5).  A one-
    sided test is used because we are specifically guarding against
    false-positive trigger labels; an ingredient that is eaten often
    with few symptoms should not be pulled toward a high group mean.

    The significance level is fixed at 0.05 (conventional level, not
    tuned on data) to avoid overfitting the protection threshold.

    Both conditions must hold:
        (a) P(X >= V | n = U+V, p = 0.5) < 0.05
        (b) raw OR < 1.0

    Args:
        U: Case-only discordant count.
        V: Control-only discordant count.
        raw_OR: Haldane-corrected odds ratio.

    Returns:
        True if the ingredient is protected safe.
    """
    n = U + V
    if n == 0:
        return False
    p_safe = _binomial_survival(V, n)
    return p_safe < 0.05 and raw_OR < 1.0


def _is_own_trigger(U: int, V: int, raw_OR: float) -> bool:
    """Test whether an ingredient's own signal is a significant trigger.

    Mirrors Rule 1 but in the trigger direction: tests whether U is
    improbably large under the null.

    Args:
        U: Case-only discordant count.
        V: Control-only discordant count.
        raw_OR: Haldane-corrected odds ratio.

    Returns:
        True if the ingredient has significant own-trigger evidence.
    """
    n = U + V
    if n == 0:
        return False
    p_trigger = _binomial_survival(U, n)
    return p_trigger < 0.05 and raw_OR > 1.0



def _dersimonian_laird(
    thetas: list[float],
    sigma_sqs: list[float],
) -> tuple[float, float]:
    """Estimate group mean and between-ingredient variance via D&L.

    This is a point-estimation (method-of-moments) procedure for the
    hyperparameters of the empirical Bayes model.  It is NOT a fully
    Bayesian computation - no prior distribution is placed on (μ_g, τ²).

    Inputs are the non-protected pool members only.  Protected-safe
    ingredients have already been removed before this function is called.

    The τ² floor at 0 (via the max() in the formula) is part of the
    original DerSimonian & Laird estimator, not an ad hoc clip.  A
    negative moment estimate of τ² simply indicates that all observed
    heterogeneity is attributable to within-ingredient sampling variance.

    Args:
        thetas:    List of θ̂ᵢ values for pool members.
        sigma_sqs: List of σᵢ² values for pool members.

    Returns:
        (mu_g, tau_sq).  Returns (NaN, NaN) if the pool is empty.
        Returns (θ̂₁, 0.0) if the pool has exactly one member.
    """
    k = len(thetas)

    if k == 0:
        return float("nan"), float("nan")

    if k == 1:
        # A single ingredient provides no information about between-
        # ingredient variance.  μ_g is just that ingredient's estimate.
        return thetas[0], 0.0

    # Inverse-variance weights
    ws = [1.0 / s for s in sigma_sqs]
    sum_w = sum(ws)

    # Weighted mean
    mu_g = sum(w * t for w, t in zip(ws, thetas)) / sum_w

    # Cochran's Q
    Q = sum(w * (t - mu_g) ** 2 for w, t in zip(ws, thetas))

    # D&L denominator
    C = sum_w - sum(w * w for w in ws) / sum_w

    # Between-ingredient variance (floored at 0 per D&L)
    tau_sq = max(0.0, (Q - (k - 1)) / C)

    return mu_g, tau_sq


@dataclass
class ShrinkageResult:
    """Per-ingredient output from the shrinkage model.

    Also used in production as a plain data container for raw OR results,
    with shrinkage fields (weight, mu_g, tau_sq) set to 0.0.
    """

    ingredient: str
    U: int
    V: int
    raw_log_OR: float
    raw_OR: float
    sigma_sq: float
    group: Optional[str]
    final_log_OR: float
    final_OR: float
    weight: float
    mu_g: float
    tau_sq: float
    confidence: str
    evidence_source: str
    protected: bool

    def to_dict(self) -> dict:
        """Return a plain dictionary of all fields."""
        return {
            "ingredient": self.ingredient,
            "U": self.U,
            "V": self.V,
            "raw_log_OR": self.raw_log_OR,
            "raw_OR": self.raw_OR,
            "sigma_sq": self.sigma_sq,
            "group": self.group,
            "final_log_OR": self.final_log_OR,
            "final_OR": self.final_OR,
            "weight": self.weight,
            "mu_g": self.mu_g,
            "tau_sq": self.tau_sq,
            "confidence": self.confidence,
            "evidence_source": self.evidence_source,
            "protected": self.protected,
        }


@dataclass
class GroupShrinkageResult:
    """Per-group summary from the shrinkage model."""

    group: str
    n_tested: int
    n_protected: int
    n_in_pool: int
    mu_g: float
    tau_sq: float
    members: list[str]

    def to_dict(self) -> dict:
        """Return a plain dictionary of all fields."""
        return {
            "group": self.group,
            "n_tested": self.n_tested,
            "n_protected": self.n_protected,
            "n_in_pool": self.n_in_pool,
            "mu_g": self.mu_g,
            "tau_sq": self.tau_sq,
            "members": list(self.members),
        }


@dataclass
class FullShrinkageOutput:
    """Top-level container for the complete shrinkage analysis."""

    ingredients: list[ShrinkageResult] = field(default_factory=list)
    groups: list[GroupShrinkageResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Return a serialisable dictionary."""
        return {
            "ingredients": [r.to_dict() for r in self.ingredients],
            "groups": [g.to_dict() for g in self.groups],
        }

    def get_all_trigger_ingredients(self, min_OR: float = 1.0) -> set[str]:
        """Return ingredient names with final_OR >= min_OR."""
        return {
            r.ingredient for r in self.ingredients
            if r.final_OR >= min_OR
        }

    def get_protected_safe(self) -> list[ShrinkageResult]:
        """Return all protected-safe ingredients."""
        return [r for r in self.ingredients if r.protected]

    def get_high_confidence(self) -> list[ShrinkageResult]:
        """Return all HIGH-confidence results."""
        return [r for r in self.ingredients if r.confidence == "HIGH"]


class GroupAwareTriggerDetector:
    """Two-level empirical Bayes trigger detector with group shrinkage.

    This class takes raw (ingredient, U, V) counts from the case-crossover
    pipeline and produces shrinkage-adjusted odds ratios using the
    DerSimonian & Laird (1986) empirical Bayes procedure.
    """

    def __init__(
        self, group_manager: Optional[IngredientGroupManager] = None
    ) -> None:
        self._gm = group_manager or get_group_manager()

    def run_shrinkage(
        self,
        ingredient_triggers: list[tuple[str, int, int]],
        all_group_members: Optional[dict[str, list[str]]] = None,
    ) -> FullShrinkageOutput:
        """Run the full shrinkage pipeline.

        Args:
            ingredient_triggers:
                List of (ingredient_name, U, V) tuples from the case-
                crossover notebook.  U and V are non-negative integers.
            all_group_members:
                If provided, untested group members are included in the
                output with group_mean scores.  Typically populated via
                ``group_manager.get_all_groups()``.

        Returns:
            FullShrinkageOutput containing per-ingredient and per-group
            results.
        """
        # Phase 1: compute raw estimates for every tested ingredient
        tested: dict[str, dict] = {}
        for name, U, V in ingredient_triggers:
            norm = normalize_ingredient_name(name)
            theta_hat, sigma_sq, raw_OR = _raw_estimates(int(U), int(V))
            group = self._gm.get_group(norm)
            tested[norm] = {
                "U": int(U),
                "V": int(V),
                "theta_hat": theta_hat,
                "sigma_sq": sigma_sq,
                "raw_OR": raw_OR,
                "group": group,
            }

        # Phase 2: Rule 1 - identify protected-safe ingredients

        protected_set: set[str] = set()
        for name, d in tested.items():
            if _is_protected_safe(d["U"], d["V"], d["raw_OR"]):
                protected_set.add(name)

        # Phase 3: build shrinkage pools per group.
        # Grouped ingredients use their group_head as pool key. Ungrouped
        # ingredients each get a singleton pool keyed by (None, name).

        # pool_key → list of (name, θ̂, σ²)
        pools: dict = {}
        # pool_key → all tested member names (including protected)
        group_tested_members: dict = {}
        # pool_key → count of protected members
        group_protected_count: dict = {}

        for name, d in tested.items():
            group = d["group"]
            key = group if group is not None else (None, name)

            if key not in pools:
                pools[key] = []
                group_tested_members[key] = []
                group_protected_count[key] = 0

            group_tested_members[key].append(name)

            if name in protected_set:
                group_protected_count[key] += 1
            else:
                pools[key].append((name, d["theta_hat"], d["sigma_sq"]))

        # Phase 4: D&L estimation per pool

        # pool_key → (μ_g, τ²)
        dl_estimates: dict = {}
        for key, pool in pools.items():
            thetas = [t for _, t, _ in pool]
            sigmas = [s for _, _, s in pool]
            dl_estimates[key] = _dersimonian_laird(thetas, sigmas)

        # Phase 5: compute final scores for tested ingredients

        results: list[ShrinkageResult] = []

        for name, d in tested.items():
            group = d["group"]
            key = group if group is not None else (None, name)
            mu_g, tau_sq = dl_estimates.get(key, (float("nan"), float("nan")))

            if name in protected_set:
                # Rule 1: own data fully trusted, no shrinkage
                results.append(ShrinkageResult(
                    ingredient=name,
                    U=d["U"],
                    V=d["V"],
                    raw_log_OR=d["theta_hat"],
                    raw_OR=d["raw_OR"],
                    sigma_sq=d["sigma_sq"],
                    group=group,
                    final_log_OR=d["theta_hat"],
                    final_OR=d["raw_OR"],
                    weight=1.0,
                    mu_g=mu_g,
                    tau_sq=tau_sq,
                    confidence="HIGH",
                    evidence_source="protected_safe",
                    protected=True,
                ))
                continue

            # Rule 2: shrinkage
            if tau_sq == 0.0:
                w_i = 0.0
            else:
                w_i = tau_sq / (tau_sq + d["sigma_sq"])

            final_log_OR = w_i * d["theta_hat"] + (1.0 - w_i) * mu_g
            final_OR = float(np.exp(final_log_OR))

            # Determine evidence source and confidence
            if _is_own_trigger(d["U"], d["V"], d["raw_OR"]):
                evidence_source = "own_trigger"
                confidence = "HIGH"
            else:
                evidence_source = "shrunk"
                confidence = "MEDIUM" if w_i > 0.5 else "LOW"

            results.append(ShrinkageResult(
                ingredient=name,
                U=d["U"],
                V=d["V"],
                raw_log_OR=d["theta_hat"],
                raw_OR=d["raw_OR"],
                sigma_sq=d["sigma_sq"],
                group=group,
                final_log_OR=final_log_OR,
                final_OR=final_OR,
                weight=w_i,
                mu_g=mu_g,
                tau_sq=tau_sq,
                confidence=confidence,
                evidence_source=evidence_source,
                protected=False,
            ))

        # Phase 6: untested group members inherit the group mean

        if all_group_members is not None:
            seen = set(tested.keys())
            for group_head, members in all_group_members.items():
                group_norm = normalize_ingredient_name(group_head)
                # Only real group keys (strings), not singleton tuples
                if group_norm not in dl_estimates:
                    continue
                mu_g, tau_sq = dl_estimates[group_norm]
                if np.isnan(mu_g):
                    continue  # Pool was empty; nothing to propagate

                for member in members:
                    member_norm = normalize_ingredient_name(member)
                    if member_norm in seen:
                        continue
                    seen.add(member_norm)

                    results.append(ShrinkageResult(
                        ingredient=member_norm,
                        U=0,
                        V=0,
                        raw_log_OR=float("nan"),
                        raw_OR=float("nan"),
                        sigma_sq=float("inf"),
                        group=group_norm,
                        final_log_OR=mu_g,
                        final_OR=float(np.exp(mu_g)),
                        weight=0.0,
                        mu_g=mu_g,
                        tau_sq=tau_sq,
                        confidence="LOW",
                        evidence_source="group_mean",
                        protected=False,
                    ))

        # Phase 7: build GroupShrinkageResult for real groups

        group_results: list[GroupShrinkageResult] = []
        for key in pools:
            if isinstance(key, tuple):
                continue  # Skip ungrouped singletons
            group_head = key
            mu_g, tau_sq = dl_estimates[group_head]
            group_results.append(GroupShrinkageResult(
                group=group_head,
                n_tested=len(group_tested_members[group_head]),
                n_protected=group_protected_count[group_head],
                n_in_pool=len(pools[group_head]),
                mu_g=mu_g,
                tau_sq=tau_sq,
                members=list(group_tested_members[group_head]),
            ))

        return FullShrinkageOutput(ingredients=results, groups=group_results)

    def run_shrinkage_from_dataframe_rows(
        self,
        rows: list[dict],
        ingredient_col: str = "ingredient",
        u_col: str = "U_case_not_ctrl",
        v_col: str = "V_ctrl_not_case",
        propagate_to_groups: bool = True,
    ) -> FullShrinkageOutput:
        """Convenience wrapper for notebook use.

        Extracts (ingredient, U, V) from dictionaries keyed by the column
        names that notebook 04 produces.

        Args:
            rows: List of dicts, each representing one row of the
                  case-crossover output.
            ingredient_col: Key for the ingredient name.
            u_col: Key for the U (case-only discordant) count.
            v_col: Key for the V (control-only discordant) count.
            propagate_to_groups: If True, calls ``get_all_groups()`` and
                                 includes untested group members.

        Returns:
            FullShrinkageOutput.
        """
        triggers = [
            (row[ingredient_col], int(row[u_col]), int(row[v_col]))
            for row in rows
        ]

        all_group_members = None
        if propagate_to_groups:
            all_group_members = self._gm.get_all_groups()

        return self.run_shrinkage(triggers, all_group_members)
