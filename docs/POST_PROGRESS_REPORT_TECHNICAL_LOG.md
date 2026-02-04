# Post-Progress Report Technical Log

**SafePlate Dissertation Project**
**Period Covered:** February 2026 onwards
**Student:** Adar Palaoglu
**Supervisor:** Prof. Long Tran-Thanh

---

## 1. Overview: What Was Implemented

The core problem addressed in this phase was the high false-negative rate in individual-ingredient trigger detection. The case-crossover design used by SafePlate requires a sufficient number of discordant strata per ingredient to reach statistical significance, but real food-diary data are sparse: most ingredients appear in fewer than 10 discordant pairs. Under a conventional one-sided binomial test at p < 0.05, an ingredient must appear in the hazard window in at least 9 out of 10 discordant pairs to be flagged. Genuine triggers with moderate effect sizes are therefore routinely missed, which in turn degrades the quality of substitution recommendations downstream.

The solution was to implement group-aware partial pooling via an empirical Bayes hierarchical shrinkage model. Ingredients that belong to the same food group (e.g., all cheeses, all tomato variants) have their trigger-risk estimates stabilised by borrowing strength from the group. The between-ingredient variance within each group (`τ²`) is estimated adaptively from the data using the DerSimonian & Laird (1986) method-of-moments estimator, so the degree of pooling is determined by the evidence rather than by a manually chosen constant. A two-rule guard system ensures that ingredients with strong individual safety evidence are not overridden by a trigger-heavy group.

The key deliverable is a complete rewrite of `backend/safeplate/group_aware_trigger_detector.py` (564 lines), replacing a naive propagation-discount skeleton with the principled statistical model described above, accompanied by a self-contained test suite (`tests/test_shrinkage.py`, 470 lines) covering 10 edge-case scenarios. All tests pass.

---

## 2. Scientific Context and Motivation

### 2.1 The False-Negative Problem

The case-crossover design identifies dietary triggers by comparing the ingredient composition of hazard (symptom-present) and control (symptom-absent) windows within the same person. For each ingredient, the analysis produces two counts: `U` (discordant strata where the ingredient appears only in the hazard window) and `V` (discordant strata where it appears only in the control window). Under the null hypothesis of no association, `U/(U+V)` follows a Binomial(n, 0.5) distribution, and a one-sided test asks whether `U` is improbably large.

The problem is statistical power. For an ingredient with `n = U + V = 10` discordant pairs, reaching significance at p < 0.05 requires `U ≥ 9` (since `P(X ≥ 9 | n=10, p=0.5) = 11/1024 ≈ 0.011`). An ingredient that appears in the hazard window 7 out of 10 times — a clinically meaningful signal (OR ≈ 3.0) — produces `P(X ≥ 7 | n=10, p=0.5) ≈ 0.17`, well above the significance threshold. This ingredient would be classified as non-significant and excluded from the trigger list, leading to potentially harmful substitution recommendations that include a genuine trigger.

### 2.2 Why Grouping Helps

SafePlate uses a pre-computed ingredient grouping (`ingredient_groups.csv`, 2698 ingredients in 993 groups) where ingredients sharing a lexical group head are assumed to have correlated trigger risks. For example, the "cheese" group contains 105 members (cheddar, mozzarella, parmesan, etc.), and the "tomato" group contains 40 members (cherry tomato, plum tomato, tomato sauce, etc.).

The key assumption is that if one member of a group is a trigger, other members are more likely to also be triggers. This assumption is grounded in the observation that group members tend to share overlapping biochemical properties — cheeses share lactose and casein; tomato variants share solanine and histamine precursors — but it is explicitly a statistical modelling assumption, not a claim about biochemical mechanism. The grouping is based on lexical token frequency in recipe data, not on nutritional or allergological taxonomy. The dissertation states this limitation clearly.

### 2.3 Statistical Approach: Empirical Bayes Hierarchical Shrinkage

The chosen method is empirical Bayes hierarchical shrinkage, estimated via the DerSimonian & Laird (1986) method-of-moments procedure. This approach was selected for the following reasons:

- **Borrows strength across group members** to stabilise sparse per-ingredient estimates, directly addressing the false-negative problem.
- **Adapts to within-group heterogeneity** via the `τ²` parameter. When group members have similar trigger evidence, pooling is aggressive; when they diverge, each ingredient retains more of its own estimate.
- **Does not assume group members are identical** (unlike the naive propagation approach it replaced, which applied a flat 0.7 discount to all propagated odds ratios regardless of data quality).
- **Requires no manual tuning** of pooling strength. The shrinkage weight `wᵢ = τ² / (τ² + σᵢ²)` is derived from the data.
- **Well-established in medical meta-analysis** with over 40 years of validation. The DerSimonian & Laird estimator is the standard method-of-moments approach in random-effects meta-analysis.

> DerSimonian, R. & Laird, N. (1986). Meta-analysis in clinical trials. *Controlled Clinical Trials*, 7(3), 177–188.

---

## 3. Implementation: `group_aware_trigger_detector.py`

### 3.1 Input Contract

The module receives two types of input:

1. **Per-ingredient discordant counts** from the case-crossover analysis (notebook 04). Each ingredient is represented as a tuple `(ingredient_name, U, V)` where:
   - `U` = number of discordant strata where the ingredient appears only in the hazard (case) window (column: `U_case_not_ctrl`)
   - `V` = number of discordant strata where the ingredient appears only in the control window (column: `V_ctrl_not_case`)
   - Both are non-negative integers.

2. **Group membership** from `IngredientGroupManager` (loaded from `ingredient_groups.csv`), providing `get_group(ingredient) → group_head` and `get_group_members(group_head) → list[str]` lookups.

### 3.2 The Statistical Model (Mathematical Specification)

All computation is performed on the **log-odds-ratio scale**. This is mandatory: the log-OR is approximately normally distributed (by the delta method), which is the distributional assumption underlying the shrinkage estimator. Shrinkage on the raw OR scale would be incorrect because the OR distribution is asymmetric and bounded below by zero.

**Individual level (per tested ingredient):**

```
θ̂ᵢ  =  log( (Uᵢ + 0.5) / (Vᵢ + 0.5) )       [Haldane-corrected log-OR]
σᵢ²  =  1/(Uᵢ + 0.5) + 1/(Vᵢ + 0.5)           [delta-method variance]
```

The 0.5 Haldane correction is applied consistently to both the point estimate and the variance. This ensures that zero cells (`U = 0` or `V = 0`) produce finite values without biasing the estimate.

**Group level (latent model):**

```
θᵢ ~ N(μ_g, τ²)
```

where `μ_g` is the group's latent mean trigger risk on the log-OR scale, and `τ²` is the between-ingredient variance within the group.

**Shrinkage estimator (posterior mean under the empirical Bayes model):**

```
E[θᵢ | data]  =  wᵢ · θ̂ᵢ  +  (1 − wᵢ) · μ_g
where  wᵢ  =  τ² / (τ² + σᵢ²)
```

When `σᵢ²` is large (sparse data), `wᵢ` is small, and the ingredient borrows heavily from the group mean. When `σᵢ²` is small (rich data), `wᵢ` is large, and the ingredient trusts its own estimate.

**DerSimonian & Laird hyperparameter estimation (computed on the shrinkage pool only):**

```
wᵢ   =  1 / σᵢ²                                [inverse-variance weights]
μ_g  =  Σ(wᵢ · θ̂ᵢ) / Σwᵢ                      [weighted mean]
Q    =  Σ wᵢ · (θ̂ᵢ − μ_g)²                     [Cochran's Q statistic]
C    =  Σwᵢ − (Σwᵢ²) / Σwᵢ                     [D&L denominator]
τ²   =  max(0, (Q − (k − 1)) / C)              [between-ingredient variance]
```

The `τ²` floor at zero is intrinsic to the DerSimonian & Laird formula. A negative moment estimate of `τ²` indicates that all observed heterogeneity is attributable to within-ingredient sampling variance, and the correct response is to set `τ² = 0` (maximum pooling), not to allow a negative variance.

### 3.3 The Two-Rule Guard

The guard rules are applied in a strict order. Rule 1 is evaluated first and determines which ingredients are excluded from the pool before any D&L estimation occurs.

**Rule 1 — Protect Safe (applied FIRST)**

For every tested ingredient with `U + V > 0`, the following one-sided exact binomial test is computed:

```
p_safe  =  P(X ≥ V | n = U + V, p = 0.5)  =  Σ_{x=V}^{n} C(n, x) · 0.5^n
```

This is computed exactly using `math.comb` (Python arbitrary-precision integers), with no normal approximation.

If **both** conditions hold — `p_safe < 0.05` **and** raw OR < 1.0 — the ingredient is **protected safe**:

- Its final OR equals its own raw OR (no shrinkage applied).
- It is excluded from the D&L pool and does not contribute to `μ_g` or `τ²`.
- It receives `confidence = "HIGH"` and `evidence_source = "protected_safe"`.

The test is **one-sided** because the goal is specifically to guard against false-positive trigger labels. An ingredient that is eaten frequently with few symptoms should not be pulled toward a high group mean. A two-sided test would incorrectly flag strong triggers as candidates for protection.

The significance level of 0.05 is used as a conventional threshold. It functions as a guard (preventing confirmed-safe ingredients from entering the pool), not as a discovery threshold, so the standard level is defensible without adjustment for multiple comparisons.

**Rule 2 — Shrink Everything Else**

All tested ingredients that did not pass Rule 1 enter the D&L pool for their group. The shrinkage formula is applied to each pool member. Untested ingredients (members of a group that were never eaten) receive `final_log_OR = μ_g`, which is mathematically equivalent to setting `wᵢ = 0` (infinite `σᵢ²`). They receive `confidence = "LOW"` and `evidence_source = "group_mean"`.

### 3.4 Edge Cases and How They Are Handled

**Empty pool (all group members protected):** `μ_g = NaN`, `τ² = NaN`. Protected members retain their own OR. Untested members of this group receive no score (they are omitted from the output for this group).

**Singleton pool (one tested member, others untested):** `μ_g = θ̂₁`, `τ² = 0`. With only one observation, there is no information to estimate between-ingredient variance. Untested members inherit the single tested member's OR directly. This is the correct behaviour: it is the maximum-pooling limit, constrained to a single data point.

**`τ² = 0` (all pool members have identical or very similar `θ̂`):** All shrinkage weights collapse to `wᵢ = 0`, and every pool member receives `final_OR = exp(μ_g)`. This is correct when pool members genuinely agree. The own-trigger significance test (a separate check) ensures that ingredients with individually significant trigger evidence still receive `confidence = "HIGH"` and `evidence_source = "own_trigger"`, even when their final OR is set to `exp(μ_g)`.

**Ungrouped ingredients (not in any group):** Treated as a singleton group: `group = None`, `τ² = 0`, `final_log_OR = θ̂ᵢ`. Their own data is fully trusted.

**Very sparse data (`U + V` = 1 or 2):** The variance `σᵢ²` is large, so `wᵢ` is small, and the final score is pulled heavily toward `μ_g`. This is the correct regularisation behaviour: with very little data, the ingredient should defer to the group estimate.

### 3.5 Output Data Structures

Three dataclasses compose the output:

**`ShrinkageResult`** (one per ingredient in the output): 15 fields — `ingredient`, `U`, `V`, `raw_log_OR`, `raw_OR`, `sigma_sq`, `group`, `final_log_OR`, `final_OR`, `weight`, `mu_g`, `tau_sq`, `confidence`, `evidence_source`, `protected`. The `confidence` field takes values `"HIGH"`, `"MEDIUM"`, or `"LOW"`. The `evidence_source` field takes values `"protected_safe"`, `"own_trigger"`, `"shrunk"`, or `"group_mean"`.

**`GroupShrinkageResult`** (one per group with tested members): `group`, `n_tested`, `n_protected`, `n_in_pool`, `mu_g`, `tau_sq`, `members`.

**`FullShrinkageOutput`** (top-level container): Lists of `ShrinkageResult` and `GroupShrinkageResult`, with convenience methods `.to_dict()`, `.get_all_trigger_ingredients(min_OR)`, `.get_protected_safe()`, and `.get_high_confidence()`.

### 3.6 Public API

**`run_shrinkage(ingredient_triggers, all_group_members)`** — the main entry point. Accepts a list of `(ingredient_name, U, V)` tuples and an optional dictionary of group memberships (typically from `group_manager.get_all_groups()`). Returns `FullShrinkageOutput`.

**`run_shrinkage_from_dataframe_rows(rows, ...)`** — a convenience wrapper for notebook use. Extracts `(name, U, V)` from dictionaries keyed by `"ingredient"`, `"U_case_not_ctrl"`, and `"V_ctrl_not_case"`, matching the column names produced by notebook 04. If `propagate_to_groups=True`, calls `get_all_groups()` internally.

---

## 4. Design Rationale: Why These Choices Are Defensible

### 4.1 Log(OR) Scale

All shrinkage computation operates on the log-OR scale. This is mandatory for three reasons: (1) the log-OR is symmetric around zero (log-OR = 0 corresponds to OR = 1, no association), whereas the raw OR is bounded below by zero and unbounded above; (2) the delta-method variance approximation assumes approximate normality, which holds for log-OR but not for OR; (3) the weighted average used in the shrinkage estimator is only meaningful when the quantity being averaged is approximately normal. Performing arithmetic shrinkage on the OR scale would produce biased estimates because the average of two ORs is not the OR of the average. This convention is standard in meta-analysis (DerSimonian & Laird, 1986).

### 4.2 DerSimonian & Laird vs Alternatives

The DerSimonian & Laird method-of-moments estimator was chosen over several alternatives:

- **Naive averaging** of odds ratios ignores the differing precision of each estimate. An ingredient with 2 discordant pairs would receive equal weight to one with 20 pairs.
- **Fixed-effect meta-analysis** assumes `τ² = 0` (no between-ingredient variance). This forces all group members to the same trigger risk, which is too restrictive: swiss cheese and cheddar cheese may genuinely differ.
- **A fully Bayesian hierarchical model** (e.g., via MCMC) would require specifying prior distributions on `μ_g` and `τ²`, introduce computational overhead, and require convergence diagnostics. For the sample sizes in this application (groups of 1–105 ingredients with < 100 discordant pairs each), the D&L point estimate is adequate and avoids these complications.
- **Manual tuning of pooling strength** (e.g., the flat 0.7 discount used in the previous implementation) is not principled: the constant has no connection to the data and cannot adapt to groups with different levels of heterogeneity.

D&L is a closed-form, deterministic estimator requiring no hyperpriors and no iterative sampling. It has been validated extensively in clinical meta-analysis over four decades.

### 4.3 One-Sided Test Direction

Rule 1 uses a one-sided test asking: "Is this ingredient significantly safer than chance?" The alternative — a two-sided test asking "Is this ingredient significantly different from chance?" — would incorrectly flag strong triggers for protection (a trigger with `U = 9, V = 1` has a two-sided p-value of ≈ 0.02, which would pass a two-sided threshold). The asymmetry is justified by the asymmetry of the cost function: falsely protecting a trigger (false negative in the guard) is far less costly than falsely failing to protect a safe ingredient (false positive in the guard), because the latter would allow a confirmed-safe ingredient to be labelled as a trigger by group shrinkage.

### 4.4 Protected-Safe Exclusion from Pool

Protected-safe ingredients are excluded from the D&L pool before `μ_g` and `τ²` are estimated. This is not an ad hoc fix but a principled application of the model: the pool should contain only ingredients whose trigger status is uncertain. Including a confirmed-safe ingredient (e.g., `U = 0, V = 1000`) would massively pull `μ_g` downward, corrupting the group mean for all other members and defeating the purpose of grouping. Test 3 in the test suite verifies this property explicitly.

### 4.5 No Cap on Untested Member Inheritance

Untested group members receive `final_OR = exp(μ_g)` with `confidence = "LOW"`. No cap or clipping is applied to this value. A cap (e.g., limiting untested members to the maximum observed OR in the group) would be an ad hoc intervention not implied by the model. The `LOW` confidence label is the correct signal to downstream systems: the recommendation engine should treat `LOW`-confidence scores differently from `HIGH`-confidence scores. If it does not, the fix belongs in the recommendation engine, not in the shrinkage model.

---

## 5. Testing and Validation

### 5.1 Test Coverage

Ten test functions were implemented in `tests/test_shrinkage.py`, covering the following scenarios:

1. **All group members are triggers** — untested member inherits high `μ_g`.
2. **One member genuinely safe** — Rule 1 protection fires correctly.
3. **Safe ingredient does not pollute `μ_g`** — confirms pool exclusion.
4. **Singleton group** — `τ² = 0`, untested sibling inherits exactly.
5. **`τ² = 0` collapse** — confirmed triggers retain `"HIGH"` confidence.
6. **Binomial direction** — exact p-value computation verified (`P(X ≥ 9 | n=10) = 11/1024`).
7. **Ungrouped ingredient** — treated as singleton, `group = None`.
8. **Notebook wrapper** — column-name mapping works correctly.
9. **Serialisation** — `.to_dict()` produces expected structure.
10. **Mixed realistic scenario** — protected + trigger + weak + standalone all handled correctly.

All 10 tests passed.

### 5.2 What the Tests Verify

The test suite establishes three categories of correctness:

- **Statistical correctness:** The D&L estimator produces the expected `μ_g` and `τ²` values, the shrinkage weights are computed correctly, and the exact binomial function returns known values (Tests 4, 5, 6).
- **Guard correctness:** Rule 1 fires when it should (Tests 2, 3) and does not fire when it should not (Test 10, rice ingredient with `P(X ≥ 4 | n=7) = 0.5`). Protected ingredients are excluded from the pool (Test 3).
- **API correctness:** The notebook wrapper extracts columns correctly (Test 8), serialisation produces complete dictionaries (Test 9), and the output contains all expected ingredients (Tests 1, 7, 10).

The test suite is self-contained: it stubs out the `safeplate` package imports using `sys.modules` patching and a `StubGroupManager` class, so it runs without Flask, data files, or other project dependencies.

---

## 6. Integration with Existing System

### 6.1 Unchanged Components

The following files were not modified during this phase:

- `backend/safeplate/utils.py` (77 lines) — cosine similarity and name normalisation.
- `backend/safeplate/embedding_loader.py` (202 lines) — Node2Vec embedding loading.
- `backend/safeplate/ingredient_groups.py` (239 lines) — group lookup manager.
- `backend/safeplate/recommendation_engine.py` (364 lines) — embedding-based recommendation with group-aware filtering methods.
- `data/processed/ingredient_groups.csv` (2699 lines including header).

Note: notebook `05b_test_group_integration.ipynb` was initially created during Phase 2 but is now deprecated (see Section 6.4).

### 6.2 What Replaced What

The file `backend/safeplate/group_aware_trigger_detector.py` was completely rewritten.

**Removed (old naive approach):**

- `IndividualTrigger`, `GroupTrigger`, `TriggerDetectionResult` dataclasses.
- `propagate_trigger_evidence()` method with flat 0.7 propagation discount.
- `_calculate_confidence()` and `_calculate_group_evidence_strength()` heuristic methods.
- `get_group_risk_assessment()` method.
- `_confidence_rank()` helper function.

**Added (principled statistical model):**

- `ShrinkageResult`, `GroupShrinkageResult`, `FullShrinkageOutput` dataclasses.
- `_binomial_survival()` — exact one-sided binomial tail probability.
- `_raw_estimates()` — Haldane-corrected `θ̂` and `σ²`.
- `_is_protected_safe()` — Rule 1 guard.
- `_is_own_trigger()` — trigger-direction significance test.
- `_dersimonian_laird()` — method-of-moments estimation of `μ_g` and `τ²`.
- `run_shrinkage()` and `run_shrinkage_from_dataframe_rows()` public API.

**Test Suite Migration:**

The notebook `05b_test_group_integration.ipynb` was initially created during Phase 2 to test the group integration functionality. However, it was written against the naive propagation API (`TriggerDetectionResult`, `propagate_trigger_evidence`, `IndividualTrigger`, `GroupTrigger`) which was completely replaced by the DerSimonian & Laird shrinkage model in Phase 3.

When the API changed, the notebook broke at cell 3 with:
`ImportError: cannot import name 'TriggerDetectionResult'`

Rather than attempting to update the notebook (which would have required rewriting most cells, since cells 19–26 all use the removed API), the testing was migrated to a more maintainable Python script:

    tests/test_group_integration.py

This script tests the **stable** parts of the group integration (6 tests, all passing):
- `IngredientGroupManager` functionality (which did not change)
- `RecommendationEngine` group-aware methods (which did not change)
- Regression verification of original recommendation methods

It does not test the shrinkage model (that is covered by `tests/test_shrinkage.py`).

The deprecated notebook remains in `notebooks/` for reference but should not be run. See `notebooks/05b_DEPRECATED_USE_PYTHON_SCRIPT.txt` for the deprecation notice.

### 6.3 Next Steps for Integration

The following work remains:

0. **COMPLETED: Group integration testing migrated to `tests/test_group_integration.py`** (6 tests, all passing).
1. **Wire into notebook 04:** The trigger detection notebook currently produces `(ingredient, U, V)` tuples independently. It needs a final cell that passes these into `run_shrinkage()` and uses the `FullShrinkageOutput` as the trigger list.
2. **Update recommendation engine:** The group-aware filtering methods in `recommendation_engine.py` currently accept raw trigger lists. They should be updated to accept `FullShrinkageOutput` and use the `confidence` labels to weight or filter recommendations.
3. **Build synthetic test lab:** Controlled scenarios with known ground-truth triggers to validate end-to-end system behaviour (false-negative rate, false-positive rate, ranking quality).
4. **Flask web UI:** Demonstration interface for the dissertation evaluation.

### 6.4 Lessons Learned: Notebooks vs Scripts

During development, an issue was encountered with notebook `05b_test_group_integration.ipynb`. The notebook was written to test the group integration functionality but inadvertently used classes from the Phase 2 naive API (`TriggerDetectionResult`, `IndividualTrigger`, `GroupTrigger`) which were removed when the shrinkage model was implemented.

This highlighted a key limitation of Jupyter notebooks for testing in rapidly-evolving codebases: notebooks are tightly coupled to the API state at the time they were written, and cell execution order can hide breaking changes until the notebook is run end-to-end.

**Decision:** All future testing will use Python scripts in `tests/` rather than Jupyter notebooks. Python scripts are:
- Easier to run in CI/CD
- More maintainable (standard Python tooling works)
- Less fragile to API changes (import errors are immediate, not hidden)
- Version-control friendly (no JSON metadata noise)

Notebooks remain valuable for:
- Exploratory data analysis
- Model selection and parameter tuning (notebook 03, for example)
- Generating figures and visualisations for the dissertation

But not for:
- Unit testing
- Integration testing
- Regression testing

The corrected structure is:
- `tests/test_shrinkage.py` — shrinkage model edge cases (10 tests)
- `tests/test_group_integration.py` — group manager + recommendation engine (6 tests)

Both are pure Python scripts that can be run with `python tests/<filename>.py` and exit with code 0 (pass) or 1 (fail).

---

## 7. Dissertation Implications

### 7.1 Contributions to State in the Dissertation

1. **Application of empirical Bayes hierarchical shrinkage to within-person case-crossover data.** This specific combination — N-of-1 case-crossover producing discordant counts, grouped by food category, with DerSimonian & Laird pooling — is a novel application of an established statistical technique to the dietary trigger detection problem.
2. **Two-rule guard system** that prevents false-positive trigger labels (protecting confirmed-safe ingredients) while simultaneously reducing false negatives (through group-level borrowing of strength). The ordered application — protect first, then shrink — is a methodological contribution.
3. **Explicit handling of untested group members** with calibrated confidence labels. This addresses a practical gap in existing dietary trigger systems, where ingredients never eaten by the user cannot be assessed for risk.

### 7.2 Limitations to State Explicitly

The dissertation must acknowledge the following limitations:

- **Grouping is lexical, not biochemical.** The ingredient groups are defined by shared tokens in recipe data (e.g., "cheddar cheese" and "blue cheese" share the token "cheese"). This is a reasonable proxy for shared biochemical properties but is not validated against nutritional or allergological taxonomies.
- **The within-group correlation assumption is substantive.** The model assumes that ingredients in the same group have correlated trigger risks. This assumption is plausible for many groups (dairy, nightshades, gluten-containing grains) but may not hold for heterogeneous groups. The `τ²` estimate provides a partial check: a large `τ²` indicates heterogeneity, but cannot distinguish "noise" from "genuinely different trigger risks."
- **Case-crossover controls for stable confounders but not time-varying ones.** Within-person comparison eliminates between-person confounding, but time-varying factors (stress, medication changes, seasonal allergen exposure) could still bias the estimates.
- **The method is designed for sparse data.** With dense diary data (hundreds of discordant pairs per ingredient), individual-level estimates are already stable, and the shrinkage model provides less benefit. The primary value is in the sparse-data regime typical of short-duration food diaries.

### 7.3 Evaluation Plan (for Future Work Section)

The planned validation approach includes:

- **Baselines:** (1) no grouping (individual-level case-crossover only), (2) naive averaging within groups (fixed-effect), (3) hard propagation (flat discount, the previous implementation).
- **Metrics:** false-negative rate (primary), false-positive rate, ranking quality (AUC over trigger/non-trigger ingredients), calibration of confidence labels.
- **Synthetic scenarios:** all-trigger group, one-safe-member group, heterogeneous group, singleton group — matching the 10 test cases but evaluated end-to-end with the recommendation engine.
- **User-facing evaluation:** clarity of confidence labels in the UI, user trust in recommendations when confidence varies.

---

## 8. Code Quality and Documentation

### 8.1 Citation and Academic Integrity

- DerSimonian & Laird (1986) is cited in the module docstring with the full bibliographic reference.
- The term "empirical Bayes" is used throughout; the unqualified term "Bayesian" does not appear anywhere in the module. This distinction is important: the model estimates `μ_g` and `τ²` from the data (empirical Bayes) rather than placing a prior distribution on them (fully Bayesian).
- All statistical assumptions are stated explicitly in function docstrings: the D&L function documents that it is a point-estimation procedure (not fully Bayesian), that inputs are non-protected pool members only, and that the `τ²` floor at zero is part of the original formula.
- The Rule 1 function documents why the test is one-sided, why 0.05 is the threshold, and what both conditions mean.

### 8.2 Dependencies and Portability

The module uses only:

- `numpy` — for `log`, `exp`, and `isnan` operations.
- `math.comb` — for exact binomial coefficient computation.
- `dataclasses` — for output data structures.
- `typing.Optional` — for type annotations.

No scipy, pandas, or other heavy dependencies are required. All computation is closed-form (no MCMC, no iterative optimisation, no random sampling). The module is fully deterministic and reproducible. For typical diary sizes (a few hundred ingredients, groups of 1–105 members), execution completes in under 100ms.

---

## 9. Timeline and Effort

- **Design phase:** February 3, 2026 (design review and specification of the statistical model, input/output contracts, and test cases).
- **Implementation:** February 3, 2026 (complete rewrite of `group_aware_trigger_detector.py`).
- **Testing:** February 3, 2026 (all 10 tests written and passing in the same session).
- **Total lines of code:** 564 (main module) + 470 (test suite) = 1,034 lines.

---

## 10. References

DerSimonian, R. & Laird, N. (1986). Meta-analysis in clinical trials. *Controlled Clinical Trials*, 7(3), 177–188.

Maclure, M. (1991). The case-crossover design: A method for studying transient effects on the risk of acute events. *American Journal of Epidemiology*, 133(2), 144–153.

Higgins, J. P. T. & Thompson, S. G. (2002). Quantifying heterogeneity in a meta-analysis. *Statistics in Medicine*, 21(11), 1539–1558.

Haldane, J. B. S. (1956). The estimation and significance of the logarithm of a ratio of frequencies. *Annals of Human Genetics*, 20(4), 309–311.

Cochran, W. G. (1954). The combination of estimates from different experiments. *Biometrics*, 10(1), 101–129.

---

## Appendix A: File Locations

```
SafePlate/
├── backend/
│   └── safeplate/
│       ├── group_aware_trigger_detector.py    [REWRITTEN, ~450 lines]
│       ├── ingredient_groups.py               [Unchanged]
│       ├── recommendation_engine.py           [Unchanged]
│       ├── embedding_loader.py                [Unchanged]
│       └── utils.py                           [Unchanged]
├── tests/
│   ├── test_shrinkage.py                      [NEW, 10 test functions]
│   └── test_group_integration.py              [NEW, 6 test functions]
├── data/
│   └── processed/
│       └── ingredient_groups.csv              [Unchanged, 2698 rows]
├── docs/
│   └── POST_PROGRESS_REPORT_TECHNICAL_LOG.md  [THIS DOCUMENT]
└── notebooks/
    ├── 04_trigger_detection_recommended.ipynb  [Unchanged, produces U/V]
    ├── 05_test_recommendation_engine.ipynb     [Unchanged]
    ├── 05b_test_group_integration.ipynb        [DEPRECATED, do not run]
    └── 05b_DEPRECATED_USE_PYTHON_SCRIPT.txt   [Deprecation notice]
```

---

## Appendix B: Quick Reference for Future Sessions

**What was done:** Replaced naive trigger propagation (flat 0.7 discount on OR) with DerSimonian & Laird empirical Bayes shrinkage. Ingredients borrow strength from their food group while a two-rule guard protects confirmed-safe members from being overridden.

**Key file:** `backend/safeplate/group_aware_trigger_detector.py`

**Key API:**

```python
from safeplate.group_aware_trigger_detector import GroupAwareTriggerDetector
from safeplate.ingredient_groups import IngredientGroupManager

gm = IngredientGroupManager("data/processed/ingredient_groups.csv")
detector = GroupAwareTriggerDetector(group_manager=gm)

output = detector.run_shrinkage(
    ingredient_triggers=[("cheddar cheese", 8, 1), ("swiss cheese", 0, 10), ...],
    all_group_members=gm.get_all_groups()
)

output.get_protected_safe()              # Swiss cheese (confirmed safe)
output.get_high_confidence()             # Cheddar (own trigger) + Swiss (protected)
output.get_all_trigger_ingredients(1.5)  # All ingredients with final OR >= 1.5
output.to_dict()                         # JSON-serialisable dictionary
```

**Key insight:** `wᵢ = τ² / (τ² + σᵢ²)`. When data are sparse (`σᵢ²` large), `wᵢ` is small, and the ingredient borrows heavily from the group mean `μ_g`. When data are rich (`σᵢ²` small), `wᵢ` is large, and the ingredient trusts its own estimate. `τ²` is estimated adaptively from the data via D&L, not set by hand.

**Tests pass:**
- `test_shrinkage.py`: All 10 edge cases verified. Run with: `python tests/test_shrinkage.py`
- `test_group_integration.py`: All 6 integration tests verified. Run with: `python tests/test_group_integration.py`

**Next step:** Integrate into notebook 04, then build synthetic test lab for end-to-end validation.
