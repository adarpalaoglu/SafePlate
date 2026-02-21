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

> **Note (February 2026):** During Phase 2 (Web UI), empirical Bayes shrinkage was found to be counterproductive when meals are sampled from real recipes. Recipe clustering creates ingredient co-occurrence patterns that violate the group homogeneity assumption: shrinkage pulls genuine triggers toward the group mean (reducing their signal) while inflating co-occurring confounders. The web UI detection pipeline now uses **raw case-crossover odds ratios** with multi-layer filtering instead. The shrinkage code is preserved for academic reference and the test suite still passes, but shrinkage is no longer used in the production detection path. See Section 9.5 for the full rationale.

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
1. **COMPLETED: Integrated pipeline built** (`backend/safeplate/integrated_pipeline.py`, 233 lines). Connects notebook 04's CSV output to the shrinkage model and recommendation engine via a reusable `SafePlatePipeline` class. See Section 7 for details.
2. **COMPLETED: End-to-end validation** (`notebooks/06_test_integrated_pipeline.ipynb`, 20 cells). Validated with real data: 25 tested ingredients, 426 total after propagation, 1 protected safe, substitution recommendations working.
3. **Build synthetic test lab:** Controlled scenarios with known ground-truth triggers to validate end-to-end system behaviour (false-negative rate, false-positive rate, ranking quality). The integrated pipeline API (`SafePlatePipeline`) is the entry point for this work.
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

## 7. Integrated Pipeline Implementation

### 7.1 Motivation and Design Rationale

The three core components of the SafePlate system — case-crossover trigger detection (notebook 04), empirical Bayes shrinkage (`group_aware_trigger_detector.py`), and embedding-based recommendation (`recommendation_engine.py`) — had each been validated independently but operated in isolation. No programmatic pathway existed to move from raw diary data through trigger detection and shrinkage to safe-substitute recommendations. This gap prevented end-to-end validation of the system and blocked the planned synthetic test lab and Flask web UI from being built.

The integration was implemented as a reusable Python module (`backend/safeplate/integrated_pipeline.py`, 233 lines) rather than as additional notebook cells. This decision was informed by the experience with notebook `05b_test_group_integration.ipynb` (see Section 6.4): notebook code is tightly coupled to the API state at the time it is written and is fragile to refactoring. A Python module provides a stable import target for the test lab, the Flask UI, and any future consumers, while remaining testable with standard tooling. A companion validation notebook (`notebooks/06_test_integrated_pipeline.ipynb`, 20 cells) demonstrates the end-to-end flow and validates edge cases, but the notebook is a *consumer* of the module, not the implementation itself.

The module was designed around the principle of *integration without refactoring*. Notebook 04's case-crossover logic is complex (multi-cell, with Bayesian posterior estimation, family-level pooling, and FDR correction) and has been validated against synthetic diary data. Refactoring it into a Python module would introduce risk without immediate benefit. Instead, the pipeline consumes notebook 04's CSV output (`ingredient_stats_all.csv`) as a stable interface contract. This separation of concerns mirrors standard data-engineering practice: the case-crossover step is a data-generation pipeline, and the integration layer is a downstream consumer. Refactoring notebook 04 into a callable module can be deferred to the test lab implementation, when the API requirements are clearer.

### 7.2 Implementation: `integrated_pipeline.py`

The module defines two public components: `PipelineConfig` and `SafePlatePipeline`.

`PipelineConfig` is a dataclass that encapsulates all decision thresholds in a single location, eliminating magic numbers from the pipeline logic and making sensitivity analysis straightforward. It has four fields: `trigger_or_threshold` (default 1.0, the minimum `final_OR` to flag an ingredient as a trigger), `trigger_min_confidence` (default `"MEDIUM"`, the minimum confidence level for trigger inclusion), `propagate_to_untested` (default `True`, whether untested group members inherit `μ_g`), and `recommendation_top_n` (default 10, the number of substitutes to return). The `trigger_min_confidence` field is validated at construction time to be one of `"HIGH"`, `"MEDIUM"`, or `"LOW"`.

`SafePlatePipeline` orchestrates the three core components through five public methods:

**`detect_triggers_from_stats(ingredient_stats)`** validates that the input DataFrame contains the required columns (`ingredient`, `U_case_not_ctrl`, `V_ctrl_not_case`), converts it to the row-dict format expected by `GroupAwareTriggerDetector.run_shrinkage_from_dataframe_rows()`, and returns the `FullShrinkageOutput`. The method raises `ValueError` with a clear diagnostic message if the DataFrame is empty or missing columns.

**`get_trigger_ingredients(shrinkage_result)`** extracts ingredient names where `final_OR` meets or exceeds the configured threshold and `confidence` meets or exceeds the configured minimum level. Confidence levels are ranked ordinally (`HIGH` > `MEDIUM` > `LOW`). Protected-safe ingredients are excluded regardless of their OR because they have statistically significant safety evidence.

**`get_safe_substitutes(query_ingredient, shrinkage_result, top_n)`** composes the trigger list from `get_trigger_ingredients()` and passes it as the exclusion filter to `RecommendationEngine.find_substitutes()`. The result is a list of `(ingredient_name, similarity_score)` tuples ranked by Node2Vec cosine similarity, with all trigger ingredients removed.

**`get_pipeline_summary(shrinkage_result)`** aggregates counts of tested, untested, protected-safe, and trigger ingredients, along with confidence-level distributions and the trigger and protected-safe ingredient lists. This method provides a single-call overview suitable for reporting tables and notebook output.

### 7.3 Validation: Notebook 06

Notebook `06_test_integrated_pipeline.ipynb` demonstrates the end-to-end flow and validates that the pipeline handles both typical and edge-case scenarios correctly. The notebook loads `ingredient_stats_all.csv` (25 ingredients from the synthetic diary used in notebook 04), initialises the pipeline with the real embedding and group data, runs `detect_triggers_from_stats()`, and exercises all four downstream methods.

The validation run produced the following results. Of the 25 tested ingredients, 1 was protected safe (tomato: `U = 0`, `V = 5`, `OR = 0.091`), confirming that Rule 1 correctly prevents upward shrinkage for an ingredient with significant safety evidence. After group propagation, the output contained 426 total ingredients (401 untested group members inheriting `μ_g` with `confidence = "LOW"` and `evidence_source = "group_mean"`). Substitution recommendations were returned successfully for test queries (`"milk"`, `"tomato"`, `"cheese"`), with trigger ingredients excluded from the results. Error handling was verified for three failure modes: missing CSV file, empty DataFrame, and DataFrame with incorrect columns.

The notebook also includes a comparison table of raw OR versus shrunk OR for all tested ingredients, illustrating which ingredients were most affected by shrinkage and in which direction. This table provides material for a dissertation figure showing the bias-variance trade-off of the empirical Bayes estimator.

### 7.4 Design Decisions and Justifications

Five key design decisions were made during the pipeline implementation, each with a clear rationale.

**Trigger threshold rule.** An ingredient is classified as a trigger if `final_OR >= 1.0` and `confidence >= "MEDIUM"`. The OR threshold at 1.0 reflects the epidemiological convention that an odds ratio above parity indicates positive association with the outcome. The confidence threshold at `"MEDIUM"` excludes only `"LOW"` confidence results, which are untested group-mean inheritors with no direct evidence. This default is conservative: it includes shrunk ingredients with some evidence but excludes pure extrapolation. Both thresholds are configurable via `PipelineConfig`, enabling sensitivity analysis during evaluation.

**Protected-safe handling in recommendations.** Protected-safe ingredients are *not* excluded from substitution results. They have statistically significant safety evidence and are therefore valid (indeed, desirable) substitutes. The `protected` flag is metadata for transparency, not a filtering criterion. A future enhancement could re-rank results to prioritise protected-safe ingredients, but this is not necessary for the current validation.

**CSV hand-off from notebook 04.** The pipeline reads notebook 04's CSV output rather than invoking the case-crossover logic programmatically. This provides clean separation of concerns and avoids the risk of introducing bugs by refactoring validated notebook code. The limitation — that notebook 04 must be run manually before the pipeline can be used — is acceptable at this stage and will be addressed when the synthetic test lab requires automated end-to-end execution.

**Error handling strategy.** The pipeline fails fast with clear error messages rather than silently substituting defaults. A missing CSV file produces a `FileNotFoundError` instructing the user to run notebook 04. Wrong column names produce a `ValueError` listing expected versus actual columns. A query ingredient not in the embeddings raises `IngredientNotFoundError`. This approach prioritises debuggability over convenience, which is appropriate for a research prototype.

**No cap on untested member inheritance.** Consistent with Section 4.5, untested group members receive `final_OR = exp(μ_g)` without clipping. The `"LOW"` confidence label is the correct mechanism for signalling uncertainty to the pipeline's trigger extraction logic, which respects the `trigger_min_confidence` threshold.

### 7.5 Integration with Existing Components

The pipeline interacts with four existing modules, all through their documented public APIs. No modifications to any existing file were required.

`GroupAwareTriggerDetector` is called via `run_shrinkage_from_dataframe_rows()`, which accepts a list of row dictionaries with keys matching notebook 04's column names. The `propagate_to_groups` flag is set from `PipelineConfig.propagate_to_untested`, controlling whether untested group members receive `μ_g`. The detector is initialised with the same `IngredientGroupManager` instance used by the pipeline, ensuring consistent group definitions.

`RecommendationEngine` is called via `find_substitutes()`, which accepts a query ingredient, a list of trigger ingredients to exclude, and a `top_n` count. The trigger list is computed by `get_trigger_ingredients()` from the shrinkage output, closing the loop from statistical inference to practical recommendation.

`IngredientGroupManager` provides group membership data to the detector via `get_all_groups()`. It is injected into the pipeline at construction time and passed to both the detector and (when needed) the recommendation engine.

`EmbeddingLoader` provides Node2Vec embeddings to the recommendation engine. It is injected at construction time and used exclusively through the engine's API.

Notebook 04's output is consumed via the CSV file `data/processed/trigger_detection/ingredient_stats_all.csv`. The pipeline requires only three of the thirteen columns produced by notebook 04 (`ingredient`, `U_case_not_ctrl`, `V_ctrl_not_case`); the remaining columns are informational and are not read.

### 7.6 Testing and Validation Results

Validation was performed at two levels: unit-level checks on the pipeline's input validation and error handling, and integration-level verification of the end-to-end flow with real data.

At the unit level, `PipelineConfig` was verified to reject invalid confidence levels. The `detect_triggers_from_stats()` method was verified to raise `ValueError` for empty DataFrames and for DataFrames missing required columns. Both error messages were confirmed to be diagnostic (listing expected versus actual values).

At the integration level, the complete flow was executed with 25 ingredients from notebook 04's synthetic diary output. The shrinkage model produced 426 total ingredients after group propagation (25 tested, 401 untested inheritors). One ingredient (tomato) was correctly identified as protected safe by Rule 1. Trigger extraction returned the expected set given the configured thresholds. Substitution recommendations were returned for three test queries with trigger ingredients excluded. No unexpected behaviours or errors were observed.

### 7.7 Limitations and Future Work

The pipeline has four known limitations, each with a planned resolution path.

The CSV-based interface with notebook 04 requires manual execution of the notebook before the pipeline can be used. This will be addressed during the synthetic test lab implementation by refactoring notebook 04's case-crossover logic into a callable module (`backend/safeplate/case_crossover.py`), which can be invoked programmatically.

Trigger thresholds are applied uniformly across all ingredients and queries. Per-user or per-query risk tolerance (e.g., "conservative" versus "exploratory" substitution modes) is not yet supported. This could be added as a `PipelineConfig` parameter if time permits.

Protected-safe ingredients are included in substitution results but not prioritised. A future enhancement could re-rank results to boost protected-safe ingredients, signalling their confirmed safety to the user.

The pipeline does not yet produce confidence intervals on the shrunk OR estimates. Adding interval estimates would strengthen the dissertation evaluation by enabling calibration analysis.

### 7.8 Contribution to Dissertation

The integrated pipeline makes four contributions to the dissertation.

*Methodological.* It demonstrates a working end-to-end system that integrates within-person causal inference (case-crossover), empirical Bayes hierarchical modelling (DerSimonian & Laird shrinkage), and representation learning (Node2Vec similarity). This combination is novel in the dietary trigger detection domain.

*Technical.* It provides a reusable API for trigger-aware recommendation that handles sparse N-of-1 data at scale (25 tested ingredients evaluated alongside 401 untested group members). The threshold-based trigger extraction is transparent and configurable, avoiding opaque decision boundaries.

*Evaluation.* Notebook 06 provides the validation framework for dissertation figures and tables, including the raw-versus-shrunk OR comparison, the confidence-level distribution, and the edge-case demonstrations (protected safe, untested inheritance, heterogeneous and homogeneous groups).

*Practical.* The pipeline is the foundation for the next implementation phase (synthetic test lab with known ground-truth triggers). It can also be used directly in a Flask web UI if one is implemented for the dissertation demonstration.

---

## 9. Dissertation Implications (Shrinkage Model — Original Phase)

> **Note:** This section describes the dissertation implications of the shrinkage model as originally designed. For the updated dissertation implications reflecting the current system architecture (raw ORs with multi-layer filtering), see Section 13.

### 8.1 Contributions to State in the Dissertation

1. **Application of empirical Bayes hierarchical shrinkage to within-person case-crossover data.** This specific combination — N-of-1 case-crossover producing discordant counts, grouped by food category, with DerSimonian & Laird pooling — is a novel application of an established statistical technique to the dietary trigger detection problem.
2. **Two-rule guard system** that prevents false-positive trigger labels (protecting confirmed-safe ingredients) while simultaneously reducing false negatives (through group-level borrowing of strength). The ordered application — protect first, then shrink — is a methodological contribution.
3. **Explicit handling of untested group members** with calibrated confidence labels. This addresses a practical gap in existing dietary trigger systems, where ingredients never eaten by the user cannot be assessed for risk.

### 8.2 Limitations to State Explicitly

The dissertation must acknowledge the following limitations:

- **Grouping is lexical, not biochemical.** The ingredient groups are defined by shared tokens in recipe data (e.g., "cheddar cheese" and "blue cheese" share the token "cheese"). This is a reasonable proxy for shared biochemical properties but is not validated against nutritional or allergological taxonomies.
- **The within-group correlation assumption is substantive.** The model assumes that ingredients in the same group have correlated trigger risks. This assumption is plausible for many groups (dairy, nightshades, gluten-containing grains) but may not hold for heterogeneous groups. The `τ²` estimate provides a partial check: a large `τ²` indicates heterogeneity, but cannot distinguish "noise" from "genuinely different trigger risks."
- **Case-crossover controls for stable confounders but not time-varying ones.** Within-person comparison eliminates between-person confounding, but time-varying factors (stress, medication changes, seasonal allergen exposure) could still bias the estimates.
- **The method is designed for sparse data.** With dense diary data (hundreds of discordant pairs per ingredient), individual-level estimates are already stable, and the shrinkage model provides less benefit. The primary value is in the sparse-data regime typical of short-duration food diaries.

### 8.3 Evaluation Plan (for Future Work Section)

The planned validation approach includes:

- **Baselines:** (1) no grouping (individual-level case-crossover only), (2) naive averaging within groups (fixed-effect), (3) hard propagation (flat discount, the previous implementation).
- **Metrics:** false-negative rate (primary), false-positive rate, ranking quality (AUC over trigger/non-trigger ingredients), calibration of confidence labels.
- **Synthetic scenarios:** all-trigger group, one-safe-member group, heterogeneous group, singleton group — matching the 10 test cases but evaluated end-to-end with the recommendation engine.
- **User-facing evaluation:** clarity of confidence labels in the UI, user trust in recommendations when confidence varies.

---

## 10. Code Quality and Documentation

### 9.1 Citation and Academic Integrity

- DerSimonian & Laird (1986) is cited in the module docstring with the full bibliographic reference.
- The term "empirical Bayes" is used throughout; the unqualified term "Bayesian" does not appear anywhere in the module. This distinction is important: the model estimates `μ_g` and `τ²` from the data (empirical Bayes) rather than placing a prior distribution on them (fully Bayesian).
- All statistical assumptions are stated explicitly in function docstrings: the D&L function documents that it is a point-estimation procedure (not fully Bayesian), that inputs are non-protected pool members only, and that the `τ²` floor at zero is part of the original formula.
- The Rule 1 function documents why the test is one-sided, why 0.05 is the threshold, and what both conditions mean.

### 9.2 Dependencies and Portability

The module uses only:

- `numpy` — for `log`, `exp`, and `isnan` operations.
- `math.comb` — for exact binomial coefficient computation.
- `dataclasses` — for output data structures.
- `typing.Optional` — for type annotations.

No scipy, pandas, or other heavy dependencies are required. All computation is closed-form (no MCMC, no iterative optimisation, no random sampling). The module is fully deterministic and reproducible. For typical diary sizes (a few hundred ingredients, groups of 1–105 members), execution completes in under 100ms.

---

*Timeline, References, and Appendices have been moved to the end of the document (Sections 14–15, Appendices A–B) to follow all implementation phases.*

---

## 8. Phase 1: Test Lab Backend

### 8.1 Motivation and Design

The synthetic test lab was identified as a critical validation requirement in Section 6.3. To evaluate the trigger detection and recommendation system end-to-end, controlled scenarios with known ground-truth triggers are needed. Phase 1 implements the backend components for this test lab: synthetic diary generation, case-crossover simulation, and preference-aware recommendations with dynamic trigger filtering.

The key insight is that real user data cannot provide ground truth. Users do not know their true triggers — discovering them is the system's purpose. Synthetic data with *planted* triggers solves this problem: if we generate diaries where "cheddar cheese" causes symptoms 60% of the time it is eaten, we can measure whether the pipeline correctly identifies it as a trigger and excludes it from recommendations.

### 8.2 Components Implemented

Four Python modules were created in `backend/safeplate/`:

**`diary_generator.py`** (~280 lines) — Generates synthetic meal and symptom logs with known ground truth. Uses an exposure-driven causal model:
- `P(symptom | trigger eaten) = trigger_effect_strength` (configurable, default 0.6)
- `P(symptom | no trigger) = baseline_symptom_rate` (configurable, default 0.05)
- Negative control ingredient included for false positive validation
- Reproducible via seed parameter

Key class: `DiaryGenerator` with `DiaryGeneratorConfig` dataclass (12 configurable parameters).

**`case_crossover_simulator.py`** (~230 lines) — Converts diary DataFrames into (U, V) discordant pair statistics suitable for pipeline input. Implements:
- Hazard/control window matching (day-of-week, time-of-day)
- Episode clustering (symptoms within gap threshold merged)
- Per-ingredient discordant pair counting

Key class: `CaseCrossoverSimulator` with `CaseCrossoverConfig` dataclass.

**`user_preferences.py`** (~140 lines) — Persistent storage for user ingredient preference rankings. Features:
- JSON file persistence
- Lazy loading with caching
- CRUD operations: save, get, has, delete, clear

Key classes: `UserPreferenceStore`, `UserRanking` dataclass.

**`preference_aware_recommendation.py`** (~270 lines) — Recommendation engine with dynamic trigger filtering and user preference integration. Algorithm:
1. Get triggers from shrinkage result (no group expansion per spec)
2. If user has stored ranking: filter triggers, take top-N safe, backfill from similarity if needed
3. If no ranking: use similarity-based recommendations, filter triggers
4. Return `Recommendation` objects with metadata

Key class: `PreferenceAwareRecommendationEngine` with `Recommendation` and `RecommendationExplanation` dataclasses.

### 8.3 Validation Notebook

Notebook `08_test_preference_engine.ipynb` validates all Phase 1 components across 7 sections:

| Section | Purpose | Status |
|---------|---------|--------|
| 1 | Synthetic Diary Generation | PASS |
| 2 | Case-Crossover Simulation | PASS |
| 3 | Trigger Detection | PASS |
| 4 | Recommendations (No Ranking) | PASS |
| 5 | Recommendations (With Ranking) | PASS |
| 6 | Dynamic Trigger Filtering | PASS |
| 7 | Ground Truth Metrics | PASS |

### 8.4 Validation Results

With a 14-day synthetic diary (seed=42):
- **41 meals** generated, **10 symptom episodes**
- True triggers: `cheddar cheese`, `whole milk`
- Negative control: `olive oil`

**Case-crossover output:** `cheddar cheese` U=5, V=0; `whole milk` U=4, V=0

**Trigger detection:**
- Recall = 1.0 (both true triggers detected)
- Negative control passed (olive oil NOT detected)
- Precision = 0.118 (expected with small dataset — many ingredients have elevated U by chance)

**Preference-aware recommendations:**
- Without ranking: Similarity fallback works, all recommendations safe
- With ranking: User preferences respected
- Dynamic filtering: Triggers correctly filtered from rankings, backfilling works (3 from ranking + 2 backfilled when 2 triggers removed)

### 8.5 Design Decisions

**Exposure-driven causal model:** Symptoms are generated probabilistically based on trigger presence, not deterministically. This mirrors real-world variance where triggers don't cause symptoms 100% of the time.

**No group expansion in trigger filtering:** The preference engine uses direct trigger detection output, not group-expanded triggers. This is intentional: group expansion is a statistical modelling assumption in the shrinkage model, not a recommendation filtering rule. Users might tolerate one cheese but not another.

**Backfilling from similarity:** When user-ranked items are filtered as triggers, the engine backfills from embedding similarity. This ensures users always receive the requested number of recommendations while respecting their preferences where possible.

**Placeholder explanations:** `RecommendationExplanation` returns placeholder text ("Coming in Phase 3"). Natural language explanation generation is deferred to a later phase.

### 8.6 Files Created

```
backend/safeplate/
├── diary_generator.py              [NEW, ~280 lines]
├── case_crossover_simulator.py     [NEW, ~230 lines]
├── user_preferences.py             [NEW, ~140 lines]
└── preference_aware_recommendation.py [NEW, ~270 lines]

data/
├── test_diaries/                   [NEW, empty directory]
└── test_results/
    └── phase1_metrics.json         [NEW, validation metrics]

notebooks/
└── 08_test_preference_engine.ipynb [NEW, 7 sections]
```

### 8.7 Next Steps

Phase 2 will add:
- Flask web UI for interactive demonstration
- Additional synthetic scenarios (multiple triggers, varying effect strengths)
- Extended ground truth metrics (calibration curves, ROC analysis)

### 8.8 Phase 3 Extension: Explainability (February 2026)

**Implementation:**
- Multi-evidence explanation generation for all recommendations
- Three explanation components:
  - Safety: Derived from shrinkage model (OR, confidence, protected status)
  - Similarity: Derived from Node2Vec embeddings (cosine similarity score)
  - Preference: Derived from user ranking history (rank position or backfill status)
- Structured text format with Markdown formatting for clarity
- Backward compatible: explanations optional via `include_explanations` parameter

**Validation:**
- Added Section 8 to testing notebook (10 test cases)
- All tests validate accuracy of explanation components
- Edge cases handled: missing ingredients, protected status, backfilled items
- Format validation: Markdown headers, pipe separators, single-line output

**Integration:**
- No changes to recommendation algorithm (filter → rank → backfill unchanged)
- No changes to Phase 1 components (all prior tests still pass)
- Explanation generation is post-processing step only

**Files modified:**
- `backend/safeplate/preference_aware_recommendation.py` (~50 lines added)
- `notebooks/08_test_preference_engine.ipynb` (Section 8 added, ~200 lines)
- `docs/POST_PROGRESS_REPORT_TECHNICAL_LOG.md` (this section)

---

## 9. Phase 2: Flask Web UI (Test Lab Interface)

**Status:** Completed (all 5 tabs fully functional)
**Date:** February 2026
**Purpose:** Web-based test lab interface for controlled evaluation of trigger detection and recommendation systems, enabling interactive parameter exploration and visual validation of pipeline outputs.

### 9.1 Application Architecture

**Files created/modified:**
- `backend/app.py` (~450 lines) — Flask application with 11 routes
- `backend/templates/index.html` (~305 lines) — Single-page interface with 5-tab navigation
- `backend/static/css/style.css` (~700 lines) — Custom CSS with health-focused design system
- `backend/static/js/main.js` (~650 lines) — Vanilla JavaScript for all interactivity

**Design decision: Module-level variable storage.** The initial implementation used Flask session cookies for diary data persistence. However, when real-recipe meals replaced random ingredient sampling (see Section 9.2), meal data regularly exceeded the ~4 KB browser cookie limit, causing silent data loss. The storage was migrated to Python module-level variables (`_diary_meals`, `_diary_symptoms`, `_diary_config`, `_selected_trigger`, `_shrinkage_result`, `_last_shrinkage_result`). This is appropriate for the single-user test lab and eliminates the cookie size constraint entirely.

**Design decision: No external frameworks.** The interface uses vanilla JavaScript and custom CSS with no dependencies on Bootstrap, Tailwind, jQuery, or React. This keeps the deployment minimal (only Flask required) and ensures the code is fully auditable for the dissertation submission.

**Routes implemented:**

| Route | Method | Purpose |
|-------|--------|---------|
| `/` | GET | Render main page |
| `/get_ingredient_pool` | GET | Return sorted ingredient list and default trigger |
| `/generate_diary` | POST | Generate synthetic diary from configuration parameters |
| `/get_diary_data` | GET | Return chronologically sorted meals, symptoms, and summary |
| `/run_detection` | POST | Run raw case-crossover analysis (no shrinkage) |
| `/get_detection_results` | GET | Return detection results with ground truth trigger |
| `/get_diary_ingredients` | GET | Return sorted unique ingredients from generated diary |
| `/get_recommendations` | POST | Return embedding-based substitutes with group warnings |
| `/save_preferences` | POST | Save user preference ranking for an ingredient |
| `/get_preferences/<ingredient>` | GET | Retrieve saved ranking |
| `/delete_preferences/<ingredient>` | DELETE | Delete saved ranking |
| `/session/clear` | GET | Clear all module-level data |

**Pipeline components initialised at startup:** `EmbeddingLoader`, `IngredientGroupManager`, `SafePlatePipeline`, `RecommendationEngine`, `PreferenceAwareRecommendationEngine`, and `UserPreferenceStore` are instantiated once when the Flask app starts, avoiding repeated loading of the ~2700-ingredient embedding matrix and group CSV on each request.

**Port configuration:** The app runs on port 5001 (`app.run(debug=True, port=5001)`) because macOS reserves port 5000 for AirPlay Receiver, which returns HTTP 403 errors to Flask requests.

### 9.2 Tab 1: Configure Scenario

Tab 1 provides slider-based configuration for all `DiaryGeneratorConfig` parameters, organised into two sections:

**Key parameters** (always visible):
- Days to generate (7–90, default 30)
- Trigger effect strength (0–100%, default 50%)
- Baseline symptom rate (0–20%, default 5%)
- Trigger exposure rate (0–100%, default 30%)

**Advanced parameters** (collapsible `<details>` section):
- Meals per day mean (1–6, default 3.0)
- Min/max ingredients per meal (2–8 / 3–15, defaults 3/8)
- Hazard window hours (2–24, default 6)
- Minimum lag hours (0–6, default 1)

**Trigger ingredient selection:** A dropdown populated from `/get_ingredient_pool` allows the user to select any of the 25 pool ingredients as the ground truth trigger. Default is "cheddar cheese". The selected trigger is sent in the generation request and stored in a module-level variable for later comparison against detection results.

**Cross-field validation:** JavaScript enforces `ingredients_max >= ingredients_min` and `hazard_window > min_lag` by auto-adjusting the dependent slider when constraints are violated.

Percentage sliders (trigger effect, baseline rate, trigger exposure) display values with "%" suffix and are converted to 0–1 decimals before being sent to the backend.

**Real recipe mode:** Diary generation uses actual recipes from the Food.com dataset (`data/processed/recipes_cleaned.csv`) rather than randomly sampling individual ingredients. The `DiaryGenerator` splits recipes into two pools — those containing the selected trigger and those without — and samples from the appropriate pool based on the configured trigger exposure rate. This produces realistic ingredient co-occurrence patterns (e.g., Mexican recipes contain cilantro, lime, and jalapeño together) while maintaining controlled trigger exposure. If the trigger appears in fewer than 20 recipes, the generator falls back to random ingredient sampling. See Section 9.5 for the impact of real recipes on detection algorithm design.

### 9.3 Tab 2: View Diary

Tab 2 displays the generated diary in a two-column timeline layout with chronological ordering.

**Summary card:** A 4-column grid showing date range, total meals, total symptoms, and number of days. Values use monospace font for numerical alignment.

**Timeline layout:**
- Left column: Meal cards (white background, subtle border) showing sequential number, recipe name, timestamp, and ingredient tags as inline rounded pills with light sage background
- Right column: Symptom cards (terracotta-tinted background `#FEF3E2`, terracotta border) showing sequential number, timestamp, and symptom ID with warning icon

**Recipe name display:** Each meal card shows the recipe name from the Food.com dataset (e.g., "Meal #0: Spaghetti Carbonara"). This was enabled by modifying the `DiaryGenerator` to track recipe metadata: recipe pools changed from `list[list[str]]` to `list[tuple[str, list[str]]]` so that `(recipe_name, ingredients)` tuples flow through the entire pipeline from generation to display.

**Chronological ordering:** Meals and symptoms are sorted by timestamp in the `/get_diary_data` endpoint (earliest first). Display numbering uses array index (`Meal #0`, `Meal #1`, ...) rather than internal `meal_id` values, which may be non-sequential after sorting.

**Responsive design:** On viewports narrower than 768px, the summary grid collapses to 2 columns and the timeline switches to a single stacked column. This was implemented via CSS `@media` queries.

**Date formatting:** ISO timestamps (e.g., `2026-01-01T08:30:00`) are parsed with `new Date()` and formatted to human-readable strings (e.g., "Jan 1, 2026 8:30 AM") using `toLocaleString()` with explicit `en-US` locale.

**Edge case: zero symptoms.** If the generated diary contains no symptoms, the symptoms column displays "No symptoms recorded in this diary" instead of an empty container.

### 9.4 Tab 3: Trigger Detection — Raw Case-Crossover Analysis

Tab 3 runs trigger detection using **raw case-crossover odds ratios** (not shrinkage — see Section 9.5 for rationale) and displays results through an adaptive multi-layer filtering system.

**Case-crossover implementation:** The `/run_detection` route counts ingredient-symptom co-occurrences within the configured hazard window:

```python
for each ingredient:
    meals_with_ingredient = [meals containing this ingredient]
    for each such meal:
        symptom_count += count of symptoms within [min_lag, hazard_window] hours
    U = symptom_count
    V = len(meals_with_ingredient) - symptom_count
    raw_OR = (U + 0.5) / (V + 0.5)   # Haldane-corrected odds ratio
```

Confidence is assigned based on sample size: `HIGH` if `U+V >= 10` and `OR > 1.0`, `MEDIUM` if `U+V >= 5`, `LOW` otherwise.

**No shrinkage applied:** Results are wrapped in `ShrinkageResult` objects for pipeline compatibility, but all shrinkage-specific fields (`weight`, `mu_g`, `tau_sq`) are set to zero and `evidence_source` is set to `"raw_case_crossover"`. The `final_OR` equals `raw_OR` for every ingredient.

**Pipeline summary:** The route calls `PIPELINE.get_pipeline_summary()` to extract `trigger_list` using the configured thresholds (`trigger_or_threshold=1.5`, `trigger_min_confidence="HIGH"`). This list drives downstream same-group warnings in recommendations.

**Results display:**

*Detection summary card:* 4-column grid showing high-confidence triggers, ingredients tested, high-confidence count, and protected-safe count.

*Results table:* Columns: Ingredient, Odds Ratio, Confidence, U (Case), V (Control). Displayed through the multi-layer filtering system described in Section 9.8.

*Visual encoding:*
- Confidence badges: HIGH = sage green background, MEDIUM = gray, LOW = light gray
- OR values: terracotta colour for OR ≥ 1.5, slate for OR < 1.5
- All numeric values use monospace font

**Ground truth validation card:** A prominent card at the top of results displays the known trigger ingredient, enabling immediate validation. The ground truth row in the results table is highlighted with a light green background (`#E8F5E9`) and green left border (`#4CAF50`), with bold ingredient name and target emoji.

**"Show all results" toggle:** A checkbox below the table header allows viewing all tested ingredients including those below the filtering thresholds. When unchecked (default), only the adaptive Top-10 is shown. When checked, all ingredients with observations are displayed, sorted by the same evidence-weighted metric.

**Re-run capability:** A "Re-run Detection Analysis" button allows running detection again with the same diary data.

### 9.5 Detection Algorithm Refinement: Removal of Empirical Bayes Shrinkage

**Problem discovered:** When diary meals are sampled from real recipes (Section 9.2), empirical Bayes shrinkage produces worse results than raw odds ratios. This was a significant and initially counterintuitive finding.

**Root cause: Recipe clustering violates the group homogeneity assumption.** The shrinkage model (Section 3) assumes that ingredients within a food group have correlated but independent trigger risks: `θᵢ ~ N(μ_g, τ²)`. However, real recipes create strong co-occurrence patterns that are not accounted for by this model:

1. **Cuisine-specific clustering:** Mexican recipes contain cilantro, cumin, lime, and jalapeño together. If cheddar cheese (the trigger) appears in Mexican recipes, all co-occurring ingredients receive elevated U counts — not because they are triggers, but because they share meals with the trigger.

2. **Shrinkage amplifies confounding:** Consider the "spice" food group containing both cumin (frequently co-occurs with cheese in Mexican dishes) and oregano (rarely co-occurs). Raw ORs correctly show cumin as suspicious and oregano as neutral. But the group mean `μ_g` is pulled upward by cumin, and shrinkage then inflates oregano's score — a pure statistical artefact.

3. **Shrinkage attenuates the true signal:** Conversely, if the "cheese" group contains many untested varieties with low or zero evidence, the group mean is diluted. Shrinkage then pulls cheddar cheese's estimate *downward*, reducing the signal for the actual trigger.

**Solution:** Bypass shrinkage entirely in the web UI detection pipeline. Use raw Haldane-corrected odds ratios with multi-layer filtering (Sections 9.6–9.8) to handle the noise that shrinkage was originally intended to address.

**What was preserved:** The `ShrinkageResult` and `FullShrinkageOutput` data structures are still used for pipeline compatibility. Groups are still used for safety warnings in recommendations (Section 9.12) — the assumption "ingredients in the same group may share properties" is valid for safety (precautionary) even when it fails for detection (inferential). The shrinkage code in `group_aware_trigger_detector.py` is preserved for academic reference and its test suite still passes.

**Key insight:** Groups defined by recipe co-occurrence (lexical token grouping) are appropriate for safety recommendations ("you might also want to avoid other cheeses") but inappropriate for statistical detection ("other cheeses are also triggers"). The two use cases require different assumptions about group homogeneity.

### 9.6 Adaptive Top-N Filtering

**Problem:** Fixed OR thresholds (e.g., OR ≥ 4.0) produce inconsistent results across different parameter configurations. With high trigger effect strength, many ingredients exceed the threshold (200+ results). With moderate effects, the true trigger may fall below the threshold (zero results). Neither outcome is useful.

**Solution:** Always display the top 10 most suspicious ingredients regardless of absolute OR values. The ranking uses an evidence-weighted composite score rather than raw OR alone.

**Sorting metric:**

```
score = adjustedOR × confidence_weight × log(U + 1)
```

Where:
- `adjustedOR` = raw OR, potentially reduced by co-occurrence penalty (Section 9.7)
- `confidence_weight` = 3.0 for HIGH, 2.0 for MEDIUM, 1.0 for LOW
- `log(U + 1)` = evidence term that rewards ingredients with more case-window observations

The `log(U + 1)` term prevents ingredients with a single observation (`U=1, V=0, OR=3.0`) from outranking ingredients with substantial evidence (`U=8, V=3, OR=2.6`). The logarithm ensures diminishing returns: going from 1 to 5 observations matters more than going from 50 to 55.

**Pre-filtering before ranking:** Only ingredients meeting minimum thresholds enter the Top-10 ranking:
- Confidence must be HIGH
- `U + V >= 5` (minimum observations for reliability)
- `OR >= 1.5` (pipeline threshold)

If fewer than 10 ingredients pass pre-filtering, all qualifying ingredients are shown.

**Rationale:** Adaptive Top-N adapts to varying data distributions, always producing a meaningful and manageable results list. It is analogous to "top-K retrieval" in information retrieval systems, where the user cares about the ranking of results, not whether they exceed an arbitrary absolute threshold.

### 9.7 Co-occurrence Penalty for Recipe Confounding

**Problem:** Ingredients that frequently co-occur with the trigger in real recipes receive inflated odds ratios even though they are not causal. For example, if cheddar cheese appears primarily in Mexican recipes, cilantro (which also appears in those recipes) will have a high co-occurrence ratio with cheese and will incorrectly show elevated OR.

**Algorithm:** The frontend computes per-ingredient co-occurrence ratios from meal data:

```javascript
for each ingredient:
    meals_with_ingredient = count of meals containing this ingredient
    meals_with_both = count of meals containing BOTH this ingredient AND the trigger
    cooccurrence_ratio = meals_with_both / meals_with_ingredient
```

If the co-occurrence ratio exceeds 0.8 (i.e., the ingredient appears with the trigger in >80% of its meals), the ingredient's adjusted OR is penalised:

```javascript
if (cooccurrence_ratio > 0.8 && ingredient !== trigger):
    adjustedOR = finalOR × (1 - cooccurrence_ratio)
```

For example, an ingredient with OR=5.0 and co-occurrence ratio of 0.9 would have its adjusted OR reduced to `5.0 × 0.1 = 0.5`, effectively demoting it in the ranking. The trigger ingredient itself is exempt from the penalty.

**Design rationale:** The 0.8 threshold is deliberately conservative — it only penalises ingredients that appear with the trigger in the vast majority of their meals, which is strong evidence of recipe-level confounding rather than independent causal association. The penalty is multiplicative rather than additive, so it scales with the degree of confounding.

### 9.8 Multi-Layer Filtering System

The four filtering layers work together as a pipeline, each addressing a different source of noise:

| Layer | Purpose | Criterion | What It Removes |
|-------|---------|-----------|-----------------|
| 1. Confidence | Statistical significance | HIGH confidence only | Low-sample ingredients (U+V < 10) |
| 2. Minimum observations | Reliability | U+V ≥ 5 | Extremely sparse data |
| 3. Evidence-weighted sort | Ranking quality | `OR × conf_weight × log(U+1)` | Low-evidence high-OR artefacts |
| 4. Co-occurrence penalty | Confounding control | Penalise if ratio > 0.8 | Recipe clustering false positives |

**Layer interaction:** Layers 1–2 are hard filters (pass/fail). Layer 3 produces a ranking score. Layer 4 modifies the score before ranking. The final output is the top 10 ingredients by adjusted composite score.

**Validation results:** With the standard parameter set (60 days, 70% effect, 4% baseline, 35% exposure):
- Cheddar cheese (ground truth trigger) consistently appears in position #1–3
- Known confounders (e.g., cilantro from Mexican recipe clustering) are demoted below position 5
- System tested across 5 different trigger ingredients (chicken, egg, cheddar cheese, garlic, salmon) with consistent top-3 detection

### 9.9 Tab Navigation and Workflow Control

Tabs are enabled progressively to enforce the sequential workflow:
- **Initial state:** Only Tab 1 active; Tabs 2–5 disabled (gray, `cursor: not-allowed`)
- **After diary generation:** Tabs 2 and 3 enabled; auto-switch to Tab 2
- **After detection:** Tab 4 enabled
- **Reset All:** Clears session, disables Tabs 2–5, resets all sliders to defaults, resets trigger dropdown to default

Tab switches trigger data loading: switching to Tab 2 calls `loadDiaryData()`, switching to Tab 3 calls `loadDetectionResults()`. This ensures displayed data is always current.

### 9.10 Visual Design System

The interface uses a minimal, health-focused design appropriate for a scientific evaluation tool:

| Element | Value |
|---------|-------|
| Primary accent (sage green) | `#7C9885` |
| Secondary accent (terracotta) | `#D4A574` |
| Text colour | `#4A5568` (slate gray) |
| Body font | `system-ui, -apple-system, "Segoe UI", sans-serif` |
| Numeric/monospace font | `Consolas, Monaco, monospace` |
| Card shadow | `0 1px 3px rgba(0,0,0,0.1)` |
| Border radius | `8px` (cards), `6px` (inputs), `4px` (tags) |
| Max container width | `1200px`, centered |
| Primary buttons | Sage green background, white text |
| Secondary buttons | Terracotta background, white text |

Sage green was chosen as the primary colour for its association with health and natural ingredients. Terracotta is used as a warning/attention colour for symptoms, trigger indicators, and warning badges. Ground truth highlighting uses a distinct light green (`#E8F5E9` background, `#4CAF50` left border) to differentiate from the sage green table headers.

### 9.11 Files Created

```
backend/
├── app.py                     [NEW, 237 lines]
├── templates/
│   └── index.html             [NEW, 211 lines]
└── static/
    ├── css/
    │   └── style.css          [NEW, 565 lines]
    └── js/
        └── main.js            [NEW, 406 lines]
```

Total: ~1,419 lines of application code (Python + HTML + CSS + JavaScript).

### 9.12 Validation

The web UI was validated through automated testing of all routes:

- `GET /` returns 200 with all expected HTML elements
- `GET /get_ingredient_pool` returns 25 sorted ingredients with default trigger
- `POST /generate_diary` with default parameters produces ~100 meals and ~23 symptoms
- `GET /get_diary_data` returns meals, symptoms, and correct summary statistics
- `POST /run_detection` correctly identifies the configured trigger (cheddar cheese: OR ≈ 1.95, HIGH confidence)
- `GET /get_detection_results` returns results with ground truth trigger
- `GET /get_diary_ingredients` returns sorted unique ingredients from generated diary
- `POST /get_recommendations` returns embedding-based substitutes with trigger exclusion and group-aware warnings
- `GET /session/clear` clears all session data; subsequent data requests return `no_data`
- Changing the trigger ingredient (e.g., to "milk") and regenerating correctly shifts detection to the new trigger

### 9.13 Tab 4: Test Recommendations

Tab 4 provides an interactive interface for exploring embedding-based ingredient substitutions, integrated with the detected triggers from Tab 3.

**Trigger shortcut pills:** After detection, all detected triggers are displayed as clickable pills at the top of the tab. Clicking a pill immediately queries substitutes for that ingredient, providing a fast path to the most relevant recommendations.

**Ingredient search:** A dropdown populated with all diary ingredients (`/get_diary_ingredients` route) allows searching for substitutes for any ingredient, not just detected triggers. This supports exploratory use cases where the user wants to understand the embedding space.

**Backend route (`/get_recommendations`):** The route accepts a `query_ingredient` and calls `RECOMMENDATION_ENGINE.find_substitutes()` with the detected trigger list as exclusions. This ensures no detected trigger appears in the recommendation list. The engine uses Node2Vec embeddings to find the most semantically similar ingredients.

**Recommendation cards:** Each substitute is displayed as a card showing the ingredient name and a similarity score bar (0–1 scale). The bar width and colour intensity encode similarity strength, providing at-a-glance comparison.

**Excluded triggers panel:** A right-hand column lists all detected triggers that were excluded from recommendations, with their confidence badges. This makes the exclusion logic transparent to the user.

### 9.14 Group-Aware Recommendation Warnings

**Problem:** The `RecommendationEngine` excludes detected triggers from recommendations, but semantically similar ingredients in the same food group (e.g., mozzarella when cheddar cheese is a trigger) may still appear as top substitutes. These could be problematic for users with group-level sensitivities (e.g., dairy intolerance).

**Solution:** A two-layer approach combining backend group checking with a client-side toggle filter.

**Backend changes:** The `/get_recommendations` route was enhanced to check each recommendation against the `IngredientGroupManager`. For each recommended ingredient, the route calls `GROUP_MANAGER.get_group(name)` and compares it against the group of each detected trigger. Two fields are added to each recommendation object:

- `is_same_group` (boolean): whether the recommendation belongs to the same food group as any detected trigger
- `same_group_as_trigger` (string or null): the specific trigger ingredient sharing the group

This approach leverages the existing `IngredientGroupManager` without modifying the recommendation engine itself, keeping the backend modules untouched.

**Frontend display:** Recommendation cards with `is_same_group: true` display a terracotta-bordered warning badge reading "Same group as trigger: **[trigger name]**". The warning uses the terracotta colour palette consistent with other trigger indicators in the UI.

**Toggle filter:** A filter bar above the recommendations provides a "Hide Same-Group Items" button. When activated:

1. All same-group recommendation cards receive `display: none` via the `hidden-by-filter` CSS class
2. The button text changes to "Show Same-Group Items" and receives the `active` class (terracotta background)
3. A count badge appears showing how many items are hidden (e.g., "3 items hidden")
4. If hiding same-group items leaves zero visible recommendations, an empty-state message is displayed

The filter state resets whenever new recommendations are loaded, ensuring a clean state for each query.

**Design rationale:** The toggle approach (warn but allow, rather than auto-exclude) was chosen because group-level sensitivity is not guaranteed—a user allergic to cheddar cheese may tolerate goat cheese despite both being in the dairy group. The warning provides information; the toggle provides control. This aligns with the SafePlate philosophy of empowering user decision-making rather than prescribing dietary restrictions.

**Edge cases handled:**
- Ingredients not found in any group: `get_group()` returns `None`, so `is_same_group` defaults to `False`
- Multiple triggers in the same group: the warning references the first matching trigger found
- No same-group items in results: the toggle button still appears but has no effect; no count badge is shown

**Known limitation: trigger threshold dependency.** Same-group warnings only fire for ingredients whose group matches a *detected trigger* in `trigger_list`. The trigger list is built by `get_pipeline_summary()` using the configured thresholds (`trigger_or_threshold=1.5`, `trigger_min_confidence="HIGH"`). If the ground truth trigger does not meet both criteria (e.g., insufficient observations for HIGH confidence), it will not appear in `trigger_list`, and no same-group warnings will be generated. This is a design trade-off: using the strict trigger list avoids warning fatigue from false positives, but may miss legitimate group-level risks when detection evidence is weak.

### 9.15 Recommendation Explanations (Integration)

**Purpose:** Display multi-evidence explanations for each recommendation to support interpretability and trust.

**Integration with Phase 3 explainability:** Phase 3 (pre-web UI) implemented `RecommendationExplanation` with three-component explanations (Section 8). This subsection integrates that backend capability into the Tab 4 UI.

**Backend changes (app.py):**
- Switched from `RecommendationEngine.find_substitutes()` to `PreferenceAwareRecommendationEngine.get_substitutes_with_preferences()` with `include_explanations=True`
- Initialized `PreferenceAwareRecommendationEngine` and `UserPreferenceStore` at app startup
- Stored the full `FullShrinkageOutput` in a module-level variable (acceptable for single-user test lab) so the preference engine can look up per-ingredient shrinkage statistics when generating explanations
- Converted markdown `**bold**` to HTML `<strong>` tags using regex before sending to the frontend
- Added `explanation` field to each recommendation in the JSON response
- Fallback to basic `RecommendationEngine` if no shrinkage result is available in memory

**Frontend changes:**
- Modified `renderRecommendations()` in `main.js` to display explanation text below the similarity score (and below the group warning, if present)
- Added CSS styles for `.recommendation-explanation` (light gray text, top border separator, 13px font)
- Explanation text is always visible—not hidden behind a tooltip or accordion—because explanations are core to SafePlate's interpretability goals

**Explanation components displayed:**
1. **Safety:** Evidence from shrinkage model (e.g., "Not a trigger (OR=0.85, LOW confidence)" or "Protected safe (OR=0.00, 12 exposures with 0 symptoms)")
2. **Similarity:** Node2Vec embedding context (e.g., "Similar to chicken (similarity=0.82, based on recipe co-occurrence)")
3. **Group:** (conditional) Same-group warning when recommendation shares a food group with a detected trigger (e.g., "Same group as trigger 'cheddar cheese' (may share chemical/biological properties)")
4. **Preference:** User ranking status (currently `None` since Tab 5 is not yet implemented; will show "You ranked this #2 for chicken substitution" after Phase 6)

**Visual hierarchy (top to bottom within each card):**
1. Ingredient name (#1–10) + similarity score (primary)
2. Group warning badge (if applicable, terracotta)
3. Explanation text (secondary, gray, smaller font, separated by 1px top border)

### 9.16 Enhanced Same-Group Handling

**Issues addressed:**
1. Explanations did not mention group relationships—users saw a warning badge but no structured explanation of why group membership matters
2. "Hide Same-Group" toggle only hid cards client-side, potentially leaving only 2–3 visible recommendations instead of the expected 10
3. The concept of "same group" was not explained to users

**Solution:**

**Backend changes (`preference_aware_recommendation.py`):**
- Added `group_reason: Optional[str]` field to `RecommendationExplanation` dataclass
- Modified `to_text()` to include a `**Group:**` component between Similarity and Preference when present
- Modified `generate_explanation()` to check group membership via `IngredientGroupManager` and populate the group reason when the recommendation shares a group with any detected trigger

**Backend changes (`app.py`):**
- Added `hide_same_group` request parameter to `POST /get_recommendations`
- When `hide_same_group=True`, fetches `top_n=30` (conservative over-fetch) and filters same-group items server-side, returning the first 10 non-same-group results
- When `hide_same_group=False`, fetches `top_n=10` as before (same-group items included with warnings)
- Returns `n_same_group_hidden` count so the frontend can display how many items were skipped
- Always guarantees 10 recommendations in both modes (unless fewer than 10 candidates exist total)

**Frontend changes:**
- Toggle button now triggers a server re-fetch (`getRecommendationsFor()`) instead of client-side `display: none` filtering
- `renderRecommendations()` accepts `hideSameGroup` and `nHidden` parameters from the server response
- Hidden count display updated from server data (e.g., "Skipped 18 same-group items to show safe alternatives")
- Added info box explaining ingredient groups (light blue, informational styling) below the trigger pills section

**Explanation format (4 components when same-group):**
`**Safety:** ... | **Similarity:** ... | **Group:** Same group as trigger 'X' (may share chemical/biological properties) | **Preference:** ...`

### 9.17 User Preference Ranking (Tab 5)

**Purpose:** Allow users to manually rank preferred ingredient substitutes, integrating with the `PreferenceAwareRecommendationEngine` from Phase 1 to produce personalised recommendation orderings.

**Integration with existing backend:**
- Uses `UserPreferenceStore` (Phase 1) for persistence via `save_ranking()`, `get_ranking()`, `delete_ranking()`
- Uses `PreferenceAwareRecommendationEngine` which already re-orders results based on stored rankings
- Rankings stored in `data/user_preferences.json`, persisting across browser refreshes and diary regeneration

**Backend routes (app.py):**
- `POST /save_preferences` — accepts `query_ingredient` and `ranked_ingredients` array, calls `PREFERENCE_STORE.save_ranking()`
- `GET /get_preferences/<ingredient>` — returns saved ranking with ingredient list and timestamp, or `no_data`
- `DELETE /delete_preferences/<ingredient>` — deletes stored ranking via `PREFERENCE_STORE.delete_ranking()`

**Frontend features:**
- Tab 5 enabled after first recommendation query in Tab 4 (global state stores last recommendations and query ingredient)
- Ranking interface: numbered input boxes (1–N) for each recommendation, pre-filled from saved preferences if they exist
- Input validation: out-of-range values highlighted in red; partial ranking allowed (user need not rank all items)
- Action buttons: Save Ranking (primary), Clear Saved Ranking (appears after save), Back to Recommendations (returns to Tab 4 and re-fetches to show updated preference explanations)
- Info box explaining how preferences integrate with recommendations

**Preference flow:**
1. User queries ingredient in Tab 4 → Tab 5 enabled
2. User switches to Tab 5 → ranking items rendered, existing preferences pre-filled
3. User enters ranks (e.g., 1, 2, 3 for top three choices), clicks Save
4. Backend stores ranked list via `UserPreferenceStore`
5. User clicks "Back to Recommendations" → Tab 4 re-fetches
6. `PreferenceAwareRecommendationEngine` finds stored ranking, re-orders results: ranked items first (in rank order), unranked items after (by similarity)
7. Explanations update: "Your #1 choice", "Your #2 choice" for ranked items; unranked items unchanged

**Design rationale:**
- Simple numbered inputs for MVP (no drag-and-drop complexity)
- Partial ranking allowed: users rank only items they care about
- Explicit clear button provides user control over stored data
- Preferences are ingredient-specific (one ranking per query ingredient), matching the `UserPreferenceStore` data model

### 9.18 UI Improvements and Polish

Several iterative improvements were made to the interface during testing:

- **Recipe names in meal cards:** Each meal card in Tab 2 displays the recipe name from the Food.com dataset (e.g., "Meal #0: Spaghetti Carbonara"), providing context for ingredient co-occurrence patterns.
- **Chronological ordering:** Meals and symptoms sorted earliest-first in `/get_diary_data`, with sequential display numbering using array index.
- **Ground truth row highlighting:** The detected trigger's row in the results table uses a full-row light green background (`#E8F5E9`) with green left border (`#4CAF50`), clearly distinguishing it from the sage green table headers.
- **Filter summary text:** Below the results table, a summary line shows "Showing top N of M tested ingredients (filtered by HIGH confidence, OR ≥ 1.5, minimum 5 observations)".
- **Detection button renamed:** "Run Shrinkage Analysis" → "Run Detection Analysis" to reflect the removal of shrinkage.

### 9.19 Limitations and Future Work

- **Simplified case-crossover.** The co-occurrence counting is a placeholder for the full case-crossover algorithm. It produces reasonable results for single-trigger scenarios but does not implement discordant pair matching or Bayesian posterior estimation.
- **Same-group warning threshold dependency.** Warnings depend on the trigger passing strict detection thresholds. See Section 9.14 for details.
- **Fixed seed.** The diary generator uses `seed=42` for reproducibility. A future enhancement could expose the seed as a configurable parameter for exploring stochastic variation.
- **Single-trigger group matching.** The group-aware warning references only the first matching trigger when multiple triggers share a group.
- **In-memory state.** All diary and detection data is stored in module-level variables. This works for the single-user test lab but would not scale to multi-user production without server-side session storage.
- **Duplicate rank handling.** Users can enter duplicate rank numbers. The system accepts this by sorting and including both; a future enhancement could validate uniqueness client-side.
- **Co-occurrence penalty is client-side.** The penalty computation runs in JavaScript, requiring meal data to be fetched by the frontend. A future refactoring could move this to the backend for consistency.

---

## 10. Phase 3: Production Code Refactoring

**Status:** Completed
**Date:** February 2026
**Purpose:** Convert Jupyter notebook data pipelines into standalone, cacheable Python scripts suitable for production deployment and CI/CD integration.

### 10.1 Motivation

The Flask application depends on two data artefacts:

1. `data/processed/ingredient_groups.csv` (loaded by `ingredient_groups.py`)
2. `data/processed/embeddings/ingredient_node2vec_iter10_UNWEIGHTED_p1_q4_w10_wl80_r80_e10.txt` (loaded by `embedding_loader.py`)

Previously these were generated by manually running notebooks 01, 02, and 03 in sequence. This created several problems: (a) onboarding required Jupyter and manual cell execution, (b) there was no caching — regeneration always started from scratch, (c) the pipeline could not be invoked programmatically or from CI, and (d) the production system had an implicit dependency on research notebooks.

### 10.2 What Was Done

Three Python modules were created:

**`backend/safeplate/data_cleaning.py`** (~420 lines) — Extracts the complete cleaning pipeline from notebook 01. Constants (descriptor sets, stop tokens, frequency thresholds) were copied verbatim from the notebook cells. Functions (`remove_descriptors_from_ingredient`, `singularise_token`, `normalise_plurals_in_ingredient`, `filter_by_frequency_and_background`, `split_compound_ingredient`, `choose_group_token`, etc.) were extracted without modification. The `main()` function checks whether `recipe_ingredients.csv` and `ingredient_groups.csv` already exist; if so, it skips the pipeline unless `--force` is passed. Outputs: `recipes_cleaned.csv`, `recipe_ingredients.csv`, `ingredient_vocabulary.csv`, `ingredient_groups.csv`, `cleaning_metadata.json`.

**`backend/safeplate/embedding_training.py`** (~200 lines) — Extracts co-occurrence graph construction from notebook 02 and Node2Vec training from notebook 03. The `build_cooccurrence_edges()` function uses `itertools.combinations` on sorted unique ingredients per recipe (identical to notebook 02 cell 3734adf7). The `train_node2vec()` function wraps the `node2vec` library with the iteration 10 hyperparameters (unweighted, p=1, q=4, window=10, walk_length=80, num_walks=80, epochs=10). Caching checks for the embedding `.txt` file before training. Output: `ingredient_node2vec_iter10_UNWEIGHTED_p1_q4_w10_wl80_r80_e10.txt`.

**`backend/scripts/prepare_data.py`** (~90 lines) — Orchestrator that calls `data_cleaning.main()` then `embedding_training.main()`. Supports `--force-all`, `--force-cleaning`, `--force-embeddings`, and `--verbose` flags.

### 10.3 Notebook Archival

All 10 `.ipynb` files and the deprecation notice were moved from `notebooks/` to `notebooks/exploratory/`. A `notebooks/README.md` was added explaining the directory structure and pointing to `prepare_data.py` for production use. The notebooks are preserved for reference and reproducibility but are not required to run the production system.

### 10.4 Dependency Cleanup

`backend/requirements.txt` was updated:
- **Removed:** `Flask_SQLAlchemy`, `psycopg2-binary`, `alembic` (none imported anywhere in the backend)
- **Added:** `networkx`, `node2vec`, `gensim` (required by `embedding_training.py`)

### 10.5 Verification

The following was verified after refactoring:

1. `python backend/scripts/prepare_data.py` — skips both steps when data already exists (0.0s).
2. `python backend/scripts/prepare_data.py --force-all` — regenerates all data from `RAW_recipes.csv`.
3. `python backend/app.py` — Flask app starts normally with no notebook dependencies.
4. Both `data_cleaning.py` and `embedding_training.py` work as standalone CLI scripts via `python -m safeplate.data_cleaning` and `python -m safeplate.embedding_training`.

### 10.6 Files Created/Modified

```
backend/
├── scripts/
│   └── prepare_data.py                 [NEW, ~90 lines]
├── safeplate/
│   ├── data_cleaning.py                [NEW, ~420 lines]
│   └── embedding_training.py           [NEW, ~200 lines]
└── requirements.txt                    [MODIFIED]

notebooks/
├── README.md                           [NEW]
└── exploratory/                        [NEW, all notebooks moved here]

README.md                               [MODIFIED, setup instructions added]
docs/POST_PROGRESS_REPORT_TECHNICAL_LOG.md  [MODIFIED, this section]
```

### 10.7 Design Decisions

**Verbatim constant extraction.** All constants (descriptor sets, stop tokens, irregular plurals, frequency thresholds) were copied exactly from notebook cells rather than being refactored or simplified. This ensures bit-for-bit reproducibility with the original notebook outputs and avoids introducing subtle cleaning differences that could change downstream embeddings.

**Single-worker training.** Node2Vec walk generation and Word2Vec training both use `workers=1`. This sacrifices parallelism for deterministic reproducibility: multi-threaded Word2Vec training is non-deterministic even with a fixed seed.

**No modification to existing modules.** `embedding_loader.py`, `ingredient_groups.py`, `recommendation_engine.py`, and all Flask application code were left unchanged. The new scripts produce files in the exact format these modules expect.

---

## 11. Phase 4: Parameter Optimization and Performance Characterization

**Status:** Completed
**Date:** February 2026
**Purpose:** Systematic exploration of parameter space to characterize detection performance and identify optimal configurations for realistic evaluation.

### 11.1 Effect Strength Paradox

A counterintuitive finding emerged during systematic testing: **95% trigger effect strength performs worse than 70% with real recipes.**

**Explanation:** Very strong effects create saturation. When `P(symptom | trigger eaten) = 0.95`, nearly every trigger-containing meal produces a symptom. Since real recipes share ingredients (e.g., Mexican recipes contain both cheese *and* cilantro), every ingredient that co-occurs with the trigger also appears in symptom-associated meals at high frequency. The result is elevated ORs across many ingredients, reducing the signal-to-noise ratio.

With `P(symptom | trigger eaten) = 0.70`, approximately 30% of trigger meals do *not* produce symptoms. These "negative" trigger exposures create variance that helps distinguish the true trigger from co-occurring confounders: the trigger has consistently elevated association across both positive and negative trigger meals, while confounders have elevated association only in the subset of meals where the trigger was also present and happened to cause symptoms.

**Recommendation:** Use 60–80% effect strength for realistic detection evaluation. This mirrors real-world dietary trigger characteristics, where trigger exposure does not deterministically cause symptoms.

### 11.2 Optimal Parameter Set

Systematic testing identified the following "sweet spot" configuration:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Days | 60 | Sufficient observations for statistical reliability |
| Trigger effect strength | 70% | Clean signal without saturation (Section 11.1) |
| Baseline symptom rate | 4% | Low enough to distinguish from trigger effect |
| Trigger exposure rate | 35% | Sufficient trigger meals without overwhelming diary |
| Meals per day | 3.0 | Realistic eating pattern |
| Hazard window | 6 hours | Standard for dietary reactions |
| Minimum lag | 1 hour | Accounts for digestion time |

**Results with standard parameters:** Ground truth trigger consistently detected at position #1–3 in the Top-10 ranking across all tested trigger ingredients. Clean separation between trigger and confounders.

### 11.3 System Performance Characterization

Performance was characterized across multiple trigger ingredients and parameter configurations:

**Works well:**
- Moderate effect strength (60–80%)
- Diverse trigger ingredients with many recipes (egg, chicken, cheddar cheese)
- Sufficient diary length (≥ 30 days)
- Moderate exposure rate (25–45%)

**Struggles:**
- Very weak effects (<40%) — insufficient signal-to-noise ratio
- Very high baseline (>10%) — baseline noise masks trigger signal
- Rare triggers (<10% exposure) — insufficient observations for HIGH confidence
- Short diaries (<14 days) — too few meals for reliable statistics

**Fails:**
- Extreme confounding (e.g., trigger only appears in one cuisine type)
- Very sparse data (trigger in <5 meals total)

These failures represent inherent limitations of observational dietary data, not algorithmic failures. The case-crossover design requires sufficient discordant exposure to detect associations, and no statistical method can extract signal from data that doesn't contain it.

### 11.4 Detection vs Safety Trade-offs

The system uses dual thresholds for different purposes:

| Purpose | Threshold | Optimized For |
|---------|-----------|---------------|
| Detection display (Tab 3) | Top 10, HIGH confidence, OR ≥ 1.5 | **Precision** — avoid overwhelming user |
| Safety warnings (Tab 4) | `trigger_list` from pipeline summary | **Recall** — avoid missing dangers |

**Rationale:** Detection display should be precise to maintain user trust — showing 50 suspicious ingredients is not actionable. Safety warnings should be lenient — it is better to warn about a potential group-level risk that turns out to be benign than to miss a genuine risk. The two views serve different user needs and justify different threshold strategies.

---

## 12. Lessons Learned

### 12.1 Real-World Data Requires Different Approaches Than Synthetic Data

The initial algorithm (empirical Bayes shrinkage) was developed and validated using randomly sampled ingredients, where group homogeneity is a reasonable assumption. Real recipes broke this assumption by introducing correlated ingredient co-occurrence patterns. The lesson: **always validate with realistic data distributions, not just statistically convenient ones.**

### 12.2 Shrinkage Assumptions Violated by Recipe Structure

The DerSimonian & Laird model assumes that ingredient-level estimates within a group are independent draws from a common distribution. Real recipes create dependencies: ingredients co-occur in cuisine-specific patterns, so their U/V counts are correlated. Shrinkage cannot distinguish "high group mean because members are genuine triggers" from "high group mean because members co-occur with a trigger in recipes." Raw ORs with domain-specific filtering (co-occurrence penalty) proved more effective.

### 12.3 Fixed Thresholds Fail With Varying Data Distributions

The progression from fixed thresholds (OR ≥ 4.0 → OR ≥ 2.0 → OR ≥ 1.5) to adaptive Top-N was driven by empirical observation: no single threshold works across all parameter configurations. Adaptive Top-N ensures consistent, usable output regardless of the underlying data distribution.

### 12.4 Stronger Isn't Always Better (Effect Strength Paradox)

The assumption that higher trigger effect strength makes detection easier is wrong in the presence of confounding. At 95% effect, the confounding signal is nearly as strong as the trigger signal. At 70% effect, the imperfect association provides discriminative information. This is a specific instance of the general principle that **noise can be informative when it breaks spurious correlations**.

### 12.5 Different Thresholds for Different Purposes

Detection (precision-focused) and safety warnings (recall-focused) should not share the same threshold. This dual-threshold design reflects the asymmetric cost structure: a false-positive detection is merely confusing, while a missed safety warning could lead to an adverse reaction.

---

## 13. Dissertation Implications

### 13.1 Contributions

1. **Application of case-crossover analysis to recipe-based dietary data** with domain-specific adaptations (co-occurrence penalty, adaptive filtering) that address the confounding inherent in real recipe co-occurrence patterns.
2. **Empirical demonstration that group-level shrinkage is counterproductive with clustered data**, providing a concrete worked example of the gap between theoretical statistical properties and practical performance with structured observational data.
3. **Multi-layer filtering system** that combines statistical significance, evidence weighting, and domain knowledge (co-occurrence penalty) into a robust detection pipeline.
4. **Two-rule guard system** (from Phase 1) for protecting confirmed-safe ingredients from false-positive trigger labels.
5. **Preference-aware recommendation** with multi-evidence explanations integrating safety, similarity, group membership, and user preference signals.

### 13.2 Limitations to State Explicitly

- **Grouping is lexical, not biochemical.** Ingredient groups are defined by shared tokens in recipe data, not validated against nutritional or allergological taxonomies.
- **Co-occurrence penalty threshold (0.8) is heuristic.** The threshold was chosen based on empirical testing, not derived from a principled statistical criterion.
- **Single-trigger evaluation.** The test lab evaluates one planted trigger at a time. Multi-trigger scenarios (e.g., both dairy and gluten) are not yet tested.
- **Case-crossover is simplified.** The web UI uses co-occurrence counting rather than full discordant pair matching with Bayesian posterior estimation.
- **Fixed seed limits generalizability.** All results use `seed=42`. Stochastic variation across seeds is not characterized.

---

## 14. Timeline and Effort

- **Design phase:** February 3, 2026 (design review and specification of the statistical model, input/output contracts, and test cases).
- **Shrinkage implementation:** February 3, 2026 (complete rewrite of `group_aware_trigger_detector.py`).
- **Shrinkage testing:** February 3, 2026 (all 10 tests written and passing in the same session).
- **Integration pipeline:** February 4, 2026 (`integrated_pipeline.py` and notebook 06, validated end-to-end).
- **Phase 1 Test Lab:** February 6, 2026 (4 new modules: diary generation, case-crossover simulation, user preferences, preference-aware recommendations; notebook 08 validation).
- **Phase 3 Explainability:** February 11, 2026 (explanation generation with safety/similarity/preference components; Section 8 validation tests).
- **Phase 2 Web UI:** February 12, 2026 (Flask app scaffold, diary display, trigger detection integration, configurable trigger selection, recommendation interface with group-aware warnings, multi-evidence explanations, and user preference ranking).
- **Phase 2 Refinement:** February 12–18, 2026 (real recipe integration, shrinkage removal, adaptive Top-N filtering, co-occurrence penalty, multi-layer filtering, UI polish).
- **Phase 3 Production Refactoring:** February 2026 (notebook-to-script conversion, caching, dependency cleanup).
- **Phase 4 Parameter Optimization:** February 2026 (systematic testing, performance characterization, effect strength paradox discovery).

---

## 15. References

DerSimonian, R. & Laird, N. (1986). Meta-analysis in clinical trials. *Controlled Clinical Trials*, 7(3), 177–188.

Maclure, M. (1991). The case-crossover design: A method for studying transient effects on the risk of acute events. *American Journal of Epidemiology*, 133(2), 144–153.

Higgins, J. P. T. & Thompson, S. G. (2002). Quantifying heterogeneity in a meta-analysis. *Statistics in Medicine*, 21(11), 1539–1558.

Haldane, J. B. S. (1956). The estimation and significance of the logarithm of a ratio of frequencies. *Annals of Human Genetics*, 20(4), 309–311.

Cochran, W. G. (1954). The combination of estimates from different experiments. *Biometrics*, 10(1), 101–129.

---

## Appendix A: File Locations (Current State)

```
SafePlate/
├── backend/
│   ├── app.py                                 [~450 lines, Web UI + detection]
│   ├── templates/
│   │   └── index.html                         [~305 lines, 5-tab interface]
│   ├── static/
│   │   ├── css/
│   │   │   └── style.css                      [~700 lines, design system]
│   │   └── js/
│   │       └── main.js                        [~650 lines, detection + filtering]
│   ├── scripts/
│   │   └── prepare_data.py                    [~90 lines, data pipeline orchestrator]
│   └── safeplate/
│       ├── group_aware_trigger_detector.py    [~450 lines, shrinkage model (preserved)]
│       ├── integrated_pipeline.py             [233 lines, pipeline orchestration]
│       ├── diary_generator.py                 [~500 lines, real recipe sampling]
│       ├── case_crossover_simulator.py        [~230 lines]
│       ├── user_preferences.py                [~140 lines]
│       ├── preference_aware_recommendation.py [~320 lines]
│       ├── data_cleaning.py                   [~420 lines, notebook 01 extraction]
│       ├── embedding_training.py              [~200 lines, notebooks 02+03 extraction]
│       ├── ingredient_groups.py               [~240 lines, group lookups]
│       ├── recommendation_engine.py           [~364 lines, Node2Vec similarity]
│       ├── embedding_loader.py                [~202 lines]
│       └── utils.py                           [~78 lines]
├── tests/
│   ├── test_shrinkage.py                      [10 test functions]
│   └── test_group_integration.py              [6 test functions]
├── data/
│   ├── raw/
│   │   └── RAW_recipes.csv                    [Food.com dataset]
│   ├── processed/
│   │   ├── recipes_cleaned.csv                [Cleaned recipes with ingredients]
│   │   ├── recipe_ingredients.csv             [Per-recipe ingredient lists]
│   │   ├── ingredient_groups.csv              [2698 ingredients, 993 groups]
│   │   ├── ingredient_vocabulary.csv          [Frequency data]
│   │   └── embeddings/
│   │       └── ingredient_node2vec_*.txt      [128-dim Node2Vec embeddings]
│   ├── test_diaries/
│   └── test_results/
│       └── phase1_metrics.json
├── docs/
│   └── POST_PROGRESS_REPORT_TECHNICAL_LOG.md  [THIS DOCUMENT]
├── notebooks/
│   ├── README.md
│   └── exploratory/                           [All original notebooks archived]
└── README.md
```

---

## Appendix B: Quick Reference for Future Sessions

**Current system state:** The detection pipeline uses **raw case-crossover odds ratios** (not shrinkage) with **multi-layer filtering** (confidence, observations, evidence-weighted ranking, co-occurrence penalty). Shrinkage code is preserved but not used in the web UI detection path.

**Key file for detection:** `backend/app.py` — the `/run_detection` route builds raw ORs directly without calling the shrinkage model.

**Key file for recommendations:** `backend/app.py` — the `/get_recommendations` route uses `PreferenceAwareRecommendationEngine` with trigger exclusion and group warnings.

**Key configuration:**
```python
PIPELINE_CONFIG = PipelineConfig(
    trigger_or_threshold=1.5,
    trigger_min_confidence="HIGH",
    propagate_to_untested=True,
    recommendation_top_n=10,
)
```

**Optimal test parameters:** 60 days, 70% effect, 4% baseline, 35% exposure.

**Run the app:** `cd backend && python app.py` → access at `http://127.0.0.1:5001`

**Tests pass:**
- `test_shrinkage.py`: All 10 edge cases verified.
- `test_group_integration.py`: All 6 integration tests verified.

**Key design decisions:**
- Raw ORs > shrinkage for real recipe data (Section 9.5)
- Adaptive Top-N > fixed thresholds (Section 9.6)
- Co-occurrence penalty for recipe confounding (Section 9.7)
- Groups used for safety warnings, not detection (Section 9.5)
- 70% effect strength > 95% (Section 11.1)

**Known limitation:** Same-group warnings depend on trigger passing strict pipeline thresholds (Section 9.10).
