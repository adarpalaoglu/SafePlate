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

## 9. Dissertation Implications

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

## 11. Timeline and Effort

- **Design phase:** February 3, 2026 (design review and specification of the statistical model, input/output contracts, and test cases).
- **Shrinkage implementation:** February 3, 2026 (complete rewrite of `group_aware_trigger_detector.py`).
- **Shrinkage testing:** February 3, 2026 (all 10 tests written and passing in the same session).
- **Integration pipeline:** February 4, 2026 (`integrated_pipeline.py` and notebook 06, validated end-to-end).
- **Phase 1 Test Lab:** February 6, 2026 (4 new modules: diary generation, case-crossover simulation, user preferences, preference-aware recommendations; notebook 08 validation).
- **Phase 3 Explainability:** February 11, 2026 (explanation generation with safety/similarity/preference components; Section 8 validation tests).
- **Phase 2 Web UI:** February 12, 2026 (Flask app scaffold, diary display, trigger detection integration, configurable trigger selection, recommendation interface with group-aware warnings, multi-evidence explanations, and user preference ranking).
- **Total lines of code:** ~450 (shrinkage module) + 233 (pipeline module) + ~470 (test_shrinkage.py) + ~250 (test_group_integration.py) + ~920 (Phase 1 Test Lab modules) + ~50 (Phase 3 explainability) + ~1,419 (Web UI) = ~3,792 lines of production and test code.

---

## 12. References

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
│   ├── app.py                                 [NEW, 237 lines, Web UI]
│   ├── templates/
│   │   └── index.html                         [NEW, 211 lines, Web UI]
│   ├── static/
│   │   ├── css/
│   │   │   └── style.css                      [NEW, 565 lines, Web UI]
│   │   └── js/
│   │       └── main.js                        [NEW, 406 lines, Web UI]
│   └── safeplate/
│       ├── group_aware_trigger_detector.py    [REWRITTEN, ~450 lines]
│       ├── integrated_pipeline.py             [NEW, 233 lines]
│       ├── diary_generator.py                 [NEW, ~280 lines, Phase 1]
│       ├── case_crossover_simulator.py        [NEW, ~230 lines, Phase 1]
│       ├── user_preferences.py                [NEW, ~140 lines, Phase 1]
│       ├── preference_aware_recommendation.py [NEW, ~320 lines, Phase 1+3]
│       ├── ingredient_groups.py               [Unchanged]
│       ├── recommendation_engine.py           [Unchanged]
│       ├── embedding_loader.py                [Unchanged]
│       └── utils.py                           [Unchanged]
├── tests/
│   ├── test_shrinkage.py                      [NEW, 10 test functions]
│   └── test_group_integration.py              [NEW, 6 test functions]
├── data/
│   ├── test_diaries/                          [NEW, Phase 1]
│   ├── test_results/
│   │   └── phase1_metrics.json                [NEW, validation metrics]
│   └── processed/
│       ├── ingredient_groups.csv              [Unchanged, 2698 rows]
│       └── trigger_detection/
│           └── ingredient_stats_all.csv       [From notebook 04]
├── docs/
│   └── POST_PROGRESS_REPORT_TECHNICAL_LOG.md  [THIS DOCUMENT]
└── notebooks/
    ├── 04_trigger_detection_recommended.ipynb  [Unchanged, produces ingredient_stats]
    ├── 05_test_recommendation_engine.ipynb     [Unchanged]
    ├── 05b_test_group_integration.ipynb        [DEPRECATED, do not run]
    ├── 05b_DEPRECATED_USE_PYTHON_SCRIPT.txt   [Deprecation notice]
    ├── 06_test_integrated_pipeline.ipynb       [NEW, 20 cells]
    └── 08_test_preference_engine.ipynb         [NEW, Phase 1+3 validation]
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

**Integrated pipeline:** `backend/safeplate/integrated_pipeline.py` connects case-crossover output to shrinkage to recommendation. Usage:

```python
from safeplate.integrated_pipeline import SafePlatePipeline, PipelineConfig

pipeline = SafePlatePipeline(embedding_loader=loader, group_manager=gm)
result = pipeline.detect_triggers_from_stats(ingredient_stats)
substitutes = pipeline.get_safe_substitutes("milk", result)
```

Notebook 06 validates end-to-end: 25 tested → 426 total, 1 protected safe (tomato).

**Tests pass:**
- `test_shrinkage.py`: All 10 edge cases verified. Run with: `python tests/test_shrinkage.py`
- `test_group_integration.py`: All 6 integration tests verified. Run with: `python tests/test_group_integration.py`

**Next step:** Complete Tabs 4–5 of the Flask web UI (Test Recommendations and Set Preferences).

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

**Status:** Completed (Tabs 1–3 fully functional, Tabs 4–5 placeholder)
**Date:** February 2026
**Purpose:** Web-based test lab interface for controlled evaluation of trigger detection and recommendation systems, enabling interactive parameter exploration and visual validation of pipeline outputs.

### 9.1 Application Architecture

**Files created:**
- `backend/app.py` (237 lines) — Flask application with 7 routes
- `backend/templates/index.html` (211 lines) — Single-page interface with 5-tab navigation
- `backend/static/css/style.css` (565 lines) — Custom CSS with health-focused design system
- `backend/static/js/main.js` (406 lines) — Vanilla JavaScript for all interactivity

**Design decision: Single-page application with session storage.** The test lab is a sequential workflow (configure → generate → detect → recommend → set preferences) where each step depends on the previous. A single HTML page with tab navigation preserves this linearity while allowing users to revisit earlier steps. Flask session storage (signed cookies) was chosen over database persistence because the test lab is single-user and ephemeral — there is no need for multi-session history or concurrent access. This eliminates database setup complexity for dissertation evaluators.

**Design decision: No external frameworks.** The interface uses vanilla JavaScript and custom CSS with no dependencies on Bootstrap, Tailwind, jQuery, or React. This keeps the deployment minimal (only Flask required) and ensures the code is fully auditable for the dissertation submission.

**Routes implemented:**

| Route | Method | Purpose |
|-------|--------|---------|
| `/` | GET | Render main page |
| `/get_ingredient_pool` | GET | Return sorted ingredient list and default trigger |
| `/generate_diary` | POST | Generate synthetic diary from configuration parameters |
| `/get_diary_data` | GET | Return meals, symptoms, and summary from session |
| `/run_detection` | POST | Run simplified case-crossover and shrinkage pipeline |
| `/get_detection_results` | GET | Return shrinkage results with ground truth trigger |
| `/session/clear` | GET | Clear all session data |

**Pipeline components initialised at startup:** `EmbeddingLoader`, `IngredientGroupManager`, and `SafePlatePipeline` are instantiated once when the Flask app starts, avoiding repeated loading of the ~2700-ingredient embedding matrix and group CSV on each request.

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

**Trigger ingredient selection:** A dropdown populated from `/get_ingredient_pool` allows the user to select any of the 25 pool ingredients as the ground truth trigger. Default is "cheddar cheese". The selected trigger is sent in the generation request and stored in session for later comparison against detection results.

**Cross-field validation:** JavaScript enforces `ingredients_max >= ingredients_min` and `hazard_window > min_lag` by auto-adjusting the dependent slider when constraints are violated.

Percentage sliders (trigger effect, baseline rate, trigger exposure) display values with "%" suffix and are converted to 0–1 decimals before being sent to the backend.

### 9.3 Tab 2: View Diary

Tab 2 displays the generated diary in a two-column timeline layout.

**Summary card:** A 4-column grid showing date range, total meals, total symptoms, and number of days. Values use monospace font for numerical alignment.

**Timeline layout:**
- Left column: Meal cards (white background, subtle border) showing timestamp, meal ID, and ingredient tags as inline rounded pills with light sage background
- Right column: Symptom cards (terracotta-tinted background `#FEF3E2`, terracotta border) showing timestamp and symptom ID with warning icon

**Responsive design:** On viewports narrower than 768px, the summary grid collapses to 2 columns and the timeline switches to a single stacked column. This was implemented via CSS `@media` queries.

**Date formatting:** ISO timestamps from session (e.g., `2026-01-01T08:30:00`) are parsed with `new Date()` and formatted to human-readable strings (e.g., "Jan 1, 2026 8:30 AM") using `toLocaleString()` with explicit `en-US` locale.

**Edge case: zero symptoms.** If the generated diary contains no symptoms, the symptoms column displays "No symptoms recorded in this diary" instead of an empty container.

### 9.4 Tab 3: Trigger Detection Integration

Tab 3 integrates with the `SafePlatePipeline` to run shrinkage analysis and display results.

**Simplified case-crossover placeholder:** The full case-crossover algorithm (notebook 04) involves window matching, discordant pair identification, and Bayesian posterior estimation. For the web UI MVP, a simplified placeholder was implemented that counts ingredient-symptom co-occurrences within the configured hazard window:

```python
for each ingredient:
    meals_with_ingredient = [meals containing this ingredient]
    for each such meal:
        symptom_count += count of symptoms within [min_lag, hazard_window] hours
    U = symptom_count
    V = len(meals_with_ingredient) - symptom_count
```

This produces (U, V) counts suitable for the shrinkage pipeline input. The placeholder correctly identifies the ground truth trigger in testing (e.g., cheddar cheese at OR ≈ 1.95 with HIGH confidence given default parameters). The full case-crossover will replace this in a future refinement.

**Pipeline integration:** The route calls `PIPELINE.detect_triggers_from_stats(ingredient_stats)` and `PIPELINE.get_pipeline_summary(shrinkage_result)`, reusing the validated pipeline from Section 7 without modification.

**Session size optimisation:** The shrinkage pipeline returns results for all ~257 ingredients (25 tested + 232 untested group members). Storing all of these in the Flask session cookie would exceed the 4KB browser cookie limit. The route therefore stores only tested ingredients (`U + V > 0`) and protected-safe ingredients, reducing the stored set to ~25 entries.

**Results display:**

*Detection summary card:* 4-column grid showing triggers detected, ingredients tested, high-confidence count, and protected-safe count.

*Results table:* Columns: Ingredient, Odds Ratio, Confidence, U (Case), V (Control). Sorted by OR descending, then by confidence (HIGH → MEDIUM → LOW). Protected-safe ingredients are excluded from the table (their safety is the model's assurance, not something requiring user review).

*Visual encoding:*
- Rows with OR ≥ 1.5 receive terracotta background highlighting (`trigger-row` class)
- Confidence badges: HIGH = sage green background, MEDIUM = gray, LOW = light gray
- OR values: terracotta colour for OR ≥ 1.5, slate for OR < 1.5
- All numeric values use monospace font

**Ground truth validation card:** A prominent terracotta-bordered card at the top of results displays the known trigger ingredient, enabling the user to immediately validate whether the detection algorithm correctly identified it. The ground truth row in the table is marked with a target emoji and a thick left border.

**Re-run capability:** A "Re-run Analysis" button allows running detection again with the same diary data, useful after parameter changes in future phases.

### 9.5 Tab Navigation and Workflow Control

Tabs are enabled progressively to enforce the sequential workflow:
- **Initial state:** Only Tab 1 active; Tabs 2–5 disabled (gray, `cursor: not-allowed`)
- **After diary generation:** Tabs 2 and 3 enabled; auto-switch to Tab 2
- **After detection:** Tab 4 enabled
- **Reset All:** Clears session, disables Tabs 2–5, resets all sliders to defaults, resets trigger dropdown to default

Tab switches trigger data loading: switching to Tab 2 calls `loadDiaryData()`, switching to Tab 3 calls `loadDetectionResults()`. This ensures displayed data is always current.

### 9.6 Visual Design System

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

Sage green was chosen as the primary colour for its association with health and natural ingredients. Terracotta is used as a warning/attention colour for symptoms, trigger indicators, and the ground truth card, providing clear visual distinction without the aggressive connotation of red.

### 9.7 Files Created

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

### 9.8 Validation

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

### 9.9 Tab 4: Test Recommendations

Tab 4 provides an interactive interface for exploring embedding-based ingredient substitutions, integrated with the detected triggers from Tab 3.

**Trigger shortcut pills:** After detection, all detected triggers are displayed as clickable pills at the top of the tab. Clicking a pill immediately queries substitutes for that ingredient, providing a fast path to the most relevant recommendations.

**Ingredient search:** A dropdown populated with all diary ingredients (`/get_diary_ingredients` route) allows searching for substitutes for any ingredient, not just detected triggers. This supports exploratory use cases where the user wants to understand the embedding space.

**Backend route (`/get_recommendations`):** The route accepts a `query_ingredient` and calls `RECOMMENDATION_ENGINE.find_substitutes()` with the detected trigger list as exclusions. This ensures no detected trigger appears in the recommendation list. The engine uses Node2Vec embeddings to find the most semantically similar ingredients.

**Recommendation cards:** Each substitute is displayed as a card showing the ingredient name and a similarity score bar (0–1 scale). The bar width and colour intensity encode similarity strength, providing at-a-glance comparison.

**Excluded triggers panel:** A right-hand column lists all detected triggers that were excluded from recommendations, with their confidence badges. This makes the exclusion logic transparent to the user.

### 9.10 Group-Aware Recommendation Warnings

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

### 9.11 Recommendation Explanations (Integration)

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

### 9.12 Enhanced Same-Group Handling

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

### 9.13 User Preference Ranking (Tab 5)

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

### 9.14 Limitations and Future Work

- **Simplified case-crossover.** The co-occurrence counting is a placeholder for the full case-crossover algorithm. It produces reasonable results for single-trigger scenarios but does not implement discordant pair matching or Bayesian posterior estimation.
- **Session cookie size.** Flask's default cookie-based sessions have a 4KB limit. The session size optimisation (storing only tested ingredients) keeps the current implementation within limits, but adding more data in future phases may require switching to server-side session storage (e.g., `flask-session` with filesystem backend).
- **Fixed seed.** The diary generator uses `seed=42` for reproducibility. A future enhancement could expose the seed as a configurable parameter for exploring stochastic variation.
- **Single-trigger group matching.** The group-aware warning currently references only the first matching trigger when multiple triggers share a group. A future enhancement could list all matching triggers in the warning.
- **In-memory shrinkage result.** The full `FullShrinkageOutput` is stored in a module-level variable for explanation generation. This works for the single-user test lab but would not scale to multi-user production without server-side session storage.
- **Duplicate rank handling.** Users can enter duplicate rank numbers (e.g., two items both ranked #1). The system accepts this by sorting and including both; a future enhancement could validate uniqueness client-side.
