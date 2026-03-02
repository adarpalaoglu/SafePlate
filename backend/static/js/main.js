document.addEventListener("DOMContentLoaded", () => {
    // --- Global state for Tab 5 preferences ---
    let lastRecommendations = null;
    let lastQueryIngredient = null;

    // --- Searchable trigger ingredient selector ---
    let allIngredients = [];
    let selectedTriggerIngredient = "cheddar cheese";

    const triggerSearch = document.getElementById("trigger-search");
    const triggerDropdownList = document.getElementById("trigger-dropdown-list");
    const triggerSelectedName = document.getElementById("trigger-selected-name");
    const triggerSelectedDisplay = document.getElementById("trigger-selected-display");
    const triggerClearBtn = document.getElementById("trigger-clear-btn");
    const triggerIngredientCount = document.getElementById("trigger-ingredient-count");

    async function loadAllIngredients() {
        try {
            const res = await fetch("/get_all_ingredients");
            const data = await res.json();
            if (data.status === "success") {
                allIngredients = data.ingredients;
                selectedTriggerIngredient = data.default_trigger;
                triggerSelectedName.textContent = data.default_trigger;
                triggerIngredientCount.textContent = `(${allIngredients.length} ingredients available)`;
            }
        } catch (err) {
            console.error("Failed to load ingredients:", err);
        }
    }

    function filterIngredients(searchText) {
        if (!searchText) return allIngredients;
        const lower = searchText.toLowerCase();
        return allIngredients.filter(ing => ing.includes(lower));
    }

    function renderDropdownItems(items) {
        triggerDropdownList.innerHTML = "";
        if (items.length === 0) {
            const noMatch = document.createElement("div");
            noMatch.className = "trigger-dropdown-item no-match";
            noMatch.textContent = "No ingredients found";
            triggerDropdownList.appendChild(noMatch);
        } else {
            items.forEach(ing => {
                const item = document.createElement("div");
                item.className = "trigger-dropdown-item";
                if (ing === selectedTriggerIngredient) item.classList.add("selected");
                item.textContent = ing;
                item.addEventListener("mousedown", (e) => {
                    e.preventDefault();
                    selectTriggerIngredient(ing);
                });
                triggerDropdownList.appendChild(item);
            });
        }
    }

    function selectTriggerIngredient(ing) {
        selectedTriggerIngredient = ing;
        triggerSelectedName.textContent = ing;
        triggerSearch.value = "";
        triggerDropdownList.classList.remove("visible");
    }

    triggerSearch.addEventListener("input", () => {
        const items = filterIngredients(triggerSearch.value);
        renderDropdownItems(items);
        triggerDropdownList.classList.add("visible");
    });

    triggerSearch.addEventListener("focus", () => {
        const items = filterIngredients(triggerSearch.value);
        renderDropdownItems(items);
        triggerDropdownList.classList.add("visible");
    });

    triggerSearch.addEventListener("blur", () => {
        // Small delay so click on item registers before hiding
        setTimeout(() => triggerDropdownList.classList.remove("visible"), 150);
    });

    // Keyboard navigation
    triggerSearch.addEventListener("keydown", (e) => {
        const items = triggerDropdownList.querySelectorAll(".trigger-dropdown-item:not(.no-match)");
        const current = triggerDropdownList.querySelector(".trigger-dropdown-item.highlighted");
        let idx = Array.from(items).indexOf(current);

        if (e.key === "ArrowDown") {
            e.preventDefault();
            if (current) current.classList.remove("highlighted");
            idx = Math.min(idx + 1, items.length - 1);
            if (items[idx]) { items[idx].classList.add("highlighted"); items[idx].scrollIntoView({ block: "nearest" }); }
        } else if (e.key === "ArrowUp") {
            e.preventDefault();
            if (current) current.classList.remove("highlighted");
            idx = Math.max(idx - 1, 0);
            if (items[idx]) { items[idx].classList.add("highlighted"); items[idx].scrollIntoView({ block: "nearest" }); }
        } else if (e.key === "Enter") {
            e.preventDefault();
            if (current) {
                selectTriggerIngredient(current.textContent);
            } else if (items.length > 0) {
                selectTriggerIngredient(items[0].textContent);
            }
        }
    });

    triggerClearBtn.addEventListener("click", () => {
        selectedTriggerIngredient = "";
        triggerSelectedName.textContent = "none";
        triggerSearch.value = "";
        triggerSearch.focus();
    });

    loadAllIngredients();

    // --- Parameter Tooltips ---
    // Icons live inside <label for="..."> elements. Clicking a label
    // generates a SECOND click on the associated input, which bubbles
    // to the document handler and would immediately close the tooltip.
    // preventDefault() stops that second synthetic click.
    document.querySelectorAll(".info-icon[data-tooltip]").forEach((icon) => {
        icon.addEventListener("click", (e) => {
            e.preventDefault();
            e.stopPropagation();
            const tooltipId = icon.getAttribute("data-tooltip");
            const tooltip = document.getElementById(tooltipId);
            if (!tooltip) return;

            // Close all other tooltips
            document.querySelectorAll(".tooltip-content.visible").forEach((t) => {
                if (t !== tooltip) t.classList.remove("visible");
            });

            tooltip.classList.toggle("visible");
        });
    });

    // Keep tooltip open when clicking inside it
    document.querySelectorAll(".tooltip-content").forEach((tip) => {
        tip.addEventListener("click", (e) => {
            e.stopPropagation();
        });
    });

    // Close tooltips when clicking outside
    document.addEventListener("click", (e) => {
        if (!e.target.closest(".info-icon[data-tooltip]") && !e.target.closest(".tooltip-content")) {
            document.querySelectorAll(".tooltip-content.visible").forEach((t) => {
                t.classList.remove("visible");
            });
        }
    });

    // --- Preset Scenarios ---
    const PRESET_SCENARIOS = {
        easy: {
            label: "Easy Detection Scenario",
            description: "The trigger exposure rate is set high and the effect strength is strong, meaning the trigger ingredient appears in many meals and consistently precedes symptoms. This gives the algorithm abundant evidence to work with, producing a clear statistical signal that separates the trigger from other ingredients.",
            n_days: 90, trigger_effect: 70, baseline_rate: 1,
            trigger_exposure: 50, meals_per_day: 3, ingredients_min: 3,
            ingredients_max: 8, hazard_window: 6, min_lag: 1,
        },
        balanced: {
            label: "Balanced Testing (Recommended)",
            description: "The trigger exposure rate is moderate and the effect strength is set to a middling level, meaning the trigger appears in a realistic proportion of meals with an effect that is noticeable but not overwhelming. This leaves room for some ambiguity in the data, so the algorithm may flag a small number of other ingredients alongside the true trigger.",
            n_days: 60, trigger_effect: 70, baseline_rate: 4,
            trigger_exposure: 35, meals_per_day: 3, ingredients_min: 3,
            ingredients_max: 8, hazard_window: 6, min_lag: 1,
        },
        difficult: {
            label: "Difficult Detection Scenario",
            description: "The trigger exposure rate is low and the effect strength is weak, meaning the trigger appears rarely and its link to symptoms is faint relative to background noise. The statistical signal is thin, so the algorithm has limited evidence to distinguish the true trigger from coincidental co-occurrences.",
            n_days: 45, trigger_effect: 40, baseline_rate: 8,
            trigger_exposure: 25, meals_per_day: 3, ingredients_min: 3,
            ingredients_max: 8, hazard_window: 6, min_lag: 1,
        },
        custom: {
            label: "Custom Parameters",
            description: "You are setting all parameters manually. Results will depend entirely on your chosen configuration.",
        },
    };

    const presetSelector = document.getElementById("preset-selector");
    const presetDescriptionEl = document.getElementById("preset-description");

    presetSelector.addEventListener("change", () => {
        const preset = PRESET_SCENARIOS[presetSelector.value];
        if (!preset) {
            presetDescriptionEl.textContent = "";
            presetDescriptionEl.classList.remove("visible");
            return;
        }

        Object.entries(preset).forEach(([key, val]) => {
            if (sliders[key]) {
                sliders[key].el.value = val;
                sliders[key].display.textContent = val + sliders[key].suffix;
            }
        });

        presetDescriptionEl.textContent = preset.description;
        presetDescriptionEl.classList.add("visible");
    });

    // --- Tab switching ---
    const tabButtons = document.querySelectorAll(".tab-btn");
    const tabPanes = document.querySelectorAll(".tab-pane");

    tabButtons.forEach((btn) => {
        btn.addEventListener("click", () => {
            if (btn.disabled) return;

            tabButtons.forEach((b) => b.classList.remove("active"));
            tabPanes.forEach((p) => p.classList.remove("active"));

            btn.classList.add("active");
            document.getElementById(btn.dataset.tab).classList.add("active");

            if (btn.dataset.tab === "tab2") {
                loadDiaryData();
            }
            if (btn.dataset.tab === "tab3") {
                loadDetectionResults();
            }
            if (btn.dataset.tab === "tab4") {
                loadRecommendationInterface();
            }
            if (btn.dataset.tab === "tab5") {
                loadPreferenceInterface();
            }
        });
    });

    // --- Slider value display ---
    const sliders = {
        n_days: { el: document.getElementById("n_days"), display: document.getElementById("n_days_val"), suffix: "" },
        trigger_effect: { el: document.getElementById("trigger_effect"), display: document.getElementById("trigger_effect_val"), suffix: "%" },
        baseline_rate: { el: document.getElementById("baseline_rate"), display: document.getElementById("baseline_rate_val"), suffix: "%" },
        trigger_exposure: { el: document.getElementById("trigger_exposure"), display: document.getElementById("trigger_exposure_val"), suffix: "%" },
        meals_per_day: { el: document.getElementById("meals_per_day"), display: document.getElementById("meals_per_day_val"), suffix: "" },
        ingredients_min: { el: document.getElementById("ingredients_min"), display: document.getElementById("ingredients_min_val"), suffix: "" },
        ingredients_max: { el: document.getElementById("ingredients_max"), display: document.getElementById("ingredients_max_val"), suffix: "" },
        hazard_window: { el: document.getElementById("hazard_window"), display: document.getElementById("hazard_window_val"), suffix: "" },
        min_lag: { el: document.getElementById("min_lag"), display: document.getElementById("min_lag_val"), suffix: "" },
    };

    Object.values(sliders).forEach(({ el, display, suffix }) => {
        el.addEventListener("input", () => {
            display.textContent = el.value + suffix;
        });
    });

    // --- Input validation: cross-field constraints ---
    sliders.ingredients_min.el.addEventListener("change", () => {
        const min = parseInt(sliders.ingredients_min.el.value);
        const max = parseInt(sliders.ingredients_max.el.value);
        if (max < min) {
            sliders.ingredients_max.el.value = min;
            sliders.ingredients_max.display.textContent = min;
        }
    });

    sliders.ingredients_max.el.addEventListener("change", () => {
        const min = parseInt(sliders.ingredients_min.el.value);
        const max = parseInt(sliders.ingredients_max.el.value);
        if (max < min) {
            sliders.ingredients_max.el.value = min;
            sliders.ingredients_max.display.textContent = min;
        }
    });

    sliders.min_lag.el.addEventListener("change", () => {
        const lag = parseInt(sliders.min_lag.el.value);
        const window = parseInt(sliders.hazard_window.el.value);
        if (window <= lag) {
            sliders.hazard_window.el.value = lag + 1;
            sliders.hazard_window.display.textContent = lag + 1;
        }
    });

    sliders.hazard_window.el.addEventListener("change", () => {
        const lag = parseInt(sliders.min_lag.el.value);
        const window = parseInt(sliders.hazard_window.el.value);
        if (window <= lag) {
            sliders.hazard_window.el.value = lag + 1;
            sliders.hazard_window.display.textContent = lag + 1;
        }
    });

    // --- Silent reset (used before diary generation) ---
    async function performSilentReset() {
        try {
            await fetch("/session/clear");
        } catch (err) {
            console.error("Session clear failed:", err);
        }

        // Disable tabs 2–5 so stale content is inaccessible
        tabButtons.forEach((btn) => {
            if (btn.dataset.tab !== "tab1") {
                btn.disabled = true;
                btn.classList.remove("active");
            }
        });

        // Clear global state
        lastRecommendations = null;
        lastQueryIngredient = null;
        _allDetectionIngredients = null;
        _detectionGroundTruth = null;
        _detectionSummary = null;
    }

    // --- Generate Diary ---
    const generateBtn = document.getElementById("generate-btn");
    const statusEl = document.getElementById("generation-status");

    generateBtn.addEventListener("click", async () => {
        const selectedTrigger = selectedTriggerIngredient;
        if (!selectedTrigger) {
            statusEl.className = "status-message error";
            statusEl.textContent = "Please select a trigger ingredient before generating the diary.";
            return;
        }

        await performSilentReset();

        generateBtn.disabled = true;
        statusEl.className = "status-message";
        statusEl.textContent = "Generating...";

        const payload = {
            n_days: parseInt(sliders.n_days.el.value),
            trigger_effect_strength: parseFloat(sliders.trigger_effect.el.value) / 100,
            baseline_symptom_rate: parseFloat(sliders.baseline_rate.el.value) / 100,
            trigger_exposure_rate: parseFloat(sliders.trigger_exposure.el.value) / 100,
            meals_per_day_mean: parseFloat(sliders.meals_per_day.el.value),
            ingredients_per_meal_min: parseInt(sliders.ingredients_min.el.value),
            ingredients_per_meal_max: parseInt(sliders.ingredients_max.el.value),
            hazard_window_hours: parseInt(sliders.hazard_window.el.value),
            min_lag_hours: parseInt(sliders.min_lag.el.value),
            selected_trigger: selectedTrigger,
        };

        try {
            const res = await fetch("/generate_diary", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
            });

            const data = await res.json();

            if (data.status === "success") {
                statusEl.className = "status-message success";
                statusEl.textContent = `Generated ${data.n_meals} meals and ${data.n_symptoms} symptoms`;

                // Enable Tab 2 and Tab 3
                const tab2Btn = document.querySelector('[data-tab="tab2"]');
                const tab3Btn = document.querySelector('[data-tab="tab3"]');
                tab2Btn.disabled = false;
                tab3Btn.disabled = false;

                // Auto-switch to Tab 2
                tab2Btn.click();
            } else {
                statusEl.className = "status-message error";
                statusEl.textContent = "Error: " + (data.message || "Generation failed");
            }
        } catch (err) {
            statusEl.className = "status-message error";
            statusEl.textContent = "Error: " + err.message;
        } finally {
            generateBtn.disabled = false;
        }
    });

    // --- Reset All ---
    const resetBtn = document.getElementById("reset-all");

    resetBtn.addEventListener("click", async () => {
        if (!confirm("Reset all data and return to configuration?")) return;

        try {
            await fetch("/session/clear");

            // Disable tabs 2-5
            tabButtons.forEach((btn) => {
                if (btn.dataset.tab !== "tab1") {
                    btn.disabled = true;
                    btn.classList.remove("active");
                }
            });

            // Switch to tab 1
            tabPanes.forEach((p) => p.classList.remove("active"));
            document.getElementById("tab1").classList.add("active");
            document.querySelector('[data-tab="tab1"]').classList.add("active");

            // Reset sliders to defaults
            const defaults = {
                n_days: "30", trigger_effect: "50", baseline_rate: "5",
                trigger_exposure: "30", meals_per_day: "3", ingredients_min: "3",
                ingredients_max: "8", hazard_window: "6", min_lag: "1",
            };

            Object.entries(defaults).forEach(([key, val]) => {
                sliders[key].el.value = val;
                sliders[key].display.textContent = val + sliders[key].suffix;
            });

            // Reset trigger selection and preset
            selectedTriggerIngredient = "cheddar cheese";
            triggerSelectedName.textContent = "cheddar cheese";
            triggerSearch.value = "";
            presetSelector.value = "";
            presetDescriptionEl.textContent = "";
            presetDescriptionEl.classList.remove("visible");

            // Clear global state
            lastRecommendations = null;
            lastQueryIngredient = null;
            _allDetectionIngredients = null;
            _detectionGroundTruth = null;
            _detectionSummary = null;

            // Clear status
            statusEl.className = "status-message";
            statusEl.textContent = "";
        } catch (err) {
            console.error("Reset failed:", err);
        }
    });

    // --- Diary Display ---
    function formatTimestamp(isoString) {
        const date = new Date(isoString);
        return date.toLocaleString("en-US", {
            month: "short",
            day: "numeric",
            year: "numeric",
            hour: "numeric",
            minute: "2-digit",
            hour12: true,
        });
    }

    async function loadDiaryData() {
        try {
            const res = await fetch("/get_diary_data");
            const data = await res.json();

            const emptyState = document.getElementById("diary-empty-state");
            const summary = document.getElementById("diary-summary");
            const timeline = document.getElementById("diary-timeline");

            if (data.status === "no_data") {
                emptyState.style.display = "";
                summary.style.display = "none";
                timeline.style.display = "none";
                return;
            }

            emptyState.style.display = "none";
            summary.style.display = "";
            timeline.style.display = "";

            document.getElementById("summary-date-range").textContent = data.summary.date_range;
            document.getElementById("summary-meals").textContent = data.summary.n_meals;
            document.getElementById("summary-symptoms").textContent = data.summary.n_symptoms;
            document.getElementById("summary-days").textContent = data.summary.n_days;

            renderMeals(data.meals);
            renderSymptoms(data.symptoms);
        } catch (err) {
            console.error("Failed to load diary data:", err);
        }
    }

    function renderMeals(meals) {
        const list = document.getElementById("meals-list");
        list.innerHTML = "";

        meals.sort((a, b) => a.timestamp.localeCompare(b.timestamp));

        meals.forEach((meal, index) => {
            const tags = meal.ingredients
                .map((ing) => `<span class="ingredient-tag">${ing}</span>`)
                .join("");

            let mealLabel = `Meal #${index}`;
            if (meal.recipe_name) {
                mealLabel += `: ${meal.recipe_name}`;
            }

            const card = document.createElement("div");
            card.className = "meal-card";
            card.innerHTML = `
                <div class="meal-timestamp">${formatTimestamp(meal.timestamp)}</div>
                <div class="meal-id">${mealLabel}</div>
                <div class="meal-ingredients">${tags}</div>
            `;
            list.appendChild(card);
        });
    }

    // --- Detection ---
    const runDetectionBtn = document.getElementById("run-detection-btn");
    const rerunDetectionBtn = document.getElementById("rerun-detection-btn");
    const detectionStatusEl = document.getElementById("detection-status");

    async function runDetection() {
        runDetectionBtn.disabled = true;
        runDetectionBtn.textContent = "Running analysis...";
        detectionStatusEl.className = "status-message";
        detectionStatusEl.textContent = "";

        try {
            const res = await fetch("/run_detection", { method: "POST" });
            const data = await res.json();

            if (data.status === "success") {
                document.getElementById("detection-empty-state").style.display = "none";
                document.getElementById("detection-results").style.display = "";
                loadDetectionResults();

                // Enable Tab 4
                const tab4Btn = document.querySelector('[data-tab="tab4"]');
                tab4Btn.disabled = false;
            } else {
                detectionStatusEl.className = "status-message error";
                detectionStatusEl.textContent = "Error: " + (data.message || "Detection failed");
            }
        } catch (err) {
            detectionStatusEl.className = "status-message error";
            detectionStatusEl.textContent = "Error: " + err.message;
        } finally {
            runDetectionBtn.disabled = false;
            runDetectionBtn.textContent = "Run Detection Analysis";
        }
    }

    runDetectionBtn.addEventListener("click", runDetection);
    rerunDetectionBtn.addEventListener("click", runDetection);

    // --- Detection results state for toggle re-rendering ---
    let _allDetectionIngredients = null;
    let _detectionGroundTruth = null;
    let _detectionSummary = null;
    let _cooccurrenceRatios = null; // ingredient -> ratio with trigger

    async function loadDetectionResults() {
        try {
            const res = await fetch("/get_detection_results");
            const data = await res.json();

            if (data.status === "no_data") {
                document.getElementById("detection-empty-state").style.display = "";
                document.getElementById("detection-results").style.display = "none";
                return;
            }

            document.getElementById("detection-empty-state").style.display = "none";
            document.getElementById("detection-results").style.display = "";

            _detectionSummary = data.summary;
            _allDetectionIngredients = data.ingredients;
            _detectionGroundTruth = data.ground_truth_trigger || "Unknown";

            const s = data.summary;
            document.getElementById("det-summary-tested").textContent = s.n_tested;
            document.getElementById("det-summary-high-conf").textContent = s.n_high_confidence;
            document.getElementById("det-summary-protected").textContent = s.n_protected_safe;

            document.getElementById("ground-truth-trigger").textContent = _detectionGroundTruth;

            // Compute co-occurrence ratios from meal data
            _cooccurrenceRatios = {};
            try {
                const diaryRes = await fetch("/get_diary_data");
                const diaryData = await diaryRes.json();
                if (diaryData.status === "success" && _detectionGroundTruth) {
                    _cooccurrenceRatios = computeCooccurrence(diaryData.meals, _detectionGroundTruth);
                }
            } catch (e) {
                console.warn("Could not compute co-occurrence ratios:", e);
            }

            renderDetectionTable();
        } catch (err) {
            console.error("Failed to load detection results:", err);
        }
    }

    function computeCooccurrence(meals, trigger) {
        const totalCount = {};   // ingredient -> total meal appearances
        const withTrigger = {};  // ingredient -> appearances in meals containing trigger

        meals.forEach((meal) => {
            const ings = meal.ingredients;
            const hasTrigger = ings.includes(trigger);
            ings.forEach((ing) => {
                totalCount[ing] = (totalCount[ing] || 0) + 1;
                if (hasTrigger) {
                    withTrigger[ing] = (withTrigger[ing] || 0) + 1;
                }
            });
        });

        const ratios = {};
        for (const ing in totalCount) {
            ratios[ing] = (withTrigger[ing] || 0) / totalCount[ing];
        }
        return ratios;
    }

    // Toggle handler for "Show all results" checkbox
    document.getElementById("show-all-results").addEventListener("change", renderDetectionTable);

    function renderDetectionTable() {
        if (!_allDetectionIngredients) return;

        const tbody = document.getElementById("detection-table-body");
        tbody.innerHTML = "";

        const showAll = document.getElementById("show-all-results").checked;
        const confMultiplier = { HIGH: 1.0, MEDIUM: 0.5, LOW: 0.1 };
        const cooc = _cooccurrenceRatios || {};

        // Filter out protected ingredients
        const nonProtected = _allDetectionIngredients.filter((i) => !i.protected);

        // Compute adjusted OR for each ingredient (co-occurrence penalty)
        const withAdjusted = nonProtected.map((ing) => {
            const ratio = cooc[ing.ingredient] || 0;
            let adjustedOR = ing.final_OR;
            if (ratio > 0.8 && ing.ingredient !== _detectionGroundTruth) {
                adjustedOR = ing.final_OR * (1 - ratio);
            }
            return { ...ing, adjustedOR, coocRatio: ratio };
        });

        // Sort ALL items by evidence-weighted score: adjustedOR × conf × log(U+1)
        withAdjusted.sort((a, b) => {
            const scoreA = a.adjustedOR * (confMultiplier[a.confidence] || 0.1) * Math.log(a.U + 1);
            const scoreB = b.adjustedOR * (confMultiplier[b.confidence] || 0.1) * Math.log(b.U + 1);
            return scoreB - scoreA;
        });

        // Filter: show HIGH confidence, OR >= 1.5, U+V >= 5, cap at top 10
        const topN = 10;
        let displayed;
        if (showAll) {
            displayed = withAdjusted;
        } else {
            const candidates = withAdjusted.filter((i) =>
                i.confidence === "HIGH" && i.adjustedOR >= 1.5 && (i.U + i.V) >= 5
            );
            displayed = candidates.slice(0, topN);
        }

        // Count high-confidence items for summary stat
        const highConf = withAdjusted.filter((i) => i.confidence === "HIGH");
        document.getElementById("det-summary-triggers").textContent = displayed.length;

        const filterSummary = document.getElementById("detection-filter-summary");
        if (showAll) {
            filterSummary.textContent = `Showing all ${displayed.length} tested ingredients (${highConf.length} high-confidence)`;
        } else if (displayed.length === 0) {
            filterSummary.textContent = `No high-confidence triggers detected (${nonProtected.length} tested). Try "Show all results".`;
        } else {
            filterSummary.textContent = `Top ${displayed.length} most suspicious ingredients (of ${nonProtected.length} tested, ${highConf.length} high-confidence)`;
        }

        displayed.forEach((ing, idx) => {
            const row = document.createElement("tr");
            if (ing.ingredient === _detectionGroundTruth) {
                row.className = "ground-truth-row";
            }

            const orClass = ing.adjustedOR >= 2.0 ? "or-high" : "or-neutral";
            const confClass = ing.confidence === "HIGH" ? "confidence-high"
                : ing.confidence === "MEDIUM" ? "confidence-medium" : "confidence-low";

            const namePrefix = ing.ingredient === _detectionGroundTruth ? "&#127919; " : "";

            const orDisplay = ing.adjustedOR.toFixed(2);
            const orNote = (ing.adjustedOR < ing.final_OR - 0.01)
                ? ` <span class="or-original" title="Original OR before co-occurrence adjustment">(raw ${ing.final_OR.toFixed(2)})</span>`
                : "";

            row.innerHTML = `
                <td class="rank-col">${idx + 1}</td>
                <td>${namePrefix}${ing.ingredient}</td>
                <td><span class="or-value ${orClass}">${orDisplay}</span>${orNote}</td>
                <td><span class="confidence-badge ${confClass}">${ing.confidence}</span></td>
                <td>${ing.U}</td>
                <td>${ing.V}</td>
            `;
            tbody.appendChild(row);
        });
    }

    // --- Recommendations ---
    const getSubstitutesBtn = document.getElementById("get-substitutes-btn");
    const recIngredientSearch = document.getElementById("rec-ingredient-search");
    const recDropdownList = document.getElementById("rec-dropdown-list");
    let recIngredients = [];
    let selectedRecIngredient = "";

    async function loadRecommendationInterface() {
        try {
            const res = await fetch("/get_detection_results");
            const data = await res.json();

            if (data.status !== "success") {
                document.getElementById("recommendations-empty-state").style.display = "";
                document.getElementById("recommendations-interface").style.display = "none";
                return;
            }

            document.getElementById("recommendations-empty-state").style.display = "none";
            document.getElementById("recommendations-interface").style.display = "";

            renderTriggerPills(data.summary.trigger_list);
            await loadIngredientDropdown();
        } catch (err) {
            console.error("Failed to load recommendation interface:", err);
        }
    }

    function renderTriggerPills(triggers) {
        const container = document.getElementById("trigger-pills-container");
        container.innerHTML = "";

        if (!triggers || triggers.length === 0) {
            container.innerHTML = '<p class="help-text">No triggers detected in this diary.</p>';
            return;
        }

        triggers.forEach((trigger) => {
            const pill = document.createElement("div");
            pill.className = "trigger-pill";
            pill.textContent = trigger;
            pill.addEventListener("click", () => getRecommendationsFor(trigger));
            container.appendChild(pill);
        });
    }

    async function loadIngredientDropdown() {
        try {
            const res = await fetch("/get_diary_ingredients");
            const data = await res.json();
            if (data.status === "success") {
                recIngredients = data.ingredients;
            }
        } catch (err) {
            console.error("Failed to load ingredients:", err);
        }
    }

    function filterRecIngredients(searchText) {
        if (!searchText) return recIngredients;
        const lower = searchText.toLowerCase();
        return recIngredients.filter(ing => ing.includes(lower));
    }

    function renderRecDropdownItems(items) {
        recDropdownList.innerHTML = "";
        if (items.length === 0) {
            const noMatch = document.createElement("div");
            noMatch.className = "trigger-dropdown-item no-match";
            noMatch.textContent = "No ingredients found";
            recDropdownList.appendChild(noMatch);
        } else {
            items.forEach(ing => {
                const item = document.createElement("div");
                item.className = "trigger-dropdown-item";
                if (ing === selectedRecIngredient) item.classList.add("selected");
                item.textContent = ing;
                item.addEventListener("mousedown", (e) => {
                    e.preventDefault();
                    selectedRecIngredient = ing;
                    recIngredientSearch.value = ing;
                    recDropdownList.classList.remove("visible");
                });
                recDropdownList.appendChild(item);
            });
        }
    }

    recIngredientSearch.addEventListener("input", () => {
        const items = filterRecIngredients(recIngredientSearch.value);
        renderRecDropdownItems(items);
        recDropdownList.classList.add("visible");
        selectedRecIngredient = "";
    });

    recIngredientSearch.addEventListener("focus", () => {
        const items = filterRecIngredients(recIngredientSearch.value);
        renderRecDropdownItems(items);
        recDropdownList.classList.add("visible");
    });

    recIngredientSearch.addEventListener("blur", () => {
        setTimeout(() => recDropdownList.classList.remove("visible"), 150);
    });

    recIngredientSearch.addEventListener("keydown", (e) => {
        const items = recDropdownList.querySelectorAll(".trigger-dropdown-item:not(.no-match)");
        const current = recDropdownList.querySelector(".trigger-dropdown-item.highlighted");
        let idx = Array.from(items).indexOf(current);

        if (e.key === "ArrowDown") {
            e.preventDefault();
            if (current) current.classList.remove("highlighted");
            idx = Math.min(idx + 1, items.length - 1);
            if (items[idx]) { items[idx].classList.add("highlighted"); items[idx].scrollIntoView({ block: "nearest" }); }
        } else if (e.key === "ArrowUp") {
            e.preventDefault();
            if (current) current.classList.remove("highlighted");
            idx = Math.max(idx - 1, 0);
            if (items[idx]) { items[idx].classList.add("highlighted"); items[idx].scrollIntoView({ block: "nearest" }); }
        } else if (e.key === "Enter") {
            e.preventDefault();
            const target = current || items[0];
            if (target) {
                selectedRecIngredient = target.textContent;
                recIngredientSearch.value = target.textContent;
                recDropdownList.classList.remove("visible");
            }
        }
    });

    async function getRecommendationsFor(ingredient) {
        const resultsEl = document.getElementById("recommendations-results");
        const listEl = document.getElementById("recommendations-list");

        resultsEl.style.display = "";
        document.getElementById("query-ingredient-display").textContent = ingredient;
        listEl.innerHTML = "<p>Loading recommendations...</p>";

        // Check if same-group filtering is active
        const toggleBtn = document.getElementById("toggle-same-group-btn");
        const hideSameGroup = toggleBtn.classList.contains("active");

        try {
            const res = await fetch("/get_recommendations", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ query_ingredient: ingredient, hide_same_group: hideSameGroup }),
            });

            const data = await res.json();

            if (data.status === "success") {
                renderRecommendations(data.recommendations, data.hide_same_group, data.n_same_group_hidden);
                renderExcludedList(data.excluded_triggers);

                // Store for Tab 5 and enable it
                lastRecommendations = data.recommendations;
                lastQueryIngredient = ingredient;
                document.querySelector('[data-tab="tab5"]').disabled = false;
            } else {
                listEl.innerHTML = `<p class="error-text">${data.message}</p>`;
            }
        } catch (err) {
            console.error("Failed to get recommendations:", err);
            listEl.innerHTML = '<p class="error-text">Failed to load recommendations.</p>';
        }
    }

    function renderRecommendations(recommendations, hideSameGroup, nHidden) {
        const container = document.getElementById("recommendations-list");
        container.innerHTML = "";

        if (!recommendations || recommendations.length === 0) {
            container.innerHTML = '<p class="help-text">No substitutes available for this ingredient.</p>';
            return;
        }

        recommendations.forEach((rec, index) => {
            const card = document.createElement("div");
            card.className = "recommendation-card";
            if (rec.is_same_group) card.setAttribute("data-same-group", "true");

            let html = `
                <div class="recommendation-header">
                    <span class="recommendation-name">${index + 1}. ${rec.ingredient}</span>
                    <span class="recommendation-score">${rec.similarity.toFixed(3)}</span>
                </div>
            `;

            if (rec.is_same_group && rec.same_group_as_trigger) {
                html += `
                    <div class="same-group-warning">
                        Same group as trigger: <strong>${rec.same_group_as_trigger}</strong>
                    </div>
                `;
            }

            if (rec.explanation) {
                html += `
                    <div class="recommendation-explanation">
                        ${rec.explanation}
                    </div>
                `;
            }

            card.innerHTML = html;
            container.appendChild(card);
        });

        // Update hidden count display from server response
        const hiddenCountDisplay = document.getElementById("hidden-count-display");
        if (hideSameGroup && nHidden > 0) {
            hiddenCountDisplay.textContent = `(Skipped ${nHidden} same-group item${nHidden > 1 ? "s" : ""} to show safe alternatives)`;
            hiddenCountDisplay.style.display = "inline";
        } else {
            hiddenCountDisplay.style.display = "none";
        }
    }

    async function toggleSameGroupFilter() {
        const toggleBtn = document.getElementById("toggle-same-group-btn");
        const queryIngredient = document.getElementById("query-ingredient-display").textContent;

        // Toggle the active state
        toggleBtn.classList.toggle("active");

        // Update button text based on state
        const isNowActive = toggleBtn.classList.contains("active");
        toggleBtn.textContent = isNowActive ? "Show All" : "Hide Same-Group Items";

        // Re-fetch recommendations with new filter mode
        if (queryIngredient && queryIngredient !== "-") {
            await getRecommendationsFor(queryIngredient);
        }
    }

    document.getElementById("toggle-same-group-btn").addEventListener("click", toggleSameGroupFilter);

    function renderExcludedList(triggers) {
        const container = document.getElementById("excluded-list");
        container.innerHTML = "";

        if (!triggers || triggers.length === 0) {
            container.innerHTML = '<p class="help-text">No triggers excluded (none detected).</p>';
            return;
        }

        triggers.forEach((trigger) => {
            const item = document.createElement("div");
            item.className = "excluded-item";
            item.innerHTML = `<span>${trigger}</span>`;
            container.appendChild(item);
        });
    }

    getSubstitutesBtn.addEventListener("click", () => {
        const ingredient = selectedRecIngredient || recIngredientSearch.value.trim();
        if (!ingredient) {
            alert("Please select an ingredient first.");
            return;
        }
        getRecommendationsFor(ingredient);
    });

    function renderSymptoms(symptoms) {
        const list = document.getElementById("symptoms-list");
        list.innerHTML = "";

        if (symptoms.length === 0) {
            list.innerHTML = '<p class="empty-state" style="padding: 24px;">No symptoms recorded in this diary</p>';
            return;
        }

        symptoms.sort((a, b) => a.timestamp.localeCompare(b.timestamp));

        symptoms.forEach((symptom, index) => {
            const card = document.createElement("div");
            card.className = "symptom-card";
            card.innerHTML = `
                <div class="symptom-timestamp">${formatTimestamp(symptom.timestamp)}</div>
                <div class="symptom-id">
                    <span class="symptom-icon">&#9888;&#65039;</span>
                    Symptom #${index}
                </div>
            `;
            list.appendChild(card);
        });
    }

    // --- Tab 5: Preferences ---
    async function loadPreferenceInterface() {
        const emptyState = document.getElementById("pref-empty-state");
        const interfaceEl = document.getElementById("pref-interface");

        if (!lastRecommendations || !lastQueryIngredient) {
            emptyState.style.display = "";
            interfaceEl.style.display = "none";
            return;
        }

        emptyState.style.display = "none";
        interfaceEl.style.display = "";
        document.getElementById("pref-query-ingredient").textContent = lastQueryIngredient;
        document.getElementById("pref-status").className = "status-message";
        document.getElementById("pref-status").textContent = "";

        await renderRankingItems(lastRecommendations, lastQueryIngredient);
    }

    async function renderRankingItems(recommendations, queryIngredient) {
        const container = document.getElementById("ranking-items-container");
        const clearBtn = document.getElementById("clear-ranking-btn");
        container.innerHTML = "";

        // Fetch existing preferences
        let existingRanking = null;
        try {
            const res = await fetch(`/get_preferences/${encodeURIComponent(queryIngredient)}`);
            const data = await res.json();
            if (data.status === "success") {
                existingRanking = data.ranked_ingredients;
                clearBtn.style.display = "";
            } else {
                clearBtn.style.display = "none";
            }
        } catch (err) {
            clearBtn.style.display = "none";
        }

        recommendations.forEach((rec) => {
            const item = document.createElement("div");
            item.className = "ranking-item";

            let prefillValue = "";
            if (existingRanking) {
                const idx = existingRanking.indexOf(rec.ingredient);
                if (idx !== -1) prefillValue = idx + 1;
            }

            item.innerHTML = `
                <span class="ranking-item-name">${rec.ingredient}</span>
                <div class="ranking-input-wrapper">
                    <span class="ranking-input-label">Rank:</span>
                    <input type="number" class="ranking-input" min="1" max="${recommendations.length}"
                           data-ingredient="${rec.ingredient}" value="${prefillValue}"
                           placeholder="-">
                </div>
            `;
            container.appendChild(item);
        });

        // Add validation listeners
        container.querySelectorAll(".ranking-input").forEach((input) => {
            input.addEventListener("input", validateRankingInput);
        });
    }

    function validateRankingInput(e) {
        const input = e.target;
        const val = parseInt(input.value);
        const max = parseInt(input.max);
        if (input.value !== "" && (isNaN(val) || val < 1 || val > max)) {
            input.classList.add("invalid");
        } else {
            input.classList.remove("invalid");
        }
    }

    async function saveRanking() {
        const inputs = document.querySelectorAll(".ranking-input");
        const statusEl = document.getElementById("pref-status");
        const rankings = [];

        inputs.forEach((input) => {
            const val = parseInt(input.value);
            if (!isNaN(val) && val >= 1) {
                rankings.push({ rank: val, ingredient: input.dataset.ingredient });
            }
        });

        if (rankings.length === 0) {
            statusEl.className = "status-message error";
            statusEl.textContent = "Please enter at least one ranking.";
            return;
        }

        rankings.sort((a, b) => a.rank - b.rank);
        const ranked = rankings.map((r) => r.ingredient);

        try {
            const res = await fetch("/save_preferences", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ query_ingredient: lastQueryIngredient, ranked_ingredients: ranked }),
            });
            const data = await res.json();
            if (data.status === "success") {
                statusEl.className = "status-message success";
                statusEl.textContent = `Ranking saved! Ranked ${data.n_ranked} ingredients for '${data.query_ingredient}'.`;
                document.getElementById("clear-ranking-btn").style.display = "";
            } else {
                statusEl.className = "status-message error";
                statusEl.textContent = "Error: " + (data.message || "Save failed");
            }
        } catch (err) {
            statusEl.className = "status-message error";
            statusEl.textContent = "Error: " + err.message;
        }
    }

    async function clearRanking() {
        if (!confirm("Delete saved ranking for this ingredient?")) return;
        const statusEl = document.getElementById("pref-status");

        try {
            const res = await fetch(`/delete_preferences/${encodeURIComponent(lastQueryIngredient)}`, { method: "DELETE" });
            const data = await res.json();
            if (data.status === "success") {
                statusEl.className = "status-message success";
                statusEl.textContent = "Ranking deleted.";
                document.getElementById("clear-ranking-btn").style.display = "none";
                document.querySelectorAll(".ranking-input").forEach((input) => {
                    input.value = "";
                    input.classList.remove("invalid");
                });
            } else {
                statusEl.className = "status-message error";
                statusEl.textContent = data.message || "Nothing to delete.";
            }
        } catch (err) {
            statusEl.className = "status-message error";
            statusEl.textContent = "Error: " + err.message;
        }
    }

    function backToRecommendations() {
        const tab4Btn = document.querySelector('[data-tab="tab4"]');
        tab4Btn.click();
        // Re-fetch recommendations so updated preferences are reflected
        if (lastQueryIngredient) {
            setTimeout(() => getRecommendationsFor(lastQueryIngredient), 100);
        }
    }

    document.getElementById("save-ranking-btn").addEventListener("click", saveRanking);
    document.getElementById("clear-ranking-btn").addEventListener("click", clearRanking);
    document.getElementById("back-to-recs-btn").addEventListener("click", backToRecommendations);
});
