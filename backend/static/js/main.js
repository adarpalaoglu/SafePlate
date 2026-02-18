document.addEventListener("DOMContentLoaded", () => {
    // --- Load ingredient pool ---
    const triggerSelect = document.getElementById("trigger-select");

    async function loadIngredientPool() {
        try {
            const res = await fetch("/get_ingredient_pool");
            const data = await res.json();
            if (data.status === "success") {
                triggerSelect.innerHTML = "";
                data.ingredients.forEach((ing) => {
                    const opt = document.createElement("option");
                    opt.value = ing;
                    opt.textContent = ing;
                    if (ing === data.default_trigger) opt.selected = true;
                    triggerSelect.appendChild(opt);
                });
            }
        } catch (err) {
            console.error("Failed to load ingredient pool:", err);
            triggerSelect.innerHTML = '<option value="">Error loading ingredients</option>';
        }
    }

    loadIngredientPool();

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

    // --- Generate Diary ---
    const generateBtn = document.getElementById("generate-btn");
    const statusEl = document.getElementById("generation-status");

    generateBtn.addEventListener("click", async () => {
        const selectedTrigger = triggerSelect.value;
        if (!selectedTrigger) {
            statusEl.className = "status-message error";
            statusEl.textContent = "Please select a trigger ingredient before generating the diary.";
            return;
        }

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

            // Reset trigger dropdown
            loadIngredientPool();

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

        meals.forEach((meal) => {
            const tags = meal.ingredients
                .map((ing) => `<span class="ingredient-tag">${ing}</span>`)
                .join("");

            const card = document.createElement("div");
            card.className = "meal-card";
            card.innerHTML = `
                <div class="meal-timestamp">${formatTimestamp(meal.timestamp)}</div>
                <div class="meal-id">Meal #${meal.meal_id}</div>
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
            runDetectionBtn.textContent = "Run Shrinkage Analysis";
        }
    }

    runDetectionBtn.addEventListener("click", runDetection);
    rerunDetectionBtn.addEventListener("click", runDetection);

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

            const s = data.summary;
            document.getElementById("det-summary-triggers").textContent = s.n_triggers;
            document.getElementById("det-summary-tested").textContent = s.n_tested;
            document.getElementById("det-summary-high-conf").textContent = s.n_high_confidence;
            document.getElementById("det-summary-protected").textContent = s.n_protected_safe;

            document.getElementById("ground-truth-trigger").textContent = data.ground_truth_trigger || "Unknown";

            renderDetectionTable(data.ingredients, data.ground_truth_trigger);
        } catch (err) {
            console.error("Failed to load detection results:", err);
        }
    }

    function renderDetectionTable(ingredients, groundTruthTrigger) {
        const tbody = document.getElementById("detection-table-body");
        tbody.innerHTML = "";

        const confOrder = { HIGH: 0, MEDIUM: 1, LOW: 2 };
        const visible = ingredients.filter((i) => !i.protected);
        visible.sort((a, b) => {
            if (b.final_OR !== a.final_OR) return b.final_OR - a.final_OR;
            return (confOrder[a.confidence] || 3) - (confOrder[b.confidence] || 3);
        });

        visible.forEach((ing) => {
            const row = document.createElement("tr");
            const classes = [];
            if (ing.final_OR >= 1.5) classes.push("trigger-row");
            if (ing.ingredient === groundTruthTrigger) classes.push("ground-truth-row");
            row.className = classes.join(" ");

            const orClass = ing.final_OR >= 1.5 ? "or-high" : "or-neutral";
            const confClass = ing.confidence === "HIGH" ? "confidence-high"
                : ing.confidence === "MEDIUM" ? "confidence-medium" : "confidence-low";

            const namePrefix = ing.ingredient === groundTruthTrigger ? "&#127919; " : "";

            row.innerHTML = `
                <td>${namePrefix}${ing.ingredient}</td>
                <td><span class="or-value ${orClass}">${ing.final_OR.toFixed(2)}</span></td>
                <td><span class="confidence-badge ${confClass}">${ing.confidence}</span></td>
                <td>${ing.U}</td>
                <td>${ing.V}</td>
            `;
            tbody.appendChild(row);
        });
    }

    // --- Recommendations ---
    const getSubstitutesBtn = document.getElementById("get-substitutes-btn");
    const ingredientSearchDropdown = document.getElementById("ingredient-search-dropdown");

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
                ingredientSearchDropdown.innerHTML = '<option value="">Select an ingredient...</option>';
                data.ingredients.forEach((ing) => {
                    const opt = document.createElement("option");
                    opt.value = ing;
                    opt.textContent = ing;
                    ingredientSearchDropdown.appendChild(opt);
                });
            }
        } catch (err) {
            console.error("Failed to load ingredients:", err);
        }
    }

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
        const ingredient = ingredientSearchDropdown.value;
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

        symptoms.forEach((symptom) => {
            const card = document.createElement("div");
            card.className = "symptom-card";
            card.innerHTML = `
                <div class="symptom-timestamp">${formatTimestamp(symptom.timestamp)}</div>
                <div class="symptom-id">
                    <span class="symptom-icon">&#9888;&#65039;</span>
                    Symptom #${symptom.symptom_id}
                </div>
            `;
            list.appendChild(card);
        });
    }
});
