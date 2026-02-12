document.addEventListener("DOMContentLoaded", () => {
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

                // Enable Tab 2
                const tab2Btn = document.querySelector('[data-tab="tab2"]');
                tab2Btn.disabled = false;

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
