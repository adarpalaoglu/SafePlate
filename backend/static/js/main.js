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
});
