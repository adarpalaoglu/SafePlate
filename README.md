# SafePlate

SafePlate is a web-based test lab for evaluating dietary trigger detection algorithms. It generates synthetic food diaries using real recipes from the Food.com dataset (~220K cleaned recipes), runs co-occurrence-based trigger detection using Haldane-corrected odds ratios with confidence-based filtering, and recommends safe substitute ingredients via Node2Vec embeddings trained on recipe co-occurrence patterns. It was my final-year dissertation project (CS310) at the University of Warwick. It is a research prototype, not a consumer or clinical tool.

## Results

How often does the detection find the hidden trigger? `backend/scripts/evaluate_presets.py` generates diaries from real recipes for each preset in the app, with 10 different trigger ingredients (egg, milk, cheddar cheese, parmesan cheese, tomato, chicken, garlic, mushroom, soy sauce, cinnamon) and 10 random seeds each. It ranks the ingredients exactly as the app does. Raw runs are in `results/evaluation_presets.csv`.

| Preset | Diary | Runs | Trigger ranked 1st | In top 3 |
|---|---|---|---|---|
| Easy | 90 days, 50% exposure, 70% effect, 1% baseline | 100 | 94% | 99% |
| Balanced | 60 days, 35% exposure, 70% effect, 4% baseline | 100 | 99% | 100% |
| Difficult | 45 days, 25% exposure, 40% effect, 8% baseline | 100 | 86% | 100% |

The misses are mostly ingredients that share recipes with the trigger. For example, garlic was beaten in 3 of 10 Difficult runs. Surprisingly, Easy does slightly worse than Balanced: with a 1% baseline, almost every symptom follows the trigger, so ingredients that often appear with it score highly too. These are synthetic diaries with a single trigger, so the numbers show how well the method separates a known signal, not how it would do on real people.

## Setup

SafePlate needs Python 3.12 and the Food.com recipe dataset. The dataset is too large for this repository, so you download it once and build the processed files yourself.

1. Download `RAW_recipes.csv` from the [Food.com Recipes and Interactions](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions) dataset on Kaggle (free account needed) and put it in `data/raw/`.
2. Install and build the data. Cleaning about 230K recipes and training the Node2Vec embeddings takes a while the first time.

```bash
python3.12 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r backend/requirements.txt
python backend/scripts/prepare_data.py --force-all --verbose
```

3. Start the app and open `http://localhost:5001`. A guided tour starts on the first visit.

```bash
python backend/app.py
```

## Tests

```bash
pip install pytest
python -m pytest tests
```

`tests/test_detection.py` covers the detection the app runs (exposure counting, the Haldane correction, confidence levels and a planted trigger) and needs no data. `tests/test_group_integration.py` needs the processed data. `tests/test_shrinkage.py` covers the shrinkage model kept for reference.

## How It Works

The UI is organised into five tabs:

1. **Configure Scenario** — select a trigger ingredient from the full vocabulary (~2,698 ingredients), set simulation parameters (diary length, trigger effect strength, exposure rate, baseline symptom rate, hazard window, meals per day, ingredient counts, min lag), or load a preset (Balanced Testing, Easy Detection, Difficult Detection).
2. **View Diary** — browse the generated timeline of meals and symptoms; a parameter validation panel compares configured values against what was actually generated.
3. **Run Detection** — the system analyses ingredient-symptom co-occurrences using Haldane-corrected odds ratios with confidence-based filtering and a co-occurrence penalty that down-weights ingredients confounded with high-confidence triggers. Results are ranked by suspicion score, with a verdict banner showing whether the hidden trigger was correctly identified.
4. **Find Safe Substitutes** — select a detected trigger and receive substitute recommendations based on Node2Vec embedding similarity, with same-group filtering to avoid semantically related ingredients.
5. **Rank Your Preferences** — personalise recommendations by drag-ranking substitutes; rankings persist across sessions (JSON-backed) and reorder future results.

## Project Structure

```
SafePlate/
├── backend/
│   ├── app.py                              # Flask app and routes
│   ├── requirements.txt
│   ├── scripts/
│   │   ├── prepare_data.py                 # Data pipeline orchestrator (cleaning + embeddings)
│   │   └── evaluate_presets.py             # Detection accuracy per preset (Results table)
│   ├── safeplate/
│   │   ├── __init__.py
│   │   ├── detection.py                    # Production detection: Haldane-corrected odds ratios
│   │   ├── diary_generator.py              # Synthetic diary generation using real Food.com recipes
│   │   ├── case_crossover_simulator.py     # Case-crossover counting (U/V discordant pairs)
│   │   ├── group_aware_trigger_detector.py # Empirical Bayes shrinkage (DerSimonian & Laird);
│   │   │                                   #   preserved for reference — NOT used in production
│   │   │                                   #   detection; ShrinkageResult/FullShrinkageOutput
│   │   │                                   #   data classes ARE reused as containers in app.py
│   │   ├── integrated_pipeline.py          # SafePlatePipeline class; get_pipeline_summary()
│   │   │                                   #   and get_trigger_ingredients() called in production;
│   │   │                                   #   detect_triggers_from_stats() (shrinkage) is not
│   │   ├── recommendation_engine.py        # Node2Vec embedding-based ingredient recommendations
│   │   ├── preference_aware_recommendation.py  # Preference-weighted recommendation re-ranking
│   │   ├── user_preferences.py             # Persistent user preference store (JSON-backed)
│   │   ├── embedding_loader.py             # Load and query trained Node2Vec embeddings
│   │   ├── embedding_training.py           # Build co-occurrence graph and train Node2Vec
│   │   ├── ingredient_groups.py            # Ingredient group manager (food taxonomy)
│   │   ├── data_cleaning.py                # Raw CSV → cleaned recipes + ingredient vocabulary
│   │   └── utils.py                        # Shared utilities (name normalisation, etc.)
│   ├── templates/
│   │   └── index.html                      # Single-page application shell
│   └── static/
│       ├── css/style.css                   # Application stylesheet
│       └── js/main.js                      # All frontend logic (vanilla JS, no framework)
├── data/
│   ├── raw/                                # RAW_recipes.csv (not tracked in git)
│   └── processed/                          # Generated by prepare_data.py (not tracked)
│       ├── recipes_cleaned.csv
│       ├── ingredient_vocabulary.csv
│       ├── ingredient_groups.csv
│       ├── ingredient_cooccurrence_edges.csv
│       ├── recipe_ingredients.csv
│       └── embeddings/
├── results/
│   └── evaluation_presets.csv              # Raw runs behind the Results table
├── tests/
│   ├── test_detection.py                   # Unit tests for the production detection
│   ├── test_shrinkage.py                   # Unit tests for the shrinkage model
│   └── test_group_integration.py           # Integration tests for group-aware detection
└── .gitignore
```

## Key Technical Decisions

**Real recipes over random ingredients.** Meals are sampled from actual Food.com recipes to maintain realistic ingredient co-occurrence patterns. This makes the detection problem harder and more ecologically valid, but it also means recipe-level clustering must be accounted for in the analysis.

**Raw odds ratios over empirical Bayes shrinkage.** The DerSimonian & Laird hierarchical shrinkage model was implemented in full and tested, but removed from the production detection path. Recipe-based ingredient clustering violates the group homogeneity assumption that shrinkage relies on, causing genuine trigger signals to be pulled toward the group mean. Raw Haldane-corrected odds ratios (+0.5 applied to both counts) with confidence-based filtering (HIGH: n >= 20 and OR > 1.0; MEDIUM: n >= 10; LOW: otherwise) proved more effective. The shrinkage code is preserved and its tests still pass.

**Co-occurrence penalty.** Ingredients that frequently co-occur with HIGH-confidence detected triggers are down-weighted in the final ranking. This reduces false positives caused by recipe clustering, where a genuine trigger and a commonly paired ingredient would otherwise both score highly.

**Node2Vec embeddings.** Ingredient similarity is learned from co-occurrence patterns across ~220K Food.com recipes, so substitute recommendations reflect genuine culinary substitutability rather than a manually curated taxonomy.

**Vanilla JavaScript.** The frontend uses no framework. This was an intentional choice to keep the prototype simple and dependency-light; all UI logic lives in a single `main.js` file.
