# Notebooks

## Directory Structure

```
notebooks/
└── exploratory/          # Archived research notebooks (not required for production)
    ├── 01_load_raw_data.ipynb
    ├── 02_build_cooccurrence_graph.ipynb
    ├── 03_node2vec_embeddings.ipynb
    ├── 03_node2vec_embeddings__FINAL.ipynb
    ├── 04_trigger_detection_MVP.ipynb
    ├── 04_trigger_detection_recommended.ipynb
    ├── 05_test_recommendation_engine.ipynb
    ├── 05b_test_group_integration.ipynb (DEPRECATED)
    ├── 06_test_integrated_pipeline.ipynb
    └── 08_test_preference_engine.ipynb
```

## Production Data Pipeline

The cleaning and embedding logic from notebooks 01-03 has been extracted into
standalone Python scripts with built-in caching:

```bash
# Run the full pipeline (skips steps whose outputs already exist)
python backend/scripts/prepare_data.py

# Force regeneration
python backend/scripts/prepare_data.py --force-all
```

See the project root `README.md` for full setup instructions.

## Exploratory Notebooks

The notebooks in `exploratory/` document the original research and experimentation
process. They are preserved for reference and reproducibility but are **not**
required to run the production system.

| Notebook | Purpose |
|----------|---------|
| 01 | Raw data loading and ingredient cleaning pipeline |
| 02 | Co-occurrence graph construction |
| 03 | Node2Vec embedding training and hyperparameter tuning |
| 04 | Trigger detection (case-crossover design) |
| 05 | Recommendation engine testing |
| 05b | Group integration testing (deprecated, replaced by `tests/`) |
| 06 | Integrated pipeline validation |
| 08 | Preference engine and explainability validation |
