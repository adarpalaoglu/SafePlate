#!/usr/bin/env python3

import sys
from pathlib import Path
import pandas as pd


def main():
    print("=" * 60)
    print("SafePlate — Node2Vec Training Script")
    print("Running with Python:", sys.executable)
    print("Working directory:", Path.cwd())
    print("=" * 60)

    # --------------------------------------------------
    # REQUIRED INPUT FILE (FULL BIPARTITE GRAPH)
    # --------------------------------------------------
    bipartite_edges_path = Path("data/processed/ingredient_recipe_bipartite_edges.csv")

    if not bipartite_edges_path.exists():
        print("ERROR: Required file not found:")
        print(" ", bipartite_edges_path.resolve())
        sys.exit(1)

    print("✓ Found bipartite edge file:")
    print(" ", bipartite_edges_path.resolve())

    # --------------------------------------------------
    # LOAD EDGE LIST (SANITY CHECK ONLY)
    # --------------------------------------------------
    print("\nLoading bipartite edge list...")
    edges_df = pd.read_csv(bipartite_edges_path)

    print("Loaded edge list successfully.")
    print("Number of edges:", len(edges_df))
    print("Columns:", list(edges_df.columns))

    expected_cols = {"node_u", "node_v", "weight"}
    if set(edges_df.columns) != expected_cols:
        print("ERROR: Unexpected columns in edge list.")
        sys.exit(1)

    print("✓ Edge list schema is correct.")


if __name__ == "__main__":
    main()
