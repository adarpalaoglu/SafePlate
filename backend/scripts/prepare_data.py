#!/usr/bin/env python3
"""
SafePlate Data Preparation Pipeline
====================================
Orchestrator script that runs the full data pipeline:

  Step 1: Clean raw recipes and build ingredient groups
          (backend/safeplate/data_cleaning.py)
  Step 2: Build co-occurrence graph and train Node2Vec embeddings
          (backend/safeplate/embedding_training.py)

Both steps have built-in caching: if outputs already exist,
the step is skipped unless --force flags are used.

Usage:
    python backend/scripts/prepare_data.py                  # skip existing
    python backend/scripts/prepare_data.py --force-all      # rebuild everything
    python backend/scripts/prepare_data.py --force-cleaning  # only redo step 1
    python backend/scripts/prepare_data.py --force-embeddings # only redo step 2
    python backend/scripts/prepare_data.py --verbose         # detailed output
"""

import argparse
import sys
import time
from pathlib import Path

# Ensure the backend package is importable regardless of where we run from.
SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = BACKEND_DIR.parent

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from safeplate.data_cleaning import main as run_cleaning
from safeplate.embedding_training import main as run_embeddings


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SafePlate Data Preparation Pipeline",
    )
    parser.add_argument(
        "--force-all", action="store_true",
        help="Force regeneration of all outputs",
    )
    parser.add_argument(
        "--force-cleaning", action="store_true",
        help="Force regeneration of cleaned data and groups",
    )
    parser.add_argument(
        "--force-embeddings", action="store_true",
        help="Force retraining of Node2Vec embeddings",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print detailed progress messages",
    )
    args = parser.parse_args()

    force_cleaning = args.force_all or args.force_cleaning
    force_embeddings = args.force_all or args.force_embeddings

    print("=" * 60)
    print("  SafePlate Data Preparation Pipeline")
    print("=" * 60)

    t0 = time.time()

    # ------------------------------------------------------------------
    # Step 1: Data cleaning
    # ------------------------------------------------------------------
    print("\n[Step 1/2] Data Cleaning")
    print("-" * 40)
    try:
        run_cleaning(force=force_cleaning, verbose=args.verbose)
    except FileNotFoundError as e:
        print(f"\n  ERROR: {e}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Step 2: Embedding training
    # ------------------------------------------------------------------
    print("\n[Step 2/2] Embedding Training")
    print("-" * 40)
    try:
        run_embeddings(force=force_embeddings, verbose=args.verbose)
    except FileNotFoundError as e:
        print(f"\n  ERROR: {e}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    elapsed = time.time() - t0
    print("\n" + "=" * 60)
    print(f"  Pipeline complete ({elapsed:.1f}s)")
    print("=" * 60)


if __name__ == "__main__":
    main()
