"""
SafePlate – Embedding Training Pipeline
========================================
Standalone script that reproduces the co-occurrence graph building
(notebook 02) and Node2Vec training (notebook 03).

Input:   data/processed/recipe_ingredients.csv  (from data_cleaning.py)
Output:  data/processed/embeddings/ingredient_node2vec_iter10_UNWEIGHTED_p1_q4_w10_wl80_r80_e10.txt
         data/processed/ingredient_cooccurrence_edges.csv  (intermediate)

Usage:
    python -m safeplate.embedding_training              # skip if output exists
    python -m safeplate.embedding_training --force       # retrain
    python -m safeplate.embedding_training --verbose      # print progress
"""

import argparse
import ast
from collections import Counter
from itertools import combinations
from pathlib import Path

import networkx as nx
import pandas as pd
from gensim.models import KeyedVectors
from node2vec import Node2Vec

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
EMB_DIR = DATA_DIR / "embeddings"

RECIPE_INGREDIENTS_PATH = DATA_DIR / "recipe_ingredients.csv"
EDGES_PATH = DATA_DIR / "ingredient_cooccurrence_edges.csv"

# ---------------------------------------------------------------------------
# Default hyperparameters (matching iter 10 – the best config)
# ---------------------------------------------------------------------------
DEFAULT_EMBEDDING_FILENAME = (
    "ingredient_node2vec_iter10_UNWEIGHTED_p1_q4_w10_wl80_r80_e10.txt"
)
DEFAULT_DIMS = 128
DEFAULT_WALK_LENGTH = 80
DEFAULT_NUM_WALKS = 80
DEFAULT_WINDOW = 10
DEFAULT_P = 1.0
DEFAULT_Q = 4.0
DEFAULT_EPOCHS = 10
DEFAULT_SEED = 42


# ---------------------------------------------------------------------------
# Functions – extracted from notebooks 02 and 03
# ---------------------------------------------------------------------------

def build_cooccurrence_edges(recipes_df: pd.DataFrame) -> Counter:
    """
    Build co-occurrence edge counts from recipe ingredient lists.

    From notebook 02 cell 3734adf7. Each recipe contributes at most one
    count per pair; pairs are unordered with alphabetical ordering.

    Parameters
    ----------
    recipes_df : DataFrame with an 'ingredients' column containing lists.

    Returns
    -------
    Counter mapping (ingredient_a, ingredient_b) -> count.
    """
    edge_counts: Counter = Counter()

    for ingredients in recipes_df["ingredients"]:
        unique_ings = sorted(set(ingredients))
        for a, b in combinations(unique_ings, 2):
            edge_counts[(a, b)] += 1

    return edge_counts


def build_graph(edge_counts: Counter, weighted: bool = False) -> nx.Graph:
    """
    Build a NetworkX graph from edge counts.

    Parameters
    ----------
    edge_counts : Counter mapping (a, b) -> count.
    weighted : If True, add edge weights. If False, unweighted graph.

    Returns
    -------
    nx.Graph
    """
    G = nx.Graph()

    if weighted:
        for (a, b), count in edge_counts.items():
            G.add_edge(a, b, weight=float(count))
    else:
        for (a, b) in edge_counts:
            G.add_edge(a, b)

    return G


def train_node2vec(
    graph: nx.Graph,
    dims: int = DEFAULT_DIMS,
    walk_length: int = DEFAULT_WALK_LENGTH,
    num_walks: int = DEFAULT_NUM_WALKS,
    window: int = DEFAULT_WINDOW,
    p: float = DEFAULT_P,
    q: float = DEFAULT_Q,
    epochs: int = DEFAULT_EPOCHS,
    seed: int = DEFAULT_SEED,
    unweighted: bool = True,
) -> KeyedVectors:
    """
    Train Node2Vec embeddings on the given graph.

    From notebook 03 cell cbe86245 (get_embeddings_cooc_node2vec).

    Returns
    -------
    gensim.models.KeyedVectors with trained embeddings.
    """
    n2v = Node2Vec(
        graph,
        dimensions=dims,
        walk_length=walk_length,
        num_walks=num_walks,
        p=p,
        q=q,
        weight_key=(None if unweighted else "weight"),
        workers=1,
        seed=seed,
    )

    model = n2v.fit(
        window=window,
        min_count=1,
        batch_words=128,
        workers=1,
        epochs=epochs,
    )

    return model.wv


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def _print(msg: str, verbose: bool) -> None:
    if verbose:
        print(msg)


def main(
    force: bool = False,
    verbose: bool = False,
    dimensions: int = DEFAULT_DIMS,
    walk_length: int = DEFAULT_WALK_LENGTH,
    num_walks: int = DEFAULT_NUM_WALKS,
    window: int = DEFAULT_WINDOW,
    p: float = DEFAULT_P,
    q: float = DEFAULT_Q,
    epochs: int = DEFAULT_EPOCHS,
    seed: int = DEFAULT_SEED,
) -> None:
    """
    Run the embedding training pipeline.

    If the output embedding file already exists and *force* is False,
    the pipeline is skipped entirely.
    """
    EMB_DIR.mkdir(parents=True, exist_ok=True)
    embedding_path = EMB_DIR / DEFAULT_EMBEDDING_FILENAME

    if not force and embedding_path.exists():
        print("  [SKIP] Embeddings already exist. Use --force to retrain.")
        return

    # Check input exists
    if not RECIPE_INGREDIENTS_PATH.exists():
        raise FileNotFoundError(
            f"recipe_ingredients.csv not found: {RECIPE_INGREDIENTS_PATH}\n"
            "Run data_cleaning.py first (or: python -m safeplate.data_cleaning)."
        )

    # ------------------------------------------------------------------
    # Step 1: Load recipes
    # ------------------------------------------------------------------
    print("  Loading recipe ingredients ...")
    recipes_df = pd.read_csv(RECIPE_INGREDIENTS_PATH)
    recipes_df["ingredients"] = recipes_df["ingredients"].apply(ast.literal_eval)
    _print(f"  Loaded {len(recipes_df)} recipes", verbose)

    # ------------------------------------------------------------------
    # Step 2: Build co-occurrence edges
    # ------------------------------------------------------------------
    print("  Building co-occurrence edges ...")
    edge_counts = build_cooccurrence_edges(recipes_df)
    _print(f"  Unique edges: {len(edge_counts)}", verbose)

    # Save edge list (intermediate output)
    edges_df = pd.DataFrame(
        [{"ingredient_a": a, "ingredient_b": b, "weight": count}
         for (a, b), count in edge_counts.items()]
    ).sort_values(by="weight", ascending=False).reset_index(drop=True)

    edges_df.to_csv(EDGES_PATH, index=False)
    _print(f"  Saved edges: {EDGES_PATH}", verbose)

    # ------------------------------------------------------------------
    # Step 3: Build graph (unweighted for iter 10 config)
    # ------------------------------------------------------------------
    print("  Building graph ...")
    G = build_graph(edge_counts, weighted=False)
    _print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges", verbose)

    # ------------------------------------------------------------------
    # Step 4: Train Node2Vec
    # ------------------------------------------------------------------
    print("  Training Node2Vec embeddings (this may take a few minutes) ...")
    kv = train_node2vec(
        G,
        dims=dimensions,
        walk_length=walk_length,
        num_walks=num_walks,
        window=window,
        p=p,
        q=q,
        epochs=epochs,
        seed=seed,
        unweighted=True,
    )

    # ------------------------------------------------------------------
    # Step 5: Save embeddings
    # ------------------------------------------------------------------
    kv.save_word2vec_format(str(embedding_path))
    print(f"  Saved: {embedding_path}")
    print(f"  Vectors: {len(kv)} | dim: {kv.vector_size}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SafePlate embedding training pipeline",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Retrain even if embedding file already exists",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print detailed progress messages",
    )
    parser.add_argument("--dimensions", type=int, default=DEFAULT_DIMS)
    parser.add_argument("--walk-length", type=int, default=DEFAULT_WALK_LENGTH)
    parser.add_argument("--num-walks", type=int, default=DEFAULT_NUM_WALKS)
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    parser.add_argument("--p", type=float, default=DEFAULT_P)
    parser.add_argument("--q", type=float, default=DEFAULT_Q)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    main(
        force=args.force,
        verbose=args.verbose,
        dimensions=args.dimensions,
        walk_length=args.walk_length,
        num_walks=args.num_walks,
        window=args.window,
        p=args.p,
        q=args.q,
        epochs=args.epochs,
        seed=args.seed,
    )
