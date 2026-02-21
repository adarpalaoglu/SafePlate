"""Load and cache Node2Vec ingredient embeddings."""

from pathlib import Path
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from .utils import normalize_ingredient_name


class EmbeddingLoader:
    """
    Load and provide access to Node2Vec ingredient embeddings.

    Embeddings are cached in memory after first load.
    """

    # Default path relative to project root
    DEFAULT_EMBEDDING_PATH = (
        "data/processed/embeddings/"
        "ingredient_node2vec_iter10_UNWEIGHTED_p1_q4_w10_wl80_r80_e10.txt"
    )

    def __init__(self, embedding_path: Optional[str] = None):
        """
        Initialize the embedding loader.

        Args:
            embedding_path: Path to embeddings file. If None, uses default path.
        """
        self._embedding_path = embedding_path
        self._embeddings: Optional[dict[str, NDArray[np.float64]]] = None
        self._embedding_matrix: Optional[NDArray[np.float64]] = None
        self._ingredient_list: Optional[list[str]] = None
        self._ingredient_to_idx: Optional[dict[str, int]] = None
        self._embedding_dim: Optional[int] = None

    def _resolve_path(self) -> Path:
        """Resolve the embedding file path."""
        if self._embedding_path:
            return Path(self._embedding_path)

        # Try to find project root by looking for common markers
        current = Path.cwd()
        for parent in [current] + list(current.parents):
            if (parent / "data" / "processed" / "embeddings").exists():
                return parent / self.DEFAULT_EMBEDDING_PATH
            if (parent / "backend").exists() and (parent / "data").exists():
                return parent / self.DEFAULT_EMBEDDING_PATH

        # Fallback: assume we're in project root
        return Path(self.DEFAULT_EMBEDDING_PATH)

    def load(self) -> None:
        """
        Load embeddings from file into memory.

        File format expected:
        - Line 1: num_ingredients embedding_dim
        - Subsequent lines: ingredient_name val1 val2 ... valN
        """
        if self._embeddings is not None:
            return  # Already loaded

        path = self._resolve_path()
        if not path.exists():
            raise FileNotFoundError(f"Embedding file not found: {path}")

        embeddings: dict[str, NDArray[np.float64]] = {}
        ingredient_list: list[str] = []

        with open(path, "r", encoding="utf-8") as f:
            # Parse header
            header = f.readline().strip().split()
            num_ingredients = int(header[0])
            self._embedding_dim = int(header[1])

            # Parse embeddings
            for line in f:
                parts = line.strip().split()
                if len(parts) < self._embedding_dim + 1:
                    continue  # Skip malformed lines

                # Find where the embedding values start
                # Values are floats, so we work backwards from the end
                values = []
                idx = len(parts) - 1
                while idx >= 0 and len(values) < self._embedding_dim:
                    try:
                        values.insert(0, float(parts[idx]))
                        idx -= 1
                    except ValueError:
                        break

                if len(values) != self._embedding_dim:
                    continue  # Skip if we couldn't parse enough values

                # Everything before the values is the ingredient name
                ingredient_name = " ".join(parts[: idx + 1])
                normalized_name = normalize_ingredient_name(ingredient_name)

                embeddings[normalized_name] = np.array(values, dtype=np.float64)
                ingredient_list.append(normalized_name)

        self._embeddings = embeddings
        self._ingredient_list = ingredient_list
        self._ingredient_to_idx = {name: i for i, name in enumerate(ingredient_list)}

        # Build matrix for batch operations
        self._embedding_matrix = np.array(
            [embeddings[name] for name in ingredient_list],
            dtype=np.float64
        )

        loaded_count = len(embeddings)
        if loaded_count != num_ingredients:
            print(f"Warning: Expected {num_ingredients} ingredients, loaded {loaded_count}")

    def get_embedding(self, ingredient: str) -> Optional[NDArray[np.float64]]:
        """
        Get embedding vector for an ingredient.

        Args:
            ingredient: Ingredient name

        Returns:
            Embedding vector or None if ingredient not found
        """
        self.load()  # Ensure loaded
        normalized = normalize_ingredient_name(ingredient)
        return self._embeddings.get(normalized)

    def has_ingredient(self, ingredient: str) -> bool:
        """Check if an ingredient exists in the embeddings."""
        self.load()
        normalized = normalize_ingredient_name(ingredient)
        return normalized in self._embeddings

    def get_all_ingredients(self) -> list[str]:
        """Get list of all ingredient names."""
        self.load()
        return list(self._ingredient_list)

    def get_embedding_matrix(self) -> NDArray[np.float64]:
        """Get the full embedding matrix (n_ingredients x embedding_dim)."""
        self.load()
        return self._embedding_matrix

    def get_ingredient_index(self, ingredient: str) -> Optional[int]:
        """Get the index of an ingredient in the matrix."""
        self.load()
        normalized = normalize_ingredient_name(ingredient)
        return self._ingredient_to_idx.get(normalized)

    def get_ingredient_by_index(self, idx: int) -> str:
        """Get ingredient name by index."""
        self.load()
        return self._ingredient_list[idx]

    @property
    def embedding_dim(self) -> int:
        """Get the embedding dimension."""
        self.load()
        return self._embedding_dim

    @property
    def num_ingredients(self) -> int:
        """Get the number of ingredients."""
        self.load()
        return len(self._embeddings)

    def __len__(self) -> int:
        """Return number of ingredients."""
        return self.num_ingredients


# Global singleton for caching
_default_loader: Optional[EmbeddingLoader] = None


def get_embedding_loader(embedding_path: Optional[str] = None) -> EmbeddingLoader:
    """
    Get the embedding loader (singleton pattern for default path).

    Args:
        embedding_path: Custom path to embeddings file.
                       If None, returns cached default loader.

    Returns:
        EmbeddingLoader instance
    """
    global _default_loader

    if embedding_path is not None:
        # Custom path - create new loader
        return EmbeddingLoader(embedding_path)

    if _default_loader is None:
        _default_loader = EmbeddingLoader()

    return _default_loader
