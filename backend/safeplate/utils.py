"""Utility functions for SafePlate recommendation engine."""

import numpy as np
from numpy.typing import NDArray


def cosine_similarity(vec_a: NDArray[np.float64], vec_b: NDArray[np.float64]) -> float:
    """
    Compute cosine similarity between two vectors.

    Args:
        vec_a: First vector (1D numpy array)
        vec_b: Second vector (1D numpy array)

    Returns:
        Cosine similarity score between -1 and 1

    Raises:
        ValueError: If vectors have different dimensions or are zero vectors
    """
    if vec_a.shape != vec_b.shape:
        raise ValueError(f"Vector dimensions don't match: {vec_a.shape} vs {vec_b.shape}")

    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)

    if norm_a == 0 or norm_b == 0:
        raise ValueError("Cannot compute cosine similarity with zero vector")

    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


def cosine_similarity_batch(
    query_vec: NDArray[np.float64],
    matrix: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    Compute cosine similarity between a query vector and all rows in a matrix.

    This is more efficient than computing similarities one at a time.

    Args:
        query_vec: Query vector (1D array of shape (d,))
        matrix: Matrix of vectors (2D array of shape (n, d))

    Returns:
        Array of cosine similarities (1D array of shape (n,))
    """
    # Normalize query vector
    query_norm = np.linalg.norm(query_vec)
    if query_norm == 0:
        raise ValueError("Cannot compute cosine similarity with zero vector")
    query_normalized = query_vec / query_norm

    # Normalize all rows in matrix
    row_norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    # Avoid division by zero
    row_norms = np.where(row_norms == 0, 1, row_norms)
    matrix_normalized = matrix / row_norms

    # Compute dot products (cosine similarities)
    similarities = matrix_normalized @ query_normalized

    return similarities


def normalize_ingredient_name(name: str) -> str:
    """
    Normalize ingredient name for consistent matching.

    Args:
        name: Raw ingredient name

    Returns:
        Normalized ingredient name (lowercase, stripped)
    """
    return name.strip().lower()
