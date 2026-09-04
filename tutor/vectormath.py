"""Cosine similarity and ranking, written by hand.

ChromaDB can do this itself - the collection is created with hnsw:space=cosine and
returns distances. This module exists anyway, for two reasons:

  1. It is the mechanism the whole project rests on. Delegating it to a library is
     fine; not being able to write it is not.
  2. It lets us CHECK the library. scripts/verify_search.py ranks the same vectors
     with this code and with Chroma's index and compares the two orderings. A silent
     disagreement would mean the store is configured wrong - the wrong metric, or
     unnormalized vectors - and nothing about that failure looks like an error.

Only numpy is used, and only for the arithmetic.
"""

from __future__ import annotations

import numpy as np


def normalize(vector) -> list[float]:
    """Scale a vector to length 1, so cosine reduces to a dot product.

    Needed because Gemini only normalizes its full-size 3072-dim output; at 768 the
    vectors come back unnormalized and cosine distance would stop meaning what we think.
    """
    array = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(array))
    return (array / norm).tolist() if norm else array.tolist()


def cosine_similarity(a, b) -> float:
    """The definition, spelled out: dot(a,b) / (|a| * |b|).

    The cosine of the angle between two vectors. 1 = same direction, 0 = perpendicular,
    -1 = opposite. The ANGLE and not the distance, because what matters is the direction
    (the meaning) and not the magnitude (which grows with text length).
    """
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator == 0.0:
        return 0.0
    return float(np.dot(a, b) / denominator)


def rank(query_vector, matrix, top_k: int = 5, threshold: float | None = None) -> list[tuple[int, float]]:
    """Score every row against the query and return the best ones, highest first.

    Uses the dot-product shortcut: on unit vectors, dot(a,b) IS the cosine, because the
    denominator is 1. One matrix multiplication instead of a Python loop over thousands
    of vectors. `test_shortcut_matches_the_definition` pins that the two agree.

    Returns (row index, score) pairs so the caller keeps its own documents and metadata.
    """
    if matrix is None or len(matrix) == 0:
        return []

    query = np.asarray(query_vector, dtype=np.float32)
    rows = np.asarray(matrix, dtype=np.float32)

    query_norm = np.linalg.norm(query)
    row_norms = np.linalg.norm(rows, axis=1)
    # Guard against a zero vector: an empty chunk would divide by zero and poison the
    # whole ranking with NaN, which sorts unpredictably instead of failing.
    row_norms[row_norms == 0] = 1.0
    if query_norm == 0:
        return []

    scores = (rows @ query) / (row_norms * query_norm)

    order = np.argsort(-scores)[:top_k]
    results = [(int(i), float(scores[i])) for i in order]
    if threshold is not None:
        results = [pair for pair in results if pair[1] >= threshold]
    return results
