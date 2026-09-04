"""Tests for the hand-written cosine similarity and ranking.

These check the maths against its own definition, with vectors chosen so the right
answer is known without computing it. The cross-check against ChromaDB's index needs
a real store, so it lives in scripts/verify_search.py.

Run:  python tests/test_vectormath.py
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tutor.vectormath import cosine_similarity, normalize, rank  # noqa: E402


def test_known_angles():
    """Vectors whose angle is obvious by eye."""
    assert cosine_similarity([1, 0], [1, 0]) == 1.0            # same direction
    assert abs(cosine_similarity([1, 0], [0, 1])) < 1e-6        # perpendicular
    assert cosine_similarity([1, 0], [-1, 0]) == -1.0           # opposite
    assert abs(cosine_similarity([1, 0], [1, 1]) - math.sqrt(2) / 2) < 1e-6   # 45 degrees
    print("known angles          OK")


def test_magnitude_does_not_matter():
    """The whole point of using the angle: a longer text must not score higher."""
    short, long_version = [1, 1], [100, 100]
    query = [1, 0.5]
    assert abs(cosine_similarity(query, short) - cosine_similarity(query, long_version)) < 1e-6
    print("magnitude ignored     OK")


def test_normalize_produces_unit_length():
    unit = normalize([3, 4])                       # 3-4-5 triangle
    assert abs(math.sqrt(sum(x * x for x in unit)) - 1.0) < 1e-6
    assert abs(unit[0] - 0.6) < 1e-6 and abs(unit[1] - 0.8) < 1e-6
    assert normalize([0, 0]) == [0.0, 0.0], "a zero vector must not divide by zero"
    print("normalize             OK")


def test_shortcut_matches_the_definition():
    """rank() uses dot products for speed; it must agree with the full formula.

    This is the test that lets the optimisation exist. Without it, a subtle mistake in
    the vectorised version would silently reorder results and nothing would look wrong.
    """
    matrix = [[1, 0, 0], [0.9, 0.1, 0], [0, 1, 0], [0.4, 0.4, 0.8], [-1, 0, 0]]
    query = [1, 0.2, 0]
    ranked = rank(query, matrix, top_k=len(matrix))
    for index, score in ranked:
        assert abs(score - cosine_similarity(query, matrix[index])) < 1e-5, index
    print("shortcut == definition OK")


def test_order_is_best_first_and_top_k_cuts():
    matrix = [[1, 0], [0, 1], [0.7, 0.7]]
    ranked = rank([1, 0], matrix, top_k=2)
    assert [i for i, _ in ranked] == [0, 2], ranked
    assert ranked[0][1] > ranked[1][1], "not sorted best first"
    assert len(ranked) == 2, "top_k did not cut"
    print("ordering and top_k    OK")


def test_threshold_can_return_nothing():
    """A query with no good match must return an empty list, not a bad best guess."""
    assert rank([1, 0], [[0, 1], [0, 1]], top_k=3, threshold=0.5) == []
    print("threshold             OK")


def test_empty_inputs_do_not_crash():
    assert rank([1, 0], [], top_k=3) == []
    assert rank([0, 0], [[1, 0]], top_k=3) == [], "a zero query has no direction"
    print("empty inputs          OK")


def test_a_zero_row_does_not_poison_the_ranking():
    """An empty chunk would divide by zero and produce NaN, which sorts unpredictably."""
    ranked = rank([1, 0], [[1, 0], [0, 0], [0.5, 0.5]], top_k=3)
    assert all(not math.isnan(score) for _, score in ranked), ranked
    assert ranked[0][0] == 0
    print("zero row handled      OK")


if __name__ == "__main__":
    test_known_angles()
    test_magnitude_does_not_matter()
    test_normalize_produces_unit_length()
    test_shortcut_matches_the_definition()
    test_order_is_best_first_and_top_k_cuts()
    test_threshold_can_return_nothing()
    test_empty_inputs_do_not_crash()
    test_a_zero_row_does_not_poison_the_ranking()
    print("\nall vector maths tests passed")
