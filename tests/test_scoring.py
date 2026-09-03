"""Tests for score derivation.

The property that matters is that the number cannot contradict the feedback. If every
criterion is marked missing, no rounding, weighting or threshold may produce a passing
verdict - which is exactly the kind of encouraging-but-wrong grading an LLM asked for
a number produces on its own.

Run:  python tests/test_scoring.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tutor.scoring import score_from_points, verdict_from_score  # noqa: E402


def test_extremes():
    assert score_from_points(["covered"] * 4) == 1.0
    assert score_from_points(["missing"] * 4) == 0.0
    assert verdict_from_score(1.0) == "correct"
    assert verdict_from_score(0.0) == "incorrect"
    print("extremes              OK")


def test_partial_credit():
    assert score_from_points(["covered", "missing"]) == 0.5
    assert score_from_points(["partial", "partial"]) == 0.5
    print("partial credit        OK")


def test_majority_is_not_correct():
    """Three of five ideas is not a correct answer, however encouraging that feels."""
    score = score_from_points(["covered", "covered", "covered", "missing", "missing"])
    assert verdict_from_score(score) == "partially_correct", verdict_from_score(score)
    print("majority != correct   OK")


def test_nothing_covered_can_never_pass():
    for count in range(1, 10):
        assert verdict_from_score(score_from_points(["missing"] * count)) == "incorrect"
    print("no false pass         OK")


def test_unknown_status_scores_zero():
    """A status the model invented must not be silently credited."""
    assert score_from_points(["covered", "brilliant"]) == 0.5
    print("unknown status        OK")


def test_empty_is_not_a_pass():
    assert score_from_points([]) == 0.0
    assert verdict_from_score(score_from_points([])) == "incorrect"
    print("empty criteria        OK")


if __name__ == "__main__":
    test_extremes()
    test_partial_credit()
    test_majority_is_not_correct()
    test_nothing_covered_can_never_pass()
    test_unknown_status_scores_zero()
    test_empty_is_not_a_pass()
    print("\nall scoring tests passed")
