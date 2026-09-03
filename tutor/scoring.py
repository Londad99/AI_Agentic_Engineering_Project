"""Turn per-criterion judgements into a score. Deliberately not the model's job.

An LLM asked for "a score out of 10" gives a number that is not reproducible and often
contradicts its own feedback ("you missed the main point... 8/10"). So the model judges
criteria and the arithmetic happens here: the score always matches the visible
reasoning, and grading twice gives the same number.
"""

from __future__ import annotations

from typing import Literal

Status = Literal["covered", "partial", "missing"]

_WEIGHTS: dict[str, float] = {"covered": 1.0, "partial": 0.5, "missing": 0.0}

# Thresholds for the verbal verdict. "Correct" demands more than a bare majority:
# a student who covered 3 of 5 required ideas has not answered the question.
CORRECT_AT = 0.85
PARTIAL_AT = 0.40


def score_from_points(statuses: list[str]) -> float:
    """Mean credit across the criteria, 0.0 to 1.0."""
    if not statuses:
        return 0.0
    return sum(_WEIGHTS.get(status, 0.0) for status in statuses) / len(statuses)


def verdict_from_score(score: float) -> str:
    if score >= CORRECT_AT:
        return "correct"
    if score >= PARTIAL_AT:
        return "partially_correct"
    return "incorrect"
