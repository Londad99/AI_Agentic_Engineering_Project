"""Tests for the anti-hallucination check on exam question sources.

Two failure modes, opposite costs:
  - accepting an invented quote ships a question about something the document never
    said, and the student studies fiction;
  - rejecting a real quote over a stray accent throws away a good question and makes
    the whole check look untrustworthy, so it gets switched off.

The threshold is tuned between those. These tests pin both directions.

Run:  python tests/test_grounding.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tutor.grounding import is_grounded, match_ratio, normalize  # noqa: E402

SOURCE = (
    "La produccion bovina puede generar impactos ambientales importantes, entre ellos "
    "emisiones de gases de efecto invernadero y deforestacion asociada a la expansion "
    "de pastizales. Sin embargo, un pastoreo bien gestionado contribuye al reciclaje "
    "de nutrientes en el suelo."
)


def test_exact_quote_is_grounded():
    quote = "un pastoreo bien gestionado contribuye al reciclaje de nutrientes"
    ok, score = is_grounded(quote, [SOURCE])
    assert ok and score == 1.0, score
    print("exact quote           OK")


def test_accents_and_whitespace_do_not_break_it():
    """pypdf mangles accents and line breaks; an honest quote must survive that."""
    quote = "La producción  bovina\npuede generar impactos ambientales importantes"
    ok, score = is_grounded(quote, [SOURCE])
    assert ok, f"a real quote was rejected over accents/whitespace (score {score:.2f})"
    print("accents tolerated     OK")


def test_invented_quote_is_rejected():
    quote = "El documento afirma que las vacas producen doce litros de leche diarios."
    ok, score = is_grounded(quote, [SOURCE])
    assert not ok, f"an invented quote passed with score {score:.2f}"
    print("invention rejected    OK")


def test_frankenstein_quote_is_rejected():
    """Real words, real document, sentence that was never written.

    This is the dangerous case: every term appears in the source, so any bag-of-words
    check would accept it. Requiring one contiguous run is what catches it.
    """
    quote = "las emisiones de gases contribuyen al reciclaje de nutrientes en el suelo"
    ok, score = is_grounded(quote, [SOURCE])
    assert not ok, f"a quote assembled from scattered words passed with score {score:.2f}"
    print("frankenstein rejected OK")


def test_best_source_wins():
    other = "Las vacas establecen jerarquias sociales dentro de sus grupos."
    quote = "Las vacas establecen jerarquias sociales"
    ok, _ = is_grounded(quote, [SOURCE, other])
    assert ok, "did not check every retrieved passage"
    print("multi-source          OK")


def test_normalize_is_idempotent():
    assert normalize(normalize("Ganadería   BOVINA!")) == normalize("Ganadería   BOVINA!")
    assert match_ratio("", SOURCE) == 0.0, "an empty quote must not count as grounded"
    print("normalize             OK")


if __name__ == "__main__":
    test_exact_quote_is_grounded()
    test_accents_and_whitespace_do_not_break_it()
    test_invented_quote_is_rejected()
    test_frankenstein_quote_is_rejected()
    test_best_source_wins()
    test_normalize_is_idempotent()
    print("\nall grounding tests passed")
