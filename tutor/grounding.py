"""Verify that a quoted sentence really appears in the source text.

The exam generator is asked for a verbatim quote with every question. Asking is not
enough: a model under pressure to supply one produces something quote-shaped, and it
looks convincing. So we check.

The match is fuzzy on purpose - pypdf mangles accents and line breaks, and rejecting
honest quotes is as bad as accepting invented ones - but it requires one contiguous
run, which is what catches a sentence assembled from scattered real words.
"""

from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher

# A quote must overlap this much of itself with the source to count as grounded.
DEFAULT_THRESHOLD = 0.85


def normalize(text: str) -> str:
    """Lowercase, strip accents, collapse whitespace and punctuation spacing.

    Accent stripping is not cosmetic here: pypdf routinely turns "ganadería" into
    "ganader´ıa" or drops the accent entirely, so a perfectly honest quote can differ
    from the stored chunk by exactly the characters we care least about.
    """
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def match_ratio(quote: str, source: str) -> float:
    """Fraction of the quote found as one contiguous run inside the source.

    The longest *contiguous* block, not a bag-of-words overlap: a sentence assembled
    from words scattered across the document is precisely the kind of plausible
    fabrication this is meant to catch, and word overlap would wave it through.
    """
    quote_n, source_n = normalize(quote), normalize(source)
    if not quote_n:
        return 0.0
    if quote_n in source_n:
        return 1.0
    match = SequenceMatcher(None, quote_n, source_n, autojunk=False).find_longest_match(
        0, len(quote_n), 0, len(source_n)
    )
    return match.size / len(quote_n)


def is_grounded(quote: str, sources: list[str], threshold: float = DEFAULT_THRESHOLD) -> tuple[bool, float]:
    """Best score across the passages the question was written from."""
    if not quote or not sources:
        return False, 0.0
    best = max(match_ratio(quote, source) for source in sources)
    return best >= threshold, best
