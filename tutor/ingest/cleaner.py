"""Remove the noise a PDF extractor leaves behind.

A footer repeated on every page gets embedded dozens of times and floods every result
list with near-identical chunks, so cleaning protects retrieval quality directly.

Boilerplate is detected by repetition AND position (page margins) AND not looking like
prose - "Universidad de Santander" is a footer, "Objetivo de la clase: comparar
algoritmos." is content that happens to repeat.

Guiding principle: a false negative leaves noise in the index; a false positive
silently deletes material the student will be examined on. Err toward keeping.
"""

from __future__ import annotations

import re
from collections import Counter

from .pdf_loader import Page

# A line made only of digits, or "Page 3 of 20", or "- 12 -".
PAGE_NUMBER_RE = re.compile(r"^\s*[-–—]?\s*(page\s+)?\d+\s*(of|de|/)?\s*\d*\s*[-–—]?\s*$", re.I)


# How many lines from the top and the bottom of a page can be a header/footer.
_MARGIN_LINES = 3


# A line ending like a sentence is prose, not a running head.
PROSE_END_RE = re.compile(r"[.!?;]$")


def _is_boilerplate_candidate(line: str) -> bool:
    return len(line) < 120 and not PROSE_END_RE.search(line)


def _margin_lines(page: Page) -> set[str]:
    lines = [line.strip() for line in page.text.splitlines() if line.strip()]
    return set(lines[:_MARGIN_LINES] + lines[-_MARGIN_LINES:])


def _repeated_lines(pages: list[Page], min_ratio: float = 0.5) -> set[str]:
    """Short lines that sit in the page margins on at least `min_ratio` of pages."""
    if len(pages) < 4:  # too few pages for the statistic to mean anything
        return set()
    counter: Counter[str] = Counter()
    for page in pages:
        # a set per page: a line repeated twice on the same page still counts once
        counter.update(line for line in _margin_lines(page) if _is_boilerplate_candidate(line))
    threshold = max(2, int(len(pages) * min_ratio))
    return {line for line, count in counter.items() if count >= threshold}


def clean_pages(pages: list[Page]) -> list[Page]:
    boilerplate = _repeated_lines(pages)
    cleaned: list[Page] = []
    for page in pages:
        margins = _margin_lines(page)
        lines = []
        for line in page.text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            # only strip boilerplate where boilerplate can live: the page margins
            if stripped in boilerplate and stripped in margins:
                continue
            if PAGE_NUMBER_RE.match(stripped):
                continue
            lines.append(stripped)
        text = "\n".join(lines)
        # de-hyphenate words split across line breaks: "algo-\nritmo" -> "algoritmo"
        text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
        # collapse runs of spaces, but keep newlines: they are our paragraph signal
        text = re.sub(r"[ \t]{2,}", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        if text.strip():
            cleaned.append(Page(source=page.source, page_number=page.page_number, text=text.strip()))
    return cleaned
