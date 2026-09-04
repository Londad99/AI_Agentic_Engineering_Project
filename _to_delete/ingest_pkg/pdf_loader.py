"""PDF -> raw text, one record per page.

Pages stay separate because the page number is the only cheap provenance a PDF gives
us, and "page 14" is far more useful to a student than "chunk 37".
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader


@dataclass
class Page:
    source: str  # file name, e.g. "algorithms-notes.pdf"
    page_number: int  # 1-indexed, as a human would say it
    text: str


def load_pdf(path: str | Path) -> list[Page]:
    path = Path(path)
    reader = PdfReader(str(path))
    pages: list[Page] = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(Page(source=path.name, page_number=i, text=text))
    return pages


def load_directory(directory: str | Path) -> list[Page]:
    directory = Path(directory)
    pages: list[Page] = []
    for pdf in sorted(directory.glob("*.pdf")):
        pages.extend(load_pdf(pdf))
    return pages
