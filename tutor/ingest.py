"""Ingestion: PDF -> text -> clean -> chunk -> embeddings -> ChromaDB.

Plain code, no LLM. The whole stage is a deterministic transformation with no judgement
call to make, so a model here would add cost, latency and a failure mode and buy
nothing. The agents start where judgement starts.

Four steps, in order, each in its own section below.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from pypdf import PdfReader

from .config import CHUNK_OVERLAP, CHUNK_SIZE, DATA_DIR
from .errors import IngestionError

# tutor.vectorstore is imported inside ingest() on purpose. It needs Chunk from this
# module, so importing it at the top would be a cycle - one that only appeared when the
# four ingest files became one.


# =========================================================================== #
# 1. PDF -> text, one record per page
# =========================================================================== #

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

# =========================================================================== #
# 2. Cleaning: drop running headers, footers and page numbers
# =========================================================================== #

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

# =========================================================================== #
# 3. Chunking: overlapping pieces that respect paragraph boundaries
# =========================================================================== #

@dataclass
class Chunk:
    id: str
    text: str
    metadata: dict = field(default_factory=dict)


def _split_long_paragraph(paragraph: str, size: int) -> list[str]:
    """Fallback for a single paragraph bigger than one chunk (tables, dense slides)."""
    return [paragraph[i : i + size] for i in range(0, len(paragraph), size)]


def _chunk_text(text: str, size: int, overlap: int) -> list[str]:
    paragraphs: list[str] = []
    for block in text.split("\n"):
        block = block.strip()
        if not block:
            continue
        if len(block) > size:
            paragraphs.extend(_split_long_paragraph(block, size))
        else:
            paragraphs.append(block)

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for paragraph in paragraphs:
        if current and current_len + len(paragraph) + 1 > size:
            chunk = "\n".join(current)
            chunks.append(chunk)
            # start the next chunk with the tail of this one
            tail = chunk[-overlap:] if overlap else ""
            current = [tail, paragraph] if tail else [paragraph]
            current_len = len(tail) + len(paragraph)
        else:
            current.append(paragraph)
            current_len += len(paragraph) + 1
    if current:
        chunks.append("\n".join(current))
    return [c.strip() for c in chunks if c.strip()]


def chunk_pages(pages: list[Page], size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[Chunk]:
    chunks: list[Chunk] = []
    for page in pages:
        for i, text in enumerate(_chunk_text(page.text, size, overlap)):
            # Deterministic id from the content itself: re-ingesting the same PDF
            # overwrites the same rows instead of duplicating them in ChromaDB.
            digest = hashlib.sha256(f"{page.source}|{page.page_number}|{i}|{text}".encode()).hexdigest()[:16]
            chunks.append(
                Chunk(
                    id=digest,
                    text=text,
                    metadata={
                        "source": page.source,
                        "page": page.page_number,
                        "chunk_index": i,
                    },
                )
            )
    return chunks

# =========================================================================== #
# 4. The pipeline: wire the three together and store the result
# =========================================================================== #

def ingest(path: str | Path | None = None, reset: bool = False) -> dict:
    from .vectorstore import add_chunks, collection_marker, get_collection

    path = Path(path) if path else DATA_DIR

    if not path.exists():
        raise IngestionError(f"{path} does not exist.")

    if path.is_file():
        pdfs = [path]
    else:
        pdfs = sorted(path.glob("*.pdf"))
        if not pdfs:
            others = [f.name for f in path.iterdir() if f.is_file() and f.name != ".gitkeep"]
            raise IngestionError(
                f"No PDF files in {path}.\n"
                f"Put your class notes or slides there as .pdf and run this again."
                + (f"\nFound instead: {', '.join(others)}" if others else "")
            )

    pages = load_pdf(path) if path.is_file() else load_directory(path)
    if not pages:
        # The files exist but pypdf got no text out of them.
        raise IngestionError(
            f"Found {len(pdfs)} PDF(s) in {path} but extracted no text: "
            f"{', '.join(f.name for f in pdfs)}.\n"
            f"They are most likely scanned images rather than digital text. Check by opening "
            f"one and trying to select a word with the cursor - if you cannot, it is an image. "
            f"Such a file needs OCR before it can be ingested."
        )

    cleaned = clean_pages(pages)
    chunks = chunk_pages(cleaned)

    collection = get_collection()
    if reset:
        if collection.count():
            collection.delete(where={"source": {"$ne": ""}})
        # Forget which embedding model built the store, so --reset is also how you
        # switch embedding providers.
        marker = collection_marker()
        if marker.exists():
            marker.unlink()

    stored = add_chunks(chunks, collection=collection)
    return {
        "files": len({p.source for p in pages}),
        "pages": len(pages),
        "chunks": stored,
        "collection_size": collection.count(),
        "avg_chunk_chars": round(sum(len(c.text) for c in chunks) / len(chunks)) if chunks else 0,
    }
