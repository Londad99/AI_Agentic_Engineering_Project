"""Step 3 of ingestion: split cleaned pages into overlapping chunks.

Design decisions, and why:

1. Characters, not tokens. Token-accurate chunking needs a tokenizer we do not
   have for Gemini. ~4 chars per token in Spanish/English means CHUNK_SIZE=1200
   is roughly 300 tokens, comfortably inside the embedding model's limit.

2. Respect paragraph boundaries. Cutting mid-sentence produces a chunk whose
   embedding represents half an idea, and a retrieved fragment the LLM cannot
   use. We pack whole paragraphs until the budget is spent.

3. Overlap. A concept explained across a paragraph boundary would otherwise be
   invisible to both chunks. CHUNK_OVERLAP=200 chars carries the tail of the
   previous chunk into the next one. Cost: ~15% more vectors stored.

4. Never merge across pages, so page provenance stays exact.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from ..config import CHUNK_OVERLAP, CHUNK_SIZE
from .pdf_loader import Page


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
