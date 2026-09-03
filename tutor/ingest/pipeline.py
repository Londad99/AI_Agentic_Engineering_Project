"""Wires the four ingestion steps together. This layer is plain code, not an agent:
it is fully deterministic, so putting an LLM in charge of it would only add cost,
latency and failure modes.
"""

from __future__ import annotations

from pathlib import Path

from ..config import DATA_DIR
from ..errors import IngestionError
from ..vectorstore import add_chunks, collection_marker, get_collection
from .chunker import chunk_pages
from .cleaner import clean_pages
from .pdf_loader import load_directory, load_pdf


def ingest(path: str | Path | None = None, reset: bool = False) -> dict:
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
