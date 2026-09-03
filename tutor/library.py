"""The set of documents on disk, and how a document leaves it.

Separate from the vector store on purpose: removing a document is two operations that
must both happen. Deleting its chunks alone is not enough - the PDF stays in data/, and
the next ingest() silently puts it back, which looks like the removal never worked.

Archiving instead of deleting: the student's own material is not ours to destroy, and
"undo" is moving one file back.
"""

from __future__ import annotations

from pathlib import Path

from .config import DATA_DIR

ARCHIVE_DIR = DATA_DIR / "_removed"


def list_pdfs() -> list[Path]:
    """PDFs that ingestion will pick up. The archive is a subfolder, so it is excluded."""
    return sorted(DATA_DIR.glob("*.pdf"))


def archive(name: str) -> Path:
    """Move one PDF out of the ingestion path. Returns where it went."""
    source = DATA_DIR / name
    if not source.exists():
        raise FileNotFoundError(f"{name} is not in {DATA_DIR}")

    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    target = ARCHIVE_DIR / name
    # Never overwrite a previously archived file with the same name: the two may be
    # different documents, and the older one is the one we cannot get back.
    counter = 1
    while target.exists():
        target = ARCHIVE_DIR / f"{source.stem} ({counter}){source.suffix}"
        counter += 1
    source.rename(target)
    return target


def restore(name: str) -> Path:
    source = ARCHIVE_DIR / name
    if not source.exists():
        raise FileNotFoundError(f"{name} is not in {ARCHIVE_DIR}")
    target = DATA_DIR / name
    source.rename(target)
    return target
