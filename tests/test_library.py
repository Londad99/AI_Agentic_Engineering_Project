"""Tests for document removal on disk.

The rule being protected: archiving must never destroy a file, and must never let two
different documents collide under one name in the archive. The student's own study
material is the one thing in this project we cannot regenerate.

Run:  python tests/test_library.py
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tutor import config, library  # noqa: E402

config.DATA_DIR = Path(tempfile.mkdtemp())
library.DATA_DIR = config.DATA_DIR
library.ARCHIVE_DIR = config.DATA_DIR / "_removed"


def write(name: str, text: str) -> Path:
    path = library.DATA_DIR / name
    path.write_text(text, encoding="utf-8")
    return path


def test_archive_moves_the_file_out_of_the_ingestion_path():
    write("notes.pdf", "first")
    library.archive("notes.pdf")
    assert not (library.DATA_DIR / "notes.pdf").exists(), "the original is still there"
    assert (library.ARCHIVE_DIR / "notes.pdf").read_text() == "first", "content lost"
    assert [p.name for p in library.list_pdfs()] == [], library.list_pdfs()
    print("archive moves         OK")


def test_archived_files_are_not_re_ingested():
    """The archive is a subfolder, and ingestion globs only the top level."""
    write("kept.pdf", "kept")
    assert [p.name for p in library.list_pdfs()] == ["kept.pdf"]
    print("archive excluded      OK")


def test_a_second_file_with_the_same_name_does_not_overwrite_the_first():
    write("notes.pdf", "second version")
    library.archive("notes.pdf")
    archived = sorted(p.name for p in library.ARCHIVE_DIR.glob("*.pdf"))
    assert len(archived) == 2, archived
    assert (library.ARCHIVE_DIR / "notes.pdf").read_text() == "first", "the older file was destroyed"
    print("no overwrite          OK")


def test_restore_puts_it_back():
    library.restore("notes.pdf")
    assert (library.DATA_DIR / "notes.pdf").read_text() == "first"
    print("restore               OK")


def test_removing_something_absent_is_an_error_not_a_silent_success():
    try:
        library.archive("never_existed.pdf")
    except FileNotFoundError:
        print("missing file          OK")
    else:
        raise AssertionError("archiving a non-existent file appeared to succeed")


if __name__ == "__main__":
    test_archive_moves_the_file_out_of_the_ingestion_path()
    test_archive_excluded = test_archived_files_are_not_re_ingested()
    test_a_second_file_with_the_same_name_does_not_overwrite_the_first()
    test_restore_puts_it_back()
    test_removing_something_absent_is_an_error_not_a_silent_success()
    print("\nall library tests passed")
