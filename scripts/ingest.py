"""CLI: python scripts/ingest.py [path/to.pdf|data/] [--reset]"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tutor.errors import TutorError  # noqa: E402
from tutor.ingest.pipeline import ingest  # noqa: E402

parser = argparse.ArgumentParser(description="Ingest PDFs into ChromaDB.")
parser.add_argument("path", nargs="?", default=None, help="PDF file or folder (default: data/)")
parser.add_argument("--reset", action="store_true", help="empty the collection before ingesting")
args = parser.parse_args()

try:
    stats = ingest(args.path, reset=args.reset)
except TutorError as error:  # translate to a process exit only here, at the CLI boundary
    raise SystemExit(str(error))

print("\n--- ingestion complete ---")
for key, value in stats.items():
    print(f"{key:20} {value}")
