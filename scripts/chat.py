"""Interactive study tutor. This is the conversational assistant, as a program.

Run:  python scripts/chat.py
      python scripts/chat.py --pdf notes.pdf     ingest that file first
      python scripts/chat.py --reset             rebuild the index from data/

Ask anything: what the document covers, a summary of a topic, a practice exam. When a
question is open, just type your answer and it gets graded.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tutor import config  # noqa: E402
from tutor.errors import TutorError  # noqa: E402
from tutor.ingest import ingest  # noqa: E402
from tutor.orchestrator import handle  # noqa: E402
from tutor.session import StudySession  # noqa: E402
from tutor.vectorstore import get_collection  # noqa: E402

parser = argparse.ArgumentParser(description="Chat with your study documents.")
parser.add_argument("--pdf", help="PDF to ingest before starting (default: everything in data/)")
parser.add_argument("--reset", action="store_true", help="rebuild the index from scratch")
parser.add_argument("--quiet", action="store_true", help="hide the [route: ...] tags")
args = parser.parse_args()

print(f"provider: {config.LLM_PROVIDER}  model: {config.active_chat_model()}")

try:
    collection = get_collection()
    if args.reset or args.pdf or collection.count() == 0:
        stats = ingest(args.pdf, reset=args.reset)
        print(f"ingested {stats['files']} file(s), {stats['chunks']} chunks")
    else:
        print(f"{collection.count()} chunks in the store (--reset to rebuild)")
except TutorError as error:
    raise SystemExit(str(error))

session = StudySession()
print("\nAsk about your document. Type 'salir' or Ctrl-D to quit.\n")

while True:
    try:
        message = input("> ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        break
    if not message:
        continue
    if message.lower() in {"salir", "exit", "quit"}:
        break
    try:
        print("\n" + handle(message, session, show_route=not args.quiet) + "\n")
    except TutorError as error:
        # A quota or config problem should not kill the session mid-conversation:
        # the student can fix .env in another window and keep going.
        print(f"\n{error}\n")

if session.scores:
    print("\nWeakest topics:")
    for name, average in session.weakest_topics():
        print(f"  {average:.0%}  {name}")
