"""CLI smoke test: python scripts/search.py "your question here" """

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tutor.vectorstore import search  # noqa: E402

query = " ".join(sys.argv[1:]) or "resumen del documento"
hits = search(query, top_k=5)
if not hits:
    print("No results. Did you run scripts/ingest.py first?")
for hit in hits:
    meta = hit["metadata"]
    print(f"\n[{hit['score']:.3f}] {meta['source']} p.{meta['page']}")
    print(hit["text"][:300].replace("\n", " "), "...")
