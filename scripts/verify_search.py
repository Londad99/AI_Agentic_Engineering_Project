"""Check ChromaDB's ranking against our own cosine implementation.

Why this exists: the collection is configured with hnsw:space=cosine and we normalize
the vectors ourselves. If either of those were wrong - the wrong metric, vectors that
were never scaled to unit length - nothing would raise. Retrieval would just get
quietly worse. Ranking the same query twice, with two independent implementations, is
how you find that.

Run:  python scripts/verify_search.py
      python scripts/verify_search.py "una pregunta concreta"
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tutor import progress  # noqa: E402
from tutor.vectorstore import compare_backends, get_collection  # noqa: E402

progress.VERBOSE = False

QUERIES = sys.argv[1:] or [
    "resumen de los puntos principales",
    "definiciones y conceptos clave",
    "ejemplos y aplicaciones",
]

collection = get_collection()
if collection.count() == 0:
    raise SystemExit("The store is empty. Run: python scripts/ingest.py")

print(f"{collection.count()} chunks indexed\n")
print(f"{'query':44} {'same order':>11} {'max score diff':>15}")
print("-" * 72)

all_agree = True
for query in QUERIES:
    result = compare_backends(query, top_k=5)
    all_agree &= result["same_order"]
    print(f"{query[:44]:44} {str(result['same_order']):>11} "
          f"{result['max_score_difference']:>15.6f}")

print("-" * 72)
if all_agree:
    print("Chroma's HNSW index and our brute-force cosine agree on every query.")
    print("The store is using the metric we think it is, on normalized vectors.")
else:
    print("DISAGREEMENT. Check that the collection was created with hnsw:space=cosine")
    print("and that the vectors were normalized before being stored.")
    raise SystemExit(1)
