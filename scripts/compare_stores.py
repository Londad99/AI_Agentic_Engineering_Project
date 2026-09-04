"""ChromaDB vs FAISS on the same vectors, plus our own cosine as referee.

The Week 4 lab indexes the same chunks in both, and the obvious question afterwards is
"so which one?". This answers it with numbers from YOUR index instead of an opinion.

FAISS is optional and NOT in requirements.txt on purpose: it is a large wheel that does
not build on every Python version, and the project does not need it to run. Install it
only to run this comparison:

    python -m pip install faiss-cpu

Run:  python scripts/compare_stores.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from tutor import progress, vectormath  # noqa: E402
from tutor.embeddings import embed_query  # noqa: E402
from tutor.vectorstore import get_collection, search  # noqa: E402

progress.VERBOSE = False

try:
    import faiss
except ImportError:
    raise SystemExit(
        "faiss is not installed. It is optional:\n    python -m pip install faiss-cpu"
    )

QUERIES = sys.argv[1:] or ["resumen de los puntos principales", "definiciones y conceptos clave"]

collection = get_collection()
if collection.count() == 0:
    raise SystemExit("The store is empty. Run: python scripts/ingest.py")

stored = collection.get(include=["documents", "metadatas", "embeddings"])
vectors = np.asarray(stored["embeddings"], dtype="float32")
print(f"{len(vectors)} chunks, {vectors.shape[1]} dimensions\n")

# Inner product, not L2. Our vectors are unit length, so the inner product IS the
# cosine - the same metric Chroma is configured with. IndexFlatL2 would rank by
# Euclidean distance instead, and the comparison would be measuring two different
# questions rather than two implementations of one.
index = faiss.IndexFlatIP(vectors.shape[1])
index.add(vectors)

TOP_K = 5
print(f"{'query':38} {'chroma':>9} {'faiss':>9} {'ours':>9}  {'same top-5':>11}")
print("-" * 82)

for query in QUERIES:
    vector = np.asarray([embed_query(query)], dtype="float32")

    started = time.perf_counter()
    chroma_hits = search(query, top_k=TOP_K)
    chroma_ms = (time.perf_counter() - started) * 1000

    started = time.perf_counter()
    _, faiss_ids = index.search(vector, TOP_K)
    faiss_ms = (time.perf_counter() - started) * 1000

    started = time.perf_counter()
    ours = vectormath.rank(vector[0], vectors, top_k=TOP_K)
    ours_ms = (time.perf_counter() - started) * 1000

    def identity(metadata):
        return (metadata.get("source"), metadata.get("page"), metadata.get("chunk_index"))

    chroma_ids = [identity(h["metadata"]) for h in chroma_hits]
    faiss_set = [identity(stored["metadatas"][i]) for i in faiss_ids[0]]
    ours_set = [identity(stored["metadatas"][i]) for i, _ in ours]
    agree = chroma_ids == faiss_set == ours_set

    print(f"{query[:38]:38} {chroma_ms:8.1f}ms {faiss_ms:8.1f}ms {ours_ms:8.1f}ms  {str(agree):>11}")

print("-" * 82)
print("""
Reading this:

  Same results  All three rank the same chunks in the same order, which is the point:
                they implement one metric three ways. A disagreement would mean the
                Chroma collection is not configured with the metric we assume.

  Timings       FAISS wins on raw search and the gap grows with corpus size. Chroma's
                number includes work FAISS does not do at all - it reads documents and
                metadata from disk in the same call.

  Why Chroma    It persists to disk in one line and stores metadata beside each vector,
                which is what lets every answer cite a file and a page. FAISS is only
                the index: persistence, the documents and the metadata would all have to
                be built and kept in sync by hand. At a few hundred chunks the speed
                difference is microseconds and irrelevant; the metadata is not.
""")
