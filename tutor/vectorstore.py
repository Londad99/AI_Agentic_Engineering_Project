"""ChromaDB wrapper.

Note we pass embeddings in explicitly instead of registering an embedding
function on the collection. Chroma's default embedder is a small English
sentence-transformer; if we let it run, our Gemini vectors would never be used.
Passing vectors by hand also keeps the asymmetric query/document distinction
from embeddings.py, which a collection-level embedding function cannot express.
"""

from __future__ import annotations

import chromadb

from .config import CHROMA_COLLECTION, STORAGE_DIR
from .embeddings import embed_documents, embed_query
from .ingest.chunker import Chunk
from .progress import status, step

_client: chromadb.ClientAPI | None = None


def get_collection(name: str = CHROMA_COLLECTION):
    global _client
    if _client is None:
        STORAGE_DIR.mkdir(parents=True, exist_ok=True)
        # First construction is the slow one: chromadb pulls in onnxruntime and
        # opens the sqlite store. Announce it so a cold start is not mistaken for a hang.
        with step("opening ChromaDB"):
            _client = chromadb.PersistentClient(path=str(STORAGE_DIR))
    return _client.get_or_create_collection(
        name=name,
        # Our vectors are L2-normalized, so cosine is the right metric
        # (and on normalized vectors it is equivalent to the dot product).
        metadata={"hnsw:space": "cosine"},
    )


def add_chunks(chunks: list[Chunk], collection=None) -> int:
    if not chunks:
        return 0
    collection = collection or get_collection()
    vectors = embed_documents([c.text for c in chunks])
    collection.upsert(  # upsert, not add: deterministic ids make re-ingestion idempotent
        ids=[c.id for c in chunks],
        documents=[c.text for c in chunks],
        embeddings=vectors,
        metadatas=[c.metadata for c in chunks],
    )
    return len(chunks)


def search(query: str, top_k: int = 5, threshold: float | None = None, collection=None) -> list[dict]:
    """Semantic search. Same idea as the week-3 notebook, backed by a real vector DB.

    Chroma returns cosine *distance* (0 = identical). We convert to the similarity
    score the notebook used, so `threshold` means the same thing it did there.
    """
    collection = collection or get_collection()
    total = collection.count()
    if total == 0:
        return []
    vector = embed_query(query)
    with step(f"searching {total} chunks"):
        result = collection.query(
            query_embeddings=[vector],
            n_results=min(top_k, total),
            include=["documents", "metadatas", "distances"],
        )
    hits = []
    for text, metadata, distance in zip(
        result["documents"][0], result["metadatas"][0], result["distances"][0]
    ):
        score = 1.0 - distance
        if threshold is not None and score < threshold:
            continue
        hits.append({"text": text, "metadata": metadata, "score": score})
    return hits
