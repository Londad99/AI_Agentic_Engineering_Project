"""ChromaDB wrapper.

Note we pass embeddings in explicitly instead of registering an embedding
function on the collection. Chroma's default embedder is a small English
sentence-transformer; if we let it run, our Gemini vectors would never be used.
Passing vectors by hand also keeps the asymmetric query/document distinction
from embeddings.py, which a collection-level embedding function cannot express.
"""

from __future__ import annotations

import chromadb

from . import config
from .config import CHROMA_COLLECTION, STORAGE_DIR
from .embeddings import embed_documents, embed_query
from .ingest.chunker import Chunk
from .progress import status, step

_client: chromadb.ClientAPI | None = None


_MODEL_MARKER = "embedding_model.txt"


def collection_marker():
    """Path of the file recording which embedding model built this store."""
    return STORAGE_DIR / _MODEL_MARKER


def _check_embedding_model() -> None:
    """Refuse to mix vector spaces.

    Vectors from two different embedding models are not comparable, so searching a
    store built with one model using another returns confident nonsense - the worst
    failure mode, because nothing errors. The model that built the store is recorded
    on first ingest and checked from then on.
    """
    from .errors import ConfigError

    marker = STORAGE_DIR / _MODEL_MARKER
    current = config.active_embed_model()
    if not marker.exists():
        STORAGE_DIR.mkdir(parents=True, exist_ok=True)
        marker.write_text(current, encoding="utf-8")
        return
    stored = marker.read_text(encoding="utf-8").strip()
    if stored != current:
        raise ConfigError(
            f"This vector store was built with '{stored}' but EMBED_PROVIDER/model now "
            f"resolves to '{current}'.\nVectors from different models are not comparable. "
            f"Rebuild it:\n    python scripts/ingest.py --reset"
        )


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
    _check_embedding_model()
    vectors = embed_documents([c.text for c in chunks])
    collection.upsert(  # upsert, not add: deterministic ids make re-ingestion idempotent
        ids=[c.id for c in chunks],
        documents=[c.text for c in chunks],
        embeddings=vectors,
        metadatas=[c.metadata for c in chunks],
    )
    return len(chunks)


def list_all_chunks(collection=None) -> list[dict]:
    """Every chunk, in document order.

    Deliberately not a search. "What topics does this document cover?" is not a
    similarity question: embedding that sentence and taking the top 5 returns the
    passages that most resemble the phrase "what topics does this cover", which is
    close to meaningless. Topic extraction needs coverage of the whole document, so
    it reads everything.

    Document order (page, then position on the page) matters too: a topic explained
    across two consecutive chunks is obvious in sequence and invisible in the
    arbitrary order the store happens to return.
    """
    collection = collection or get_collection()
    if collection.count() == 0:
        return []
    result = collection.get(include=["documents", "metadatas"])
    chunks = [
        {"text": text, "metadata": metadata}
        for text, metadata in zip(result["documents"], result["metadatas"])
    ]
    chunks.sort(key=lambda c: (c["metadata"].get("source", ""), c["metadata"].get("page", 0),
                               c["metadata"].get("chunk_index", 0)))
    return chunks


def search(query: str, top_k: int = 5, threshold: float | None = None, collection=None) -> list[dict]:
    """Semantic search. Same idea as the week-3 notebook, backed by a real vector DB.

    Chroma returns cosine *distance* (0 = identical). We convert to the similarity
    score the notebook used, so `threshold` means the same thing it did there.
    """
    collection = collection or get_collection()
    _check_embedding_model()
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
