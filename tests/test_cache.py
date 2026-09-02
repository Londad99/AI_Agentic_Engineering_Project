"""Tests for the embedding cache. No API key, no network: the API call is faked.

The interesting assertion is the last one. The cache key includes the model and
the task_type, so the same sentence embedded as a document and as a query cannot
collide. If it could, retrieval would silently degrade - the hardest kind of bug
to notice, because nothing crashes, the answers just get slightly worse.
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tutor import config, progress  # noqa: E402

progress.VERBOSE = False
config.STORAGE_DIR = Path(tempfile.mkdtemp())  # never touch the real store

from tutor import embedding_cache  # noqa: E402

embedding_cache.STORAGE_DIR = config.STORAGE_DIR
embedding_cache._CACHE_PATH = config.STORAGE_DIR / "embedding_cache.sqlite3"


def test_roundtrip():
    embedding_cache.put_many({"k1": [0.1, 0.2, 0.3], "k2": [0.4, 0.5, 0.6]})
    found = embedding_cache.get_many(["k1", "k2", "missing"])
    assert set(found) == {"k1", "k2"}, "missing key was invented"
    assert abs(found["k1"][0] - 0.1) < 1e-6, "vector did not survive the round trip"
    print("roundtrip             OK")


def test_keys_separate_task_types():
    document = embedding_cache.make_key("las vacas", "m", "RETRIEVAL_DOCUMENT", 768)
    query = embedding_cache.make_key("las vacas", "m", "RETRIEVAL_QUERY", 768)
    assert document != query, "document and query embeddings would share a cache entry"
    print("task_type separated   OK")


def test_keys_separate_models_and_dims():
    base = embedding_cache.make_key("texto", "gemini-embedding-001", "RETRIEVAL_QUERY", 768)
    other_model = embedding_cache.make_key("texto", "gemini-embedding-2", "RETRIEVAL_QUERY", 768)
    other_dim = embedding_cache.make_key("texto", "gemini-embedding-001", "RETRIEVAL_QUERY", 1536)
    assert len({base, other_model, other_dim}) == 3, "changing model or dims reuses stale vectors"
    print("model/dims separated  OK")


def test_second_call_makes_no_api_request():
    from tutor import embeddings

    calls = []

    def fake_batch(texts, task_type):
        calls.append(len(texts))
        return [[0.1] * config.EMBED_DIM for _ in texts]

    embeddings._embed_batch = fake_batch
    embedding_cache.clear()

    embeddings.embed_query("¿por qué las vacas son valiosas?")
    assert calls == [1], f"expected one API call, got {calls}"

    embeddings.embed_query("¿por qué las vacas son valiosas?")
    assert calls == [1], "the repeat call hit the API instead of the cache"
    print("repeat call cached    OK")


def test_partial_hit_only_embeds_the_new_ones():
    from tutor import embeddings

    calls = []

    def fake_batch(texts, task_type):
        calls.append(list(texts))
        return [[0.2] * config.EMBED_DIM for _ in texts]

    embeddings._embed_batch = fake_batch
    embedding_cache.clear()

    embeddings.embed_documents(["chunk A", "chunk B"])
    embeddings.embed_documents(["chunk A", "chunk B", "chunk C"])
    assert calls[-1] == ["chunk C"], f"re-embedded unchanged chunks: {calls[-1]}"
    print("partial hit           OK")


if __name__ == "__main__":
    test_roundtrip()
    test_keys_separate_task_types()
    test_keys_separate_models_and_dims()
    test_second_call_makes_no_api_request()
    test_partial_hit_only_embeds_the_new_ones()
    print("\nall embedding cache tests passed")
