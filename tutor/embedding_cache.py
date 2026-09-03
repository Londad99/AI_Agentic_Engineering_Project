"""On-disk cache of embedding vectors.

An embedding is a pure function of (model, task_type, dimensions, text), so paying for
the same one twice is waste - and a notebook re-runs the same query on every iteration.

SQLite with raw float32 blobs: 768 floats is 3 KB binary against ~15 KB of JSON, and
lookups do not load the whole file.
"""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

import numpy as np

from .config import STORAGE_DIR

_CACHE_PATH = STORAGE_DIR / "embedding_cache.sqlite3"
_connection: sqlite3.Connection | None = None


def _connect() -> sqlite3.Connection:
    global _connection
    if _connection is None:
        STORAGE_DIR.mkdir(parents=True, exist_ok=True)
        _connection = sqlite3.connect(_CACHE_PATH, check_same_thread=False)
        _connection.execute(
            "CREATE TABLE IF NOT EXISTS embeddings ("
            "  key TEXT PRIMARY KEY,"
            "  vector BLOB NOT NULL"
            ")"
        )
        _connection.commit()
    return _connection


def make_key(text: str, model: str, task_type: str, dimensions: int) -> str:
    """Every input that changes the vector goes into the key.

    Leaving out model or task_type would be a correctness bug, not an optimisation
    detail: the same sentence embedded as RETRIEVAL_QUERY is a different vector
    from the same sentence embedded as RETRIEVAL_DOCUMENT, and silently serving one
    for the other would quietly degrade every search.
    """
    payload = f"{model}|{task_type}|{dimensions}|{text}".encode()
    return hashlib.sha256(payload).hexdigest()


def get_many(keys: list[str]) -> dict[str, list[float]]:
    if not keys:
        return {}
    connection = _connect()
    found: dict[str, list[float]] = {}
    # SQLite caps the number of bound parameters, so read in blocks.
    for start in range(0, len(keys), 500):
        block = keys[start : start + 500]
        placeholders = ",".join("?" * len(block))
        rows = connection.execute(
            f"SELECT key, vector FROM embeddings WHERE key IN ({placeholders})", block
        ).fetchall()
        for key, blob in rows:
            found[key] = np.frombuffer(blob, dtype=np.float32).tolist()
    return found


def put_many(items: dict[str, list[float]]) -> None:
    if not items:
        return
    connection = _connect()
    connection.executemany(
        "INSERT OR REPLACE INTO embeddings (key, vector) VALUES (?, ?)",
        [(key, np.asarray(vector, dtype=np.float32).tobytes()) for key, vector in items.items()],
    )
    connection.commit()


def stats() -> dict:
    connection = _connect()
    count = connection.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
    size = _CACHE_PATH.stat().st_size if _CACHE_PATH.exists() else 0
    return {"cached_vectors": count, "cache_size_kb": round(size / 1024)}


def clear() -> None:
    """Call this if you change the embedding model - old vectors are not comparable."""
    connection = _connect()
    connection.execute("DELETE FROM embeddings")
    connection.commit()
