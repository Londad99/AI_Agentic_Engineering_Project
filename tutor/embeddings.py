"""Gemini embeddings, wrapped so both ingestion and retrieval go through one place.

The single most important detail here is `task_type`. gemini-embedding-001 is
asymmetric: it produces a different vector for the same text depending on whether
you declare it a stored document or a search query. Embedding both sides with the
default task type measurably degrades retrieval. Documents go in as
RETRIEVAL_DOCUMENT, questions as RETRIEVAL_QUERY.
"""

from __future__ import annotations

import time

import numpy as np
from google import genai
from google.genai import types

from . import embedding_cache
from . import config
from .progress import status, step

# Read through config.* at call time, never copied into a module constant: a value
# captured at import is frozen for the life of the process, which is what made .env
# edits appear to do nothing inside a running Jupyter kernel.
_MAX_RETRIES = 5

_client: genai.Client | None = None


def get_client() -> genai.Client:
    global _client
    if _client is None:
        config.validate_models()
        _client = genai.Client(api_key=config.require_api_key(), http_options=build_http_options())
    return _client


def build_http_options() -> types.HttpOptions:
    """Timeout plus a lid on the SDK's own retrying.

    google-genai retries internally (it uses tenacity under the hood), so a single
    generate/embed call can spend minutes silently cycling through attempts while
    our timeout - which applies per HTTP attempt, not per call - never fires. We
    keep the SDK's retries to a minimum and let our own visible retry loop own the
    waiting, so the user can see it happening.

    HttpRetryOptions is not present in every SDK version, hence the guard.
    """
    options = types.HttpOptions(timeout=int(config.REQUEST_TIMEOUT_SECONDS * 1000))  # SDK takes ms
    retry_type = getattr(types, "HttpRetryOptions", None)
    if retry_type is not None:
        try:
            # attempts=1 means "do not retry inside the SDK". We want to own the
            # retrying: the SDK's attempts are invisible, they multiply our per-attempt
            # timeout (60s became an observed 122s), and they cannot print progress.
            options.retry_options = retry_type(attempts=1)
        except Exception:  # noqa: BLE001 - field names differ across versions
            pass
    return options


def _normalize(vector: list[float]) -> list[float]:
    """Required when output_dimensionality != 3072: Google only normalizes the full-size output."""
    array = np.array(vector, dtype=np.float32)
    norm = np.linalg.norm(array)
    return (array / norm).tolist() if norm else array.tolist()


def _embed_batch(texts: list[str], task_type: str) -> list[list[float]]:
    config.validate_models()  # cheap, and the cached client would otherwise skip the check
    request_config = types.EmbedContentConfig(
        task_type=task_type, output_dimensionality=config.EMBED_DIM
    )
    for attempt in range(_MAX_RETRIES):
        try:
            response = get_client().models.embed_content(
                model=config.GEMINI_EMBED_MODEL, contents=texts, config=request_config
            )
            return [_normalize(e.values) for e in response.embeddings]
        except Exception as exc:  # noqa: BLE001 - free tier throws on rate limits
            if attempt == _MAX_RETRIES - 1:
                raise
            wait = 2**attempt  # exponential backoff: 1s, 2s, 4s, 8s
            status(f"     embedding retry {attempt + 1}/{_MAX_RETRIES} in {wait}s ({type(exc).__name__})")
            time.sleep(wait)
    raise RuntimeError("unreachable")


def _embed_cached(texts: list[str], task_type: str) -> list[list[float]]:
    """Cache lookup first, one API call for whatever is left, then store the new ones."""
    keys = [embedding_cache.make_key(t, config.GEMINI_EMBED_MODEL, task_type, config.EMBED_DIM) for t in texts]
    cached = embedding_cache.get_many(keys)

    missing = [(i, t) for i, (k, t) in enumerate(zip(keys, texts)) if k not in cached]
    if cached and missing:
        status(f"  cache hit on {len(cached)}/{len(texts)}; embedding the remaining {len(missing)}")
    elif cached:
        status(f"  cache hit on all {len(cached)} - no API call needed")

    fresh: dict[str, list[float]] = {}
    for start in range(0, len(missing), config.EMBED_BATCH_SIZE):
        block = missing[start : start + config.EMBED_BATCH_SIZE]
        with step(f"embedding {len(block)} text(s) with {config.GEMINI_EMBED_MODEL}"):
            vectors = _embed_batch([t for _, t in block], task_type)
        for (index, _), vector in zip(block, vectors):
            fresh[keys[index]] = vector
        if len(missing) > config.EMBED_BATCH_SIZE:
            status(f"  embedded {min(start + config.EMBED_BATCH_SIZE, len(missing))}/{len(missing)}")

    embedding_cache.put_many(fresh)
    merged = {**cached, **fresh}
    return [merged[key] for key in keys]


def embed_documents(texts: list[str]) -> list[list[float]]:
    return _embed_cached(texts, "RETRIEVAL_DOCUMENT")


def embed_query(text: str) -> list[float]:
    return _embed_cached([text], "RETRIEVAL_QUERY")[0]
