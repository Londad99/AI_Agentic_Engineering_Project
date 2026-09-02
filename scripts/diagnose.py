"""Timing breakdown of one retrieval, stage by stage.

Written because a search took 2m38s with no output and there was no way to tell
which of four very different operations was responsible. Guessing at a performance
problem is how you end up optimising the fast part.

Run:  python scripts/diagnose.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tutor import progress  # noqa: E402

progress.VERBOSE = False  # this script does its own timing


def timed(label, function):
    started = time.monotonic()
    try:
        result = function()
        print(f"{label:32} {time.monotonic() - started:7.2f}s")
        return result
    except Exception as error:
        print(f"{label:32} {time.monotonic() - started:7.2f}s  FAILED: {type(error).__name__}: {error}")
        raise


# What the SDK is actually configured with. A timeout that never fires because the
# SDK retries underneath it looks exactly like a hang.
from google.genai import types as genai_types  # noqa: E402

from tutor.embeddings import build_http_options  # noqa: E402

options = build_http_options()
print(f"SDK timeout setting : {getattr(options, 'timeout', None)} ms")
print(f"HttpRetryOptions    : {'available' if hasattr(genai_types, 'HttpRetryOptions') else 'NOT in this SDK version'}")
print(f"SDK retry_options   : {getattr(options, 'retry_options', None)}")
print()

print("stage                             elapsed")
print("-" * 46)

timed("import chromadb", lambda: __import__("chromadb"))
timed("import google.genai", lambda: __import__("google.genai"))

from tutor.config import GEMINI_EMBED_MODEL, REQUEST_TIMEOUT_SECONDS  # noqa: E402
from tutor.embeddings import embed_query, get_client  # noqa: E402
from tutor.vectorstore import get_collection  # noqa: E402

collection = timed("open ChromaDB collection", get_collection)
count = timed("count chunks", collection.count)
timed("build Gemini client", get_client)

# Two calls: the first pays for the TLS handshake and connection setup, the second
# is the steady-state cost. If only the first is slow, it is a cold start, not a
# slow model.
timed("embed query (first, cold)", lambda: embed_query("prueba de conexion"))
vector = timed("embed query (second, warm)", lambda: embed_query("segunda prueba"))

timed(
    "chroma vector search",
    lambda: collection.query(query_embeddings=[vector], n_results=min(3, count or 1)),
)

# A third call, identical to the second: it must be served from the cache and be
# effectively free. If it is not, caching is broken.
timed("embed query (third, cached)", lambda: embed_query("segunda prueba"))

from tutor import embedding_cache  # noqa: E402

print("-" * 46)
print(f"cache          : {embedding_cache.stats()}")
print(f"chunks in store: {count}")
print(f"embed model    : {GEMINI_EMBED_MODEL}")
print(f"request timeout: {REQUEST_TIMEOUT_SECONDS}s")
print("\nIf 'first, cold' is slow but 'warm' is fast   -> cold start, harmless.")
print("If BOTH cold and warm are slow               -> the embedding API is the bottleneck;")
print("                                                try GEMINI_EMBED_MODEL=gemini-embedding-2.")
print("If 'third, cached' is not ~0.00s             -> the cache is not working.")
print("If 'open ChromaDB collection' is slow        -> local, first-run onnxruntime load.")
