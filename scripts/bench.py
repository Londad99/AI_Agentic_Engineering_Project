"""Measure latency per model - chat and embeddings - so the choice is made with data.

Context: gemini-embedding-001 took 169 seconds to embed one short sentence and
still succeeded. A success that slow is not a rate limit and not a bug in our
code - it is the SDK retrying 503s internally until one gets through. The fix is
to use a model that is not saturated, and the only way to know which one that is
on YOUR key, from YOUR network, is to measure.

Run:  python scripts/bench.py
      python scripts/bench.py --http-debug     (show every HTTP attempt)
"""

import argparse
import logging
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

parser = argparse.ArgumentParser()
parser.add_argument("--http-debug", action="store_true", help="log every HTTP request the SDK makes")
parser.add_argument("--runs", type=int, default=3, help="timed calls per model")
args = parser.parse_args()

if args.http_debug:
    # This is what exposes the SDK's internal retries: each 503 and each new
    # attempt shows up as its own request line.
    logging.basicConfig(level=logging.DEBUG, format="    %(name)s %(message)s")
    for noisy in ("httpx", "httpcore", "google_genai"):
        logging.getLogger(noisy).setLevel(logging.DEBUG)

from google.genai import types  # noqa: E402

from tutor import progress  # noqa: E402
from tutor.config import EMBED_DIM, GEMINI_EMBED_MODEL, GEMINI_MODEL  # noqa: E402
from tutor.embeddings import get_client  # noqa: E402

progress.VERBOSE = False

CANDIDATES = [
    "gemini-embedding-2",    # current standard, released April 2026
    "gemini-embedding-001",  # legacy, supported until 2028 - what we use today
]

client = get_client()

print("Models your key can actually reach:")
available = set()
try:
    for model in client.models.list():
        actions = getattr(model, "supported_actions", None) or []
        if "embedContent" in actions or "embed" in str(actions).lower():
            name = model.name.replace("models/", "")
            available.add(name)
            print(f"  {name}")
except Exception as error:  # noqa: BLE001
    print(f"  (could not list models: {error})")

print(f"\nTiming {args.runs} single-text calls per model, {EMBED_DIM} dims")
print(f"{'model':28} {'median':>9} {'min':>9} {'max':>9}")
print("-" * 60)

for model in CANDIDATES:
    if available and model not in available:
        print(f"{model:28} {'not available to this key':>29}")
        continue
    timings = []
    failure = None
    for run in range(args.runs):
        # A different sentence each run, otherwise the API may serve a cached result
        # and we would be measuring Google's cache instead of the model.
        text = f"prueba de latencia numero {run} para medir el modelo de embeddings"
        started = time.monotonic()
        try:
            client.models.embed_content(
                model=model,
                contents=[text],
                config=types.EmbedContentConfig(
                    task_type="RETRIEVAL_QUERY", output_dimensionality=EMBED_DIM
                ),
            )
            timings.append(time.monotonic() - started)
        except Exception as error:  # noqa: BLE001
            failure = f"{type(error).__name__}: {str(error)[:60]}"
            break
    if failure:
        print(f"{model:28} {failure:>29}")
    else:
        print(
            f"{model:28} {statistics.median(timings):8.2f}s "
            f"{min(timings):8.2f}s {max(timings):8.2f}s"
        )

# --- chat models -----------------------------------------------------------
# The '-latest' aliases hot-swap to the newest release, which is also the one
# under the most load. A pinned stable build is usually far more available.
CHAT_CANDIDATES = [
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.5-flash",
    "gemini-3.6-flash",
    "gemini-3.7-flash",
    "gemini-flash-latest",
]

print(f"\nTiming {args.runs} short chat calls per model")
print(f"{'model':28} {'median':>9} {'min':>9} {'max':>9}")
print("-" * 60)

for model in CHAT_CANDIDATES:
    timings = []
    failure = None
    for run in range(args.runs):
        started = time.monotonic()
        try:
            client.models.generate_content(
                model=model,
                contents=f"Responde solo con el numero {run}.",
                config=types.GenerateContentConfig(temperature=0.0),
            )
            timings.append(time.monotonic() - started)
        except Exception as error:  # noqa: BLE001
            failure = f"{type(error).__name__}: {str(error)[:52]}"
            break
    if failure:
        print(f"{model:28} {failure:>29}")
    else:
        print(
            f"{model:28} {statistics.median(timings):8.2f}s "
            f"{min(timings):8.2f}s {max(timings):8.2f}s"
        )

print("-" * 60)
print(f"currently configured: GEMINI_MODEL={GEMINI_MODEL}")
print(f"                      GEMINI_EMBED_MODEL={GEMINI_EMBED_MODEL}")
print("\nPick the fastest one that works and set it in .env. Changing the embedding")
print("model invalidates the index, so afterwards run:")
print("    python scripts/ingest.py --reset")
