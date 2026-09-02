"""Central configuration. Every module reads settings from here, never from os.environ directly."""

import os
from pathlib import Path

import shutil

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
STORAGE_DIR = ROOT_DIR / "storage"
PROMPTS_DIR = ROOT_DIR / "prompts"
NOTEBOOK_DIR = ROOT_DIR / "notebooks"

ENV_FILE = ROOT_DIR / ".env"
ENV_TEMPLATE = ROOT_DIR / ".env.example"


def ensure_env_file() -> bool:
    """Create .env from .env.example on first run. Returns True if it just created it.

    Why not simply ship a .env with empty values? Because .env is git-ignored, so a
    committed one would never reach anyone who clones the repo — and un-ignoring it to
    fix that makes git track the file, which is exactly how API keys end up pushed to
    GitHub. The template is public documentation; the values stay local. This function
    removes the friction of copying it by hand without giving up that separation.
    """
    if ENV_FILE.exists() or not ENV_TEMPLATE.exists():
        return False
    shutil.copy(ENV_TEMPLATE, ENV_FILE)
    return True


ensure_env_file()

# Anchor to the repo root, not to the current working directory: the notebook
# lives in notebooks/ and would otherwise never find the .env file.
load_dotenv(ENV_FILE)

# An empty string or a leftover placeholder must count as "not set", otherwise the
# failure surfaces later as an opaque 400 from the API instead of a clear message here.
_PLACEHOLDERS = {"", "your_key_here", "your_key_from_aistudio.google.com"}


def _read_api_key() -> str | None:
    raw = (os.environ.get("GOOGLE_API_KEY") or "").strip()
    return None if raw in _PLACEHOLDERS else raw


GOOGLE_API_KEY = _read_api_key()

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")
GEMINI_EMBED_MODEL = os.environ.get("GEMINI_EMBED_MODEL", "gemini-embedding-001")

# gemini-embedding-001 natively outputs 3072 dims. 768 keeps quality while cutting
# storage and search cost ~4x. Any value other than 3072 must be re-normalized by us.
EMBED_DIM = int(os.environ.get("EMBED_DIM", "768"))

# Chunking, in characters (see tutor/ingest/chunker.py for why characters and not tokens).
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "1200"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "200"))

CHROMA_COLLECTION = os.environ.get("CHROMA_COLLECTION", "study_material")

# Texts per embedding request. The observed cost of an embedding call is dominated
# by the round trip, not by how many texts it carries, so bigger batches are close
# to free: 240 chunks is 3 requests at 100, but 8 requests at 32. Lower it if the
# API starts rejecting large batches.
EMBED_BATCH_SIZE = int(os.environ.get("EMBED_BATCH_SIZE", "100"))

# Hard ceiling on a single API call. Without it, httpx waits forever on a stalled
# connection and the notebook cell looks like it is "still loading" with no end.
# Failing after a minute is strictly better: the retry loop can then do its job.
REQUEST_TIMEOUT_SECONDS = float(os.environ.get("REQUEST_TIMEOUT_SECONDS", "60"))

# Retry policy for Gemini. A 503 "high demand" is routine on the free tier and
# clears in tens of seconds, so the waits are sized for a capacity spike rather
# than a network blip: 2s, 4s, 8s, 16s (capped), ~30s of total patience.
RETRY_ATTEMPTS = int(os.environ.get("RETRY_ATTEMPTS", "5"))
RETRY_BASE_DELAY = float(os.environ.get("RETRY_BASE_DELAY", "2"))
RETRY_MAX_DELAY = float(os.environ.get("RETRY_MAX_DELAY", "16"))

# Off while the project is being built: running an 8B model locally is slow and is
# not what we are debugging right now. Flip it to true in .env once the agents work,
# so the demo has a safety net against Gemini quota limits and outages.
ENABLE_LOCAL_FALLBACK = os.environ.get("ENABLE_LOCAL_FALLBACK", "false").strip().lower() in {
    "1", "true", "yes",
}

OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3:8b")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")


def reload() -> dict:
    """Re-read .env into this module, without restarting the Python process.

    Needed because a Jupyter kernel outlives every edit you make to .env. Modules
    are imported once and their globals keep whatever the file said at that moment,
    so changing a setting appears to do nothing - the single most confusing failure
    mode when configuring this project.

    override=True matters: load_dotenv defaults to not overwriting variables that
    already exist in the environment, which is exactly the case on a second call.
    """
    load_dotenv(ENV_FILE, override=True)
    module = globals()
    module["GOOGLE_API_KEY"] = _read_api_key()
    module["GEMINI_MODEL"] = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")
    module["GEMINI_EMBED_MODEL"] = os.environ.get("GEMINI_EMBED_MODEL", "gemini-embedding-001")
    module["EMBED_DIM"] = int(os.environ.get("EMBED_DIM", "768"))
    module["EMBED_BATCH_SIZE"] = int(os.environ.get("EMBED_BATCH_SIZE", "100"))
    module["CHUNK_SIZE"] = int(os.environ.get("CHUNK_SIZE", "1200"))
    module["CHUNK_OVERLAP"] = int(os.environ.get("CHUNK_OVERLAP", "200"))
    module["REQUEST_TIMEOUT_SECONDS"] = float(os.environ.get("REQUEST_TIMEOUT_SECONDS", "60"))
    module["RETRY_ATTEMPTS"] = int(os.environ.get("RETRY_ATTEMPTS", "5"))
    module["RETRY_BASE_DELAY"] = float(os.environ.get("RETRY_BASE_DELAY", "2"))
    module["RETRY_MAX_DELAY"] = float(os.environ.get("RETRY_MAX_DELAY", "16"))
    module["ENABLE_LOCAL_FALLBACK"] = os.environ.get(
        "ENABLE_LOCAL_FALLBACK", "false"
    ).strip().lower() in {"1", "true", "yes"}
    module["OLLAMA_MODEL"] = os.environ.get("OLLAMA_MODEL", "qwen3:8b")
    module["OLLAMA_HOST"] = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

    # The clients cache the API key and timeout, so they must be rebuilt too.
    from . import embeddings, llm

    embeddings._client = None
    llm._client = None

    return {
        "GEMINI_MODEL": GEMINI_MODEL,
        "GEMINI_EMBED_MODEL": GEMINI_EMBED_MODEL,
        "EMBED_DIM": EMBED_DIM,
        "ENABLE_LOCAL_FALLBACK": ENABLE_LOCAL_FALLBACK,
    }


def validate_models() -> None:
    """Catch model names put in the wrong variable, before the API does.

    Gemini has two model families that do not overlap: generative models answer
    generateContent, embedding models answer embedContent. Swapping them returns a
    404 whose message ("is not found for API version v1beta, or is not supported
    for generateContent") reads like the model does not exist, which sends you
    hunting for the right model name instead of looking at the variable it is in.

    Cheap to check, and it turns a confusing 404 into a sentence that names the fix.
    """
    from .errors import ConfigError

    if "embedding" in GEMINI_MODEL.lower():
        raise ConfigError(
            f"GEMINI_MODEL is set to '{GEMINI_MODEL}', which is an embedding model.\n"
            f"It cannot generate text - embedding models have no generateContent method.\n"
            f"Set a generative model in {ENV_FILE}, for example:\n"
            f"    GEMINI_MODEL=gemini-2.5-flash\n"
            f"    GEMINI_EMBED_MODEL={GEMINI_EMBED_MODEL}   (this one is correct)"
        )
    if "embedding" not in GEMINI_EMBED_MODEL.lower():
        raise ConfigError(
            f"GEMINI_EMBED_MODEL is set to '{GEMINI_EMBED_MODEL}', which does not look like\n"
            f"an embedding model. Expected something like gemini-embedding-001 or\n"
            f"gemini-embedding-2. Fix it in {ENV_FILE}."
        )


def require_api_key() -> str:
    if not GOOGLE_API_KEY:
        raise RuntimeError(
            f"GOOGLE_API_KEY is empty.\n"
            f"Open {ENV_FILE} and paste your key from https://aistudio.google.com/apikey\n"
            f"(the file was created for you from .env.example; it is git-ignored, so your "
            f"key never leaves this machine)."
        )
    return GOOGLE_API_KEY
