"""Single entry point for every LLM call: one retry policy, one output contract.

Failures are classified rather than lumped together:
  429 / quota  -> stop retrying; retrying cannot create quota.
  5xx, timeout -> retry with backoff; Google's shared models spike and recover.
  4xx          -> stop, wrapped with the API's own message; our config is wrong.
  anything else-> re-raise raw; that is our bug and it must be loud.

The local Ollama fallback (config.ENABLE_LOCAL_FALLBACK, off by default) exists so a
quota limit mid-demo degrades instead of dying.
"""

from __future__ import annotations

import json
import random
import re
import time
from dataclasses import dataclass
from typing import Type, TypeVar

import httpx
import requests
from google import genai
from google.genai import types
from pydantic import BaseModel, ValidationError

from . import config
from .errors import TutorError
from .progress import status, step

T = TypeVar("T", bound=BaseModel)

OLLAMA_TIMEOUT = 180  # an 8B model on CPU is slow; this is not a network hang

# Qwen3 emits its chain of thought inside these tags before the real answer.
THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


class LLMError(TutorError):
    """Every provider failed, or one failed in a way we must not paper over."""


@dataclass
class LLMResponse:
    """The text plus who produced it — the provider is shown in the demo on purpose."""

    text: str
    provider: str  # "gemini" or "ollama"
    model: str

    def parse(self, schema: Type[T]) -> T:
        """Validate the JSON against the caller's Pydantic model."""
        try:
            return schema.model_validate_json(self.text)
        except ValidationError as error:
            raise LLMError(
                f"{self.provider} returned JSON that does not match {schema.__name__}:\n"
                f"{self.text[:500]}\n\n{error}"
            ) from error


_client: genai.Client | None = None


def get_client() -> genai.Client:
    global _client
    if _client is None:
        # An explicit timeout (the SDK takes milliseconds) so a stalled request
        # fails and gets retried instead of hanging the notebook forever.
        config.validate_models()
        from .embeddings import build_http_options  # shared timeout + retry policy

        _client = genai.Client(api_key=config.require_api_key(), http_options=build_http_options())
    return _client


def _is_daily_quota(error: Exception) -> bool:
    """A per-day quota, not a per-minute one.

    The distinction is everything: the same 429 carries a RetryInfo saying "retry in
    2.5s", which is true for a per-minute limit and useless for a daily one.
    """
    return "PerDay" in str(error)


def _is_rate_limit(error: Exception) -> bool:
    """429 / RESOURCE_EXHAUSTED. Quota, not correctness."""
    code = getattr(error, "code", None) or getattr(error, "status_code", None)
    if code == 429:
        return True
    # Narrow on purpose: matching a bare "429" anywhere in the message would
    # misread an unrelated error as a quota problem.
    return "RESOURCE_EXHAUSTED" in str(error).upper()


# The Gemini SDK speaks httpx; our Ollama client speaks requests. A retry policy
# that only knows one of them is blind to half its own failures - which is exactly
# what happened here: an httpx.ReadTimeout from Gemini was classified as "our bug"
# and re-raised instead of retried, after the loop had already survived two 503s.
_TRANSIENT_TYPES = (
    httpx.TimeoutException,      # ReadTimeout, ConnectTimeout, PoolTimeout
    httpx.TransportError,        # connection reset, DNS, protocol errors
    requests.exceptions.RequestException,
    TimeoutError,                # builtin, raised by the socket layer
    ConnectionError,             # builtin
)


def _is_transient(error: Exception) -> bool:
    """5xx, a timeout, or a network blip: worth retrying the same provider."""
    code = getattr(error, "code", None) or getattr(error, "status_code", None)
    if isinstance(code, int) and 500 <= code < 600:
        return True
    if isinstance(error, _TRANSIENT_TYPES):
        return True
    # Backstop for exception classes we do not import (a vendored httpx, a wrapper
    # type added in a future SDK version). Cheap, and it fails safe: worst case we
    # retry something that was never going to succeed.
    name = type(error).__name__.lower()
    return "timeout" in name or "unavailable" in name


def _backoff_delay(attempt: int) -> float:
    """Exponential backoff, capped, with jitter.

    The cap stops the last waits growing to minutes. The jitter matters once the
    exam generator issues several calls in a loop: without it, calls that failed
    together retry together, hit the same busy window together, and fail together
    again - a self-inflicted thundering herd. A random spread breaks the lockstep.

    Note the google-genai SDK already retries internally at the HTTP level. This
    layer sits on top of that and is what handles a spike the SDK gives up on.
    """
    delay = min(config.RETRY_BASE_DELAY * (2**attempt), config.RETRY_MAX_DELAY)
    # Clamp after jittering, so config.RETRY_MAX_DELAY is a real ceiling and not a
    # suggestion the jitter can exceed by 25%.
    return min(delay * random.uniform(0.75, 1.25), config.RETRY_MAX_DELAY)


# --------------------------------------------------------------------------- #
# Providers
# --------------------------------------------------------------------------- #


def _call_gemini(prompt: str, system: str | None, schema: Type[BaseModel] | None, temperature: float) -> str:
    request_config = types.GenerateContentConfig(
        system_instruction=system,
        temperature=temperature,
        # Forcing the schema at the API level is stricter than asking for JSON in
        # the prompt: the model is constrained during decoding, so it cannot
        # produce a shape that does not validate.
        response_mime_type="application/json" if schema else None,
        response_schema=schema,
    )
    client = get_client()
    response = client.models.generate_content(
        model=config.GEMINI_MODEL, contents=prompt, config=request_config
    )
    if not response.text:
        raise LLMError(f"Gemini returned no text (finish reason: {response.candidates[0].finish_reason}).")
    return response.text


def _call_ollama(prompt: str, system: str | None, schema: Type[BaseModel] | None, temperature: float) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload: dict = {
        "model": config.OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "think": False,  # ask Qwen3 to skip its reasoning block
        "options": {"temperature": temperature},
    }
    if schema:
        # Ollama constrains decoding to a JSON schema too, so the same Pydantic
        # model drives both providers.
        payload["format"] = schema.model_json_schema()

    response = requests.post(f"{config.OLLAMA_HOST}/api/chat", json=payload, timeout=OLLAMA_TIMEOUT)
    response.raise_for_status()
    content = response.json()["message"]["content"]
    # Belt and braces: older Ollama builds ignore "think", so strip the block anyway.
    return THINK_BLOCK_RE.sub("", content).strip()


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def generate(
    prompt: str,
    system: str | None = None,
    schema: Type[BaseModel] | None = None,
    temperature: float = 0.2,
    allow_fallback: bool | None = None,
) -> LLMResponse:
    """Ask Gemini; if it is unusable, optionally fall back to the local model.

    temperature defaults low: this project extracts topics, writes exam questions
    grounded in a source text, and grades answers. All three want faithfulness,
    not creativity.

    allow_fallback=None means "use the project setting" (config.ENABLE_LOCAL_FALLBACK).
    """
    # Validated here, not only inside get_client(): the client is a cached singleton,
    # so once it exists the check inside it never runs again. A guard that only fires
    # on the first call is a guard you cannot rely on.
    config.validate_models()

    if config.LLM_PROVIDER == "ollama":
        # Forced local: no Gemini attempt at all. With the daily quota spent, trying
        # Gemini first would burn a minute per call to prove what we already know.
        with step(f"{config.OLLAMA_MODEL} (local)"):
            text = _call_ollama(prompt, system, schema, temperature)
        return LLMResponse(text, "ollama", config.OLLAMA_MODEL)

    if allow_fallback is None:
        allow_fallback = config.ENABLE_LOCAL_FALLBACK

    last_error: Exception | None = None

    for attempt in range(config.RETRY_ATTEMPTS):
        try:
            label = f"{config.GEMINI_MODEL}" + (f" (attempt {attempt + 1}/{config.RETRY_ATTEMPTS})" if attempt else "")
            with step(label):
                text = _call_gemini(prompt, system, schema, temperature)
            return LLMResponse(text, "gemini", config.GEMINI_MODEL)
        except Exception as error:  # noqa: BLE001
            last_error = error
            if _is_rate_limit(error):
                break  # retrying cannot create quota
            if _is_transient(error):
                if attempt < config.RETRY_ATTEMPTS - 1:
                    wait = _backoff_delay(attempt)
                    status(
                        f"     Gemini busy ({_short(error)}) - "
                        f"retry {attempt + 1}/{config.RETRY_ATTEMPTS - 1} in {wait:.1f}s"
                    )
                    time.sleep(wait)
                    continue
                break  # patience spent; the provider is down, not our code
            code = getattr(error, "code", None)
            if isinstance(code, int) and 400 <= code < 500:
                # A 4xx is our fault (wrong model, bad schema, dead key), so we still
                # stop rather than fall back - but the API's own message is the most
                # useful thing on screen, and re-raising raw buries it under 40 lines
                # of SDK traceback. Wrap it so the diagnosis is the first thing read.
                raise LLMError(_explain(error)) from error
            raise  # a real bug in our code: TypeError, ValueError. Do not dress it up.

    if not allow_fallback:
        raise LLMError(_explain(last_error)) from last_error

    status(f"  [fallback] Gemini unusable -> {config.OLLAMA_MODEL} via Ollama")
    try:
        with step(f"{config.OLLAMA_MODEL} (local, this is slower)"):
            text = _call_ollama(prompt, system, schema, temperature)
        return LLMResponse(text, "ollama", config.OLLAMA_MODEL)
    except requests.exceptions.RequestException as error:
        raise LLMError(
            f"{_explain(last_error)}\n\n"
            f"The local fallback is enabled but Ollama is unreachable at {config.OLLAMA_HOST}.\n"
            f"Start it with `ollama serve`, or set ENABLE_LOCAL_FALLBACK=false in .env."
        ) from error


def _short(error: Exception) -> str:
    """First line of a provider error, for a progress message that stays readable."""
    return str(error).split("\n")[0][:80]


def _explain(error: Exception | None) -> str:
    """Turn a provider failure into something the reader can act on.

    A raw 503 traceback makes a student think they broke something. They did not:
    the shared model is busy. Saying so, and saying what to do, is the difference
    between a dead end and a five-second fix.
    """
    if error is None:
        return "Gemini failed for an unknown reason."
    code = getattr(error, "code", None)
    if isinstance(code, int) and 400 <= code < 500 and code != 429:
        # Google's own message usually names the fix ("use models/gemini-3.6-flash"),
        # so it goes first and unabridged. Ours only adds what it cannot know.
        return (
            f"Gemini rejected the request with {code} using GEMINI_MODEL={config.GEMINI_MODEL}.\n\n"
            f"What the API said:\n  {error}\n\n"
            f"If the message names a replacement model, put that name in .env. Beware that a\n"
            f"model can exist and still 404 for you: older ones are closed to new API keys.\n"
            f"`python scripts/bench.py` times every candidate against YOUR key, which is the\n"
            f"only list that matters."
        )
    if isinstance(error, _TRANSIENT_TYPES) and not getattr(error, "code", None):
        return (
            f"Gemini did not answer within {config.REQUEST_TIMEOUT_SECONDS:.0f}s per attempt, "
            f"{config.RETRY_ATTEMPTS} times over ({config.GEMINI_MODEL}).\n"
            f"The '-latest' aliases hot-swap to the newest model, which is also the most "
            f"contended one. Pin a stable build in .env instead:\n"
            f"    GEMINI_MODEL=gemini-2.5-flash\n"
            f"Run `python scripts/bench.py` to see which models actually respond on your key.\n"
            f"Original error: {type(error).__name__}: {error}"
        )
    if getattr(error, "code", None) == 404:
        return (
            f"Gemini says '{config.GEMINI_MODEL}' does not exist or cannot generate text.\n"
            f"Check config.GEMINI_MODEL in .env - a 404 here usually means an embedding model or a\n"
            f"retired model name. Current stable options include gemini-2.5-flash,\n"
            f"gemini-3.5-flash and gemini-3.7-flash. `python scripts/bench.py` lists what\n"
            f"your key can actually reach.\n"
            f"Original error: {error}"
        )
    if _is_rate_limit(error):
        if _is_daily_quota(error):
            return (
                f"Daily free-tier quota exhausted for {config.GEMINI_MODEL}.\n"
                f"The error also says 'retry in a few seconds' - ignore that. It is the per-minute\n"
                f"RetryInfo attached to every 429; a daily quota resets tomorrow.\n\n"
                f"Three ways forward:\n"
                f"  1. Switch model - the quota is PER MODEL, so another has its own budget:\n"
                f"       GEMINI_MODEL=gemini-3.5-flash-lite\n"
                f"  2. Run locally for the rest of today:  LLM_PROVIDER=ollama\n"
                f"  3. Stay on Gemini but degrade automatically:  ENABLE_LOCAL_FALLBACK=true\n\n"
                f"Original error: {error}"
            )
        return (
            f"Per-minute rate limit on {config.GEMINI_MODEL}. Quota, not a bug in your code.\n"
            f"Wait a minute and re-run, or set ENABLE_LOCAL_FALLBACK=true in .env.\n"
            f"Original error: {error}"
        )
    return (
        f"Gemini is still unavailable after {config.RETRY_ATTEMPTS} attempts spread over "
        f"~{int(sum(min(config.RETRY_BASE_DELAY * 2**i, config.RETRY_MAX_DELAY) for i in range(config.RETRY_ATTEMPTS - 1)))}s "
        f"({config.GEMINI_MODEL}).\n"
        f"A 503 means the shared model is overloaded on Google's side - nothing is wrong with "
        f"your key or your code. Options:\n"
        f"  1. Re-run the cell in a minute; these spikes are usually short.\n"
        f"  2. Pin a specific model in .env, e.g. GEMINI_MODEL=gemini-3.5-flash-lite, instead of the\n"
        f"     'latest' alias, which points at whichever build is busiest.\n"
        f"  3. Set ENABLE_LOCAL_FALLBACK=true to answer from Ollama instead.\n"
        f"Original error: {error}"
    )


def ollama_available(model: str | None = None) -> bool:
    """Used by the notebook to report honestly whether the fallback is armed."""
    try:
        response = requests.get(f"{config.OLLAMA_HOST}/api/tags", timeout=3)
        models = [m["name"] for m in response.json().get("models", [])]
        wanted = (model or config.OLLAMA_MODEL).split(":")[0]
        return any(m.split(":")[0] == wanted for m in models)
    except Exception:  # noqa: BLE001
        return False
