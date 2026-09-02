"""One entry point for every LLM call in the project.

No agent talks to Gemini directly. They all call `generate()` here, which means:

* the retry, timeout and fallback policy is written once, not four times;
* switching model or provider is a one-line change instead of a hunt;
* structured output works the same way on both backends, so an agent's Pydantic
  contract holds no matter who answered.

The fallback exists for a specific, real risk: the Gemini free tier is rate
limited per minute and its shared models return 503 under load, while a live
10-minute demo makes several calls in a row. If either hits mid-demo, the project
can degrade to a local Qwen model through Ollama instead of dying.

It is OFF by default (config.ENABLE_LOCAL_FALLBACK), because an 8B model on CPU
is slow and turning it on while the agents are still being written just hides
what we are trying to debug. Turn it on for the demo.

The important discipline is *when* to fall back:

  429 / RESOURCE_EXHAUSTED  -> immediately. Retrying cannot create quota.
  5xx / network             -> retry Gemini with backoff first; fall back only
                               after the retries are spent.
  anything else             -> re-raise. A malformed prompt or a broken schema is
                               our bug, and answering it with a weaker model would
                               hide the bug behind a worse answer.
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

from .config import (
    ENABLE_LOCAL_FALLBACK,
    GEMINI_MODEL,
    OLLAMA_HOST,
    OLLAMA_MODEL,
    REQUEST_TIMEOUT_SECONDS,
    RETRY_ATTEMPTS,
    RETRY_BASE_DELAY,
    RETRY_MAX_DELAY,
    require_api_key,
)
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
        from .embeddings import build_http_options  # shared timeout + retry policy

        _client = genai.Client(api_key=require_api_key(), http_options=build_http_options())
    return _client


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
    delay = min(RETRY_BASE_DELAY * (2**attempt), RETRY_MAX_DELAY)
    # Clamp after jittering, so RETRY_MAX_DELAY is a real ceiling and not a
    # suggestion the jitter can exceed by 25%.
    return min(delay * random.uniform(0.75, 1.25), RETRY_MAX_DELAY)


# --------------------------------------------------------------------------- #
# Providers
# --------------------------------------------------------------------------- #


def _call_gemini(prompt: str, system: str | None, schema: Type[BaseModel] | None, temperature: float) -> str:
    config = types.GenerateContentConfig(
        system_instruction=system,
        temperature=temperature,
        # Forcing the schema at the API level is stricter than asking for JSON in
        # the prompt: the model is constrained during decoding, so it cannot
        # produce a shape that does not validate.
        response_mime_type="application/json" if schema else None,
        response_schema=schema,
    )
    client = get_client()
    response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt, config=config)
    if not response.text:
        raise LLMError(f"Gemini returned no text (finish reason: {response.candidates[0].finish_reason}).")
    return response.text


def _call_ollama(prompt: str, system: str | None, schema: Type[BaseModel] | None, temperature: float) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload: dict = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "think": False,  # ask Qwen3 to skip its reasoning block
        "options": {"temperature": temperature},
    }
    if schema:
        # Ollama constrains decoding to a JSON schema too, so the same Pydantic
        # model drives both providers.
        payload["format"] = schema.model_json_schema()

    response = requests.post(f"{OLLAMA_HOST}/api/chat", json=payload, timeout=OLLAMA_TIMEOUT)
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

    allow_fallback=None means "use the project setting" (ENABLE_LOCAL_FALLBACK).
    """
    if allow_fallback is None:
        allow_fallback = ENABLE_LOCAL_FALLBACK

    last_error: Exception | None = None

    for attempt in range(RETRY_ATTEMPTS):
        try:
            label = f"{GEMINI_MODEL}" + (f" (attempt {attempt + 1}/{RETRY_ATTEMPTS})" if attempt else "")
            with step(label):
                text = _call_gemini(prompt, system, schema, temperature)
            return LLMResponse(text, "gemini", GEMINI_MODEL)
        except Exception as error:  # noqa: BLE001
            last_error = error
            if _is_rate_limit(error):
                break  # retrying cannot create quota
            if _is_transient(error):
                if attempt < RETRY_ATTEMPTS - 1:
                    wait = _backoff_delay(attempt)
                    status(
                        f"     Gemini busy ({_short(error)}) - "
                        f"retry {attempt + 1}/{RETRY_ATTEMPTS - 1} in {wait:.1f}s"
                    )
                    time.sleep(wait)
                    continue
                break  # patience spent; the provider is down, not our code
            raise  # a real bug: schema error, bad argument, missing key. Do not hide it.

    if not allow_fallback:
        raise LLMError(_explain(last_error)) from last_error

    status(f"  [fallback] Gemini unusable -> {OLLAMA_MODEL} via Ollama")
    try:
        with step(f"{OLLAMA_MODEL} (local, this is slower)"):
            text = _call_ollama(prompt, system, schema, temperature)
        return LLMResponse(text, "ollama", OLLAMA_MODEL)
    except requests.exceptions.RequestException as error:
        raise LLMError(
            f"{_explain(last_error)}\n\n"
            f"The local fallback is enabled but Ollama is unreachable at {OLLAMA_HOST}.\n"
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
    if isinstance(error, _TRANSIENT_TYPES) and not getattr(error, "code", None):
        return (
            f"Gemini did not answer within {REQUEST_TIMEOUT_SECONDS:.0f}s per attempt, "
            f"{RETRY_ATTEMPTS} times over ({GEMINI_MODEL}).\n"
            f"The '-latest' aliases hot-swap to the newest model, which is also the most "
            f"contended one. Pin a stable build in .env instead:\n"
            f"    GEMINI_MODEL=gemini-2.5-flash\n"
            f"Run `python scripts/bench.py` to see which models actually respond on your key.\n"
            f"Original error: {type(error).__name__}: {error}"
        )
    if _is_rate_limit(error):
        return (
            f"Gemini rate limit reached ({GEMINI_MODEL}). This is quota, not a bug in your code.\n"
            f"Wait about a minute, or enable the local fallback with ENABLE_LOCAL_FALLBACK=true in .env.\n"
            f"Original error: {error}"
        )
    return (
        f"Gemini is still unavailable after {RETRY_ATTEMPTS} attempts spread over "
        f"~{int(sum(min(RETRY_BASE_DELAY * 2**i, RETRY_MAX_DELAY) for i in range(RETRY_ATTEMPTS - 1)))}s "
        f"({GEMINI_MODEL}).\n"
        f"A 503 means the shared model is overloaded on Google's side - nothing is wrong with "
        f"your key or your code. Options:\n"
        f"  1. Re-run the cell in a minute; these spikes are usually short.\n"
        f"  2. Pin a specific model in .env, e.g. GEMINI_MODEL=gemini-2.5-flash, instead of the\n"
        f"     'latest' alias, which points at whichever build is busiest.\n"
        f"  3. Set ENABLE_LOCAL_FALLBACK=true to answer from Ollama instead.\n"
        f"Original error: {error}"
    )


def ollama_available() -> bool:
    """Used by the notebook to report honestly whether the fallback is armed."""
    try:
        response = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=3)
        models = [m["name"] for m in response.json().get("models", [])]
        return any(m.split(":")[0] == OLLAMA_MODEL.split(":")[0] for m in models)
    except Exception:  # noqa: BLE001
        return False
