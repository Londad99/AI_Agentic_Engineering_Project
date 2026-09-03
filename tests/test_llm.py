"""Tests for the provider routing policy in tutor/llm.py.

These make no network calls: the two provider functions are replaced with fakes,
so what is under test is the decision logic — when to retry, when to fall back,
and when to refuse to fall back.

That last one is the point of the file. A fallback that triggers on *any* error
is worse than no fallback: a bug in our prompt or schema would quietly produce a
degraded answer from the local model instead of a stack trace, and we would ship
it without noticing.

Run:  python tests/test_llm.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests  # noqa: E402

from tutor import llm, progress  # noqa: E402

progress.VERBOSE = False  # keep the test output to one line per test


class FakeRateLimit(Exception):
    code = 429


class FakeServerError(Exception):
    code = 503


def run(gemini, ollama=lambda *a: "local answer", **kwargs):
    """Swap both providers for fakes and call generate().

    The fallback is off by default in config while the project is being built, so
    the tests that exercise it pass allow_fallback explicitly.
    """
    llm._call_gemini, llm._call_ollama = gemini, ollama
    llm.time.sleep = lambda _s: None  # never actually wait in tests
    return llm.generate("prompt", **kwargs)


def test_happy_path():
    response = run(lambda *a: "gemini answer")
    assert response.provider == "gemini" and response.text == "gemini answer"
    print("happy path            OK")


def test_rate_limit_falls_back():
    calls = []

    def gemini(*a):
        calls.append(1)
        raise FakeRateLimit("429 RESOURCE_EXHAUSTED")

    response = run(gemini, allow_fallback=True)
    assert response.provider == "ollama", "did not fall back on a rate limit"
    assert len(calls) == 1, "retried a rate limit instead of going local immediately"
    print("rate limit -> ollama  OK")


def test_httpx_timeout_is_transient():
    """The regression that broke the structured-output cell.

    Gemini's SDK raises httpx exceptions, not requests ones. The first version of
    _is_transient only knew about requests, so an httpx.ReadTimeout - the single
    most common Gemini failure - was classified as a bug in our code and re-raised,
    after the retry loop had already correctly survived two 503s.
    """
    import httpx

    for error in (
        httpx.ReadTimeout("timed out"),
        httpx.ConnectTimeout("timed out"),
        httpx.ConnectError("refused"),
        TimeoutError("timed out"),
    ):
        assert llm._is_transient(error), f"{type(error).__name__} would not be retried"

    assert not llm._is_transient(TypeError("bad argument")), "a code bug looks transient"
    assert not llm._is_transient(ValueError("bad schema")), "a code bug looks transient"
    print("httpx timeout retried OK")


def test_timeout_message_suggests_pinning():
    import httpx

    def gemini(*a):
        raise httpx.ReadTimeout("The read operation timed out")

    llm._call_gemini = gemini
    llm.time.sleep = lambda _s: None
    try:
        llm.generate("prompt", allow_fallback=False)
    except llm.LLMError as error:
        assert "GEMINI_MODEL=" in str(error), "does not suggest pinning a stable model"
        print("timeout guidance      OK")
    else:
        raise AssertionError("expected LLMError after repeated timeouts")


def test_model_variables_are_validated():
    """The mistake that produced a 404: an embedding model in the chat variable.

    Gemini's own error ("is not found for API version v1beta, or is not supported
    for generateContent") reads like a missing model, which sends you hunting for
    model names instead of looking at which variable holds them. Catching it before
    the call turns an hour of confusion into one sentence.
    """
    from tutor import config
    from tutor.errors import ConfigError

    original_chat, original_embed = config.GEMINI_MODEL, config.GEMINI_EMBED_MODEL
    try:
        config.GEMINI_MODEL = "gemini-embedding-001"
        try:
            config.validate_models()
        except ConfigError as error:
            assert "embedding model" in str(error)
            assert "GEMINI_MODEL=gemini-2.5-flash" in str(error), "does not name the fix"
        else:
            raise AssertionError("an embedding model in GEMINI_MODEL was accepted")

        config.GEMINI_MODEL = "gemini-2.5-flash"
        config.GEMINI_EMBED_MODEL = "gemini-2.5-flash"
        try:
            config.validate_models()
        except ConfigError as error:
            assert "does not look like" in str(error)
        else:
            raise AssertionError("a chat model in GEMINI_EMBED_MODEL was accepted")

        config.GEMINI_EMBED_MODEL = "gemini-embedding-001"
        config.validate_models()  # the correct pairing must pass
        print("model vars validated  OK")
    finally:
        config.GEMINI_MODEL, config.GEMINI_EMBED_MODEL = original_chat, original_embed


def test_api_4xx_is_wrapped_not_buried():
    """A 404 must stop the run, but readably.

    This is what happened with gemini-2.5-flash: Google answered
    "no longer available to new users, use models/gemini-3.6-flash" - the exact fix,
    in the exception - and our code re-raised it raw, burying that sentence under
    forty lines of SDK traceback. A 4xx is still our fault and still must not fall
    back, but the API's own message deserves to be the first thing on screen.
    """

    class NotFound(Exception):
        code = 404

    def gemini(*a):
        raise NotFound("This model is no longer available, use models/gemini-3.6-flash")

    llm._call_gemini = gemini
    try:
        llm.generate("prompt", allow_fallback=True)  # must NOT fall back on a 4xx
    except llm.LLMError as error:
        assert "gemini-3.6-flash" in str(error), "the API's own guidance was dropped"
        assert "GEMINI_MODEL=" in str(error), "does not point at the setting to change"
        print("4xx wrapped           OK")
    else:
        raise AssertionError("a 404 was answered by the fallback model")


DAILY_429 = (
    "429 RESOURCE_EXHAUSTED. Quota exceeded for metric: generate_content_free_tier_requests, "
    "limit: 20 ... 'quotaId': 'GenerateRequestsPerDayPerProjectPerModel-FreeTier' ... "
    "'retryDelay': '2s'"
)
MINUTE_429 = "429 RESOURCE_EXHAUSTED. 'quotaId': 'GenerateRequestsPerMinutePerProject-FreeTier'"


def test_daily_quota_is_told_apart_from_per_minute():
    """Both are 429 and both carry 'retry in 2s'. Only one of them means it.

    A daily quota resets tomorrow, so 'wait a minute and re-run' is wrong advice that
    costs the student an afternoon of retrying.
    """
    assert llm._is_daily_quota(Exception(DAILY_429))
    assert not llm._is_daily_quota(Exception(MINUTE_429))

    daily = llm._explain(FakeRateLimit(DAILY_429))
    assert "resets tomorrow" in daily
    assert "LLM_PROVIDER=ollama" in daily, "does not offer the local switch"
    assert "PER MODEL" in daily, "does not mention that another model has its own quota"

    minute = llm._explain(FakeRateLimit(MINUTE_429))
    assert "Wait a minute" in minute and "resets tomorrow" not in minute
    print("daily vs minute 429   OK")


def test_forced_local_never_calls_gemini():
    """LLM_PROVIDER=ollama must not touch Gemini at all.

    With the daily quota spent, an attempt-then-fall-back would burn a minute of
    retries per call to rediscover something we already know.
    """
    called = []

    def gemini(*a):
        called.append(1)
        raise AssertionError("Gemini was called with LLM_PROVIDER=ollama")

    llm._call_gemini, llm._call_ollama = gemini, lambda *a: "local answer"
    original = llm.config.LLM_PROVIDER
    try:
        llm.config.LLM_PROVIDER = "ollama"
        response = llm.generate("prompt")
        assert response.provider == "ollama" and not called
        print("forced local          OK")
    finally:
        llm.config.LLM_PROVIDER = original


def test_invalid_provider_is_rejected():
    from tutor import config
    from tutor.errors import ConfigError

    original = config.LLM_PROVIDER
    try:
        config.LLM_PROVIDER = "openai"
        try:
            config.validate_models()
        except ConfigError as error:
            assert "LLM_PROVIDER" in str(error)
            print("bad provider caught   OK")
        else:
            raise AssertionError("an unknown provider was accepted")
    finally:
        config.LLM_PROVIDER = original


def test_env_names_in_messages_are_real():
    """Error messages must name the .env variable, not the Python attribute path.

    A refactor once rewrote GEMINI_MODEL into config.GEMINI_MODEL everywhere, including
    inside the strings telling the user what to put in .env - advice that cannot work.
    """
    import re

    source = open(llm.__file__, encoding='utf-8').read()
    for match in re.findall(r'config\.[A-Z_]+=', source):
        raise AssertionError(f"message text tells the user to write {match!r} in .env")
    print("env names in messages OK")


def test_error_says_whether_the_fallback_ran():
    """An error must never leave you guessing whether the local model was tried."""
    def gemini(*a):
        raise FakeRateLimit(DAILY_429)

    llm._call_gemini = gemini
    llm.time.sleep = lambda _s: None
    try:
        llm.generate("prompt", allow_fallback=False)
    except llm.LLMError as error:
        assert "local fallback was NOT used" in str(error)
        assert "ENABLE_LOCAL_FALLBACK is" in str(error), "does not show the live value"
        print("fallback status shown OK")
    else:
        raise AssertionError("expected LLMError")


def test_env_edit_is_picked_up_without_a_restart():
    """Editing .env mid-session must take effect on the next call, not the next kernel."""
    import os
    import tempfile

    from tutor import config

    original_file = config.ENV_FILE
    # A temp dir, not the repo: tests must not leave files behind next to the code.
    tmp = Path(tempfile.mkdtemp()) / "test.env"
    try:
        tmp.write_text("GEMINI_MODEL=model-a\nGEMINI_EMBED_MODEL=gemini-embedding-001\n")
        config.ENV_FILE = tmp
        config._env_mtime = 0.0
        config.reload_if_changed()
        assert config.GEMINI_MODEL == "model-a", config.GEMINI_MODEL

        tmp.write_text("GEMINI_MODEL=model-b\nGEMINI_EMBED_MODEL=gemini-embedding-001\n")
        # Stamp the mtime forward explicitly: some filesystems have one-second
        # granularity, so two writes in the same second are indistinguishable and the
        # test would fail for a reason that has nothing to do with the logic.
        stamp = tmp.stat().st_mtime + 5
        os.utime(tmp, (stamp, stamp))
        assert config.reload_if_changed(), "the change on disk was not noticed"
        assert config.GEMINI_MODEL == "model-b", "still serving the old value"

        assert not config.reload_if_changed(), "reloaded again with no change - wasteful"
        print("env hot reload        OK")
    finally:
        config.ENV_FILE = original_file
        config._env_mtime = 0.0
        config.reload()


def test_real_bug_is_not_hidden():
    def gemini(*a):
        raise TypeError("bad argument to generate_content")

    llm._call_gemini, llm._call_ollama = gemini, lambda *a: "local answer"
    try:
        llm.generate("prompt")
    except TypeError:
        print("bug re-raised         OK")
    else:
        raise AssertionError("a code bug was silently answered by the fallback model")


def test_transient_error_retries_same_provider():
    attempts = []

    def gemini(*a):
        attempts.append(1)
        if len(attempts) < 2:
            raise FakeServerError("503 unavailable")
        return "gemini answer after retry"

    response = run(gemini)
    assert response.provider == "gemini" and len(attempts) == 2, attempts
    print("transient retry       OK")


def test_retry_schedule():
    """A 503 spike must be waited out, not given up on after two seconds.

    Checks the shape of the backoff rather than exact numbers: as many attempts as
    configured, waits that grow, and a cap so the last one cannot balloon.
    """
    waits = []
    attempts = []

    def gemini(*a):
        attempts.append(1)
        raise FakeServerError("503 UNAVAILABLE")

    llm._call_gemini = gemini
    llm._call_ollama = lambda *a: "local answer"
    llm.time.sleep = waits.append

    try:
        llm.generate("prompt", allow_fallback=False)
    except llm.LLMError:
        pass

    assert len(attempts) == llm.config.RETRY_ATTEMPTS, f"tried {len(attempts)}, expected {llm.config.RETRY_ATTEMPTS}"
    assert len(waits) == llm.config.RETRY_ATTEMPTS - 1, "one wait between each pair of attempts"
    assert waits[0] < waits[-1], "backoff is not growing"
    assert max(waits) <= llm.config.RETRY_MAX_DELAY, "a wait exceeded RETRY_MAX_DELAY - the cap is not a cap"
    assert len(set(waits)) == len(waits), "waits are identical - jitter is not applied"
    print(f"retry schedule        OK ({len(attempts)} tries, {sum(waits):.0f}s total patience)")


def test_exhausted_retries_fall_back():
    """A 503 that never clears means the provider is down, not that we have a bug.

    This is the case that actually happened during development: Gemini answered
    503 UNAVAILABLE ("high demand") three times in a row. The first version of
    generate() re-raised it and never reached the fallback, contradicting its own
    docstring. This test exists so that cannot regress.
    """

    def gemini(*a):
        raise FakeServerError("503 UNAVAILABLE")

    response = run(gemini, allow_fallback=True)
    assert response.provider == "ollama", "gave up instead of falling back after retries"
    print("503 exhausted -> local OK")


def test_outage_message_is_actionable():
    """With the fallback off, the user must learn it is Google's fault, not theirs."""

    def gemini(*a):
        raise FakeServerError("503 UNAVAILABLE")

    llm._call_gemini = gemini
    llm.time.sleep = lambda _s: None
    try:
        llm.generate("prompt", allow_fallback=False)
    except llm.LLMError as error:
        message = str(error)
        assert "overloaded on Google" in message, "does not say whose fault it is"
        assert "GEMINI_MODEL=" in message, "does not offer the pin-a-model workaround"
        print("outage message        OK")
    else:
        raise AssertionError("expected LLMError when Gemini is down and fallback is off")


def test_fallback_can_be_disabled():
    def gemini(*a):
        raise FakeRateLimit("429")

    llm._call_gemini = gemini
    try:
        llm.generate("prompt", allow_fallback=False)
    except llm.LLMError:
        print("allow_fallback=False  OK")
    else:
        raise AssertionError("fell back even though allow_fallback was False")


def test_running_ollama_with_the_wrong_model_is_not_reported_as_unreachable():
    """The mistake that cost an evening: Ollama was running the whole time.

    _call_ollama used raise_for_status(), so a 404 "model not found" arrived as a
    RequestException - the same class as a refused connection - and got reported as
    "Ollama is unreachable". That sends you to restart a service that was already up,
    instead of comparing OLLAMA_MODEL against `ollama list`.
    """
    llm.ollama_models = lambda: ["llama3:8b", "qwen3:4b"]
    original = llm.config.OLLAMA_MODEL
    try:
        llm.config.OLLAMA_MODEL = "qwen3:8b"
        report = llm.ollama_report(llm.OllamaRejected(404, 'model "qwen3:8b" not found'))
        assert "IS running" in report, "still claims the server is down"
        assert "llama3:8b" in report and "qwen3:4b" in report, "does not list what IS installed"
        assert "OLLAMA_MODEL" in report, "does not point at the setting to change"
        print("wrong model reported  OK")
    finally:
        llm.config.OLLAMA_MODEL = original


def test_unreachable_ollama_message():
    def gemini(*a):
        raise FakeRateLimit("429")

    def ollama(*a):
        raise llm.OllamaUnreachable("connection refused")

    llm.ollama_models = lambda: None       # server genuinely not answering
    llm._call_gemini, llm._call_ollama = gemini, ollama
    try:
        llm.generate("prompt", allow_fallback=True)
    except llm.LLMError as error:
        assert "ollama serve" in str(error), "error does not tell the user how to fix it"
        print("both down -> guidance OK")
    else:
        raise AssertionError("expected LLMError when both providers fail")


def test_think_block_is_stripped():
    dirty = "<think>the user wants a topic list, let me...</think>\n{\"ok\": true}"
    assert llm.THINK_BLOCK_RE.sub("", dirty).strip() == '{"ok": true}'
    print("think block stripped  OK")


if __name__ == "__main__":
    test_happy_path()
    test_rate_limit_falls_back()
    test_model_variables_are_validated()
    test_daily_quota_is_told_apart_from_per_minute()
    test_forced_local_never_calls_gemini()
    test_invalid_provider_is_rejected()
    test_env_names_in_messages_are_real()
    test_error_says_whether_the_fallback_ran()
    test_env_edit_is_picked_up_without_a_restart()
    test_api_4xx_is_wrapped_not_buried()
    test_real_bug_is_not_hidden()
    test_httpx_timeout_is_transient()
    test_timeout_message_suggests_pinning()
    test_transient_error_retries_same_provider()
    test_retry_schedule()
    test_exhausted_retries_fall_back()
    test_outage_message_is_actionable()
    test_fallback_can_be_disabled()
    test_running_ollama_with_the_wrong_model_is_not_reported_as_unreachable()
    test_unreachable_ollama_message()
    test_think_block_is_stripped()
    print("\nall llm routing tests passed")
