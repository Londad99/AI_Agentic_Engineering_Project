"""The shared persona must reach every agent.

The failure this guards against is silent: an agent whose system prompt is only its
own role text still works, still answers, and quietly loses the grounding and refusal
rules - so it starts inventing facts the document never stated. Nothing crashes; the
tutor just becomes wrong in a way a student cannot detect.

Run:  python tests/test_prompts.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tutor import prompts  # noqa: E402
from tutor.errors import ConfigError  # noqa: E402


def test_system_prompt_exists_and_has_the_rules():
    text = prompts.load("system_prompt")
    for rule in ["grounded", "does not cover", "language of the source document"]:
        assert rule.lower() in text.lower(), f"system prompt lost its '{rule}' rule"
    print("system prompt intact  OK")


def test_role_is_composed_on_top_of_the_persona():
    composed = prompts.system("You extract topics.")
    assert prompts.load("system_prompt") in composed, "an agent would run without the persona"
    assert "You extract topics." in composed, "the role text was dropped"
    assert composed.index("You extract topics.") > composed.index("grounded"), (
        "the role must come after the shared rules, not before them"
    )
    print("role composed         OK")


def test_missing_prompt_names_what_exists():
    try:
        prompts.load("no_such_prompt")
    except ConfigError as error:
        assert "Available:" in str(error), "does not list the prompts that do exist"
        print("missing prompt msg    OK")
    else:
        raise AssertionError("loading a missing prompt silently succeeded")


if __name__ == "__main__":
    test_system_prompt_exists_and_has_the_rules()
    test_role_is_composed_on_top_of_the_persona()
    test_missing_prompt_names_what_exists()
    print("\nall prompt tests passed")
