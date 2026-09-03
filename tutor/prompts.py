"""Prompts live in files, not in string literals scattered through the code.

The shared instructions (scope, refusal, language, citations) are identical for every
agent, so they are composed rather than duplicated. Editing behaviour then needs no
Python and produces a readable diff - and prompts/system_prompt.txt is the Week 2
deliverable, running for real rather than a copy that drifts.
"""

from __future__ import annotations

from functools import lru_cache

from .config import PROMPTS_DIR
from .errors import ConfigError


@lru_cache(maxsize=None)
def load(name: str) -> str:
    """Read prompts/<name>.txt. Cached, since prompts do not change mid-run."""
    path = PROMPTS_DIR / f"{name}.txt"
    if not path.exists():
        available = sorted(p.stem for p in PROMPTS_DIR.glob("*.txt"))
        raise ConfigError(f"No prompt named '{name}' in {PROMPTS_DIR}. Available: {available}")
    return path.read_text(encoding="utf-8").strip()


def system(role: str) -> str:
    """The shared tutor persona plus one agent's specific instructions.

    Composed rather than concatenated by hand at each call site, so no agent can
    accidentally run without the grounding and refusal rules.
    """
    return f"{load('system_prompt')}\n\n---\n\nYOUR ROLE IN THIS TURN\n\n{role}"


def clear_cache() -> None:
    """Call after editing a prompt file while a Jupyter kernel is running."""
    load.cache_clear()
