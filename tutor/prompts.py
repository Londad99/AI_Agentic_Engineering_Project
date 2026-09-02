"""Prompts live in files, not in string literals scattered through the code.

Three reasons, all of which show up in this project:

1. The shared instructions (scope, refusal behaviour, language, citation style) are
   identical for every agent. Duplicating them means fixing a refusal rule in four
   places and missing one.
2. A prompt is content, not logic. Editing prompts/*.txt to change the tutor's
   behaviour needs no Python and produces a readable git diff.
3. The Week 2 checkpoint asks for prompts/system_prompt.txt in the repository. It is
   the same file the assistant actually runs on, not a copy that drifts out of date.
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
