"""Conversation memory and study state, kept deliberately separate.

Most of what this tutor must remember is not conversation: the open exam, the pending
question, per-topic scores. That is structured state and lives in typed objects.

Keeping it out of the transcript is what makes a sliding window safe. The usual
objection - that a window silently drops something important - only applies when the
transcript is the only place a fact lives. Here the window can forget turn 1 without
forgetting the exam.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

# Turns kept verbatim. Everything older is compacted into the summary.
KEEP_RECENT = 6


@dataclass
class Conversation:
    """Sliding window over recent turns, plus a rolling summary of everything older."""

    turns: list[dict] = field(default_factory=list)
    summary: str = ""
    keep_recent: int = KEEP_RECENT

    def add(self, role: str, content: str) -> None:
        self.turns.append({"role": role, "content": content})

    def needs_compaction(self) -> bool:
        # Compact at twice the window so summarisation is occasional rather than
        # every single turn: an extra LLM call per message would double the cost of
        # the conversation to save context we have not run out of yet.
        return len(self.turns) > self.keep_recent * 2

    def compact(self, summarize: Callable[[str], str]) -> None:
        """Fold the older turns into the summary, keeping the recent ones verbatim."""
        if not self.needs_compaction():
            return
        older, recent = self.turns[: -self.keep_recent], self.turns[-self.keep_recent :]
        transcript = "\n".join(f"{turn['role']}: {turn['content']}" for turn in older)
        # The previous summary is fed back in, so facts from turn 1 survive any number
        # of compactions instead of being summarised away one generation at a time.
        if self.summary:
            transcript = f"Summary so far:\n{self.summary}\n\nNewer turns:\n{transcript}"
        self.summary = summarize(transcript)
        self.turns = recent

    def as_context(self) -> str:
        parts = []
        if self.summary:
            parts.append(f"Earlier in this session:\n{self.summary}")
        if self.turns:
            parts.append(
                "Recent turns:\n"
                + "\n".join(f"{turn['role']}: {turn['content']}" for turn in self.turns)
            )
        return "\n\n".join(parts)


@dataclass
class StudySession:
    """Structured state. Deliberately never serialised into the transcript."""

    topic_map: object | None = None        # TopicMap, once extracted
    current_exam: object | None = None     # Exam, once generated
    current_index: int = 0                 # which question is pending
    scores: dict[str, list[float]] = field(default_factory=dict)  # topic -> scores
    conversation: Conversation = field(default_factory=Conversation)

    @property
    def current_question(self):
        exam = self.current_exam
        if exam is None or self.current_index >= len(exam.questions):
            return None
        return exam.questions[self.current_index]

    def record_score(self, topic: str, score: float) -> None:
        self.scores.setdefault(topic, []).append(score)

    def advance(self) -> None:
        self.current_index += 1

    def weakest_topics(self, limit: int = 3) -> list[tuple[str, float]]:
        """Lowest average score first - what the student should revise.

        This is the payoff of keeping scores as numbers rather than as sentences in a
        transcript: it is a sort, not a question for the model.
        """
        averages = [(topic, sum(s) / len(s)) for topic, s in self.scores.items() if s]
        return sorted(averages, key=lambda pair: pair[1])[:limit]
