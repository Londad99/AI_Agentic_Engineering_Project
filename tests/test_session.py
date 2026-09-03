"""Tests for conversation memory and study state.

The important one is test_structured_state_is_not_in_the_transcript. The whole reason
a sliding window is acceptable here is that the exam in progress does not live in the
conversation. If that ever stops being true, the window starts eating real state and
the design silently breaks.

Run:  python tests/test_session.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tutor.session import Conversation, StudySession  # noqa: E402


def fake_summarizer(transcript: str) -> str:
    return f"[summary of {len(transcript.splitlines())} lines]"


def test_window_keeps_recent_turns_verbatim():
    conversation = Conversation(keep_recent=4)
    for i in range(20):
        conversation.add("user", f"message {i}")
    conversation.compact(fake_summarizer)
    assert len(conversation.turns) == 4, conversation.turns
    assert conversation.turns[-1]["content"] == "message 19"
    assert "message 19" in conversation.as_context()
    print("window                OK")


def test_no_compaction_before_the_threshold():
    conversation = Conversation(keep_recent=4)
    for i in range(5):
        conversation.add("user", f"m{i}")
    conversation.compact(fake_summarizer)
    assert conversation.summary == "", "summarised early, paying for a call it did not need"
    assert len(conversation.turns) == 5
    print("no early compaction   OK")


def test_previous_summary_is_carried_forward():
    """Turn 1 must survive repeated compactions, not be summarised away in stages."""
    seen = []

    def summarizer(transcript):
        seen.append(transcript)
        return "SUMMARY"

    conversation = Conversation(keep_recent=2)
    for i in range(6):
        conversation.add("user", f"m{i}")
    conversation.compact(summarizer)
    for i in range(6, 12):
        conversation.add("user", f"m{i}")
    conversation.compact(summarizer)

    assert "Summary so far:" in seen[-1], "the old summary was dropped instead of folded in"
    print("summary carried       OK")


def test_structured_state_is_not_in_the_transcript():
    class FakeExam:
        questions = ["q1", "q2"]

    session = StudySession(current_exam=FakeExam())
    session.conversation.add("user", "hola")
    assert "FakeExam" not in session.conversation.as_context()
    assert session.current_question == "q1", "state must come from the object, not the text"
    print("state out of context  OK")


def test_scores_and_weakest_topics():
    session = StudySession()
    session.record_score("Impacto ambiental", 0.2)
    session.record_score("Impacto ambiental", 0.4)
    session.record_score("Comportamiento social", 1.0)
    weakest = session.weakest_topics()
    assert weakest[0][0] == "Impacto ambiental", weakest
    assert abs(weakest[0][1] - 0.3) < 1e-9, weakest
    print("weakest topics        OK")


def test_advance_past_the_end_is_not_a_crash():
    class FakeExam:
        questions = ["only one"]

    session = StudySession(current_exam=FakeExam())
    session.advance()
    assert session.current_question is None, "running off the end must be a None, not an IndexError"
    print("end of exam           OK")


if __name__ == "__main__":
    test_window_keeps_recent_turns_verbatim()
    test_no_compaction_before_the_threshold()
    test_previous_summary_is_carried_forward()
    test_structured_state_is_not_in_the_transcript()
    test_scores_and_weakest_topics()
    test_advance_past_the_end_is_not_a_crash()
    print("\nall session tests passed")
