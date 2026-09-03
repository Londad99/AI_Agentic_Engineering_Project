"""Routes a student message to the right agent.

Defined by what it does NOT do: it never answers from the document itself. Every
grounding guarantee - verbatim quotes, criterion-based grading, page citations - lives
inside the specialists, so an orchestrator that answered directly would be an unchecked
fourth agent whose output nobody could tell apart from the verified ones.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

from . import agents, grounding, llm, prompts
from .session import StudySession

INTENTS = Literal[
    "map_topics", "summarize_topic", "generate_exam", "grade_answer", "progress", "out_of_scope"
]


class Route(BaseModel):
    intent: INTENTS
    topic: Optional[str] = Field(default=None, description="topic the student named, if any")
    n_questions: Optional[int] = Field(default=None, description="how many questions they asked for")


def resolve_topic(name: str | None, topic_map):
    """Match a model-produced topic name against the real map.

    Fuzzy ('lo ambiental' -> 'Impacto ambiental y manejo del pastoreo'). Returns None
    rather than guessing: a wrong match examines the student on the wrong material.
    """
    if not name or topic_map is None:
        return None
    best, best_score = None, 0.0
    for topic in topic_map.topics:
        haystack = topic.name + " " + " ".join(topic.subtopics)
        score = grounding.match_ratio(name, haystack)
        if score > best_score:
            best, best_score = topic, score
    return best if best_score >= 0.5 else None


def route(message: str, session: StudySession) -> Route:
    topics = [t.name for t in session.topic_map.topics] if session.topic_map else []
    pending = session.current_question
    context = (
        f"Topics available: {topics}\n"
        f"Open question: {pending.question if pending else 'none'}\n\n"
        f"Conversation so far:\n{session.conversation.as_context() or '(new session)'}\n\n"
        f"Student message:\n{message}"
    )
    response = llm.generate(
        prompt=context,
        system=prompts.system(prompts.load("router")),
        schema=Route,
        temperature=0.0,  # routing is not creative work
    )
    return response.parse(Route)


def summarize_transcript(transcript: str) -> str:
    return llm.generate(
        prompt=transcript, system=prompts.load("summarizer"), temperature=0.0
    ).text


def handle(message: str, session: StudySession, show_route: bool = True) -> str:
    """One turn: route, delegate, update memory. All judgement happened upstream."""
    decision = route(message, session)
    session.conversation.add("student", message)
    lines = [f"[route: {decision.intent}]"] if show_route else []

    if decision.intent == "map_topics":
        if session.topic_map is None:
            session.topic_map = agents.extract_topics()
        lines += [f"- {t.name}: {t.summary}" for t in session.topic_map.topics]

    elif decision.intent in ("summarize_topic", "generate_exam"):
        topic = resolve_topic(decision.topic, session.topic_map)
        if topic is None:
            names = [t.name for t in session.topic_map.topics] if session.topic_map else []
            lines.append(f"Which topic did you mean? Available: {names}")
        elif decision.intent == "summarize_topic":
            lines.append(f"{topic.name} (p. {sorted(set(topic.source_pages))})")
            lines.append(topic.summary)
            lines += [f"  - {s}" for s in topic.subtopics]
        else:
            exam, hits = agents.generate_exam(topic, n_questions=decision.n_questions or 3)
            checks = agents.verify_exam(exam, hits)
            # Only verified questions reach the student.
            exam.questions = [q for q, c in zip(exam.questions, checks) if c["grounded"]]
            dropped = len(checks) - len(exam.questions)
            session.current_exam, session.current_index = exam, 0
            if dropped:
                lines.append(f"({dropped} question(s) dropped: quote not found in the document)")
            if not exam.questions:
                lines.append("No question could be traced to the document. Try another topic.")
            else:
                lines.append(f"Question 1/{len(exam.questions)}: {session.current_question.question}")

    elif decision.intent == "grade_answer":
        question = session.current_question
        if question is None:
            lines.append("There is no open question. Ask me for a practice exam first.")
        else:
            result = agents.grade_answer(question, message)
            session.record_score(session.current_exam.topic, result["score"])
            session.advance()
            lines.append(agents.format_feedback(result))
            nxt = session.current_question
            if nxt:
                n = session.current_index + 1
                lines.append(f"\nQuestion {n}/{len(session.current_exam.questions)}: {nxt.question}")
            else:
                lines.append("\nThat was the last question of this topic.")

    elif decision.intent == "progress":
        if not session.scores:
            lines.append("You have not answered any questions yet.")
        else:
            lines += [f"  {avg:.0%}  {name}" for name, avg in session.weakest_topics()]
            lines.append("Weakest topic first - start your revision there.")

    else:
        lines.append("I only work from the document you uploaded. Ask me about it.")

    reply = "\n".join(lines)
    session.conversation.add("tutor", reply)
    session.conversation.compact(summarize_transcript)
    return reply
