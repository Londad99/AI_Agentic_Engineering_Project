"""The three specialist agents: topic extraction, exam generation, grading.

Each is a prompt (in prompts/), a Pydantic output schema, and its own retrieval
strategy. They share nothing but tutor.llm.generate() and the persona in
prompts/system_prompt.txt.

Where RAG is used, and where it is not:
  Topic Extractor - NO retrieval. Needs coverage of the whole document; similarity
                    search is built to discard most of it. Opposite goals.
  Answerer        - retrieval on the question, with a score floor: below it the
                    document does not cover the question and we say so.
  Exam Generator  - retrieval on the topic name plus its subtopics.
  Grader          - retrieval on the question AND the student's answer, so unsupported
                    claims can be identified as such.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

from . import grounding, llm, prompts, scoring
from .progress import status
from .vectorstore import list_all_chunks, search

BATCH_CHARS = 20_000   # fits the context window with room for the answer
RETRIEVAL_K = 6


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #


class Topic(BaseModel):
    name: str = Field(description="short topic name, in the language of the document")
    summary: str = Field(description="one or two sentences on what the document says about it")
    subtopics: list[str] = Field(description="specific ideas under this topic, 2 to 6 of them")
    source_pages: list[int] = Field(description="page numbers where this topic is discussed")


class TopicMap(BaseModel):
    topics: list[Topic]


class ExamQuestion(BaseModel):
    question: str = Field(description="the question, in the language of the document")
    expected_answer: str = Field(description="a complete correct answer, 2-4 sentences")
    key_points: list[str] = Field(description="2-4 ideas a correct answer must contain")
    difficulty: Literal["recall", "understanding", "application"]
    source_page: int = Field(description="page this question comes from")
    source_quote: str = Field(
        description="one sentence copied VERBATIM from the passage that supports the answer"
    )


class Exam(BaseModel):
    topic: str
    questions: list[ExamQuestion]


class PointVerdict(BaseModel):
    key_point: str = Field(description="the criterion being judged, copied as given")
    status: Literal["covered", "partial", "missing"]
    comment: str = Field(description="one sentence: what the student said about this point")


class RawFeedback(BaseModel):
    """What the model returns. No score field: that is derived in code."""

    points: list[PointVerdict]
    what_was_right: str = Field(description="what the student did get right; empty if nothing")
    what_to_review: str = Field(description="the specific idea to go back and study")
    misconceptions: list[str] = Field(
        description="claims the student made that the document contradicts or never states"
    )
    source_pages: list[int] = Field(description="pages where the missing material is explained")


# --------------------------------------------------------------------------- #
# Agent 1 - Topic Extractor (no RAG: needs coverage)
# --------------------------------------------------------------------------- #


def _batch_by_size(chunks, max_chars=BATCH_CHARS):
    """Group consecutive chunks into prompts that fit; order preserved.

    Sized in characters, not chunk count: the limit that matters is the context
    window, and 20 dense chunks are a very different prompt from 20 sparse ones.
    """
    batch, size = [], 0
    for chunk in chunks:
        if batch and size + len(chunk["text"]) > max_chars:
            yield batch
            batch, size = [], 0
        batch.append(chunk)
        size += len(chunk["text"])
    if batch:
        yield batch


def as_passages(chunks) -> str:
    """Label each passage with its page - the only honest source for source_pages."""
    return "\n\n".join(f"[page {c['metadata']['page']}] {c['text']}" for c in chunks)


def extract_topics(chunks=None) -> TopicMap:
    """Map over the whole document, then reduce the per-batch topic lists into one."""
    chunks = chunks if chunks is not None else list_all_chunks()
    if not chunks:
        raise ValueError("No chunks in the vector store - ingest a PDF first.")

    batches = list(_batch_by_size(chunks))
    status(f"  {len(chunks)} chunks -> {len(batches)} batch(es)")

    partials = []
    for number, batch in enumerate(batches, start=1):
        status(f"  map {number}/{len(batches)}")
        response = llm.generate(
            prompt=as_passages(batch),
            system=prompts.system(prompts.load("topic_extractor")),
            schema=TopicMap,
        )
        partials.append(response.parse(TopicMap))

    if len(partials) == 1:
        return partials[0]   # nothing to merge; a reduce call could only paraphrase

    status("  reduce")
    listing = "\n\n".join(
        f"--- list {i} ---\n{partial.model_dump_json(indent=1)}"
        for i, partial in enumerate(partials, start=1)
    )
    merged = llm.generate(
        prompt=listing, system=prompts.system(prompts.load("merge_topics")), schema=TopicMap
    )
    return merged.parse(TopicMap)


# --------------------------------------------------------------------------- #
# Agent 2 - Answerer (RAG on the question)
# --------------------------------------------------------------------------- #

# Below this similarity, the retrieved passages are not about the question. Tuned
# against observed scores: a real hit on this corpus lands at 0.70-0.85, and unrelated
# text sits well under 0.40.
ANSWER_THRESHOLD = 0.45


class GroundedAnswer(BaseModel):
    answered: bool = Field(description="false if the passages do not contain the answer")
    answer: str = Field(description="the answer, empty when answered is false")
    missing: str = Field(description="what the document does not cover, when answered is false")
    source_pages: list[int] = Field(description="pages the answer comes from")


def answer_question(question: str, top_k: int = RETRIEVAL_K,
                    source: str | None = None) -> tuple[GroundedAnswer, list[dict]]:
    """Answer from the document, or say plainly that the document does not cover it.

    Declining is a feature, not a fallback. A student revising for an exam is harmed
    more by a confident answer their professor never taught than by "this is not in
    your notes".

    Two independent gates, because either alone is weak:

      1. Retrieval score. If nothing clears ANSWER_THRESHOLD, the passages are not about
         the question and we decline WITHOUT calling the model - no quota spent to be
         told what the numbers already said.
      2. The model's own judgement. Passages can score well and still not contain the
         answer, so the schema forces an explicit answered=true/false rather than
         letting a hedge ("the document suggests...") pass for an answer.
    """
    hits = search(question, top_k=top_k, source=source)
    best = max((h["score"] for h in hits), default=0.0)

    if not hits or best < ANSWER_THRESHOLD:
        status(f"  no passage above {ANSWER_THRESHOLD:.2f} (best {best:.2f}) - declining")
        scope = f" in {source}" if source else ""
        return (
            GroundedAnswer(
                answered=False,
                answer="",
                missing=f"Nothing{scope} is close to this question.",
                source_pages=[],
            ),
            hits,
        )

    response = llm.generate(
        prompt=f"QUESTION:\n{question}\n\nPASSAGES:\n{as_passages(hits)}",
        system=prompts.system(prompts.load("answerer")),
        schema=GroundedAnswer,
        temperature=0.1,
    )
    return response.parse(GroundedAnswer), hits


def format_answer(answer: GroundedAnswer) -> str:
    if not answer.answered:
        return f"The document does not cover this. {answer.missing}".strip()
    pages = ", ".join(str(p) for p in sorted(set(answer.source_pages)))
    return f"{answer.answer}\n\n(p. {pages})" if pages else answer.answer


# --------------------------------------------------------------------------- #
# Agent 3 - Exam Generator (RAG on the topic)
# --------------------------------------------------------------------------- #


def generate_exam(topic: Topic, n_questions: int = 4, top_k: int = RETRIEVAL_K):
    """RAG over one topic, then structured generation. Returns (Exam, retrieved hits)."""
    # Subtopics widen the query; the name alone retrieves too narrowly.
    query = f"{topic.name}. " + ". ".join(topic.subtopics)
    hits = search(query, top_k=top_k)
    if not hits:
        raise ValueError(f"No passages retrieved for topic {topic.name!r}.")

    prompt = (
        f"Topic: {topic.name}\n"
        f"Subtopics: {', '.join(topic.subtopics)}\n\n"
        f"Write exactly {n_questions} questions from these passages:\n\n{as_passages(hits)}"
    )
    response = llm.generate(
        prompt=prompt,
        system=prompts.system(prompts.load("exam_generator")),
        schema=Exam,
        temperature=0.4,  # variety in phrasing, still faithful
    )
    return response.parse(Exam), hits


def verify_exam(exam: Exam, hits: list[dict]) -> list[dict]:
    """Check every source_quote against the passages it came from.

    The prompt asks for a verbatim quote; asking is not enough. A model under pressure
    to supply one produces something quote-shaped, and it looks convincing.
    """
    sources = [h["text"] for h in hits]
    pages = {h["metadata"]["page"] for h in hits}
    report = []
    for index, question in enumerate(exam.questions, start=1):
        ok, score = grounding.is_grounded(question.source_quote, sources)
        report.append(
            {"n": index, "grounded": ok, "score": score, "page_ok": question.source_page in pages}
        )
    return report


# --------------------------------------------------------------------------- #
# Agent 4 - Grader (RAG on question + answer)
# --------------------------------------------------------------------------- #


def grade_answer(question: ExamQuestion, student_answer: str, top_k: int = 5) -> dict:
    """Grade one answer; returns the judgements plus a score derived in code."""
    if not student_answer.strip():
        return {   # no API call: the verdict is not in doubt
            "feedback": RawFeedback(
                points=[
                    PointVerdict(key_point=p, status="missing", comment="No answer given.")
                    for p in question.key_points
                ],
                what_was_right="",
                what_to_review=question.expected_answer,
                misconceptions=[],
                source_pages=[question.source_page],
            ),
            "score": 0.0,
            "verdict": "incorrect",
        }

    # Both sides steer retrieval: where the answer should come from, and where the
    # student's claims come from - or that they come from nowhere in the document.
    hits = search(f"{question.question} {student_answer}", top_k=top_k)

    prompt = (
        f"QUESTION:\n{question.question}\n\n"
        f"KEY POINTS A CORRECT ANSWER MUST CONTAIN:\n"
        + "\n".join(f"- {point}" for point in question.key_points)
        + f"\n\nREFERENCE ANSWER:\n{question.expected_answer}\n\n"
        f"STUDENT'S ANSWER:\n{student_answer}\n\n"
        f"PASSAGES FROM THEIR DOCUMENT:\n{as_passages(hits)}"
    )
    response = llm.generate(
        prompt=prompt,
        system=prompts.system(prompts.load("grader")),
        schema=RawFeedback,
        temperature=0.0,  # grading must be reproducible
    )
    feedback = response.parse(RawFeedback)
    score = scoring.score_from_points([p.status for p in feedback.points])
    return {"feedback": feedback, "score": score, "verdict": scoring.verdict_from_score(score)}


def format_feedback(result: dict) -> str:
    feedback = result["feedback"]
    lines = [f"{result['verdict'].replace('_', ' ').upper()}  ({result['score']:.0%})", ""]
    for point in feedback.points:
        mark = {"covered": "+", "partial": "~", "missing": "-"}[point.status]
        lines.append(f"  [{mark}] {point.key_point}")
        lines.append(f"      {point.comment}")
    if feedback.what_was_right:
        lines.append(f"\nRight: {feedback.what_was_right}")
    if feedback.misconceptions:
        lines.append("\nIncorrect claims:")
        lines += [f"  - {item}" for item in feedback.misconceptions]
    pages = ", ".join(str(p) for p in sorted(set(feedback.source_pages)))
    lines.append(f"\nReview: {feedback.what_to_review}  (p. {pages})")
    return "\n".join(lines)
