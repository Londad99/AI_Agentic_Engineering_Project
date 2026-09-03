"""Dispatch tests for the orchestrator. No API calls: the LLM and the vector store
are replaced with fakes, so what is under test is the wiring.

The two that matter:
  - an ungrounded question never reaches the student, even when the model produced it;
  - a bare answer with no keywords is routed to the grader because a question is open.

Run:  python tests/test_orchestrator.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tutor import agents, orchestrator, progress  # noqa: E402
from tutor.session import StudySession  # noqa: E402

progress.VERBOSE = False

PASSAGE = "Las vacas establecen jerarquias sociales dentro de sus grupos y se reconocen entre si."
HITS = [{"text": PASSAGE, "metadata": {"page": 3, "source": "notes.pdf", "chunk_index": 0}}]

TOPIC = agents.Topic(
    name="Comportamiento social",
    summary="Como se organizan los grupos.",
    subtopics=["jerarquias", "reconocimiento"],
    source_pages=[3],
)
TOPIC_MAP = agents.TopicMap(topics=[TOPIC])


class FakeResponse:
    def __init__(self, obj):
        self.obj = obj
        self.text = "fake"
        self.provider = "fake"

    def parse(self, schema):
        return self.obj


def question(quote: str, page: int = 3) -> agents.ExamQuestion:
    return agents.ExamQuestion(
        question="¿Como se organizan los grupos de vacas?",
        expected_answer="Con jerarquias sociales.",
        key_points=["jerarquias sociales", "se reconocen entre si"],
        difficulty="understanding",
        source_page=page,
        source_quote=quote,
    )


def install(route_obj=None, exam=None, feedback=None):
    """Point every LLM call at canned objects and the vector store at one passage."""
    def fake_generate(prompt=None, system=None, schema=None, **kwargs):
        if schema is orchestrator.Route:
            return FakeResponse(route_obj)
        if schema is agents.Exam:
            return FakeResponse(exam)
        if schema is agents.RawFeedback:
            return FakeResponse(feedback)
        if schema is agents.TopicMap:
            return FakeResponse(TOPIC_MAP)
        return FakeResponse(None)

    orchestrator.llm.generate = fake_generate
    agents.llm.generate = fake_generate
    agents.search = lambda *a, **k: HITS
    agents.list_all_chunks = lambda *a, **k: HITS


def test_ungrounded_questions_never_reach_the_student():
    good = question(PASSAGE[:60])                      # verbatim
    invented = question("Las vacas producen doce litros de leche al dia.")
    install(
        route_obj=orchestrator.Route(intent="generate_exam", topic="Comportamiento social"),
        exam=agents.Exam(topic="Comportamiento social", questions=[good, invented]),
    )
    session = StudySession(topic_map=TOPIC_MAP)
    reply = orchestrator.handle("hazme preguntas", session)

    assert len(session.current_exam.questions) == 1, "an invented question survived"
    assert "1 question(s) dropped" in reply, "the drop was not reported"
    print("ungrounded dropped    OK")


def test_bare_answer_routes_to_the_grader():
    """No keywords, no verb 'califica'. The open question is what makes it an answer."""
    install(
        route_obj=orchestrator.Route(intent="grade_answer"),
        feedback=agents.RawFeedback(
            points=[
                agents.PointVerdict(key_point="jerarquias sociales", status="covered", comment="ok"),
                agents.PointVerdict(key_point="se reconocen entre si", status="missing", comment="no"),
            ],
            what_was_right="Mencionó las jerarquías.",
            what_to_review="El reconocimiento individual.",
            misconceptions=[],
            source_pages=[3],
        ),
    )
    session = StudySession(topic_map=TOPIC_MAP)
    session.current_exam = agents.Exam(topic="Comportamiento social", questions=[question(PASSAGE[:60])])
    reply = orchestrator.handle("forman jerarquias", session)

    assert "PARTIALLY CORRECT" in reply and "(50%)" in reply, reply
    assert session.scores["Comportamiento social"] == [0.5]
    assert session.current_question is None, "did not advance past the last question"
    print("bare answer graded    OK")


def test_grading_with_no_open_question_does_not_crash():
    install(route_obj=orchestrator.Route(intent="grade_answer"))
    reply = orchestrator.handle("mi respuesta", StudySession(topic_map=TOPIC_MAP))
    assert "no open question" in reply.lower()
    print("no open question      OK")


def test_unknown_topic_asks_instead_of_guessing():
    install(route_obj=orchestrator.Route(intent="summarize_topic", topic="termodinamica cuantica"))
    reply = orchestrator.handle("resume termodinamica cuantica", StudySession(topic_map=TOPIC_MAP))
    assert "Which topic did you mean?" in reply, "guessed a topic instead of asking"
    print("unknown topic asks    OK")


def test_out_of_scope_is_refused():
    install(route_obj=orchestrator.Route(intent="out_of_scope"))
    reply = orchestrator.handle("¿capital de Francia?", StudySession(topic_map=TOPIC_MAP))
    assert "only work from the document" in reply
    print("out of scope refused  OK")


def test_empty_answer_costs_no_api_call():
    calls = []
    agents.llm.generate = lambda **k: calls.append(1)
    result = agents.grade_answer(question(PASSAGE[:60]), "   ")
    assert result["score"] == 0.0 and not calls, "spent quota to grade an empty answer"
    print("empty answer          OK")


if __name__ == "__main__":
    test_ungrounded_questions_never_reach_the_student()
    test_bare_answer_routes_to_the_grader()
    test_grading_with_no_open_question_does_not_crash()
    test_unknown_topic_asks_instead_of_guessing()
    test_out_of_scope_is_refused()
    test_empty_answer_costs_no_api_call()
    print("\nall orchestrator tests passed")
