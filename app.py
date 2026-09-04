"""Streamlit front end over tutor/. No agent logic lives here - this file only
renders state and calls the same functions the notebook and scripts/chat.py use.

Streamlit re-executes this entire script on every interaction. Two consequences drive
the whole structure:

  * anything expensive (the ChromaDB client) must be cached with @st.cache_resource,
    or every keystroke reopens the store;
  * anything that must survive between clicks (the StudySession, the topic map, the
    open exam) must live in st.session_state, not in a local variable.

Get either wrong and the app silently re-ingests the PDF or forgets which question was
open - the two failure modes that make a Streamlit demo fall apart on stage.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tutor import agents, config, library, llm, orchestrator, progress  # noqa: E402
from tutor.errors import TutorError  # noqa: E402
from tutor.ingest import ingest  # noqa: E402
from tutor.session import StudySession  # noqa: E402
from tutor.vectorstore import get_collection, list_sources, remove_source  # noqa: E402

st.set_page_config(page_title="Study Tutor", layout="wide")

# Restrained styling: tighter type for dense text, monospace for anything the system
# reports about itself. No decoration that does not carry information.
st.markdown(
    """
    <style>
      .block-container { padding-top: 2.2rem; max-width: 1300px; }
      section[data-testid="stSidebar"] { font-size: 0.86rem; }
      code, pre, .stCode { font-size: 0.80rem; }
      .quote { border-left: 2px solid #6b7280; padding-left: .7rem; color: #6b7280;
               font-size: .82rem; margin: .3rem 0 .6rem 0; }
      .meta  { color: #6b7280; font-size: .78rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# State
# --------------------------------------------------------------------------- #


@st.cache_resource
def collection():
    """One ChromaDB handle for the process, not one per rerun."""
    return get_collection()


def state() -> StudySession:
    if "session" not in st.session_state:
        st.session_state.session = StudySession()
        st.session_state.chat = []          # (speaker, text) pairs
        st.session_state.exam_report = []   # verification report for the open exam
    return st.session_state.session


session = state()


# Streamlit drops writes from threads without its script context. The heartbeat in
# progress.step() runs on one, so its ticks would silently vanish from the page.
try:
    from streamlit.runtime.scriptrunner import add_script_run_ctx

    progress.THREAD_HOOK = add_script_run_ctx
except ImportError:  # older Streamlit: the heartbeat just stays in the terminal
    pass


def run(label: str, function, *args, **kwargs):
    """Call an agent, streaming its progress to the page and catching provider errors.

    The log is the point. A spinner says "wait"; these lines say what is being waited
    on - which passage is being embedded, that Gemini answered 503 and is being retried,
    how many seconds have passed. On a call that can take a minute, that is the
    difference between "it is working" and "it is frozen".

    A quota error must not blank the page either: the student keeps their topics, their
    open exam and their scores, and can fix .env and carry on.
    """
    with st.status(label, expanded=True) as box:
        log: list[str] = []
        # One placeholder, rewritten in place. Calling box.code() per line APPENDS a new
        # block each time, which is how the log ended up printed once per update.
        view = box.empty()

        def sink(line: str) -> None:
            log.append(line.rstrip())
            view.code("\n".join(log[-14:]), language=None)

        try:
            with progress.capture(sink):
                result = function(*args, **kwargs)
        except TutorError as error:
            box.update(label=f"{label} - failed", state="error")
            st.error(str(error))
            return None
        except Exception as error:  # noqa: BLE001
            box.update(label=f"{label} - failed", state="error")
            st.error(f"{type(error).__name__}: {error}")
            return None

        box.update(label=label, state="complete", expanded=False)
        return result


# --------------------------------------------------------------------------- #
# Sidebar: document and runtime state
# --------------------------------------------------------------------------- #

with st.sidebar:
    st.markdown("### Document")

    chunks_stored = collection().count()
    indexed = list_sources(collection())

    if indexed:
        for name, count in indexed.items():
            row, action = st.columns([4, 1])
            row.markdown(f"<span class='meta'>{name}<br>{count} chunks</span>",
                         unsafe_allow_html=True)
            if action.button("✕", key=f"drop_{name}", help=f"Remove {name} from the index"):
                removed = remove_source(name, collection())
                # Both halves, or the next ingest silently puts the document back.
                try:
                    library.archive(name)
                except FileNotFoundError:
                    pass   # already gone from data/; the index entries are what mattered
                # Topics, exam and scores were derived from a document that is now gone.
                st.session_state.session = StudySession()
                st.session_state.chat = []
                st.session_state.exam_report = []
                collection.clear()
                st.toast(f"{name}: {removed} chunks removed, file moved to data/_removed/")
                st.rerun()
    else:
        st.markdown("<span class='meta'>nothing indexed</span>", unsafe_allow_html=True)

    orphans = [p.name for p in library.list_pdfs() if p.name not in indexed]
    if orphans:
        st.markdown(f"<span class='meta'>in data/ but not indexed: {', '.join(orphans)}</span>",
                    unsafe_allow_html=True)
        if st.button("Index them", use_container_width=True):
            if run("Reading", ingest):
                collection.clear()
                st.rerun()

    upload = st.file_uploader("Add a PDF", type="pdf", label_visibility="collapsed")
    if upload is not None and st.button("Ingest", use_container_width=True):
        target = config.DATA_DIR / upload.name
        target.write_bytes(upload.getbuffer())
        stats = run(f"Reading {upload.name}", ingest, target)
        if stats:
            collection.clear()
            # A new document invalidates the topic map and any exam drawn from the old one.
            st.session_state.session = StudySession()
            st.session_state.chat = []
            st.success(f"{stats['pages']} pages, {stats['chunks']} chunks")
            st.rerun()

    st.divider()
    st.markdown("### Runtime")
    st.markdown(
        f"<span class='meta'>provider &nbsp; <code>{config.LLM_PROVIDER}</code><br>"
        f"chat &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <code>{config.active_chat_model()}</code><br>"
        f"embed &nbsp;&nbsp;&nbsp;&nbsp; <code>{config.active_embed_model()}</code></span>",
        unsafe_allow_html=True,
    )
    if config.LLM_PROVIDER == "ollama" or config.ENABLE_LOCAL_FALLBACK:
        installed = llm.ollama_models()
        if installed is None:
            st.markdown("<span class='meta'>ollama &nbsp;&nbsp;&nbsp;&nbsp; not answering</span>",
                        unsafe_allow_html=True)
        elif config.OLLAMA_MODEL in installed:
            st.markdown("<span class='meta'>ollama &nbsp;&nbsp;&nbsp;&nbsp; ready</span>",
                        unsafe_allow_html=True)
        else:
            # Running but without the configured tag: the common case, and the one the
            # old "unreachable" message sent people to debug in the wrong place.
            st.markdown(
                f"<span class='meta'>ollama &nbsp;&nbsp;&nbsp;&nbsp; running, but no "
                f"<code>{config.OLLAMA_MODEL}</code><br>installed: "
                f"{', '.join(installed) or '(none)'}</span>",
                unsafe_allow_html=True,
            )

    if session.scores:
        st.divider()
        st.markdown("### Weakest topics")
        for name, average in session.weakest_topics():
            st.markdown(f"<span class='meta'>{average:.0%} &nbsp; {name}</span>",
                        unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

st.markdown("## Study Tutor")
st.markdown(
    "<span class='meta'>Answers come only from the indexed document. Every exam "
    "question carries a quote checked against its source page.</span>",
    unsafe_allow_html=True,
)

if chunks_stored == 0:
    st.info("Add a PDF in the sidebar to begin.")
    st.stop()

topics_tab, exam_tab, chat_tab = st.tabs(["Topics", "Practice exam", "Chat"])


# ---- Topics ---------------------------------------------------------------- #

with topics_tab:
    if session.topic_map is None:
        st.markdown(
            "<span class='meta'>The extractor reads every chunk rather than retrieving: "
            "mapping a document needs coverage, and retrieval is built to discard.</span>",
            unsafe_allow_html=True,
        )
        if st.button("Extract topics"):
            topic_map = run("Reading the whole document", agents.extract_topics)
            if topic_map:
                session.topic_map = topic_map
                st.rerun()
    else:
        for topic in session.topic_map.topics:
            pages = ", ".join(str(p) for p in sorted(set(topic.source_pages)))
            with st.expander(f"{topic.name}  ·  p. {pages}"):
                st.write(topic.summary)
                for subtopic in topic.subtopics:
                    st.markdown(f"- {subtopic}")
        if st.button("Re-extract"):
            session.topic_map = None
            st.rerun()


# ---- Practice exam --------------------------------------------------------- #

with exam_tab:
    if session.topic_map is None:
        st.info("Extract the topics first.")
    else:
        names = [t.name for t in session.topic_map.topics]
        left, middle, right = st.columns([3, 1, 1])
        chosen = left.selectbox("Topic", names, label_visibility="collapsed")
        count = middle.number_input("Questions", 1, 8, 3, label_visibility="collapsed")
        generate = right.button("Generate", use_container_width=True)

        if generate:
            topic = next(t for t in session.topic_map.topics if t.name == chosen)
            produced = run(f"Retrieving passages on {topic.name}", agents.generate_exam, topic, count)
            if produced:
                exam, hits = produced
                report = agents.verify_exam(exam, hits)
                # Questions whose quote is not in the source never reach the student.
                keep = [(q, c) for q, c in zip(exam.questions, report) if c["grounded"]]
                dropped = len(report) - len(keep)
                exam.questions = [q for q, _ in keep]
                session.current_exam, session.current_index = exam, 0
                st.session_state.exam_report = [c for _, c in keep]
                # Widget and result keys are indexed by position, so leftovers from the
                # previous exam would surface under the new questions. Clear them.
                for key in [k for k in st.session_state
                            if k.startswith(("answer_", "submit_", "result_"))]:
                    del st.session_state[key]
                if dropped:
                    st.warning(f"{dropped} question(s) discarded: quote not found in the document.")
                st.rerun()

        exam = session.current_exam
        if exam is None:
            st.markdown("<span class='meta'>No exam yet.</span>", unsafe_allow_html=True)
        elif not exam.questions:
            st.warning("No question could be traced to the document. Try another topic.")
        else:
            for index, question in enumerate(exam.questions):
                check = st.session_state.exam_report[index]
                answered = index < session.current_index
                header = f"{index + 1}. {question.question}"
                # Open the pending question, and any question whose feedback just arrived.
                just_graded = st.session_state.get(f"result_{index}") is not None
                with st.expander(header, expanded=index == session.current_index or just_graded):
                    st.markdown(
                        f"<span class='meta'>{question.difficulty} · page {question.source_page} · "
                        f"quote verified {check['score']:.0%}</span>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(f"<div class='quote'>{question.source_quote}</div>",
                                unsafe_allow_html=True)

                    if not answered:
                        answer = st.text_area("Your answer", key=f"answer_{index}",
                                              label_visibility="collapsed", height=110)
                        if st.button("Submit", key=f"submit_{index}"):
                            result = run("Grading", agents.grade_answer, question, answer)
                            if result:
                                session.record_score(exam.topic, result["score"])
                                session.advance()
                                st.session_state[f"result_{index}"] = result
                                st.rerun()

                    # Rendered whether or not the question is answered. Skipping this for
                    # answered ones is what made grading look like it produced nothing:
                    # advance() flips `answered` before the rerun that would have drawn it.
                    result = st.session_state.get(f"result_{index}")
                    if result:
                        feedback = result["feedback"]
                        st.markdown(
                            f"**{result['verdict'].replace('_', ' ')}** · {result['score']:.0%}"
                        )
                        for point in feedback.points:
                            mark = {"covered": "✓", "partial": "~", "missing": "✗"}[point.status]
                            st.markdown(
                                f"{mark} **{point.key_point}** — <span class='meta'>"
                                f"{point.comment}</span>",
                                unsafe_allow_html=True,
                            )
                        if feedback.what_was_right:
                            st.markdown(f"<span class='meta'>Right: {feedback.what_was_right}"
                                        f"</span>", unsafe_allow_html=True)
                        if feedback.misconceptions:
                            for item in feedback.misconceptions:
                                st.markdown(f"<span class='meta'>Incorrect: {item}</span>",
                                            unsafe_allow_html=True)
                        pages = ", ".join(str(x) for x in sorted(set(feedback.source_pages)))
                        st.markdown(f"<span class='meta'>Review: {feedback.what_to_review} "
                                    f"(p. {pages})</span>", unsafe_allow_html=True)
                        st.markdown(f"<div class='quote'>Your answer: "
                                    f"{st.session_state.get(f'answer_{index}', '')}</div>",
                                    unsafe_allow_html=True)
                    elif answered:
                        st.markdown("<span class='meta'>answered</span>", unsafe_allow_html=True)


# ---- Chat ------------------------------------------------------------------ #

with chat_tab:
    st.markdown(
        "<span class='meta'>The orchestrator classifies each message and delegates. "
        "With a question open, a plain answer is routed to the grader.</span>",
        unsafe_allow_html=True,
    )
    for speaker, text in st.session_state.chat:
        with st.chat_message(speaker):
            st.markdown(text)

    message = st.chat_input("Ask about the document, or answer the open question")
    if message:
        st.session_state.chat.append(("user", message))
        reply = run("Thinking", orchestrator.handle, message, session, False)
        st.session_state.chat.append(("assistant", reply or "_(the call failed - see above)_"))
        st.rerun()
