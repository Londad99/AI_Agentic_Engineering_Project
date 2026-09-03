# Study Tutor — Corte 1 Project

**AI Agentic Engineering · Ingeniería de Sistemas · Universidad de Santander**

A conversational study assistant. The student uploads a PDF of their own material —
class notes, slides, a textbook chapter — and the tutor maps its topics, generates
practice exams grounded in the text, and grades written answers point by point.

Built with **context engineering + RAG**, the **Gemini API** and **ChromaDB**, on a
**sub-agent architecture**.

**Start with [`notebooks/tutor.ipynb`](notebooks/tutor.ipynb)** — it walks through the
whole system with its output committed, so it can be read on GitHub without running
anything. The agents themselves live in `tutor/`, so the notebook and the chat program
(`scripts/chat.py`) and the web app (`app.py`) are front ends over the same tested
code.

---

## Setup

### 1. Requirements

Python 3.10 or newer, and a Gemini API key from
[Google AI Studio](https://aistudio.google.com/apikey) (free tier is enough).

### 2. Clone and install

```bash
git clone <this-repo-url>
cd AI_Agentic_Engineering_Project
python -m pip install -r requirements.txt
```

> **On Windows, prefer `python -m <tool>` over the bare command.** Python installers
> frequently leave their `Scripts` folder off the PATH, so `pip` and `streamlit` are
> "not recognized" even though they are installed. `python -m pip` and
> `python -m streamlit` always work if Python itself does. (`py -m ...` if `python`
> is not on the PATH either.)

### 3. Check the install without spending any quota

```bash
python tests/test_ingest.py
python tests/test_grounding.py
```

These make no API calls. If they pass, the environment is fine.

### 4. Configure your API key

Run this once — it creates `.env` from `.env.example`:

```bash
python -c "from tutor import config"
```

Open the `.env` it created and paste your key:

```
GOOGLE_API_KEY=AIza...
```

`.env` is git-ignored, so the key never leaves your machine. `.env.example` is the
template that ships with the repo; it lists the variable names and no values.

### 5. Add your study material

Put one or more PDFs in `data/`. They must have selectable text — try selecting a word
in a PDF reader. A scanned document is images and needs OCR first. 10–40 pages works
well: fewer gives thin exams, more makes the first ingestion slow.

### 6. Check which models your key can reach

```bash
python scripts/bench.py
```

Model availability depends on your key, not only on the docs: some models are closed to
new keys and answer 404. This times every candidate and tells you which respond. Put the
fastest working ones in `.env` as `GEMINI_MODEL` and `GEMINI_EMBED_MODEL`.

### 7. Run it

Two front ends over the same code in `tutor/`.

**As a web app** — the demo interface:

```bash
python -m streamlit run app.py
```

It opens at http://localhost:8501. The sidebar lists each indexed document with its
chunk count, a `✕` to remove it, and an uploader to add more. Three tabs:

- **Topics** — press *Extract topics*; the agent reads the whole document.
- **Practice exam** — pick a topic and a number of questions. Each question shows its
  verified source quote, its page, and the quote match percentage. Type an answer and
  press *Submit* to be graded criterion by criterion.
- **Chat** — free conversation. With a question open, a plain answer is routed to the
  grader.

Removing a document does two things: its chunks are deleted from the index **and** the
PDF is moved to `data/_removed/`. Only doing the first would leave the file where the
next ingestion picks it up again, which looks exactly like the removal not working. The
file is archived rather than deleted — undo is moving it back.

Long operations stream their progress on screen (which passage is being embedded, a
retry after a 503, seconds elapsed), so a slow call is visibly working rather than
frozen.

**Reloading:**

| Changed | What to do |
|---|---|
| `.env` | Nothing. It is re-read before the next call. |
| `app.py` | Streamlit offers *Rerun*; or enable *Always rerun* in its ⋮ menu. |
| anything in `tutor/` | **Restart the process** (`Ctrl+C`, run again). Python caches imported modules, so a rerun alone keeps the old code. |

Restarting costs nothing: the index and the embedding cache live on disk.

**As a chat program:**

```bash
python scripts/chat.py                 # uses whatever is in data/
python scripts/chat.py --pdf notes.pdf # ingest one file first
```

Ask what the document covers, request a summary or a practice exam; when a question is
open, type your answer and it gets graded. `salir` to quit.

**As the notebook** — the same flow with the reasoning written out:

```bash
jupyter notebook notebooks/tutor.ipynb
```

Then **Run All**. Its last section is an interactive loop, so it is not a fixed script.

Other entry points:

```bash
python scripts/ingest.py --reset          # PDFs -> ChromaDB
python scripts/search.py "your question"  # check retrieval only
python scripts/diagnose.py                # time each stage
python scripts/bench.py                   # compare models on your key
python scripts/check_ollama.py            # is the local model actually usable?
```

### Running on a local model

The Gemini free tier allows **20 requests per day, per model**. When it runs out the
429 says "retry in 2.5s" — that is the per-minute `RetryInfo` attached to every 429 and
does not apply; a daily quota resets tomorrow.

Three ways out, in `.env`:

```
GEMINI_MODEL=gemini-3.5-flash-lite   # the quota is per model: another has its own 20
LLM_PROVIDER=ollama                  # local only, no Gemini call attempted
ENABLE_LOCAL_FALLBACK=true           # stay on Gemini, degrade to local on 429
```

For the local options, install [Ollama](https://ollama.com) and pull the models:

```bash
ollama serve
ollama pull qwen3:8b            # chat
ollama pull nomic-embed-text    # only if you also set EMBED_PROVIDER=ollama
```

Then check it, because `OLLAMA_MODEL` must match a tag from `ollama list` **exactly** —
`qwen3:8b`, `qwen3:8b-instruct` and `qwen3:latest` are three different things:

```bash
python scripts/check_ollama.py
```

It reports whether the server answers, lists the installed tags, and sends a test
message.

Moving *embeddings* to Ollama invalidates the vector store — vectors from different
models are not comparable — so rebuild it:

```bash
python scripts/ingest.py --reset
```

The store records which model built it and refuses to search across a mismatch, since
that failure is silent: nothing errors, the results are just quietly wrong.

### Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `503` / `504` from Gemini | The shared model is busy. The retry policy handles it; if it persists, pick another model with `scripts/bench.py`. |
| `404 ... not supported for generateContent` | An embedding model is in `GEMINI_MODEL`. They are different families. |
| `404 ... no longer available to new users` | That model is closed to new keys. Use the replacement its message names. |
| A cell hangs with no output | Run `python scripts/diagnose.py` — it times each stage separately. |
| `429 ... PerDay ... limit: 20` | Daily quota spent for that model. See *Running on a local model* above. |
| Changing `.env` seems to do nothing | It is re-read automatically before each call. In the notebook, re-run the setup cell. |
| `pip` / `streamlit` "not recognized" (Windows) | Their folder is not on the PATH. Use `python -m pip`, `python -m streamlit`. |
| Editing `tutor/*.py` changes nothing | Python caches imported modules. Restart Streamlit, or restart the Jupyter kernel. |
| "Ollama is not answering" but it is running | `OLLAMA_MODEL` does not match a tag in `ollama list`. Run `python scripts/check_ollama.py`. |
| Ingestion says no text found | The PDF is scanned images. It needs OCR. |

---

## Where RAG is used — and where it is not

RAG is not applied uniformly. Each agent asks a different question, and only two of them
are retrieval questions.

| Stage | RAG? | Query | Why |
|---|---|---|---|
| Ingestion | — | — | Deterministic transformation, no LLM at all. |
| **Answerer** | **Yes** | the question itself | Plain Q&A, with a score floor: below it the document does not cover the question and the assistant says so. |
| **Topic Extractor** | **No** | reads every chunk | Needs **coverage**; retrieval is built to discard most of the document. Opposite goals. |
| **Exam Generator** | **Yes** | topic name + subtopics | "Which passages are about this topic?" is exactly a similarity question. |
| **Grader** | **Yes** | question + the student's answer | Finds where the answer should come from *and* where the student's claims come from. |
| Orchestrator | — | — | Routes only; never touches the document. |

Two retrieval details that carry most of the quality:

- **Asymmetric embeddings.** `gemini-embedding-001` embeds the same text differently as
  a stored document (`RETRIEVAL_DOCUMENT`) than as a question (`RETRIEVAL_QUERY`). Using
  one type for both measurably hurts recall.
- **The exam query uses subtopics too.** "Impacto ambiental" alone retrieves too
  narrowly to support four genuinely different questions.

## The sub-agents

```
                      student message
                            |
                     [ ORCHESTRATOR ]   routes, resolves arguments, delegates
                            |           never answers from the document itself
        +-------------------+-------------------+
        |                   |                   |
  Topic Extractor     Exam Generator         Grader
        |                   |                   |
        +-------------------+-------------------+
                            |
              ChromaDB  <-  ingestion (plain code, no LLM)
```

Each agent is a distinct prompt, a distinct Pydantic output schema and its own
retrieval strategy. They all call `tutor.llm.generate()`, so retry, timeout and
fallback policy is written once.

**Orchestrator** — classifies the message into one of six intents using structured
output (a `Literal` enforced during decoding, so an invalid route cannot be returned),
resolves the topic the student named against the real topic map, and delegates. It
never answers from the document: every grounding guarantee lives inside the
specialists, so an orchestrator that answered directly would be an unchecked fourth
agent.

**Answerer** — retrieves passages for the question and answers from them, citing pages.
When nothing clears the similarity floor it declines *without calling the model*, and
even on good passages the schema forces an explicit `answered: true/false` rather than
letting a hedge ("the document suggests…") pass for an answer. Declining is a feature:
a student revising for an exam is harmed more by a confident answer their professor
never taught than by "this is not in your notes".

**Topic Extractor** — reads the whole document in order and returns topics, subtopics
and page numbers. Map-reduce: topics per batch, then one call merging duplicates
across batches.

**Exam Generator** — retrieves passages for one topic and writes open questions, each
with a correct answer, the criteria a correct answer must contain, a difficulty level
(recall / understanding / application) and a verbatim source quote.

**Grader** — marks each criterion `covered` / `partial` / `missing`, names what was
right, what to review and which claims the document contradicts.

### Graceful refusal

Two independent gates, because either alone is weak:

1. **Retrieval score** (`ANSWER_THRESHOLD = 0.45`). Real hits on this corpus land at
   0.70–0.85 and unrelated text well under 0.40. Below the floor the assistant declines
   and spends no quota confirming what the numbers already said.
2. **The model's own judgement.** Passages can score well and still not contain the
   answer, so `GroundedAnswer.answered` is a required boolean.

Out-of-scope messages (the weather, another course) are refused earlier, by the router.

### Two verification steps

These are what separate this from four chained prompts.

**Quotes are checked, not trusted.** The generator is asked for a verbatim quote;
asking is not enough. `tutor/grounding.py` matches every quote against the passages the
question came from — fuzzy enough to survive the accents pypdf mangles, strict enough
to reject a sentence assembled from scattered real words. Ungrounded questions are
dropped before the student sees them.

**The score is computed, not generated.** The grader returns no number. An LLM asked
for "a score out of 10" is not reproducible and often contradicts its own feedback
(*"you missed the main point… 8/10"*). `tutor/scoring.py` derives the score from the
per-criterion verdicts, so it cannot disagree with what the student is reading.

### Context engineering

Structured state — the open exam, the pending question, per-topic scores — lives in
typed objects in `tutor/session.py`, never in the transcript. The conversation itself
uses a sliding window of 6 turns plus a rolling summary.

That split is the point. The usual objection to a sliding window is that it silently
drops something important, but that only happens when the transcript is the only place
a fact lives. Here the window can forget turn 1 without forgetting the exam in
progress.

---

## Layout

```
notebooks/tutor.ipynb      the project: agents, orchestrator, demo
prompts/                   system_prompt.txt (shared persona) + one file per agent
                           (answerer, topic_extractor, exam_generator, grader, router)
tutor/config.py            settings from .env; config.reload() re-reads them live
tutor/llm.py               one entry point for LLM calls; retry and fallback policy
tutor/embeddings.py        Gemini embeddings, batched, cached
tutor/embedding_cache.py   SQLite cache so an embedding is never paid for twice
tutor/vectorstore.py       ChromaDB: storage, semantic search, full listing
tutor/grounding.py         verifies a quoted sentence exists in the source
tutor/scoring.py           derives the score from per-criterion verdicts
tutor/session.py           study state (typed) + conversation memory
tutor/library.py           the PDFs on disk: listing, archiving, restoring
tutor/prompts.py           loads prompts/*.txt, composes persona + role
tutor/agents.py            the three specialists: topics, exam, grading
tutor/orchestrator.py      routing and dispatch
app.py                     Streamlit front end (rendering only, no agent logic)
tutor/ingest/              pdf_loader -> cleaner -> chunker -> pipeline
tests/                     offline tests; no API key, no network
scripts/                   ingest.py, search.py, bench.py, diagnose.py
data/                      your PDFs (git-ignored); _removed/ holds detached ones
storage/                   ChromaDB + embedding cache (git-ignored)
```

## Tests

All offline — no API key, no network:

```bash
python tests/test_ingest.py         # cleaning and chunking
python tests/test_llm.py            # retry / fallback classification
python tests/test_cache.py          # embedding cache correctness
python tests/test_grounding.py      # invented quotes rejected, real ones kept
python tests/test_scoring.py        # the score can never contradict the feedback
python tests/test_session.py        # the window never eats structured state
python tests/test_prompts.py        # the shared persona reaches every agent
python tests/test_orchestrator.py   # routing and dispatch, with fake agents
python tests/test_progress.py       # progress sinks used by the UI
python tests/test_library.py        # archiving never destroys a document
python tests/test_notebook_refs.py  # notebook cells reference things that exist
```
