# Study Tutor — Corte 1 Project

**AI Agentic Engineering · Ingeniería de Sistemas · Universidad de Santander**

A conversational study assistant. The student uploads a PDF of their own material —
class notes, slides, a textbook chapter — and the tutor maps its topics, generates
practice exams grounded in the text, and grades written answers point by point.

Built with **context engineering + RAG**, the **Gemini API** and **ChromaDB**, on a
**sub-agent architecture**.

**The deliverable is [`notebooks/tutor.ipynb`](notebooks/tutor.ipynb)** — open it
first. It holds the agents, their prompts and their schemas, committed with outputs so
it can be read on GitHub without running anything.

---

## Setup

### 1. Requirements

Python 3.10 or newer, and a Gemini API key from
[Google AI Studio](https://aistudio.google.com/apikey) (free tier is enough).

### 2. Clone and install

```bash
git clone <this-repo-url>
cd AI_Agentic_Engineering_Project
pip install -r requirements.txt
```

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

```bash
jupyter notebook notebooks/tutor.ipynb
```

Then **Run All**. The first run ingests the PDFs; later runs reuse the vector store and
the embedding cache.

Or from the command line:

```bash
python scripts/ingest.py --reset          # PDFs -> ChromaDB
python scripts/search.py "your question"  # check retrieval
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
| Changing `.env` seems to do nothing | Re-run the setup cell; it calls `config.reload()`. |
| Ingestion says no text found | The PDF is scanned images. It needs OCR. |

---

## Where RAG is used — and where it is not

RAG is not applied uniformly. Each agent asks a different question, and only two of them
are retrieval questions.

| Stage | RAG? | Query | Why |
|---|---|---|---|
| Ingestion | — | — | Deterministic transformation, no LLM at all. |
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

**Topic Extractor** — reads the whole document in order and returns topics, subtopics
and page numbers. Map-reduce: topics per batch, then one call merging duplicates
across batches.

**Exam Generator** — retrieves passages for one topic and writes open questions, each
with a correct answer, the criteria a correct answer must contain, a difficulty level
(recall / understanding / application) and a verbatim source quote.

**Grader** — marks each criterion `covered` / `partial` / `missing`, names what was
right, what to review and which claims the document contradicts.

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
prompts/system_prompt.txt  shared persona: scope, refusal, language, citations
tutor/config.py            settings from .env; config.reload() re-reads them live
tutor/llm.py               one entry point for LLM calls; retry and fallback policy
tutor/embeddings.py        Gemini embeddings, batched, cached
tutor/embedding_cache.py   SQLite cache so an embedding is never paid for twice
tutor/vectorstore.py       ChromaDB: storage, semantic search, full listing
tutor/grounding.py         verifies a quoted sentence exists in the source
tutor/scoring.py           derives the score from per-criterion verdicts
tutor/session.py           study state (typed) + conversation memory
tutor/prompts.py           loads prompts/*.txt, composes persona + role
tutor/ingest/              pdf_loader -> cleaner -> chunker -> pipeline
tests/                     offline tests; no API key, no network
scripts/                   ingest.py, search.py, bench.py, diagnose.py
data/                      your PDFs (git-ignored)
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
python tests/test_notebook_refs.py  # notebook cells reference things that exist
```
