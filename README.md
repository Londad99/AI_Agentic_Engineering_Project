# Study Tutor — Corte 1 Project

**AI Agentic Engineering · Ingeniería de Sistemas · Universidad de Santander**

A conversational assistant that turns a student's own PDF (class notes or slides) into study
material: it extracts the topics, generates practice exams grounded in the source text, and
gives specific feedback on the student's answers instead of a bare right/wrong.

Built with **context engineering + RAG**, the **Gemini API** and **ChromaDB**.

## The deliverable is the notebook

**[`notebooks/tutor.ipynb`](notebooks/tutor.ipynb)** — open it first. It contains the agents,
their prompts, their structured-output schemas and the orchestrator, each with its output
visible. It is committed with outputs so it can be read on GitHub without running anything.

The `tutor/` package holds the deterministic plumbing the notebook imports: PDF reading,
cleaning, chunking, embeddings and the vector store. That code is not agentic and not
interesting to read cell by cell, so it lives in modules where it can be unit-tested.

## Layout

```
notebooks/tutor.ipynb   the project: agents, orchestrator, demo
tutor/config.py         all settings, read from .env (config.reload() re-reads it)
tutor/prompts.py        loads prompts/*.txt and composes persona + role
tutor/llm.py            single entry point for LLM calls, with retry/fallback policy
tutor/embedding_cache.py  on-disk cache so an embedding is never paid for twice
tutor/progress.py       progress output that survives Jupyter's stdout buffering
tutor/embeddings.py     Gemini embeddings (batched, with backoff)
tutor/vectorstore.py    ChromaDB: storage and semantic search
tutor/ingest/           pdf_loader -> cleaner -> chunker -> pipeline
tests/test_ingest.py    offline tests for cleaning and chunking
tests/test_llm.py       offline tests for the provider routing policy
tests/test_cache.py     offline tests for the embedding cache
tests/test_prompts.py   the shared persona reaches every agent
tests/test_notebook_refs.py  notebook cells reference attributes that exist
scripts/                CLI entry points: ingest.py, search.py, diagnose.py
prompts/system_prompt.txt  the tutor's persona, scope and refusal rules
data/                   put your PDFs here (git-ignored)
storage/                ChromaDB files (git-ignored)
```

## Setup

```bash
pip install -r requirements.txt
python tests/test_ingest.py   # sanity check, uses no quota
```

Put one or more PDFs in `data/` and open the notebook. Its first cell creates `.env` for you
from `.env.example`; paste your key from [AI Studio](https://aistudio.google.com/apikey) into
it, then run **Kernel → Restart & Run All**.

> The repo ships the template and not the file: `.env` is git-ignored, so a committed one
> would never reach anyone who clones this — and un-ignoring it to fix that would make git
> track the file, which is how API keys end up pushed. The variable names are documentation
> and belong in git; the values do not.

## Design notes

Ingestion is plain code, not an agent: it is a deterministic transformation with no judgement
call to make, so an LLM there would add cost, latency and a failure mode for nothing. The
agents begin where judgement begins.

Three retrieval decisions carry most of the quality:

- **Header/footer removal by position + repetition.** A footer repeated on 40 pages is
  otherwise embedded 40 times and floods every result list with near-identical noise. The
  heuristic errs toward keeping a line: a false positive silently deletes material the student
  will be examined on.
- **Content-hashed chunk ids.** Re-ingesting the same PDF overwrites the same rows instead of
  duplicating them.
- **Asymmetric embeddings.** `gemini-embedding-001` embeds the same text differently as a
  stored document (`RETRIEVAL_DOCUMENT`) than as a question (`RETRIEVAL_QUERY`); using one
  type for both measurably hurts recall.

## Status

- [x] Ingestion pipeline + vector store + tests
- [x] LLM provider with local fallback (Gemini -> Qwen/Ollama on rate limit)
- [x] Topic Extractor agent (map-reduce over the whole document)
- [ ] Exam Generator agent
- [ ] Grader agent
- [ ] Orchestrator + conversation memory
