<div align="center">

# 🎙️ Tafseel — تفصیل

**Agentic RAG system with voice I/O — built for native pakistanis**

[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-Agentic_RAG-FF6B6B?style=flat-square)](https://langchain-ai.github.io/langgraph/)
[![Gemini](https://img.shields.io/badge/Gemini-2.5_Flash-4285F4?style=flat-square&logo=google&logoColor=white)](https://deepmind.google/technologies/gemini/)
[![Qdrant](https://img.shields.io/badge/Qdrant-Cloud-DC143C?style=flat-square)](https://qdrant.tech)
[![Supabase](https://img.shields.io/badge/Supabase-PostgreSQL-3ECF8E?style=flat-square&logo=supabase&logoColor=white)](https://supabase.com)
[![Flutter](https://img.shields.io/badge/Flutter-Mobile-02569B?style=flat-square&logo=flutter&logoColor=white)](https://flutter.dev)
[![AWS EC2](https://img.shields.io/badge/AWS-EC2_+_CI%2FCD-FF9900?style=flat-square&logo=amazonaws&logoColor=white)](https://aws.amazon.com/ec2/)

**[Local Web App](http://127.0.0.1:8000)** · **[Local API Docs](http://127.0.0.1:8000/docs)**

</div>

---

## Run the web app

This repository contains the Python backend and a responsive browser frontend in `web/`. The Flutter source is not included. Both clients can use the same `/chat`, `/voice`, and conversation endpoints.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # only for a fresh setup; preserve an existing .env
# Set GEMINI_API_KEY and UPLIFTAI_API_KEY in .env
uvicorn api.app:app --host 127.0.0.1 --port 8000
```

Open **http://127.0.0.1:8000**. Text chat, microphone recording, spoken replies, English/Urdu/Sindhi/Balochi selection, and conversation history are available. Recording requires localhost or HTTPS and browser microphone permission. Tap the microphone again to send; recordings stop after 60 seconds. English speech uses Gemini TTS; Urdu, Sindhi and Balochi use Uplift AI. Regional voice IDs can be overridden with `UPLIFTAI_SINDHI_VOICE_ID` and `UPLIFTAI_BALOCHI_VOICE_ID` if your account uses different voices.

API keys stay in the server environment. No OpenAI or Groq key is needed. Gemini handles answer generation, structured graph checks, translation and audio transcription. The graph completes its checks before delivering the answer over SSE. Gemini requests share a default budget of five per minute; daily quotas still apply. Set `GEMINI_REQUESTS_PER_MINUTE` to your actual provider quota. Relevance checks are batched to reduce calls, and longer queries send status updates while waiting. If speech synthesis fails, the text answer remains available.

`STORAGE_MODE=local` keeps all database storage on your computer, even if cloud credentials exist in your environment. Search uses BM25 over the bundled `data/*.txt` documents and conversations persist in `conversations.db`. No ingestion step or cloud credentials are needed locally. The source documents are snapshots: confirm current government fees and requirements with NADRA.

For a future optional cloud path, explicitly set `STORAGE_MODE=cloud`, `QDRANT_URL`, `QDRANT_API_KEY`, and `SUPABASE_DB_URL`, then run `python ingest_all.py`. Cloud embeddings now use Gemini (`gemini-embedding-001`, 3072 dimensions). Use a **new collection**, such as `nia_gemini_chunks`; existing OpenAI vectors are incompatible and must be re-ingested. The original PostgreSQL conversation store is retained when Supabase is configured.

The local app is a single-user workspace. Before sharing a hosted instance, add authentication and conversation ownership checks. Deploy the Python service with the `web/` and `data/` directories, persistent storage for SQLite (or Supabase), runtime secrets, and HTTPS. Static-only hosting cannot run this backend. The original EC2 deployment workflow remains in `.github/workflows/`; no remote deployment is performed by local setup.

For a repeatable environment use `pip install -r requirements.lock.txt`. To run integration tests, install pytest and run `python -m pytest -q`. Health configuration flags at `/health` indicate that keys are present, not that provider quota or credentials are valid.

## What Is Tafseel?

Tafseel is an **agentic RAG system with end-to-end voice I/O**, designed to give low-literacy and Urdu-speaking Pakistanis frictionless access to government information. Instead of reading through dense official websites, users ask a question — by voice or text — and get a verified, natural-sounding Urdu answer spoken back to them.

Phase 1 covers the NADRA knowledge domain. The architecture is domain-agnostic and designed to scale across any knowledge base.

---

## Architecture

This is not a standard RAG pipeline. Tafseel implements a **multi-node agentic graph** with self-correction loops, hallucination detection, and query rewriting — all orchestrated by LangGraph. Retrieved answers pass through relevance, grounding, and usefulness checks before delivery. Candidate chunks are graded in one batch to reduce API calls.

```
                         ┌─────────────────┐
                         │   User Query    │
                         └────────┬────────┘
                                  │
                    ┌─────────────▼──────────────┐
                    │      decide_retrieval       │  ← LLM classifies intent
                    └──────┬──────────────┬───────┘
                           │              │
               need_retrieval=True   need_retrieval=False
                           │              │
                    ┌──────▼──────┐  ┌────▼──────────────┐
                    │   retrieve  │  │  generate_direct   │ → END
                    └──────┬──────┘  └────────────────────┘
                           │
                    ┌──────▼──────┐
                    │  is_relevant│  ← Grades each doc independently
                    └──────┬──────┘
               ┌───────────┴───────────┐
         relevant=True           relevant=False
               │                       │
  ┌────────────▼──────────┐    ┌───────▼──────────┐
  │  generate_from_context│    │  no_answer_found  │ → END
  └────────────┬──────────┘    └──────────────────┘
               │
        ┌──────▼──────┐
        │    is_sup   │  ← Hallucination check: is answer grounded in context?
        └──────┬──────┘
     ┌─────────┴──────────┐
  fully_supported     not supported (retries < MAX)
     │                    │
┌────▼────┐        ┌──────▼──────┐
│  is_use │        │revise_answer│ ──loop──► is_sup
└────┬────┘        └─────────────┘
     │  ← Is the answer actually useful to the user?
  ┌──┴───────────────────┐
  │                      │
useful=True         useful=False (rewrites < MAX)
  │                      │
 END            ┌────────▼────────┐
                │rewrite_question │ ──loop──► retrieve
                └─────────────────┘
```

### Why This Architecture Matters

**Standard RAG:** retrieve → generate → done. No quality control. No fallback. Silent failures.

**Tafseel's Agentic RAG:** every generation is challenged, graded, and if necessary, revised or restarted with a better query. The system fails loudly and recovers automatically rather than returning a hallucinated answer with confidence.

| Capability | Standard RAG | Tafseel |
|---|:---:|:---:|
| Relevance grading before generation | ❌ | ✅ |
| Hallucination detection post-generation | ❌ | ✅ |
| Auto-revision loop on unsupported answers | ❌ | ✅ |
| Query rewriting on poor retrieval | ❌ | ✅ |
| Usefulness scoring before delivery | ❌ | ✅ |
| Graceful fallback to `no_answer_found` | ❌ | ✅ |

---

## Parent-Child Chunking — Why It Matters

Most RAG systems embed and retrieve the same chunk. Tafseel uses a **two-tier chunking strategy** that separates retrieval precision from generation context.

```
Source Document
       │
       ├── Parent Chunk (1000 tokens) ──► stored in Supabase PostgreSQL
       │        │
       │        ├── Child Chunk (300 tokens) ──► embedded → Qdrant
       │        ├── Child Chunk (300 tokens) ──► embedded → Qdrant
       │        └── Child Chunk (300 tokens) ──► embedded → Qdrant
       │
       ├── Parent Chunk (1000 tokens) ──► stored in Supabase PostgreSQL
       │        │
       │       ...
```

**The retrieval problem this solves:**

- Small chunks (300 tokens) embed with high semantic precision — the query vector closely matches the relevant passage without noise from surrounding text.
- But generating an answer from 300 tokens gives the LLM insufficient context — it hallucinates missing details or gives incomplete answers.

**The solution:** retrieve using child chunks for precision, but pass the full parent chunk to the LLM for generation. You get the best of both worlds — sharp retrieval and rich context. This directly reduces hallucinations and improves answer completeness.

---

## Quality Gates — Three Layers of Answer Verification

### Gate 1: Relevance Grading (`is_relevant`)
Before generation, every retrieved document is independently graded by the LLM. Only documents that can directly answer the question are passed to the generator. Irrelevant documents — even semantically close ones — are discarded. This keeps the context window clean and prevents the LLM from being confused by tangentially related content.

### Gate 2: Hallucination Detection (`is_sup`)
After generation, the answer is checked against the context using a structured `IsSUPDecision` output:
- `fully_supported` → passes to usefulness check
- `partially_supported` / `no_support` → triggers automatic revision

The revision loop runs up to `MAX_RETRIES` times. This catches the most common RAG failure mode — the LLM adding plausible-sounding details that aren't in the source.

### Gate 3: Usefulness Scoring (`is_use`)
Even a grounded answer can fail to address what the user actually asked. The final gate checks whether the answer is genuinely useful. If not, the original question is automatically rewritten for better vector retrieval and the entire pipeline re-runs — up to `MAX_REWRITE_TRIES` times.

---

## Voice Pipeline

```
┌──────────────┐     audio file      ┌─────────────────────────────────────────┐
│  Flutter App │ ──────────────────► │             FastAPI /voice              │
│  (mobile)    │ ◄────────────────── │                                         │
└──────────────┘   SSE stream        │  1. Gemini audio transcription                  │
                   · status events   │     └─ Transcribes Urdu speech to text  │
                   · transcript      │                                         │
                   · answer text     │  2. LangGraph Agentic RAG Pipeline      │
                   · base64 MP3      │     └─ Full graph runs on transcript    │
                                     │                                         │
                                     │  3. UpliftAI TTS                        │
                                     │     └─ Converts Urdu answer to speech   │
                                     │        using Pakistani voice models     │
                                     └─────────────────────────────────────────┘
```

The entire pipeline streams status events back to the client in real-time, so the UI always reflects what the system is doing — transcribing, thinking, searching, preparing — with no black-box waiting.

UpliftAI was chosen specifically for its **Pakistani Urdu voice models**, which produce natural-sounding speech with correct Pakistani pronunciation, intonation, and rhythm — something generic TTS providers consistently fail at.

---

## Dual-Mode Prompting

The system operates in two modes: `text` and `voice`. Separate prompt templates are used for each:

- **Text mode** produces structured, formatted answers suitable for reading.
- **Voice mode** produces conversational, natural Urdu — shorter sentences, no bullet points, no markdown — optimized for text-to-speech synthesis.

The mode propagates through the full LangGraph state and is checked at every generation node, ensuring the output style is always appropriate for the delivery channel.

---

## Data Ingestion Pipeline

```bash
python ingest_all.py
```

The ingestion system is **idempotent and incremental**:

- Files are hashed on disk. Unchanged files are skipped entirely — no duplicate embeddings, no wasted API calls.
- If a file changes, all its old vectors are deleted from Qdrant and its parent chunks deleted from Supabase before re-ingestion. Clean state guaranteed.
- New files are ingested fresh. Multiple `.txt` files in `data/` are processed in a single run.

This makes it safe to re-run the ingestion script at any time without corrupting the vector store.

---

## Technology Stack

| Layer | Technology | Why |
|---|---|---|
| **LLM** | Gemini 2.5 Flash | Fast, cost-efficient, strong structured output |
| **Orchestration** | LangGraph | Stateful graph with cycles — essential for revision and rewrite loops |
| **Vector Store** | Qdrant Cloud | High-performance ANN search, production-grade cloud hosting |
| **Relational Store** | Supabase (PostgreSQL) | Parent chunk storage with connection pooling |
| **Embeddings** | `gemini-embedding-001` (cloud) / BM25 (local) | Gemini vectors or local document search |
| **STT** | Gemini audio understanding | Audio transcription without OpenAI |
| **TTS** | UpliftAI | Pakistani Urdu voice models, natural prosody |
| **API** | FastAPI + SSE | Async streaming, low-latency event delivery |
| **Mobile** | Flutter | Cross-platform iOS & Android |
| **Infra** | AWS EC2 + CI/CD | Push-to-deploy, always live |

---

## Project Structure

```
tafseel/
├── api/app.py           # FastAPI + web UI — /chat, /voice, /health
├── web/                 # Browser interface (HTML/CSS/JS)
├── main.py              # CLI test runner
├── ingest_all.py        # Ingestion entrypoint
└── src/
    ├── graph.py         # LangGraph graph definition (nodes + edges)
    ├── nodes.py         # Node implementations
    ├── routing.py       # Conditional edges + LLM instantiation
    ├── prompts.py       # Prompt templates (text + voice modes)
    ├── state.py         # MyState TypedDict
    ├── models.py        # Pydantic structured output schemas
    ├── ingestion.py     # Parent-child chunking, Qdrant + Supabase I/O
    └── config.py        # Env config + hyperparameters
```

---

## Setup

```bash
git clone https://github.com/your-username/tafseel.git
cd tafseel
pip install -r requirements.txt
```

**.env**
```env
GEMINI_API_KEY=
UPLIFTAI_API_KEY=
UPLIFTAI_VOICE_ID=v_8eelc901
QDRANT_URL=
QDRANT_API_KEY=
QDRANT_COLLECTION=nia_gemini_chunks
supabase_db_url=postgresql://postgres:[password]@db.xxx.supabase.co:5432/postgres
```

```bash
python ingest_all.py          # optional: ingest data/ into configured cloud stores
uvicorn api.app:app --port 8000   # start server
```

---

## API

### `POST /chat` — Streaming text response
```json
{ "question": "NADRA card renew karne ka tarika kya hai?" }
```
```
data: {"type": "status",  "text": "Thinking…"}
data: {"type": "status",  "text": "Searched NADRA knowledge base"}
data: {"type": "token",   "text": "آپ کا کارڈ..."}
data: {"type": "done"}
```

### `POST /voice` — Streaming voice response
Multipart audio upload (`.m4a` / `.mp3` / `.wav`)
```
data: {"type": "status",     "text": "Listening..."}
data: {"type": "transcript", "text": "سوال کا متن"}
data: {"type": "answer",     "text": "جواب کا متن"}
data: {"type": "audio",      "data": "<base64 MP3>"}
data: {"type": "done"}
```

---

## Roadmap

- [x] Phase 1 — NADRA knowledge base
- [ ] Phase 2 — BISP, Passport, FBR, PEMRA
- [x] Phase 3 — Multi-turn conversation memory
- [ ] Phase 4 — District-level localization & offline support

---

## License

MIT — see [LICENSE](LICENSE) for details.

---

<div align="center">
<sub>Built for Pakistan · Agentic RAG · Voice-first · Urdu-native</sub>
</div>
