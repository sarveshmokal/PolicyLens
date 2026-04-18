# PolicyLens

**Multi-Agent RAG System for Policy Document Analysis**

A production-ready system with 15 pluggable AI agents that analyzes real-world policy documents (OECD, IMF, WHO, UN, EU), answers questions with cited sources, and verifies answers through adversarial debate and NLI entailment scoring.

Built with FastAPI + React + LangGraph + ChromaDB + Groq.

---

## What It Does

Ask PolicyLens a question about any of 11 policy documents, and it:

1. **Plans** — Analyzes query complexity and selects retrieval strategy
2. **Retrieves** — Hybrid BM25 + SBERT dense search across 7,504 text chunks
3. **Reranks** — Cross-encoder re-scoring for precision
4. **Synthesizes** — LLM generates a cited answer grounded in source passages
5. **Verifies** — NLI entailment scoring + adversarial debate agents stress-test the answer
6. **Evaluates** — Multi-dimension quality scoring (relevance, completeness, clarity, citations)

Every answer includes source citations like `[Source: EU_AI_Act_Regulation_2024.pdf, p.33]` so claims are traceable to the original documents.

---

## Architecture

Four-layer architecture with LangGraph orchestration:

| Layer | Components |
|-------|-----------|
| **Presentation** | React 18 + Tailwind CSS dashboard, FastAPI REST gateway |
| **Core Processing** | LangGraph state machine, 15 pluggable agents (ingestion, analysis, verification, support) |
| **Caching** | Redis (session + query cache) |
| **Database** | ChromaDB (dense vectors), BM25 index (sparse tokens), SQLite (experiment tracking) |

### Agent Pipeline

| Step | Agent | What It Does |
|------|-------|-------------|
| 1 | GuardrailsAgent | Safety check (PII, injection filtering) |
| 2 | PlannerAgent | Complexity analysis, query decomposition |
| 3 | RetrieverAgent | Hybrid BM25 + SBERT, alpha-weighted fusion |
| 4 | RerankerAgent | Cross-encoder re-scoring |
| 5 | SynthesizerAgent | LLM generation with citations |
| 6 | VerifierAgent | NLI entailment + adversarial debate |
| 7 | EvaluatorAgent | Quality scoring |

### LLM Provider Chain (Auto-Failover)

| Priority | Provider | Model | Speed | Requirement |
|----------|----------|-------|-------|-------------|
| 1 | Groq API | Llama 3.3 70B | ~200 tok/s | API key (free tier) |
| 2 | Ollama | Mistral 7B | ~30 tok/s | Local install |
| 3 | Flan-T5 | google/flan-t5-base | ~5 tok/s | None (CPU) |

Each provider auto-fails to the next. The system never crashes due to LLM unavailability.

---

## The 15 Agents

| Group | Agent | Technique |
|-------|-------|-----------|
| **Ingestion** | DocProcessor | PyMuPDF PDF parsing + recursive chunking |
| | Preprocessor | spaCy NER, text cleaning, entity extraction |
| | EmbeddingAgent | SBERT all-MiniLM-L6-v2 (384-dim) into ChromaDB |
| | TopicModel | LDA/NMF topic discovery on TF-IDF |
| **Analysis** | Planner | Query decomposition + strategy selection |
| | Retriever | Hybrid BM25 + dense (alpha-weighted fusion) |
| | Reranker | Cross-encoder ms-marco-MiniLM-L-6-v2 |
| | Synthesizer | LLM answer generation with source citations |
| **Verification** | DebateAgent A | Advocate — argues for answer correctness |
| | DebateAgent B | Challenger — finds weaknesses and gaps |
| | Verifier | Orchestrates NLI + debate into verdict |
| | NLI Agent | BART-large-MNLI entailment scoring |
| **Support** | Evaluator | Multi-dimension quality scoring |
| | MultiLingual | EN/DE language detection |
| | Guardrails | PII detection, prompt injection filtering |

All agents inherit from `BaseAgent` (Template Method pattern) and are registered via YAML config for plug-and-play toggling.

---

## Baseline Evaluation Results

Evaluated on 30 benchmark queries across three retrieval methods:

| Method | Hit Rate@5 | MRR | Source Precision@5 |
|--------|-----------|------|-------------------|
| BM25 | 100% | 0.9417 | 0.7600 |
| Dense (SBERT) | 100% | 0.9417 | 0.7700 |
| Hybrid (alpha=0.5) | 100% | 0.8833 | 0.7500 |

Verification metrics (full pipeline):
- **Faithfulness:** 90% (9/10 claims supported by source evidence)
- **Entailment Score:** 0.8477
- **Verdict:** PARTIALLY VERIFIED

Experiment results are versioned in `results/experiments/` for regression tracking.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python 3.13, FastAPI, Uvicorn |
| Frontend | React 18, Tailwind CSS, Vite, Recharts |
| LLM Providers | Groq (Llama 3.3), Ollama, HuggingFace Transformers |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector Store | ChromaDB (persistent, cosine similarity) |
| Sparse Search | rank-bm25 |
| NLI Verification | facebook/bart-large-mnli |
| Reranking | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| Topic Modeling | scikit-learn (LDA/NMF) |
| NLP Pipeline | spaCy (en_core_web_sm) |
| Orchestration | LangGraph (state machine) |

---

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+
- Groq API key (free at https://console.groq.com)

### Setup

```bash
# Clone
git clone https://github.com/sarveshmokal/PolicyLens.git
cd PolicyLens

# Backend
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/Mac
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Environment
cp .env.example .env
# Edit .env and add your GROQ_API_KEY

# Ingest documents
python scripts/ingest_all.py

# Start backend
uvicorn src.api.main:app --reload --port 8000

# Frontend (new terminal)
cd frontend
npm install
npm run dev
```

Open http://localhost:5173 and start asking questions.

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | /api/query | Full pipeline with verification |
| POST | /api/query/fast | Quick answer without verification |
| POST | /api/retrieve | Retrieval only (debugging) |
| GET | /api/agents | List all 15 agents |
| GET | /api/agents/health | Agent health checks |
| GET | /api/stats | System statistics |

API docs available at http://localhost:8000/docs

---

## Project Structure

```
PolicyLens/
├── src/
│   ├── agents/
│   │   ├── ingestion/      # DocProcessor, Preprocessor, Embedding, TopicModel
│   │   ├── analysis/       # Planner, Retriever, Reranker, Synthesizer
│   │   ├── verification/   # DebateA, DebateB, Verifier, NLI
│   │   └── support/        # Evaluator, MultiLingual, Guardrails
│   ├── core/               # BaseAgent, Config, Registry, LLM Provider, Orchestrator
│   ├── api/                # FastAPI routes and application
│   ├── models/             # Pydantic schemas
│   └── services/           # Business logic
├── frontend/src/           # React dashboard (Chat, Agents, Stats)
├── data/
│   ├── raw/                # 11 policy PDFs (OECD, IMF, WHO, EU, UNCTAD)
│   └── policy_queries.json # 30 evaluation queries
├── results/experiments/    # Versioned experiment runs
├── config/                 # settings.yaml, agents.yaml
├── scripts/                # ingest_all.py, evaluate_retrieval.py
└── docs/architecture/      # Architecture diagrams (PPTX + PNG)
```

---

## Policy Documents

| Document | Organization | Pages |
|----------|-------------|-------|
| EU AI Act Regulation 2024 | European Union | 144 |
| World Economic Outlook Ch.1 | IMF | 42 |
| AI Strategies Toolkit Note 14 | OECD | 26 |
| Economic Outlook Dec 2024 | OECD | 268 |
| Education at a Glance 2024 | OECD | 498 |
| Employment Outlook 2024 | OECD | 293 |
| Health at a Glance Asia-Pacific 2024 | OECD | 179 |
| How's Life? Wellbeing 2024 | OECD | 109 |
| Policy Considerations GenAI 2024 | OECD | 40 |
| Trade and Development Report 2024 | UNCTAD | 204 |
| World Health Statistics 2024 | WHO | 96 |

---

## Author

**Sarvesh Mokal** — [GitHub](https://github.com/sarveshmokal)
