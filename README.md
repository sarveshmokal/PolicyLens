# PolicyLens

Multi-Agent RAG System for Policy Document Analysis

## Overview
PolicyLens is a production-ready system that uses 15 pluggable AI agents to analyze real-world policy documents (OECD, IMF, WHO, UN, EU). It combines hybrid retrieval (BM25 + SBERT), adversarial debate, NLI-based verification, and experiment tracking within a deployable full-stack application.

## Architecture
4-layer architecture: Presentation (React + FastAPI) → Core Processing (LangGraph + 15 Agents) → Caching (Redis) → Database (ChromaDB + SQLite)

## Tech Stack
- **Backend:** Python 3.11+, FastAPI, LangGraph
- **Frontend:** React 18, Tailwind CSS, Vite
- **ML/NLP:** sentence-transformers, spaCy, BART-MNLI, cross-encoder
- **LLM Providers:** Groq API → Ollama → Flan-T5 (auto-failover)
- **Storage:** ChromaDB, Redis, SQLite
- **Deployment:** Docker, GitHub Actions CI/CD

## Status
🚧 Under active development — Phase 1: Project Setup

## Author
Sarvesh Mokal — [GitHub](https://github.com/sarveshmokal)