# RAG Platform

A production-grade Retrieval-Augmented Generation (RAG) system built to demonstrate ML engineering best practices. Designed to mirror real-world AI infrastructure used at companies like Google, Meta, and Amazon.

![CI](https://github.com/bblcoder333/rag-platform/actions/workflows/ci.yml/badge.svg)

## What it does

Ingests documents (PDFs, text files), chunks and embeds them using a local embedding model, stores vectors in Postgres with pgvector, and answers natural language questions with cited responses using a local LLM.

## Architecture
Document → Chunker → Embedder → pgvector

↓

Query → Embedder → Hybrid Search (Vector + BM25) → Reranker → LLM → Answer

↓

Evaluation Pipeline

## Stack

| Layer | Technology |
|---|---|
| Embedding model | BAAI/bge-small-en (sentence-transformers) |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| Vector database | Postgres + pgvector |
| Keyword search | Postgres full-text search (BM25-style) |
| Search fusion | Reciprocal Rank Fusion (RRF) |
| LLM | Llama 3.2 via Ollama |
| Backend API | FastAPI |
| Evaluation | Custom golden dataset + RAGAS |
| CI/CD | GitHub Actions |
| Containerization | Docker + Docker Compose |

## Key engineering decisions

**Chunking strategy:** Implemented both fixed-size and recursive character splitting. Recursive chunking produces higher minimum chunk sizes (318 vs 274 chars) by respecting natural text boundaries, reducing context fragmentation during retrieval.

**Hybrid retrieval pipeline:** Vector similarity search (cosine) combined with BM25 keyword search via Reciprocal Rank Fusion, followed by cross-encoder reranking. This surfaces results that pure semantic search misses (e.g. exact terminology matches) while still capturing semantic similarity. The cross-encoder adds ~0.35s latency but improves precision on ambiguous queries.

**Evaluation methodology:** Built two evaluation layers — a custom golden dataset harness measuring retrieval recall and answer accuracy via keyword matching, and a RAGAS semantic evaluation layer using an LLM judge for faithfulness and relevancy scoring. RAGAS 0.4.3's high-level `evaluate()` orchestrator has a broken import chain (`langchain_community.chat_models.vertexai` was removed upstream) and an asyncio incompatibility with recent Python versions. Resolved by calling the underlying `Metric.ascore(row)` dict-based API directly, bypassing the broken orchestrator.

**Local-first architecture:** Entire stack runs locally using Ollama + sentence-transformers with no external API dependencies, enabling cost-free development and offline operation.

## Results

### Custom golden dataset (keyword matching)
| Metric | Value |
|---|---|
| Retrieval recall (with reranker) | 0.4 |
| Answer accuracy (with reranker) | 0.3 |
| Avg latency with reranker | 9.12s |
| Avg latency without reranker | 8.77s |

### RAGAS semantic evaluation (LLM-judged)
| Metric | Value |
|---|---|
| Faithfulness | 0.750 |
| Answer Relevancy | 0.768 |

**Faithfulness** measures whether answers are grounded in retrieved context (no hallucination). **Answer Relevancy** measures whether the answer actually addresses the question asked. These RAGAS scores are more meaningful than the keyword-matching scores above since they use semantic understanding rather than exact string matches.

## Setup

```bash
# Clone and install
git clone https://github.com/bblcoder333/rag-platform.git
cd rag-platform
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Start database
docker run -d --name pgvector \
  -e POSTGRES_PASSWORD=password \
  -e POSTGRES_DB=ragplatform \
  -p 5432:5432 ankane/pgvector

# Initialize database
python backend/db/database.py

# Start Ollama and pull model
brew install ollama
brew services start ollama
ollama pull llama3.2

# Start API
uvicorn backend.api.main:app --reload --port 8000
```

## Deployment

Deployed to Railway with a swappable LLM backend architecture — Ollama for local development, Gemini for cloud deployment (avoiding the need to run a local model server in production).

**Live deployment verification:**

The system was deployed end-to-end and verified working: documents ingested via the `/ingest` endpoint, questions answered via `/ask` with proper source citations, all served through the custom frontend chat UI.

![Live deployment screenshot](docs/screenshots/live-deployment.png)

*The deployment was paused after verification to avoid ongoing trial costs — this is a portfolio project, not a production service requiring 24/7 uptime. The full setup (Procfile, environment-based LLM backend switching, Railway Postgres + pgvector) is documented below for anyone who wants to redeploy it.*

### Deployment setup

```bash
# Procfile
web: uvicorn backend.api.main:app --host 0.0.0.0 --port $PORT
```

Environment variables required for cloud deployment:
DATABASE_URL=<Railway Postgres connection string>

LLM_BACKEND=gemini

GEMINI_API_KEY=<your Gemini API key>

Production uses a slim `requirements.txt` (FastAPI, embeddings, pgvector) separate from `requirements-dev.txt` (full local stack including Ollama, RAGAS, evaluation tools) to avoid dependency conflicts and reduce build time on Railway's infrastructure.

## Usage

```bash
# Ingest a document (drop any PDF/text file into docs/ first)
python backend/rag/pipeline.py docs/your_document.pdf

# Ask a question
python backend/rag/generator.py "your question here"

# Run custom evaluation
python backend/evaluation/evaluator.py

# Run RAGAS evaluation
python backend/evaluation/ragas_eval.py

# API docs
open http://localhost:8000/docs
```

## Project structure
rag-platform/

├── backend/

│   ├── api/          # FastAPI endpoints

│   ├── rag/

│   │   ├── ingestion/    # Document loading and chunking

│   │   ├── embeddings/   # Embedding pipeline

│   │   └── retrieval/    # Vector search, BM25, hybrid fusion, reranking

│   ├── evaluation/   # Golden dataset + RAGAS evaluation harnesses

│   └── db/           # Database setup and connections

├── experiments/      # Evaluation results

├── docs/             # Sample documents (gitignored)

├── docker/           # Dockerfile

└── .github/workflows # CI pipeline

## Future improvements

- Cloud deployment (AWS/GCP)
- Frontend chat interface
- MLflow experiment tracking
- Larger, more diverse golden dataset