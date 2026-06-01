# RAG Platform

A production-grade Retrieval-Augmented Generation (RAG) system built to demonstrate ML engineering best practices. Designed to mirror real-world AI infrastructure used at companies like Google, Meta, and Amazon.

![CI](https://github.com/bblcoder333/rag-platform/actions/workflows/ci.yml/badge.svg)

## What it does

Ingests documents (PDFs, text files), chunks and embeds them using a local embedding model, stores vectors in Postgres with pgvector, and answers natural language questions with cited responses using a local LLM.

## Architecture
Document → Chunker → Embedder → pgvector
↓
Query → Embedder → Vector Search → Reranker → LLM → Answer
↓
Evaluation Pipeline

## Stack

| Layer | Technology |
|---|---|
| Embedding model | BAAI/bge-small-en (sentence-transformers) |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| Vector database | Postgres + pgvector |
| LLM | Llama 3.2 via Ollama |
| Backend API | FastAPI |
| Experiment tracking | Custom evaluation pipeline |
| CI/CD | GitHub Actions |
| Containerization | Docker + Docker Compose |

## Key engineering decisions

**Chunking strategy:** Implemented both fixed-size and recursive character splitting. Recursive chunking produces higher minimum chunk sizes (318 vs 274 chars) by respecting natural text boundaries, reducing context fragmentation during retrieval.

**Hybrid retrieval pipeline:** Vector similarity search (cosine) followed by cross-encoder reranking. The cross-encoder scores query-chunk pairs directly rather than independently, improving precision on ambiguous queries at the cost of ~1s additional latency.

**Evaluation methodology:** Built a golden dataset evaluation harness measuring retrieval recall and answer accuracy. Keyword-based matching is a known limitation — semantic similarity scoring (RAGAS) is a planned improvement.

**Local-first architecture:** Entire stack runs locally using Ollama + sentence-transformers with no external API dependencies, enabling cost-free development and offline operation.

## Results

| Metric | Value |
|---|---|
| Retrieval recall (with reranker) | 0.4 |
| Answer accuracy (with reranker) | 0.3 |
| Avg latency with reranker | 9.12s |
| Avg latency without reranker | 8.77s |
| Reranker overhead | ~0.35s |

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

## Usage

```bash
# Ingest a document
python backend/rag/pipeline.py docs/your_document.pdf

# Ask a question
python backend/rag/generator.py "your question here"

# Run evaluation
python backend/evaluation/evaluator.py

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
│   │   └── retrieval/    # Vector search and reranking
│   ├── evaluation/   # Golden dataset evaluation harness
│   └── db/           # Database setup and connections
├── experiments/      # Evaluation results
├── docs/             # Sample documents
├── docker/           # Dockerfile
└── .github/workflows # CI pipeline

## Future improvements

- Hybrid search (BM25 + vector) with RRF score fusion
- RAGAS evaluation for semantic answer quality
- MLflow experiment tracking
- Cloud deployment (AWS/GCP)
- Frontend chat interface