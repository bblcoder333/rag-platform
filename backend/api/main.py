import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import shutil
import tempfile

from rag.pipeline import ingest_document
from rag.generator import generate_answer
from rag.retrieval.retriever import retrieve
from db.database import engine
from sqlalchemy import text

app = FastAPI(
    title="RAG Platform API",
    description="Production-grade RAG system",
    version="0.1.0"
)

# Allow frontend to talk to backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Request/Response Models ---

class QuestionRequest(BaseModel):
    question: str
    top_k: int = 5


class QuestionResponse(BaseModel):
    answer: str
    sources: list
    chunks_used: int
    similarity_scores: list


# --- Routes ---

@app.get("/")
def root():
    return {"status": "ok", "message": "RAG Platform is running"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    """Upload and ingest a document into the RAG pipeline."""
    allowed_types = ["application/pdf", "text/plain", "text/markdown"]

    # Save uploaded file to temp location
    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        result = ingest_document(tmp_path)
        result["source"] = file.filename
        return {"status": "success", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(tmp_path)


@app.post("/ask", response_model=QuestionResponse)
def ask(request: QuestionRequest):
    """Ask a question and get an answer from the RAG system."""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        result = generate_answer(request.question, top_k=request.top_k)
        return QuestionResponse(
            answer=result["answer"],
            sources=result["sources"],
            chunks_used=result["chunks_used"],
            similarity_scores=result["similarity_scores"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents")
def list_documents():
    """List all ingested documents."""
    with engine.connect() as conn:
        results = conn.execute(text("""
            SELECT d.id, d.source, d.num_pages, d.file_size_kb,
                   d.created_at, COUNT(c.id) as chunk_count
            FROM documents d
            LEFT JOIN chunks c ON d.id = c.document_id
            GROUP BY d.id
            ORDER BY d.created_at DESC
        """))
        docs = []
        for row in results:
            docs.append({
                "id": row[0],
                "source": row[1],
                "num_pages": row[2],
                "file_size_kb": row[3],
                "created_at": str(row[4]),
                "chunk_count": row[5]
            })
    return {"documents": docs}


@app.get("/stats")
def stats():
    """System stats — useful for your monitoring dashboard later."""
    with engine.connect() as conn:
        doc_count = conn.execute(
            text("SELECT COUNT(*) FROM documents")).scalar()
        chunk_count = conn.execute(
            text("SELECT COUNT(*) FROM chunks")).scalar()

    return {
        "total_documents": doc_count,
        "total_chunks": chunk_count,
        "embedding_model": "BAAI/bge-small-en",
        "llm": "llama3.2",
        "vector_db": "pgvector"
    }