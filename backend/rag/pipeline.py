import sys
import os
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.ingestion.document_loader import load_document
from rag.ingestion.chunker import chunk_document
from rag.embeddings.embedder import embed_chunks
from db.database import engine
from sqlalchemy import text


def ingest_document(file_path: str, strategy: str = "recursive",
                    chunk_size: int = 512) -> dict:
    """
    Full pipeline: load → chunk → embed → store in Postgres.
    Returns a summary of what was ingested.
    """
    start_time = time.time()
    print(f"\n{'='*50}")
    print(f"Starting ingestion: {file_path}")
    print(f"Strategy: {strategy}, Chunk size: {chunk_size}")
    print(f"{'='*50}")

    # Step 1: Load
    print("\n[1/4] Loading document...")
    doc = load_document(file_path)
    print(f"  Loaded: {doc['source']} ({doc['num_pages']} pages, "
          f"{doc['file_size_kb']} KB)")

    # Step 2: Chunk
    print("\n[2/4] Chunking document...")
    chunks = chunk_document(doc["content"], strategy=strategy,
                           chunk_size=chunk_size)
    print(f"  Created {len(chunks)} chunks")

    # Step 3: Embed
    print("\n[3/4] Embedding chunks...")
    embedded_chunks = embed_chunks(chunks)

    # Step 4: Store
    print("\n[4/4] Storing in database...")
    with engine.connect() as conn:
        # Insert document record
        result = conn.execute(text("""
            INSERT INTO documents (source, file_path, num_pages, file_size_kb)
            VALUES (:source, :file_path, :num_pages, :file_size_kb)
            RETURNING id
        """), {
            "source": doc["source"],
            "file_path": doc["file_path"],
            "num_pages": doc["num_pages"],
            "file_size_kb": doc["file_size_kb"]
        })
        document_id = result.fetchone()[0]

        # Insert each chunk
        for chunk in embedded_chunks:
            conn.execute(text("""
                INSERT INTO chunks
                (document_id, chunk_index, text, strategy,
                 chunk_size, char_count, embedding)
                VALUES
                (:document_id, :chunk_index, :text, :strategy,
                 :chunk_size, :char_count, :embedding)
            """), {
                "document_id": document_id,
                "chunk_index": chunk["chunk_index"],
                "text": chunk["text"],
                "strategy": chunk["strategy"],
                "chunk_size": chunk["chunk_size"],
                "char_count": chunk["char_count"],
                "embedding": str(chunk["embedding"])
            })

        conn.commit()
        print(f"  Stored document (id={document_id}) and "
              f"{len(embedded_chunks)} chunks")

    elapsed = time.time() - start_time
    summary = {
        "document_id": document_id,
        "source": doc["source"],
        "num_pages": doc["num_pages"],
        "num_chunks": len(embedded_chunks),
        "strategy": strategy,
        "chunk_size": chunk_size,
        "elapsed_seconds": round(elapsed, 2)
    }

    print(f"\n{'='*50}")
    print(f"Ingestion complete in {elapsed:.2f}s")
    print(f"Summary: {summary}")
    print(f"{'='*50}\n")

    return summary


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pipeline.py <file_path>")
        sys.exit(1)

    result = ingest_document(sys.argv[1])