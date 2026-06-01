import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from rag.embeddings.embedder import embed_query
from db.database import engine
from sqlalchemy import text
from typing import List


def retrieve(query: str, top_k: int = 5, strategy: str = None) -> List[dict]:
    """
    Embed a query and find the most similar chunks in the database.
    Returns top_k chunks ranked by cosine similarity.
    """
    print(f"\nQuery: {query}")
    print(f"Retrieving top {top_k} chunks...")

    # Embed the query
    query_embedding = embed_query(query)
    embedding_str = str(query_embedding)

    # Vector similarity search using pgvector
    with engine.connect() as conn:
        results = conn.execute(text("""
            SELECT
                c.id,
                c.chunk_index,
                c.text,
                c.strategy,
                c.char_count,
                d.source,
                1 - (c.embedding <=> CAST(:embedding AS vector)) as similarity
            FROM chunks c
            JOIN documents d ON c.document_id = d.id
            WHERE (:strategy IS NULL OR c.strategy = :strategy)
            ORDER BY c.embedding <=> CAST(:embedding AS vector)
            LIMIT :top_k
        """), {
            "embedding": embedding_str,
            "strategy": strategy,
            "top_k": top_k
        })

        chunks = []
        for row in results:
            chunks.append({
                "id": row[0],
                "chunk_index": row[1],
                "text": row[2],
                "strategy": row[3],
                "char_count": row[4],
                "source": row[5],
                "similarity": round(float(row[6]), 4)
            })

    return chunks


def print_results(chunks: List[dict]):
    """Pretty print retrieval results."""
    print(f"\nFound {len(chunks)} chunks:\n")
    for i, chunk in enumerate(chunks):
        print(f"--- Result {i+1} (similarity: {chunk['similarity']}) ---")
        print(f"Source: {chunk['source']}")
        print(f"Text: {chunk['text'][:200]}...")
        print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python retriever.py '<your question>'")
        sys.exit(1)

    query = sys.argv[1]
    results = retrieve(query, top_k=3)
    print_results(results)