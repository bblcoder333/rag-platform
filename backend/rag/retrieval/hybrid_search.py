import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from rag.embeddings.embedder import embed_query
from db.database import engine
from sqlalchemy import text
from typing import List
import math


def bm25_search(query: str, top_k: int = 10) -> List[dict]:
    """
    Keyword search using Postgres full-text search (approximates BM25).
    Finds chunks containing the exact words from the query.
    """
    # Convert query to tsquery format
    query_words = " & ".join(query.strip().split())

    with engine.connect() as conn:
        results = conn.execute(text("""
            SELECT
                c.id,
                c.chunk_index,
                c.text,
                c.strategy,
                c.char_count,
                d.source,
                ts_rank(
                    to_tsvector('english', c.text),
                    to_tsquery('english', :query)
                ) as bm25_score
            FROM chunks c
            JOIN documents d ON c.document_id = d.id
            WHERE to_tsvector('english', c.text) @@ 
                  to_tsquery('english', :query)
            ORDER BY bm25_score DESC
            LIMIT :top_k
        """), {"query": query_words, "top_k": top_k})

        chunks = []
        for row in results:
            chunks.append({
                "id": row[0],
                "chunk_index": row[1],
                "text": row[2],
                "strategy": row[3],
                "char_count": row[4],
                "source": row[5],
                "bm25_score": float(row[6]),
                "similarity": 0.0  # placeholder
            })

    return chunks


def vector_search(query: str, top_k: int = 10) -> List[dict]:
    """Standard vector similarity search."""
    query_embedding = embed_query(query)
    embedding_str = str(query_embedding)

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
            ORDER BY c.embedding <=> CAST(:embedding AS vector)
            LIMIT :top_k
        """), {"embedding": embedding_str, "top_k": top_k})

        chunks = []
        for row in results:
            chunks.append({
                "id": row[0],
                "chunk_index": row[1],
                "text": row[2],
                "strategy": row[3],
                "char_count": row[4],
                "source": row[5],
                "similarity": round(float(row[6]), 4),
                "bm25_score": 0.0  # placeholder
            })

    return chunks


def reciprocal_rank_fusion(
        vector_results: List[dict],
        bm25_results: List[dict],
        k: int = 60,
        top_k: int = 5) -> List[dict]:
    """
    Combine vector and BM25 results using Reciprocal Rank Fusion (RRF).
    RRF score = 1/(k + rank) for each result in each list.
    Higher score = more relevant.
    """
    scores = {}
    chunk_data = {}

    # Score vector results
    for rank, chunk in enumerate(vector_results):
        chunk_id = chunk["id"]
        scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (k + rank + 1)
        chunk_data[chunk_id] = chunk

    # Score BM25 results
    for rank, chunk in enumerate(bm25_results):
        chunk_id = chunk["id"]
        scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (k + rank + 1)
        chunk_data[chunk_id] = chunk

    # Sort by combined RRF score
    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

    results = []
    for chunk_id in sorted_ids[:top_k]:
        chunk = chunk_data[chunk_id]
        chunk["rrf_score"] = round(scores[chunk_id], 6)
        results.append(chunk)

    return results


def hybrid_search(query: str, top_k: int = 5) -> List[dict]:
    """
    Full hybrid search: vector + BM25 fused with RRF.
    Best of both worlds — semantic + keyword matching.
    """
    print(f"\nHybrid search: '{query}'")

    # Get candidates from both methods
    vector_results = vector_search(query, top_k=top_k * 2)
    bm25_results = bm25_search(query, top_k=top_k * 2)

    print(f"  Vector results: {len(vector_results)}")
    print(f"  BM25 results:   {len(bm25_results)}")

    # Fuse with RRF
    fused = reciprocal_rank_fusion(vector_results, bm25_results, top_k=top_k)
    print(f"  Fused results:  {len(fused)}")

    return fused


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python hybrid_search.py '<query>'")
        sys.exit(1)

    query = sys.argv[1]

    print("\n--- VECTOR ONLY ---")
    vector = vector_search(query, top_k=3)
    for i, c in enumerate(vector):
        print(f"  {i+1}. similarity={c['similarity']} | {c['text'][:80]}...")

    print("\n--- BM25 ONLY ---")
    bm25 = bm25_search(query, top_k=3)
    if bm25:
        for i, c in enumerate(bm25):
            print(f"  {i+1}. bm25={c['bm25_score']:.4f} | {c['text'][:80]}...")
    else:
        print("  No BM25 results found")

    print("\n--- HYBRID (RRF) ---")
    hybrid = hybrid_search(query, top_k=3)
    for i, c in enumerate(hybrid):
        print(f"  {i+1}. rrf={c['rrf_score']} | {c['text'][:80]}...")