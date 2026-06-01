import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from sentence_transformers import CrossEncoder
from typing import List
import time

# This model scores query-chunk pairs directly
# Much more accurate than cosine similarity alone
MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

print(f"Loading reranker model: {MODEL_NAME}")
reranker = CrossEncoder(MODEL_NAME)
print("Reranker loaded.")


def rerank(query: str, chunks: List[dict], top_k: int = 3) -> List[dict]:
    """
    Takes retrieved chunks and reranks them using a cross-encoder.
    Returns top_k chunks sorted by rerank score.
    """
    if not chunks:
        return []

    start = time.time()

    # Create query-chunk pairs for the cross-encoder
    pairs = [[query, chunk["text"]] for chunk in chunks]

    # Score each pair
    scores = reranker.predict(pairs)

    # Add rerank scores to chunks
    for i, chunk in enumerate(chunks):
        chunk["rerank_score"] = round(float(scores[i]), 4)

    # Sort by rerank score descending
    reranked = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)
    top_chunks = reranked[:top_k]

    elapsed = time.time() - start
    print(f"Reranked {len(chunks)} chunks → top {top_k} in {elapsed:.2f}s")

    return top_chunks


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python reranker.py '<question>'")
        sys.exit(1)

    query = sys.argv[1]

    # First retrieve
    from retriever import retrieve
    chunks = retrieve(query, top_k=8)

    print(f"\nBefore reranking (by cosine similarity):")
    for i, c in enumerate(chunks[:3]):
        print(f"  {i+1}. similarity={c['similarity']} | "
              f"{c['text'][:80]}...")

    # Then rerank
    reranked = rerank(query, chunks, top_k=3)

    print(f"\nAfter reranking (by cross-encoder):")
    for i, c in enumerate(reranked):
        print(f"  {i+1}. rerank_score={c['rerank_score']} | "
              f"{c['text'][:80]}...")