import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
from rag.retrieval.retriever import retrieve
from rag.retrieval.reranker import rerank
from rag.retrieval.query_decomposition import decompose_query

def build_context(chunks: list) -> str:
    context_parts = []
    for i, chunk in enumerate(chunks):
        context_parts.append(
            f"[Source {i+1} - {chunk['source']} "
            f"(rerank_score: {chunk.get('rerank_score', chunk['similarity'])})]:"
            f"\n{chunk['text']}"
        )
    return "\n\n".join(context_parts)

def generate_answer(query: str, top_k: int = 5,
                    use_reranker: bool = True,
                    use_hybrid: bool = True,
                    use_decomposition: bool = True) -> dict:
    """
    Full RAG pipeline: decompose → retrieve → rerank → generate.
    """
    # Step 0: Decompose multi-part questions
    if use_decomposition:
        sub_queries = decompose_query(query)
    else:
        sub_queries = [query]

    # Step 1: Retrieve for each sub-query, merge results
    all_chunks = []
    seen_ids = set()

    for sub_q in sub_queries:
        if use_hybrid:
            from rag.retrieval.hybrid_search import hybrid_search
            chunks = hybrid_search(sub_q, top_k=top_k)
        else:
            chunks = retrieve(sub_q, top_k=top_k)

        for chunk in chunks:
            if chunk["id"] not in seen_ids:
                all_chunks.append(chunk)
                seen_ids.add(chunk["id"])

    if not all_chunks:
        return {
            "answer": "I couldn't find any relevant information.",
            "sources": [],
            "chunks_used": 0
        }

    # Step 2: Rerank merged results against the ORIGINAL query
    if use_reranker:
        all_chunks = rerank(query, all_chunks, top_k=top_k)
        # Filter out low-relevance chunks that survived reranking but
        # don't actually belong in context (cross-document contamination guard)
        MIN_RERANK_SCORE = 0  # cross-encoder scores below 0 are generally irrelevant
        all_chunks = [c for c in all_chunks if c.get("rerank_score", 1) > MIN_RERANK_SCORE]
    else:
        all_chunks = all_chunks[:top_k]

    # Step 3: Build context
    context = build_context(all_chunks)

    # Step 4: Generate
    prompt = f"""You are a helpful assistant. Answer the question using 
only the context below. Cite sources like [Source 1].
If the answer isn't in the context, say "I don't have enough information."

Context:
{context}

Question: {query}

Answer:"""

    response = requests.post("http://localhost:11434/api/generate", json={
        "model": "llama3.2",
        "prompt": prompt,
        "stream": False
    })

    answer = response.json()["response"]

    return {
        "answer": answer,
        "sources": list(set(c["source"] for c in all_chunks)),
        "chunks_used": len(all_chunks),
        "similarity_scores": [
            round(c.get("rerank_score", c.get("rrf_score", c.get("similarity", 0))), 4)
            for c in all_chunks
        ],
        "reranked": use_reranker,
        "hybrid": use_hybrid,
        "decomposed": use_decomposition,
        "sub_queries": sub_queries
    }

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python generator.py '<your question>'")
        sys.exit(1)

    query = sys.argv[1]
    print(f"\nQuestion: {query}\n")

    result = generate_answer(query)
    print(f"Answer:\n{result['answer']}")
    print(f"\nSources: {result['sources']}")
    print(f"Chunks used: {result['chunks_used']}")
    print(f"Reranked: {result['reranked']}")