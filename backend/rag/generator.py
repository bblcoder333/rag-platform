import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
from rag.retrieval.retriever import retrieve
from rag.retrieval.reranker import rerank


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
                    use_reranker: bool = True) -> dict:
    """
    Full RAG pipeline: retrieve → rerank → generate.
    """
    # Step 1: Retrieve more candidates than we need
    retrieval_k = top_k * 2 if use_reranker else top_k
    chunks = retrieve(query, top_k=retrieval_k)

    if not chunks:
        return {
            "answer": "I couldn't find any relevant information.",
            "sources": [],
            "chunks_used": 0
        }

    # Step 2: Rerank and take top_k
    if use_reranker:
        chunks = rerank(query, chunks, top_k=top_k)

    # Step 3: Build context
    context = build_context(chunks)

    # Step 4: Generate with Ollama
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
        "sources": list(set(c["source"] for c in chunks)),
        "chunks_used": len(chunks),
        "similarity_scores": [c["similarity"] for c in chunks],
        "reranked": use_reranker
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