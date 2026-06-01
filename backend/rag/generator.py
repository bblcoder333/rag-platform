import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
from rag.retrieval.retriever import retrieve


def build_context(chunks: list) -> str:
    """Format retrieved chunks into a context string."""
    context_parts = []
    for i, chunk in enumerate(chunks):
        context_parts.append(
            f"[Source {i+1} - {chunk['source']} "
            f"(similarity: {chunk['similarity']})]:\n{chunk['text']}"
        )
    return "\n\n".join(context_parts)


def generate_answer(query: str, top_k: int = 5) -> dict:
    """
    Full RAG pipeline: retrieve relevant chunks then generate an answer.
    """
    # Step 1: Retrieve
    chunks = retrieve(query, top_k=top_k)

    if not chunks:
        return {
            "answer": "I couldn't find any relevant information.",
            "sources": [],
            "chunks_used": 0
        }

    # Step 2: Build context
    context = build_context(chunks)

    # Step 3: Generate with Ollama
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
        "similarity_scores": [c["similarity"] for c in chunks]
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
    print(f"Similarity scores: {result['similarity_scores']}")