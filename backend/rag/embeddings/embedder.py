from sentence_transformers import SentenceTransformer
from typing import List
import time

# Using BAAI/bge-small-en — small, fast, and surprisingly good
# This is a well-respected model used in production RAG systems
MODEL_NAME = "BAAI/bge-small-en"

# Load once at module level so it doesn't reload on every call
print(f"Loading embedding model: {MODEL_NAME}")
model = SentenceTransformer(MODEL_NAME)
print("Model loaded.")


def embed_chunks(chunks: List[dict]) -> List[dict]:
    """
    Takes a list of chunks from the chunker and adds an embedding to each.
    Returns the same chunks with an 'embedding' field added.
    """
    texts = [chunk["text"] for chunk in chunks]

    start = time.time()
    embeddings = model.encode(texts, show_progress_bar=True)
    elapsed = time.time() - start

    for i, chunk in enumerate(chunks):
        chunk["embedding"] = embeddings[i].tolist()

    print(f"\nEmbedded {len(chunks)} chunks in {elapsed:.2f}s")
    print(f"Embedding dimension: {len(embeddings[0])}")
    print(f"Avg time per chunk: {elapsed/len(chunks)*1000:.1f}ms")

    return chunks


def embed_query(query: str) -> List[float]:
    """Embed a single query string for retrieval."""
    embedding = model.encode([query])[0]
    return embedding.tolist()


if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

    from ingestion.document_loader import load_document
    from ingestion.chunker import chunk_document

    if len(sys.argv) < 2:
        print("Usage: python embedder.py <file_path>")
        sys.exit(1)

    # Load and chunk
    doc = load_document(sys.argv[1])
    chunks = chunk_document(doc["content"], strategy="recursive")
    print(f"\nLoaded {len(chunks)} chunks from {doc['source']}")

    # Embed
    embedded_chunks = embed_chunks(chunks)

    # Show results
    print(f"\nSample embedding (first 5 values):")
    print(embedded_chunks[0]["embedding"][:5])

    print(f"\nAll chunks embedded successfully.")
    print(f"Each chunk now has {len(embedded_chunks[0])} fields: "
          f"{list(embedded_chunks[0].keys())}")