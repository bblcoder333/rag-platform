from typing import List


def chunk_fixed(text: str, chunk_size: int = 512, overlap: int = 50) -> List[dict]:
    """
    Strategy 1: Fixed-size chunking.
    Splits text into chunks of exactly chunk_size characters with overlap.
    Simple but doesn't respect sentence boundaries.
    """
    chunks = []
    start = 0
    chunk_index = 0

    while start < len(text):
        end = start + chunk_size
        chunk_text = text[start:end]

        if chunk_text.strip():
            chunks.append({
                "chunk_index": chunk_index,
                "text": chunk_text,
                "strategy": "fixed",
                "chunk_size": chunk_size,
                "overlap": overlap,
                "char_count": len(chunk_text)
            })
            chunk_index += 1

        start += chunk_size - overlap

    return chunks


def chunk_recursive(text: str, chunk_size: int = 512, overlap: int = 50) -> List[dict]:
    """
    Strategy 2: Recursive character splitting.
    Tries to split on paragraphs, then sentences, then words.
    Respects natural text boundaries — better quality chunks.
    """
    separators = ["\n\n", "\n", ". ", " "]
    chunks = []

    def split_text(text: str, separators: List[str]) -> List[str]:
        if not separators or len(text) <= chunk_size:
            return [text]

        separator = separators[0]
        splits = text.split(separator)
        result = []
        current = ""

        for split in splits:
            if len(current) + len(split) + len(separator) <= chunk_size:
                current += split + separator
            else:
                if current.strip():
                    result.append(current.strip())
                if len(split) > chunk_size:
                    result.extend(split_text(split, separators[1:]))
                    current = ""
                else:
                    current = split + separator

        if current.strip():
            result.append(current.strip())

        return result

    raw_chunks = split_text(text, separators)

    for i, chunk_text in enumerate(raw_chunks):
        if chunk_text.strip():
            chunks.append({
                "chunk_index": i,
                "text": chunk_text,
                "strategy": "recursive",
                "chunk_size": chunk_size,
                "overlap": overlap,
                "char_count": len(chunk_text)
            })

    return chunks


def chunk_document(text: str, strategy: str = "recursive",
                   chunk_size: int = 512, overlap: int = 50) -> List[dict]:
    """Main entry point — choose your chunking strategy."""
    if strategy == "fixed":
        return chunk_fixed(text, chunk_size, overlap)
    elif strategy == "recursive":
        return chunk_recursive(text, chunk_size, overlap)
    else:
        raise ValueError(f"Unknown strategy: {strategy}. Use 'fixed' or 'recursive'.")


if __name__ == "__main__":
    # Benchmark both strategies against each other
    from document_loader import load_document
    import sys

    if len(sys.argv) < 2:
        print("Usage: python chunker.py <file_path>")
        sys.exit(1)

    doc = load_document(sys.argv[1])
    text = doc["content"]

    print(f"Document: {doc['source']}")
    print(f"Total characters: {len(text)}")
    print("=" * 50)

    for strategy in ["fixed", "recursive"]:
        chunks = chunk_document(text, strategy=strategy)
        sizes = [c["char_count"] for c in chunks]
        print(f"\nStrategy: {strategy.upper()}")
        print(f"  Total chunks:    {len(chunks)}")
        print(f"  Avg chunk size:  {sum(sizes) // len(sizes)} chars")
        print(f"  Min chunk size:  {min(sizes)} chars")
        print(f"  Max chunk size:  {max(sizes)} chars")
        print(f"\n  Sample chunk:")
        print(f"  {chunks[0]['text'][:200]}...")