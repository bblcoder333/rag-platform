import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.ingestion.chunker import chunk_document
from rag.ingestion.document_loader import load_text_file


def test_chunker_fixed():
    text = "This is a test document. " * 100
    chunks = chunk_document(text, strategy="fixed", chunk_size=100)
    assert len(chunks) > 0
    assert all("text" in c for c in chunks)
    assert all("strategy" in c for c in chunks)


def test_chunker_recursive():
    text = "This is a test document.\n\nSecond paragraph here. " * 50
    chunks = chunk_document(text, strategy="recursive", chunk_size=100)
    assert len(chunks) > 0
    assert chunks[0]["strategy"] == "recursive"


def test_chunk_sizes():
    text = "word " * 500
    chunks = chunk_document(text, strategy="fixed", chunk_size=200)
    for chunk in chunks[:-1]:
        assert chunk["char_count"] <= 220


def test_load_text_file(tmp_path):
    test_file = tmp_path / "test.txt"
    test_file.write_text("Hello this is a test document.")
    doc = load_text_file(str(test_file))
    assert doc["source"] == "test.txt"
    assert "Hello" in doc["content"]
    assert doc["num_pages"] == 1