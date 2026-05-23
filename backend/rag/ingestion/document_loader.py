import os
from pathlib import Path
from pypdf import PdfReader


def load_pdf(file_path: str) -> dict:
    """Load a PDF and return its text content with metadata."""
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    reader = PdfReader(file_path)
    full_text = ""
    
    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            full_text += text
    
    return {
        "source": path.name,
        "file_path": str(path),
        "content": full_text,
        "num_pages": len(reader.pages),
        "file_size_kb": round(path.stat().st_size / 1024, 2)
    }


def load_text_file(file_path: str) -> dict:
    """Load a plain text or markdown file."""
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    
    return {
        "source": path.name,
        "file_path": str(path),
        "content": content,
        "num_pages": 1,
        "file_size_kb": round(path.stat().st_size / 1024, 2)
    }


def load_document(file_path: str) -> dict:
    """Auto-detect file type and load accordingly."""
    ext = Path(file_path).suffix.lower()
    
    if ext == ".pdf":
        return load_pdf(file_path)
    elif ext in [".txt", ".md"]:
        return load_text_file(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        doc = load_document(sys.argv[1])
        print(f"Loaded: {doc['source']}")
        print(f"Pages: {doc['num_pages']}")
        print(f"Size: {doc['file_size_kb']} KB")
        print(f"Characters: {len(doc['content'])}")
        print(f"\nFirst 500 chars:\n{doc['content'][:500]}")
    else:
        print("Usage: python document_loader.py <path_to_file>")