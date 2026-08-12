"""
MODULE 1a — Document loading & chunking.
Reads PDF / DOCX resumes into plain text, with light chunking utilities
reused later for career notes ingestion.
"""
from pathlib import Path
from typing import List

from pypdf import PdfReader
import docx


def load_pdf(path: str | Path) -> str:
    reader = PdfReader(str(path))
    text = []
    for page in reader.pages:
        text.append(page.extract_text() or "")
    return "\n".join(text).strip()


def load_docx(path: str | Path) -> str:
    d = docx.Document(str(path))
    return "\n".join(p.text for p in d.paragraphs if p.text.strip())


def load_resume(path: str | Path) -> str:
    """Dispatch on file extension. Raises for unsupported types."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return load_pdf(path)
    elif suffix == ".docx":
        return load_docx(path)
    else:
        raise ValueError(f"Unsupported resume format: {suffix} (use .pdf or .docx)")


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
    """
    Simple word-based sliding-window chunker.
    Used for career notes / job descriptions before embedding.
    """
    words = text.split()
    if not words:
        return []
    chunks = []
    step = max(chunk_size - overlap, 1)
    for start in range(0, len(words), step):
        chunk = " ".join(words[start:start + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
        if start + chunk_size >= len(words):
            break
    return chunks
