"""Split transcript text into overlapping word-window chunks."""

from __future__ import annotations


def chunk_text(text: str, chunk_size: int = 200, overlap: int = 40) -> list[str]:
    """Split `text` into chunks of ~`chunk_size` words with `overlap` words shared
    between consecutive chunks, so sentences cut at a boundary stay retrievable."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    words = text.split()
    if not words:
        return []
    if len(words) <= chunk_size:
        return [" ".join(words)]

    step = chunk_size - overlap
    chunks = []
    for start in range(0, len(words), step):
        window = words[start:start + chunk_size]
        chunks.append(" ".join(window))
        if start + chunk_size >= len(words):
            break
    return chunks
