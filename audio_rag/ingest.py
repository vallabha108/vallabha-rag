"""Upload transcript text files into the Chroma collection as embedded chunks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from . import config as cfg
from .chunking import chunk_text
from .embeddings import Embedder


@dataclass
class IngestResult:
    source: str
    chunks: int
    skipped: bool = False


def source_exists(collection, source: str) -> bool:
    existing = collection.get(where={"source": source}, limit=1)
    return bool(existing["ids"])


def delete_source(collection, source: str) -> None:
    collection.delete(where={"source": source})


def upload_transcript(
    collection,
    embedder: Embedder,
    transcript_path: Path | str,
    chunk_size: int = cfg.CHUNK_SIZE_WORDS,
    overlap: int = cfg.CHUNK_OVERLAP_WORDS,
    force: bool = False,
) -> IngestResult:
    """Chunk, embed and store one transcript. `source` metadata = file name,
    so the same file is skipped on re-upload unless force=True."""
    transcript_path = Path(transcript_path)
    text = transcript_path.read_text()
    source = transcript_path.name

    if source_exists(collection, source):
        if not force:
            return IngestResult(source=source, chunks=0, skipped=True)
        delete_source(collection, source)

    chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
    if not chunks:
        return IngestResult(source=source, chunks=0, skipped=True)

    embeddings = embedder.embed(chunks)
    collection.add(
        ids=[f"{transcript_path.stem}-{i:04d}" for i in range(len(chunks))],
        documents=chunks,
        embeddings=embeddings,
        metadatas=[
            {"source": source, "chunk_index": i, "embedding_model": embedder.model_name}
            for i in range(len(chunks))
        ],
    )
    return IngestResult(source=source, chunks=len(chunks))


def list_sources(collection) -> dict[str, int]:
    """Map of source file name -> number of chunks stored."""
    data = collection.get(include=["metadatas"])
    counts: dict[str, int] = {}
    for meta in data["metadatas"] or []:
        src = str(meta.get("source", "?"))
        counts[src] = counts.get(src, 0) + 1
    return counts
