"""Semantic search over the transcript collection."""

from __future__ import annotations

from dataclasses import dataclass

from .embeddings import Embedder


@dataclass
class Chunk:
    text: str
    source: str
    chunk_index: int
    score: float  # cosine similarity in [0, 1]; higher is more relevant


def search(collection, embedder: Embedder, query: str, top_k: int = 5) -> list[Chunk]:
    if not query.strip():
        raise ValueError("Query must not be empty")
    if collection.count() == 0:
        return []

    [embedding] = embedder.embed([query])
    result = collection.query(
        query_embeddings=[embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )
    chunks = []
    for doc, meta, dist in zip(
        result["documents"][0], result["metadatas"][0], result["distances"][0]
    ):
        chunks.append(
            Chunk(
                text=doc,
                source=str(meta.get("source", "?")),
                chunk_index=int(meta.get("chunk_index", -1)),
                score=1.0 - float(dist),  # cosine distance -> similarity
            )
        )
    return chunks
