"""Shared fixtures: an in-memory Chroma collection and a deterministic fake embedder."""

from __future__ import annotations

import hashlib

import pytest


class FakeEmbedder:
    """Deterministic bag-of-hashed-words embedder (no ML model needed).

    Texts sharing words get similar vectors, so relevance ordering is testable.
    Mirrors the audio_rag.embeddings.Embedder interface.
    """

    model_name = "fake-embedder"
    DIM = 64

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            vec = [0.0] * self.DIM
            for word in text.lower().split():
                idx = int(hashlib.md5(word.encode()).hexdigest(), 16) % self.DIM
                vec[idx] += 1.0
            norm = sum(v * v for v in vec) ** 0.5 or 1.0
            vectors.append([v / norm for v in vec])
        return vectors


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def collection():
    chromadb = pytest.importorskip("chromadb")
    client = chromadb.EphemeralClient()
    yield client.get_or_create_collection("test-audio", metadata={"hnsw:space": "cosine"})
    client.delete_collection("test-audio")
