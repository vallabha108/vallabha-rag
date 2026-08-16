"""Integration: real sentence-transformers embeddings give sensible relevance."""

import pytest

from audio_rag.embeddings import Embedder

pytestmark = pytest.mark.integration


def test_real_model_semantic_relevance():
    pytest.importorskip("sentence_transformers")
    embedder = Embedder("all-MiniLM-L6-v2")

    [query, relevant, irrelevant] = embedder.embed([
        "How do I cut infrastructure spending?",
        "We agreed to reduce our cloud hosting costs by twenty percent.",
        "The cafeteria menu now includes vegetarian lasagna on Fridays.",
    ])

    def cosine(a, b):
        return sum(x * y for x, y in zip(a, b))  # vectors are normalized

    assert cosine(query, relevant) > cosine(query, irrelevant)
    assert len(query) == 384  # MiniLM-L6 dimensionality
