"""Integration: end-to-end RAG answer through a local Ollama model."""

import pytest

from audio_rag.answer import generate_answer
from audio_rag.config import Config
from audio_rag.search import Chunk

pytestmark = pytest.mark.integration


def ollama_model_available(model: str) -> bool:
    try:
        import ollama

        names = [m.model for m in ollama.list().models]
        return any(n == model or n.startswith(f"{model}:") for n in names)
    except Exception:  # noqa: BLE001 - daemon not running / not installed
        return False


def test_ask_llm_summarizes_from_context():
    config = Config.load()
    if not ollama_model_available(config.llm_model):
        pytest.skip(f"Ollama model '{config.llm_model}' not available locally")

    chunks = [
        Chunk(
            text="In the meeting we decided to reduce cloud hosting costs by "
                 "twenty percent before the end of next quarter.",
            source="meeting.txt", chunk_index=0, score=0.9,
        )
    ]
    answer = generate_answer("What was decided about cloud costs?", chunks, config.llm_model)

    assert answer.text
    assert "twenty" in answer.text.lower() or "20" in answer.text
