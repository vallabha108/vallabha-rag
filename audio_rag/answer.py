"""RAG answer generation: retrieved chunks + query -> local LLM via Ollama."""

from __future__ import annotations

from dataclasses import dataclass

from .search import Chunk

SYSTEM_PROMPT = (
    "You are a helpful assistant answering questions about audio recordings, "
    "using only the transcript excerpts provided as context. If the context "
    "does not contain the answer, say so instead of guessing. "
    "Give a concise, well-summarized answer."
)


@dataclass
class Answer:
    text: str
    model: str
    chunks: list[Chunk]


def build_prompt(query: str, chunks: list[Chunk]) -> str:
    context_blocks = [
        f"[{i}] (source: {c.source}, chunk {c.chunk_index}, relevance {c.score:.2f})\n{c.text}"
        for i, c in enumerate(chunks, 1)
    ]
    context = "\n\n".join(context_blocks) if context_blocks else "(no context retrieved)"
    return (
        f"Context excerpts from audio transcripts:\n\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer using only the context above."
    )


def generate_answer(query: str, chunks: list[Chunk], llm_model: str) -> Answer:
    import ollama

    response = ollama.chat(
        model=llm_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(query, chunks)},
        ],
    )
    return Answer(text=response["message"]["content"].strip(), model=llm_model, chunks=chunks)
