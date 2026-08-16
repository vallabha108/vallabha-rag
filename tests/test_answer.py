from audio_rag.answer import Answer, build_prompt, generate_answer
from audio_rag.search import Chunk

CHUNKS = [
    Chunk(text="We will cut cloud costs by 20 percent.", source="meeting.txt",
          chunk_index=3, score=0.91),
    Chunk(text="Marketing spend stays flat.", source="meeting.txt",
          chunk_index=7, score=0.72),
]


def test_build_prompt_contains_query_and_all_chunks():
    prompt = build_prompt("What was decided about costs?", CHUNKS)
    assert "What was decided about costs?" in prompt
    for chunk in CHUNKS:
        assert chunk.text in prompt
        assert chunk.source in prompt


def test_build_prompt_without_chunks_says_so():
    assert "(no context retrieved)" in build_prompt("q", [])


def test_generate_answer_calls_ollama_with_context(monkeypatch):
    import ollama

    captured = {}

    def fake_chat(model, messages):
        captured["model"] = model
        captured["messages"] = messages
        return {"message": {"content": "  Costs will drop 20%.  "}}

    monkeypatch.setattr(ollama, "chat", fake_chat)

    answer = generate_answer("What about costs?", CHUNKS, llm_model="gemma4")

    assert isinstance(answer, Answer)
    assert answer.text == "Costs will drop 20%."
    assert answer.model == "gemma4"
    assert answer.chunks == CHUNKS
    assert captured["model"] == "gemma4"
    system, user = captured["messages"]
    assert system["role"] == "system"
    assert "cloud costs" in user["content"]
