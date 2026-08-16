from audio_rag.ingest import IngestResult, list_sources, upload_transcript
from audio_rag.search import search


def make_transcript(tmp_path, name="meeting.txt", text=None):
    text = text or (
        "The quarterly budget review covered marketing spend and cloud costs. "
        "We agreed to reduce cloud costs by twenty percent next quarter. "
        + "Filler discussion about scheduling. " * 100
    )
    path = tmp_path / name
    path.write_text(text)
    return path


def test_upload_creates_chunks_with_metadata(tmp_path, collection, fake_embedder):
    path = make_transcript(tmp_path)
    result = upload_transcript(collection, fake_embedder, path)

    assert isinstance(result, IngestResult)
    assert result.chunks > 1 and not result.skipped
    assert collection.count() == result.chunks

    stored = collection.get(include=["metadatas"])
    assert all(m["source"] == "meeting.txt" for m in stored["metadatas"])
    assert {m["chunk_index"] for m in stored["metadatas"]} == set(range(result.chunks))
    assert stored["metadatas"][0]["embedding_model"] == "fake-embedder"


def test_duplicate_upload_skipped_unless_forced(tmp_path, collection, fake_embedder):
    path = make_transcript(tmp_path)
    first = upload_transcript(collection, fake_embedder, path)
    again = upload_transcript(collection, fake_embedder, path)
    assert again.skipped and collection.count() == first.chunks

    forced = upload_transcript(collection, fake_embedder, path, force=True)
    assert not forced.skipped
    assert collection.count() == forced.chunks  # replaced, not duplicated


def test_empty_transcript_is_skipped(tmp_path, collection, fake_embedder):
    path = tmp_path / "empty.txt"
    path.write_text("   ")
    result = upload_transcript(collection, fake_embedder, path)
    assert result.skipped and collection.count() == 0


def test_list_sources_counts_chunks(tmp_path, collection, fake_embedder):
    upload_transcript(collection, fake_embedder, make_transcript(tmp_path, "a.txt"))
    upload_transcript(collection, fake_embedder,
                      make_transcript(tmp_path, "b.txt", text="short transcript"))
    sources = list_sources(collection)
    assert set(sources) == {"a.txt", "b.txt"}
    assert sources["b.txt"] == 1


def test_search_returns_relevant_chunk_first(tmp_path, collection, fake_embedder):
    upload_transcript(collection, fake_embedder, make_transcript(tmp_path))
    chunks = search(collection, fake_embedder,
                    "reduce cloud costs budget quarterly", top_k=3)
    assert chunks
    assert "cloud costs" in chunks[0].text
    assert chunks[0].source == "meeting.txt"
    # results come back best-first
    scores = [c.score for c in chunks]
    assert scores == sorted(scores, reverse=True)


def test_search_respects_top_k_and_empty_collection(tmp_path, collection, fake_embedder):
    assert search(collection, fake_embedder, "anything", top_k=3) == []
    upload_transcript(collection, fake_embedder, make_transcript(tmp_path))
    assert len(search(collection, fake_embedder, "budget", top_k=2)) == 2


def test_search_rejects_empty_query(collection, fake_embedder):
    import pytest

    with pytest.raises(ValueError):
        search(collection, fake_embedder, "   ")
