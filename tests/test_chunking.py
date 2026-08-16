import pytest

from audio_rag.chunking import chunk_text


def test_empty_text_gives_no_chunks():
    assert chunk_text("") == []
    assert chunk_text("   \n  ") == []


def test_short_text_is_single_chunk():
    text = "one two three four"
    assert chunk_text(text, chunk_size=10, overlap=2) == [text]


def test_chunks_cover_all_words_with_overlap():
    words = [f"w{i}" for i in range(500)]
    chunks = chunk_text(" ".join(words), chunk_size=200, overlap=40)

    seen = [w for chunk in chunks for w in chunk.split()]
    assert set(seen) == set(words)  # nothing lost
    # consecutive chunks share exactly `overlap` words
    first, second = chunks[0].split(), chunks[1].split()
    assert first[-40:] == second[:40]


def test_last_partial_chunk_kept():
    chunks = chunk_text(" ".join(str(i) for i in range(250)), chunk_size=200, overlap=0)
    assert len(chunks) == 2
    assert len(chunks[1].split()) == 50


def test_invalid_params_rejected():
    with pytest.raises(ValueError):
        chunk_text("a b c", chunk_size=0)
    with pytest.raises(ValueError):
        chunk_text("a b c", chunk_size=10, overlap=10)
    with pytest.raises(ValueError):
        chunk_text("a b c", chunk_size=10, overlap=-1)
