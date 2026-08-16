"""Integration: real Chroma server lifecycle + ingest/search over HTTP."""

import pytest

from audio_rag import vectordb
from audio_rag.ingest import upload_transcript
from audio_rag.search import search

pytestmark = pytest.mark.integration

PORT = 8971  # off the default port so a running dev server is untouched


@pytest.fixture
def server(tmp_path):
    data_dir = tmp_path / "chroma"
    msg = vectordb.start_server(data_dir=data_dir, host="127.0.0.1", port=PORT)
    assert "Chroma" in msg
    yield data_dir
    print(vectordb.stop_server(data_dir=data_dir))


def test_server_lifecycle_and_http_roundtrip(server, tmp_path, fake_embedder):
    assert vectordb.is_running("127.0.0.1", PORT)

    client = vectordb.get_client("127.0.0.1", PORT)
    collection = vectordb.get_collection(client, "integration-test")

    transcript = tmp_path / "talk.txt"
    transcript.write_text(
        "The speaker explained retrieval augmented generation and vector databases. "
        + "Unrelated filler about weather. " * 80
    )
    embedder = fake_embedder
    result = upload_transcript(collection, embedder, transcript)
    assert result.chunks >= 1

    chunks = search(collection, embedder, "retrieval augmented generation vector databases")
    assert chunks and chunks[0].source == "talk.txt"


def test_start_is_idempotent(server):
    msg = vectordb.start_server(data_dir=server, host="127.0.0.1", port=PORT)
    assert "already running" in msg


def test_require_client_error_when_down():
    with pytest.raises(ConnectionError, match="db-start"):
        vectordb.require_client("127.0.0.1", 8999)
