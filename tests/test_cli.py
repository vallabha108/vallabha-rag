import json

import pytest

from audio_rag import cli
from audio_rag.config import CONFIG_ENV


@pytest.fixture(autouse=True)
def isolated_config(tmp_path, monkeypatch):
    path = tmp_path / "audiorag.json"
    monkeypatch.setenv(CONFIG_ENV, str(path))
    monkeypatch.delenv("AUDIORAG_LLM_MODEL", raising=False)
    monkeypatch.delenv("AUDIORAG_EMBEDDING_MODEL", raising=False)
    return path


def run_cli(monkeypatch, *argv):
    monkeypatch.setattr("sys.argv", ["audiorag", *argv])
    cli.main()


def test_config_show_prints_defaults(monkeypatch, capsys):
    run_cli(monkeypatch, "config", "show")
    out = capsys.readouterr().out
    assert "llm_model = gemma4" in out
    assert "embedding_model = all-MiniLM-L6-v2" in out


def test_config_set_roundtrip(monkeypatch, capsys, isolated_config):
    run_cli(monkeypatch, "config", "set", "llm_model", "llama3")
    assert json.loads(isolated_config.read_text()) == {"llm_model": "llama3"}

    run_cli(monkeypatch, "config", "show")
    assert "llm_model = llama3" in capsys.readouterr().out


def test_config_set_rejects_unknown_key(monkeypatch):
    with pytest.raises(SystemExit):
        run_cli(monkeypatch, "config", "set", "whisper_model", "large")


def test_cli_flag_overrides_config(monkeypatch):
    import argparse

    args = argparse.Namespace(llm_model="phi3", embedding_model=None)
    config = cli._config_from_args(args)
    assert config.llm_model == "phi3"
    assert config.source["llm_model"] == "cli:--llm-model"
    assert config.embedding_model == "all-MiniLM-L6-v2"


def test_transcribe_no_files_exits(monkeypatch, tmp_path):
    empty = tmp_path / "nothing"
    empty.mkdir()
    with pytest.raises(SystemExit, match="No audio files"):
        run_cli(monkeypatch, "transcribe", str(empty))
