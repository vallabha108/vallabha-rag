import json

import pytest

from audio_rag.config import CONFIG_ENV, Config, DEFAULTS


@pytest.fixture(autouse=True)
def isolated_config(tmp_path, monkeypatch):
    """Point AUDIORAG_CONFIG at a temp file and clear model env overrides."""
    path = tmp_path / "audiorag.json"
    monkeypatch.setenv(CONFIG_ENV, str(path))
    monkeypatch.delenv("AUDIORAG_LLM_MODEL", raising=False)
    monkeypatch.delenv("AUDIORAG_EMBEDDING_MODEL", raising=False)
    return path


def test_defaults_when_no_file(isolated_config):
    config = Config.load()
    assert config.llm_model == DEFAULTS["llm_model"]
    assert config.embedding_model == DEFAULTS["embedding_model"]
    assert config.source["llm_model"] == "default"


def test_file_overrides_defaults(isolated_config):
    isolated_config.write_text(json.dumps({"llm_model": "llama3"}))
    config = Config.load()
    assert config.llm_model == "llama3"
    assert config.embedding_model == DEFAULTS["embedding_model"]
    assert config.source["llm_model"] == str(isolated_config)


def test_env_overrides_file(isolated_config, monkeypatch):
    isolated_config.write_text(json.dumps({"llm_model": "llama3"}))
    monkeypatch.setenv("AUDIORAG_LLM_MODEL", "mistral")
    config = Config.load()
    assert config.llm_model == "mistral"
    assert config.source["llm_model"] == "env:AUDIORAG_LLM_MODEL"


def test_set_and_save_persists(isolated_config):
    config = Config.load()
    config.set_and_save("embedding_model", "all-mpnet-base-v2")
    assert json.loads(isolated_config.read_text())["embedding_model"] == "all-mpnet-base-v2"
    assert Config.load().embedding_model == "all-mpnet-base-v2"


def test_set_unknown_key_rejected(isolated_config):
    with pytest.raises(KeyError):
        Config.load().set_and_save("whisper_model", "large")


def test_invalid_json_raises_clear_error(isolated_config):
    isolated_config.write_text("{not json")
    with pytest.raises(ValueError, match="Invalid JSON"):
        Config.load()
