"""Configuration for the Audio RAG pipeline.

Two knobs are user-configurable (LLM and embedding model); everything else is
a fixed convention of the pipeline. Precedence: CLI flag > environment
variable > config file (audiorag.json) > built-in default.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

CONFIG_ENV = "AUDIORAG_CONFIG"
DEFAULT_CONFIG_FILE = "audiorag.json"

CONFIGURABLE_KEYS = ("llm_model", "embedding_model")

DEFAULTS = {
    "llm_model": "gemma4",                # any model available in `ollama list`
    "embedding_model": "all-MiniLM-L6-v2",  # any sentence-transformers model
}

ENV_OVERRIDES = {
    "llm_model": "AUDIORAG_LLM_MODEL",
    "embedding_model": "AUDIORAG_EMBEDDING_MODEL",
}

# Fixed conventions (one-time setup, not part of the user-facing config).
CHROMA_DIR = ".chroma"
CHROMA_HOST = "127.0.0.1"
CHROMA_PORT = 8765
COLLECTION_NAME = "audio_transcripts"
TRANSCRIPTS_DIR = "transcripts"
WHISPER_MODEL = "base"
CHUNK_SIZE_WORDS = 200
CHUNK_OVERLAP_WORDS = 40


def config_path() -> Path:
    return Path(os.environ.get(CONFIG_ENV, DEFAULT_CONFIG_FILE))


@dataclass
class Config:
    llm_model: str = DEFAULTS["llm_model"]
    embedding_model: str = DEFAULTS["embedding_model"]
    source: dict = field(default_factory=dict)  # where each value came from

    @classmethod
    def load(cls, path: Path | None = None) -> "Config":
        path = path or config_path()
        values = dict(DEFAULTS)
        source = {k: "default" for k in values}

        if path.is_file():
            try:
                data = json.loads(path.read_text())
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}: {exc}") from exc
            for key in CONFIGURABLE_KEYS:
                if key in data:
                    values[key] = str(data[key])
                    source[key] = str(path)

        for key, env in ENV_OVERRIDES.items():
            if os.environ.get(env):
                values[key] = os.environ[env]
                source[key] = f"env:{env}"

        return cls(llm_model=values["llm_model"],
                   embedding_model=values["embedding_model"],
                   source=source)

    def set_and_save(self, key: str, value: str, path: Path | None = None) -> None:
        if key not in CONFIGURABLE_KEYS:
            raise KeyError(
                f"Unknown config key '{key}'. Configurable keys: {', '.join(CONFIGURABLE_KEYS)}"
            )
        path = path or config_path()
        data = {}
        if path.is_file():
            data = json.loads(path.read_text())
        data[key] = value
        path.write_text(json.dumps(data, indent=2) + "\n")
        setattr(self, key, value)
        self.source[key] = str(path)
