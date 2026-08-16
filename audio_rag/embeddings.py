"""Local embedding model wrapper (sentence-transformers, configurable)."""

from __future__ import annotations


class Embedder:
    """Embeds texts with a local sentence-transformers model.

    The model is loaded lazily on first use so that CLI commands which never
    embed anything (e.g. db-status) stay fast.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._load()
        vectors = model.encode(list(texts), normalize_embeddings=True)
        return [list(map(float, v)) for v in vectors]
