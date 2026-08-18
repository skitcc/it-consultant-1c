"""Ollama embedding client shared by reindex and mail_gateway."""

from __future__ import annotations

import logging

import httpx

logger = logging.getLogger(__name__)


class OllamaEmbedder:
    """Calls Ollama ``/api/embed`` and returns a dense vector."""

    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:11434",
        model: str = "nomic-embed-text",
        timeout_sec: float = 120.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout_sec

    @property
    def model(self) -> str:
        return self._model

    def embed(self, text: str) -> list[float]:
        prompt = text.strip()
        if not prompt:
            raise ValueError("Cannot embed empty text")

        with httpx.Client(timeout=self._timeout) as client:
            response = client.post(
                f"{self._base_url}/api/embed",
                json={"model": self._model, "input": prompt},
            )
            response.raise_for_status()
            data = response.json()

        embedding = _vector_from_ollama(data)
        if embedding is None:
            keys = list(data) if isinstance(data, dict) else type(data).__name__
            raise RuntimeError(f"Ollama returned no embedding; keys={keys}")
        return embedding

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]


def _vector_from_ollama(data: object) -> list[float] | None:
    """Accept both ``embedding`` (legacy) and ``embeddings`` (``/api/embed``)."""
    if not isinstance(data, dict):
        return None
    embedding = data.get("embedding")
    parsed = _as_vector(embedding)
    if parsed is not None:
        return parsed
    embeddings = data.get("embeddings")
    if isinstance(embeddings, list) and embeddings:
        return _as_vector(embeddings[0])
    return None


def _as_vector(value: object) -> list[float] | None:
    if not isinstance(value, list) or not value:
        return None
    if isinstance(value[0], list):
        return _as_vector(value[0])
    try:
        return [float(x) for x in value]
    except (TypeError, ValueError):
        return None
