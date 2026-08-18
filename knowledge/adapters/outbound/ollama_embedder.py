"""Ollama embedding adapter."""

from __future__ import annotations

import httpx


class OllamaEmbedder:
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
        vectors = self.embed_documents([text])
        return vectors[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        prompts = [text.strip() for text in texts]
        if not prompts or any(not prompt for prompt in prompts):
            raise ValueError("Cannot embed an empty text")
        with httpx.Client(timeout=self._timeout) as client:
            response = client.post(
                f"{self._base_url}/api/embed",
                json={"model": self._model, "input": prompts},
            )
            response.raise_for_status()
            data = response.json()
        vectors = vectors_from_ollama(data)
        if vectors is None or len(vectors) != len(prompts):
            raise RuntimeError("Ollama returned an unexpected embedding response")
        return vectors


def vectors_from_ollama(data: object) -> list[list[float]] | None:
    if not isinstance(data, dict):
        return None
    raw = data.get("embeddings")
    if isinstance(raw, list) and raw:
        if isinstance(raw[0], list):
            try:
                return [[float(value) for value in vector] for vector in raw]
            except (TypeError, ValueError):
                return None
        vector = _as_vector(raw)
        return [vector] if vector is not None else None
    vector = _as_vector(data.get("embedding"))
    return [vector] if vector is not None else None


def _as_vector(value: object) -> list[float] | None:
    if not isinstance(value, list) or not value:
        return None
    try:
        return [float(item) for item in value]
    except (TypeError, ValueError):
        return None
