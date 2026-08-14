"""Ollama embedding client shared by reindex and mail_gateway."""

from __future__ import annotations

import logging

import httpx

logger = logging.getLogger(__name__)


class OllamaEmbedder:
    """Calls Ollama ``/api/embeddings`` and returns a dense vector."""

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

        embedding = data.get("embedding") if isinstance(data, dict) else None
        if not isinstance(embedding, list) or not embedding:
            raise RuntimeError(f"Ollama returned no embedding: {data!r}")
        return [float(x) for x in embedding]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]
