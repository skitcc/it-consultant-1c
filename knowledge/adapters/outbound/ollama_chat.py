"""Ollama native chat adapter."""

from __future__ import annotations

from collections.abc import Sequence

import httpx

from knowledge.core.domain import ConversationMessage


class OllamaChatModel:
    def __init__(
        self,
        *,
        model: str,
        base_url: str = "http://127.0.0.1:11434",
        timeout_sec: float = 300.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout_sec

    def complete(self, messages: Sequence[ConversationMessage]) -> str:
        payload = {
            "model": self._model,
            "messages": [
                {"role": message.role, "content": message.content}
                for message in messages
            ],
            "stream": False,
        }
        with httpx.Client(timeout=self._timeout) as client:
            response = client.post(f"{self._base_url}/api/chat", json=payload)
            response.raise_for_status()
            data = response.json()
        if not isinstance(data, dict) or not isinstance(data.get("message"), dict):
            raise RuntimeError("Ollama returned no chat message")
        content = str(data["message"].get("content") or "").strip()
        if not content:
            raise RuntimeError("Ollama returned empty chat content")
        return content
