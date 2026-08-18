"""Ollama chat backend via native /api/chat HTTP API."""

from __future__ import annotations

import logging

import httpx

from mail_gateway.adapters.assistant.payload import (
    build_assistant_payload,
    log_assistant_payload,
)
from mail_gateway.domain.models import IncomingMessage
from mail_gateway.ports import Assistant

logger = logging.getLogger(__name__)


class OllamaAssistant(Assistant):
    """Maps our payload to Ollama chat messages and returns the model reply."""

    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:11434",
        model: str,
        timeout_sec: float = 300.0,
        system_prompt: str | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout_sec
        self._system_prompt = system_prompt

    def ask(self, message: IncomingMessage) -> str | None:
        payload = build_assistant_payload(
            message,
            system_prompt=self._system_prompt,
        )
        log_assistant_payload(
            payload,
            destination=f"ollama:{self._base_url} model={self._model}",
        )

        ollama_messages: list[dict[str, str]] = [
            {"role": "system", "content": payload["system_prompt"]},
        ]
        for item in payload["messages"]:
            role = item["role"]
            if role not in {"user", "assistant"}:
                role = "user"
            ollama_messages.append({"role": role, "content": item["body"]})

        request_body = {
            "model": self._model,
            "messages": ollama_messages,
            "stream": False,
        }
        logger.info(
            "Calling Ollama model=%s conversation_id=%s messages=%s",
            self._model,
            message.conversation_id,
            len(payload["messages"]),
        )
        with httpx.Client(timeout=self._timeout) as client:
            response = client.post(
                f"{self._base_url}/api/chat",
                json=request_body,
            )
            response.raise_for_status()
            data = response.json()

        content = (
            (data.get("message") or {}).get("content")
            if isinstance(data, dict)
            else None
        )
        if content is None:
            logger.warning("Ollama returned empty message content: %s", data)
            return None
        text = str(content).strip()
        return text or None
