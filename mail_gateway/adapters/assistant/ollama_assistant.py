"""Ollama chat backend via native /api/chat HTTP API."""

from __future__ import annotations

import logging
import time

import httpx

from mail_gateway.adapters.assistant.payload import (
    build_assistant_payload,
    build_verifier_payload,
    log_assistant_payload,
)
from mail_gateway.domain.models import IncomingMessage
from mail_gateway.ports import Assistant

logger = logging.getLogger(__name__)


class OllamaAssistant(Assistant):
    """Two-layer grounded chat: draft, then verify against the same chunks."""

    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:11434",
        model: str,
        timeout_sec: float = 300.0,
        temperature: float = 0.0,
        top_p: float = 0.1,
        system_prompt: str | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout_sec
        self._temperature = temperature
        self._top_p = top_p
        self._system_prompt = system_prompt

    def ask(self, message: IncomingMessage) -> str | None:
        payload = build_assistant_payload(
            message,
            system_prompt=self._system_prompt,
        )
        log_assistant_payload(
            payload,
            destination=f"ollama:{self._base_url} model={self._model} layer=1",
        )
        draft = self._chat(payload, layer=1, conversation_id=message.conversation_id)
        if not draft:
            return None

        rag_context = (message.rag_context or "").strip()
        if not rag_context:
            return draft

        verifier = build_verifier_payload(
            conversation_id=message.conversation_id,
            draft=draft,
            rag_context=rag_context,
        )
        log_assistant_payload(
            verifier,
            destination=f"ollama:{self._base_url} model={self._model} layer=2",
        )
        verified = self._chat(
            verifier,
            layer=2,
            conversation_id=message.conversation_id,
        )
        if not verified:
            logger.warning(
                "Verifier returned empty conversation_id=%s; dropping draft",
                message.conversation_id,
            )
            return None
        return verified

    def _chat(self, payload: dict, *, layer: int, conversation_id: str) -> str | None:
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
            "think": True,
            "options": {
                "temperature": self._temperature,
                "top_p": self._top_p,
            },
        }
        started_at = time.perf_counter()
        logger.info(
            "Calling Ollama layer=%s model=%s conversation_id=%s messages=%s "
            "temperature=%s top_p=%s",
            layer,
            self._model,
            conversation_id,
            len(payload["messages"]),
            self._temperature,
            self._top_p,
        )
        with httpx.Client(timeout=self._timeout) as client:
            response = client.post(
                f"{self._base_url}/api/chat",
                json=request_body,
            )
            response.raise_for_status()
            data = response.json()

        elapsed = time.perf_counter() - started_at
        message = data.get("message") if isinstance(data, dict) else None
        thinking = ""
        content = None
        if isinstance(message, dict):
            thinking = str(message.get("thinking") or "").strip()
            content = message.get("content")
        if thinking:
            logger.debug(
                "Ollama thinking layer=%s conversation_id=%s chars=%s",
                layer,
                conversation_id,
                len(thinking),
            )
        logger.info(
            "Ollama layer=%s done conversation_id=%s elapsed=%.3fs content_chars=%s",
            layer,
            conversation_id,
            elapsed,
            len(str(content or "")),
        )
        if content is None:
            logger.warning(
                "Ollama layer=%s returned empty message content conversation_id=%s",
                layer,
                conversation_id,
            )
            return None
        text = str(content).strip()
        return text or None
