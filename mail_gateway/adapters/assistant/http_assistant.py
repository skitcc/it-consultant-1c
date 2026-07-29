"""HTTP client for the AI assistant service."""

from __future__ import annotations

import logging

import httpx

from mail_gateway.domain.models import IncomingMessage
from mail_gateway.ports import Assistant

logger = logging.getLogger(__name__)


class HttpAssistant(Assistant):
    def __init__(self, base_url: str, *, timeout_sec: float = 120.0) -> None:
        self._url = base_url.rstrip("/")
        self._timeout = timeout_sec

    def ask(self, message: IncomingMessage) -> str | None:
        payload = {
            "conversation_id": message.conversation_id,
            "from": message.from_address,
            "subject": message.subject,
            "body": message.body,
        }
        logger.info("Calling AI service conversation_id=%s", message.conversation_id)
        with httpx.Client(timeout=self._timeout) as client:
            response = client.post(self._url, json=payload)
            response.raise_for_status()
            data = response.json()

        reply = data.get("reply")
        if reply is None:
            return None
        text = str(reply).strip()
        return text or None
