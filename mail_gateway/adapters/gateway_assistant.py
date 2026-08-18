"""HTTP assistant adapter backed by the shared API Gateway."""

from __future__ import annotations

import logging

import httpx

from mail_gateway.application.clean_email_body import clean_email_body
from mail_gateway.domain.models import ConversationTurn, IncomingMessage
from mail_gateway.ports import Assistant

logger = logging.getLogger(__name__)


class GatewayAssistant(Assistant):
    """Map an email thread to the Gateway's OpenAI-compatible chat endpoint."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str = "it-consultant",
        timeout_sec: float = 300.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._timeout = timeout_sec

    def ask(self, message: IncomingMessage) -> str | None:
        messages = _thread_messages(message)
        request_body = {
            "model": self._model,
            "messages": messages,
            "stream": False,
        }
        logger.info(
            "Calling API Gateway conversation_id=%s model=%s messages=%s",
            message.conversation_id,
            self._model,
            len(messages),
        )
        with httpx.Client(timeout=self._timeout) as client:
            response = client.post(
                f"{self._base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self._api_key}"},
                json=request_body,
            )
            response.raise_for_status()
            data = response.json()

        choices = data.get("choices") if isinstance(data, dict) else None
        if not isinstance(choices, list) or not choices:
            logger.warning("API Gateway returned no choices")
            return None
        choice = choices[0]
        content = (
            (choice.get("message") or {}).get("content")
            if isinstance(choice, dict)
            else None
        )
        text = str(content or "").strip()
        return text or None


def _thread_messages(message: IncomingMessage) -> list[dict[str, str]]:
    turns = message.messages or (
        ConversationTurn(
            role="user",
            body=message.body,
            from_address=message.from_address,
            subject=message.subject,
            item_id=message.item_id,
        ),
    )
    result: list[dict[str, str]] = []
    for turn in turns:
        content = clean_email_body(turn.body) or turn.body.strip()
        if not content:
            continue
        role = turn.role if turn.role in {"user", "assistant"} else "user"
        result.append({"role": role, "content": content})
    if result:
        return result
    return [{"role": "user", "content": message.body.strip()}]
