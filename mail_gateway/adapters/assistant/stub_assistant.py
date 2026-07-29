"""Fixed reply for local debugging without an AI service."""

from __future__ import annotations

from mail_gateway.domain.models import IncomingMessage
from mail_gateway.ports import Assistant


class StubAssistant(Assistant):
    def __init__(self, reply_text: str | None = None) -> None:
        self._reply_text = (
            reply_text
            if reply_text is not None
            else "Это тестовый ответ почтового шлюза (stub)."
        )

    def ask(self, message: IncomingMessage) -> str | None:
        return (
            f"{self._reply_text}\n\n"
            f"conversation_id={message.conversation_id}\n"
            f"subject={message.subject}"
        )
