"""Build and log the payload sent to the AI assistant."""

from __future__ import annotations

import json
import logging

from mail_gateway.application.clean_email_body import clean_email_body
from mail_gateway.domain.models import ConversationTurn, IncomingMessage

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = (
    "Ты IT-консультант по технической документации компании. "
    "Отвечай кратко и по делу на русском языке. "
    "Опирайся только на переписку пользователя и доступную документацию. "
    "Если данных недостаточно для уверенного ответа — скажи об этом прямо "
    "и предложи обратиться к администратору. "
    "Не выдумывай факты, ссылки и настройки."
)


def turns_to_payload_messages(
    turns: tuple[ConversationTurn, ...] | list[ConversationTurn],
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for turn in turns:
        body = clean_email_body(turn.body)
        if not body:
            continue
        messages.append({"role": turn.role, "body": body})
    return messages


def build_assistant_payload(
    message: IncomingMessage,
    *,
    system_prompt: str | None = None,
) -> dict:
    messages = message.messages
    if not messages:
        messages = (
            ConversationTurn(
                role="user",
                body=message.body,
                from_address=message.from_address,
                subject=message.subject,
                item_id=message.item_id,
            ),
        )

    payload_messages = turns_to_payload_messages(messages)
    if not payload_messages:
        body = clean_email_body(message.body) or message.body.strip()
        payload_messages = [{"role": "user", "body": body}]

    prompt = (system_prompt or DEFAULT_SYSTEM_PROMPT).strip()
    return {
        "conversation_id": message.conversation_id,
        "system_prompt": prompt,
        "messages": payload_messages,
    }


def log_assistant_payload(payload: dict, *, destination: str) -> None:
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    logger.info(
        "Assistant payload destination=%s conversation_id=%s messages=%s\n%s",
        destination,
        payload.get("conversation_id"),
        len(payload.get("messages") or []),
        rendered,
    )
