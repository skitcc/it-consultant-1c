"""Map Exchange Message objects to domain models."""

from __future__ import annotations

from exchangelib import Message

from mail_gateway.domain.models import IncomingMessage


def _email_address(message: Message) -> str:
    if message.sender and message.sender.email_address:
        return message.sender.email_address
    if message.author and message.author.email_address:
        return message.author.email_address
    return ""


def _body_text(message: Message) -> str:
    body = message.text_body
    if body:
        return str(body).strip()
    if message.body is None:
        return ""
    return str(message.body).strip()


def _conversation_id(message: Message) -> str:
    cid = message.conversation_id
    if cid is None:
        return message.message_id or message.id or ""
    value = getattr(cid, "id", None)
    return str(value if value is not None else cid)


def message_to_incoming(message: Message) -> IncomingMessage:
    return IncomingMessage(
        conversation_id=_conversation_id(message),
        item_id=str(message.id),
        change_key=str(message.changekey),
        from_address=_email_address(message),
        subject=str(message.subject or ""),
        body=_body_text(message),
        message_id=str(message.message_id) if message.message_id else None,
    )
