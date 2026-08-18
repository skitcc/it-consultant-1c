from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal


Role = Literal["user", "assistant"]


@dataclass(frozen=True, slots=True)
class ConversationTurn:
    role: Role
    body: str
    from_address: str = ""
    subject: str = ""
    item_id: str | None = None
    at: str | None = None


@dataclass(frozen=True, slots=True)
class DocumentChunk:
    text: str
    source_path: str
    chunk_index: int = 0
    score: float | None = None
    headings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class IncomingMessage:
    conversation_id: str
    item_id: str
    change_key: str
    from_address: str
    subject: str
    body: str
    message_id: str | None = None
    messages: tuple[ConversationTurn, ...] = field(default_factory=tuple)
    rag_context: str | None = None


@dataclass(frozen=True, slots=True)
class Reply:
    conversation_id: str
    in_reply_to_item_id: str
    in_reply_to_change_key: str
    body: str


def turn_from_incoming(message: IncomingMessage, *, role: Role = "user") -> ConversationTurn:
    return ConversationTurn(
        role=role,
        body=message.body,
        from_address=message.from_address,
        subject=message.subject,
        item_id=message.item_id,
    )


def with_messages(
    message: IncomingMessage,
    messages: tuple[ConversationTurn, ...] | list[ConversationTurn],
) -> IncomingMessage:
    return replace(message, messages=tuple(messages))


def with_rag_context(message: IncomingMessage, rag_context: str | None) -> IncomingMessage:
    return replace(message, rag_context=rag_context)
