from collections.abc import Iterator, Sequence
from typing import Protocol, runtime_checkable

from mail_gateway.domain.models import ConversationTurn, DocumentChunk, IncomingMessage, Reply


@runtime_checkable
class MailListener(Protocol):
    def listen(self) -> Iterator[IncomingMessage]:
        """Yield incoming messages until the subscription ends or errors."""
        ...


@runtime_checkable
class MailSender(Protocol):
    def send_reply(self, reply: Reply) -> None:
        """Send a reply into the same conversation."""
        ...


@runtime_checkable
class ConversationHistoryLoader(Protocol):
    def load(self, conversation_id: str) -> Sequence[ConversationTurn]:
        """Return conversation turns ordered from oldest to newest."""
        ...


@runtime_checkable
class DocumentRetriever(Protocol):
    def retrieve(self, query: str) -> Sequence[DocumentChunk]:
        """Return documentation chunks relevant to ``query``."""
        ...


@runtime_checkable
class Reranker(Protocol):
    def rerank(
        self,
        query: str,
        chunks: Sequence[DocumentChunk],
    ) -> Sequence[DocumentChunk]:
        """Return ``chunks`` ordered by relevance to ``query`` (best first)."""
        ...


@runtime_checkable
class Assistant(Protocol):
    def ask(self, message: IncomingMessage) -> str | None:
        """Return answer text, or None when no relevant answer exists.

        ``message.messages`` holds the full thread to send to the model.
        ``message.rag_context`` may hold retrieved documentation snippets.
        """
        ...
