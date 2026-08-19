"""Load conversation history from Exchange by conversation_id."""

from __future__ import annotations

import logging
from collections.abc import Sequence

from exchangelib import Account, Message, Q
from exchangelib.properties import ConversationId

from mail_gateway.application.clean_email_body import clean_email_body
from mail_gateway.application.render_answer import strip_sources_footer
from mail_gateway.domain.models import ConversationTurn
from mail_gateway.ports import ConversationHistoryLoader

logger = logging.getLogger(__name__)


def _body_text(message: Message) -> str:
    body = message.text_body
    if body:
        raw = str(body)
    elif message.body is None:
        raw = ""
    else:
        raw = str(message.body)
    return clean_email_body(strip_sources_footer(_strip_html(raw)))


def _strip_html(text: str) -> str:
    if "<" not in text or ">" not in text:
        return text
    from html.parser import HTMLParser

    class _TextExtractor(HTMLParser):
        def __init__(self) -> None:
            super().__init__(convert_charrefs=True)
            self.parts: list[str] = []

        def handle_data(self, data: str) -> None:
            if data:
                self.parts.append(data)

        def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
            del attrs
            if tag.lower() in {"p", "br", "tr", "li", "h2", "h3"}:
                self.parts.append("\n")

    parser = _TextExtractor()
    parser.feed(text)
    parser.close()
    return "\n".join(line.strip() for line in "".join(parser.parts).splitlines())


def _from_address(message: Message) -> str:
    if message.sender and message.sender.email_address:
        return message.sender.email_address
    if message.author and message.author.email_address:
        return message.author.email_address
    return ""


def _received_at(message: Message) -> str | None:
    value = message.datetime_received
    return value.isoformat() if value is not None else None


class EwsConversationHistoryLoader(ConversationHistoryLoader):
    def __init__(self, account: Account, *, bot_email: str, limit: int = 50) -> None:
        self._account = account
        self._bot_email = bot_email.lower()
        self._limit = limit

    def load(self, conversation_id: str) -> Sequence[ConversationTurn]:
        if not conversation_id:
            logger.warning("Empty conversation_id; history skipped")
            return []

        cid = ConversationId(id=conversation_id)
        query = Q(conversation_id=cid)
        items: list[Message] = []

        for folder_name, folder in (
            ("inbox", self._account.inbox),
            ("sent", self._account.sent),
        ):
            try:
                found = list(
                    folder.filter(query).order_by("datetime_received")[: self._limit]
                )
            except Exception:
                logger.exception(
                    "Failed to load conversation history folder=%s conversation_id=%s",
                    folder_name,
                    conversation_id,
                )
                continue
            logger.info(
                "History scan folder=%s conversation_id=%s found=%s",
                folder_name,
                conversation_id,
                len(found),
            )
            items.extend(item for item in found if isinstance(item, Message))

        # Deduplicate by item id (same mail can match filters oddly).
        unique: dict[str, Message] = {}
        for item in items:
            unique[str(item.id)] = item

        ordered = sorted(
            unique.values(),
            key=lambda m: m.datetime_received or m.datetime_created or m.datetime_sent,
        )
        if len(ordered) > self._limit:
            ordered = ordered[-self._limit :]

        turns: list[ConversationTurn] = []
        for item in ordered:
            body = _body_text(item)
            if not body:
                logger.info(
                    "Skipping empty/cleaned message item_id=%s conversation_id=%s",
                    item.id,
                    conversation_id,
                )
                continue
            sender = _from_address(item).lower()
            role = "assistant" if sender == self._bot_email else "user"
            turns.append(
                ConversationTurn(
                    role=role,
                    body=body,
                    from_address=_from_address(item),
                    subject=str(item.subject or ""),
                    item_id=str(item.id) if item.id else None,
                    at=_received_at(item),
                )
            )

        logger.info(
            "Loaded conversation history conversation_id=%s turns=%s",
            conversation_id,
            len(turns),
        )
        return turns
