"""EWS Streaming listener for NewMail events."""

from __future__ import annotations

import logging
from collections.abc import Iterator

from exchangelib import Account, Message
from exchangelib.properties import NewMailEvent

from mail_gateway.adapters.ews.mapping import message_to_incoming
from mail_gateway.domain.models import IncomingMessage
from mail_gateway.ports import MailListener

logger = logging.getLogger(__name__)


class EwsMailListener(MailListener):
    def __init__(
        self,
        account: Account,
        *,
        connection_timeout_minutes: int = 30,
    ) -> None:
        self._account = account
        self._connection_timeout_minutes = connection_timeout_minutes

    def listen(self) -> Iterator[IncomingMessage]:
        inbox = self._account.inbox
        subscription_id = inbox.subscribe_to_streaming(
            event_types=[NewMailEvent.ELEMENT_NAME]
        )
        logger.info("EWS streaming subscription started id=%s", subscription_id)
        try:
            for notification in inbox.get_streaming_events(
                subscription_id,
                connection_timeout=self._connection_timeout_minutes,
            ):
                for event in notification.events:
                    yield from self._handle_event(event)
        finally:
            try:
                self._account.unsubscribe(subscription_id)
            except Exception:
                logger.exception("Failed to unsubscribe id=%s", subscription_id)

    def _handle_event(self, event: object) -> Iterator[IncomingMessage]:
        if not isinstance(event, NewMailEvent):
            return

        item_id = event.item_id
        if item_id is None:
            return

        try:
            item = self._account.inbox.get(id=item_id.id, changekey=item_id.changekey)
        except Exception:
            logger.exception(
                "Failed to load item id=%s", getattr(item_id, "id", item_id)
            )
            return

        if not isinstance(item, Message):
            logger.debug(
                "Skipping non-message item id=%s type=%s",
                item_id.id,
                type(item),
            )
            return

        # Ignore mail we sent ourselves to avoid reply loops.
        own = self._account.primary_smtp_address.lower()
        sender = ""
        if item.sender and item.sender.email_address:
            sender = item.sender.email_address.lower()
        if sender == own:
            logger.debug("Skipping own outbound message id=%s", item_id.id)
            return

        incoming = message_to_incoming(item)
        logger.info(
            "New mail conversation_id=%s from=%s subject=%r",
            incoming.conversation_id,
            incoming.from_address,
            incoming.subject,
        )
        yield incoming
