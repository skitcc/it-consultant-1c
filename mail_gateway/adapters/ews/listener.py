"""EWS Streaming listener for new inbox messages."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from datetime import timedelta

from exchangelib import Account, EWSDateTime, Message, UTC
from exchangelib.properties import CreatedEvent, NewMailEvent

from mail_gateway.adapters.ews.mapping import message_to_incoming
from mail_gateway.domain.models import IncomingMessage
from mail_gateway.ports import MailListener

logger = logging.getLogger(__name__)

_MAIL_EVENTS = (NewMailEvent, CreatedEvent)


class EwsMailListener(MailListener):
    def __init__(
        self,
        account: Account,
        *,
        connection_timeout_minutes: int = 30,
        ignore_own_mail: bool = True,
        catchup_minutes: int = 30,
    ) -> None:
        self._account = account
        self._connection_timeout_minutes = connection_timeout_minutes
        self._ignore_own_mail = ignore_own_mail
        self._catchup_minutes = catchup_minutes
        self._seen_item_ids: set[str] = set()

    def listen(self) -> Iterator[IncomingMessage]:
        inbox = self._account.inbox
        yield from self._catch_up_unread(inbox)

        subscription_id = inbox.subscribe_to_streaming(
            event_types=[NewMailEvent.ELEMENT_NAME, CreatedEvent.ELEMENT_NAME]
        )
        logger.info(
            "EWS streaming subscription started id=%s "
            "(waiting for NewMail/Created; send a NEW mail AFTER this line)",
            subscription_id,
        )
        try:
            for notification in inbox.get_streaming_events(
                subscription_id,
                connection_timeout=self._connection_timeout_minutes,
            ):
                events = list(notification.events)
                if not events:
                    logger.info("EWS streaming keepalive (empty notification)")
                    continue
                for event in events:
                    yield from self._handle_event(event)
        finally:
            try:
                self._account.unsubscribe(subscription_id)
            except Exception:
                logger.exception("Failed to unsubscribe id=%s", subscription_id)

    def _catch_up_unread(self, inbox: object) -> Iterator[IncomingMessage]:
        """Process recent unread inbox mail (helps if mail arrived before subscribe)."""
        if self._catchup_minutes <= 0:
            return

        since = EWSDateTime.now(tz=UTC) - timedelta(minutes=self._catchup_minutes)
        try:
            items = list(
                inbox.filter(datetime_received__gte=since, is_read=False).order_by(
                    "-datetime_received"
                )[:20]
            )
        except Exception:
            logger.exception("Catch-up unread scan failed")
            return

        logger.info(
            "Catch-up: found %s unread message(s) in last %s min",
            len(items),
            self._catchup_minutes,
        )
        for item in items:
            yield from self._message_to_yield(item)

    def _handle_event(self, event: object) -> Iterator[IncomingMessage]:
        event_name = type(event).__name__
        item_id = getattr(event, "item_id", None)
        logger.info(
            "EWS event type=%s item_id=%s",
            event_name,
            getattr(item_id, "id", item_id),
        )

        if not isinstance(event, _MAIL_EVENTS):
            return
        if item_id is None:
            return

        try:
            item = self._account.inbox.get(id=item_id.id, changekey=item_id.changekey)
        except Exception:
            logger.exception(
                "Failed to load item id=%s", getattr(item_id, "id", item_id)
            )
            return

        yield from self._message_to_yield(item)

    def _message_to_yield(self, item: object) -> Iterator[IncomingMessage]:
        if not isinstance(item, Message):
            logger.info("Skipping non-message item type=%s", type(item).__name__)
            return

        item_id = str(item.id)
        if item_id in self._seen_item_ids:
            logger.info("Skipping already handled item_id=%s", item_id)
            return

        own = self._account.primary_smtp_address.lower()
        sender = ""
        if item.sender and item.sender.email_address:
            sender = item.sender.email_address.lower()

        if self._ignore_own_mail and sender == own:
            logger.warning(
                "Skipping mail from our own mailbox (%s). "
                "Send the test from ANOTHER address, or set EWS_IGNORE_OWN_MAIL=false",
                own,
            )
            self._seen_item_ids.add(item_id)
            return

        incoming = message_to_incoming(item)
        self._seen_item_ids.add(item_id)
        logger.info(
            "New mail conversation_id=%s from=%s subject=%r",
            incoming.conversation_id,
            incoming.from_address,
            incoming.subject,
        )
        yield incoming
