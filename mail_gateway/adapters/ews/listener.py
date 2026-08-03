"""EWS Streaming listener for new inbox messages."""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from datetime import timedelta

from exchangelib import Account, EWSDateTime, Message, UTC
from exchangelib.properties import CreatedEvent, ItemId, NewMailEvent

from mail_gateway.adapters.ews.mapping import message_to_incoming
from mail_gateway.domain.models import IncomingMessage
from mail_gateway.ports import MailListener

logger = logging.getLogger(__name__)

_MAIL_EVENTS = (NewMailEvent, CreatedEvent)


class EwsMailListener(MailListener):
    def __init__(
        self,
        stream_account: Account,
        fetch_account: Account,
        *,
        connection_timeout_minutes: int = 30,
        ignore_own_mail: bool = True,
        catchup_minutes: int = 30,
        own_addresses: set[str] | frozenset[str] | None = None,
    ) -> None:
        # Separate accounts: streaming holds a long-lived HTTP connection;
        # fetch/reply must use another session or get() hangs inside the loop.
        self._stream_account = stream_account
        self._fetch_account = fetch_account
        self._connection_timeout_minutes = connection_timeout_minutes
        self._ignore_own_mail = ignore_own_mail
        self._catchup_minutes = catchup_minutes
        self._seen_item_ids: set[str] = set()
        extras = {addr.strip().lower() for addr in (own_addresses or set()) if addr.strip()}
        primary = (fetch_account.primary_smtp_address or "").strip().lower()
        if primary:
            extras.add(primary)
        self._own_addresses = frozenset(extras)

    def listen(self) -> Iterator[IncomingMessage]:
        logger.info(
            "Listener cycle started stream_protocol=%s fetch_protocol=%s "
            "shared_protocol=%s pool_size=%s pool_max=%s",
            id(self._stream_account.protocol),
            id(self._fetch_account.protocol),
            self._stream_account.protocol is self._fetch_account.protocol,
            self._stream_account.protocol.session_pool_size,
            self._stream_account.protocol._session_pool_maxsize,
        )
        yield from self._catch_up_unread()

        inbox = self._stream_account.inbox
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
                logger.info(
                    "EWS notification received event_count=%s connection_status=%s",
                    len(events),
                    getattr(notification, "connection_status", None),
                )
                if not events:
                    logger.debug("EWS streaming keepalive (empty notification)")
                    continue
                for event in events:
                    yield from self._handle_event(event)
        finally:
            try:
                self._stream_account.unsubscribe(subscription_id)
            except Exception:
                logger.exception("Failed to unsubscribe id=%s", subscription_id)

    def _catch_up_unread(self) -> Iterator[IncomingMessage]:
        if self._catchup_minutes <= 0:
            return

        inbox = self._fetch_account.inbox
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
            "EWS event type=%s item_id=%s changekey=%s",
            event_name,
            getattr(item_id, "id", item_id),
            getattr(item_id, "changekey", None),
        )

        if not isinstance(event, _MAIL_EVENTS):
            return
        if item_id is None:
            return

        raw_id = str(item_id.id)
        if raw_id in self._seen_item_ids:
            logger.info("Skipping already handled item_id=%s", raw_id)
            return

        item = self._fetch_message(item_id)
        if item is None:
            return

        yield from self._message_to_yield(item)

    def _fetch_message(self, item_id: ItemId) -> Message | None:
        """Load message via the non-streaming account (with short retries)."""
        last_error: Exception | None = None
        for attempt in range(1, 4):
            started_at = time.perf_counter()
            try:
                logger.info(
                    "Loading message attempt=%s id=%s pool_size=%s/%s",
                    attempt,
                    item_id.id,
                    self._fetch_account.protocol.session_pool_size,
                    self._fetch_account.protocol._session_pool_maxsize,
                )
                kwargs: dict[str, object] = {"id": item_id.id}
                if getattr(item_id, "changekey", None):
                    kwargs["changekey"] = item_id.changekey
                item = self._fetch_account.inbox.get(**kwargs)
                if isinstance(item, Message):
                    logger.info(
                        "Loaded message attempt=%s elapsed=%.3fs subject=%r from=%s",
                        attempt,
                        time.perf_counter() - started_at,
                        item.subject,
                        getattr(getattr(item, "sender", None), "email_address", None),
                    )
                    return item
                logger.info(
                    "Loaded non-message type=%s id=%s",
                    type(item).__name__,
                    item_id.id,
                )
                return None
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "Fetch message failed attempt=%s elapsed=%.3fs id=%s "
                    "error_type=%s error=%s",
                    attempt,
                    time.perf_counter() - started_at,
                    item_id.id,
                    type(exc).__name__,
                    exc,
                )
                time.sleep(0.5 * attempt)

        logger.error(
            "Giving up loading item id=%s last_error=%s",
            item_id.id,
            last_error,
        )
        return None

    def _message_to_yield(self, item: Message) -> Iterator[IncomingMessage]:
        item_id = str(item.id)
        if item_id in self._seen_item_ids:
            logger.info("Skipping already handled item_id=%s", item_id)
            return

        sender = ""
        if item.sender and item.sender.email_address:
            sender = item.sender.email_address.strip().lower()
        elif item.author and item.author.email_address:
            sender = item.author.email_address.strip().lower()

        if self._ignore_own_mail and sender and sender in self._own_addresses:
            logger.info(
                "Skipping own mailbox mail from=%s own_addresses=%s",
                sender,
                sorted(self._own_addresses),
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
