"""In-process mail queue with wait notices for users behind the first request."""

from __future__ import annotations

import logging
import threading
from queue import Queue

from mail_gateway.domain.models import IncomingMessage, Reply
from mail_gateway.ports import MailSender

logger = logging.getLogger(__name__)

QUEUE_NOTICE_MARKER = "Ваш запрос принят и стоит в очереди."


class IncomingMailQueue:
    """FIFO of incoming mail plus an in-flight slot for wait-position math."""

    def __init__(self) -> None:
        self._items: Queue[IncomingMessage] = Queue()
        self._lock = threading.Lock()
        self._queued = 0
        self._busy = False

    def enqueue(self, message: IncomingMessage) -> int:
        """Put ``message`` and return its 1-based position including in-flight work."""
        with self._lock:
            position = int(self._busy) + self._queued + 1
            self._queued += 1
            self._items.put(message)
        return position

    def take(self) -> IncomingMessage:
        message = self._items.get()
        with self._lock:
            self._queued -= 1
            self._busy = True
        return message

    def done(self) -> None:
        with self._lock:
            self._busy = False


def estimated_wait_minutes(*, position: int, minutes_each: int) -> int:
    if position <= 1:
        return 0
    return (position - 1) * minutes_each


def queue_notice_html(*, ahead: int, wait_minutes: int) -> str:
    return (
        f"<p>{QUEUE_NOTICE_MARKER}</p>"
        f"<p>Перед вами {_ahead_phrase(ahead)}. "
        f"Примерное время ожидания — {wait_minutes} мин.</p>"
    )


def is_queue_notice(text: str) -> bool:
    return QUEUE_NOTICE_MARKER in (text or "")


def send_queue_notice(
    sender: MailSender,
    message: IncomingMessage,
    *,
    position: int,
    minutes_each: int,
) -> None:
    ahead = position - 1
    wait_minutes = estimated_wait_minutes(
        position=position,
        minutes_each=minutes_each,
    )
    body = queue_notice_html(ahead=ahead, wait_minutes=wait_minutes)
    logger.info(
        "Queue notice conversation_id=%s item_id=%s position=%s ahead=%s "
        "wait_minutes=%s",
        message.conversation_id,
        message.item_id,
        position,
        ahead,
        wait_minutes,
    )
    sender.send_reply(
        Reply(
            conversation_id=message.conversation_id,
            in_reply_to_item_id=message.item_id,
            in_reply_to_change_key=message.change_key,
            body=body,
            html=True,
        )
    )


def enqueue_and_notify(
    queue: IncomingMailQueue,
    sender: MailSender,
    message: IncomingMessage,
    *,
    minutes_each: int,
) -> int:
    position = queue.enqueue(message)
    logger.info(
        "Mail queued conversation_id=%s item_id=%s position=%s",
        message.conversation_id,
        message.item_id,
        position,
    )
    if position > 1:
        try:
            send_queue_notice(
                sender,
                message,
                position=position,
                minutes_each=minutes_each,
            )
        except Exception:
            logger.exception(
                "Failed to send queue notice conversation_id=%s item_id=%s",
                message.conversation_id,
                message.item_id,
            )
    return position


def _ahead_phrase(ahead: int) -> str:
    mod10 = ahead % 10
    mod100 = ahead % 100
    if mod10 == 1 and mod100 != 11:
        word = "запрос"
    elif mod10 in {2, 3, 4} and mod100 not in {12, 13, 14}:
        word = "запроса"
    else:
        word = "запросов"
    return f"{ahead} {word}"
