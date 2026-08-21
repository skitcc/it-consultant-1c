from mail_gateway.application.mail_queue import (
    IncomingMailQueue,
    QUEUE_NOTICE_MARKER,
    enqueue_and_notify,
    estimated_wait_minutes,
    is_queue_notice,
    queue_notice_html,
)
from mail_gateway.domain.models import IncomingMessage, Reply


class FakeSender:
    def __init__(self) -> None:
        self.sent: list[Reply] = []
        self.fail = False

    def send_reply(self, reply: Reply) -> None:
        if self.fail:
            raise RuntimeError("ews failed")
        self.sent.append(reply)

    def send_mail(self, *, to: str, subject: str, body: str) -> None:
        del to, subject, body


def _message(item_id: str = "item-1") -> IncomingMessage:
    return IncomingMessage(
        conversation_id="conv-1",
        item_id=item_id,
        change_key="ck",
        from_address="user@company.ru",
        subject="q",
        body="help",
    )


def test_first_request_is_position_one() -> None:
    queue = IncomingMailQueue()
    assert queue.enqueue(_message("a")) == 1


def test_waiting_requests_get_increasing_positions() -> None:
    queue = IncomingMailQueue()
    assert queue.enqueue(_message("a")) == 1
    assert queue.enqueue(_message("b")) == 2
    assert queue.enqueue(_message("c")) == 3


def test_position_counts_in_flight_request() -> None:
    queue = IncomingMailQueue()
    queue.enqueue(_message("a"))
    taken = queue.take()
    assert taken.item_id == "a"
    assert queue.enqueue(_message("b")) == 2
    assert queue.enqueue(_message("c")) == 3
    queue.done()
    assert queue.take().item_id == "b"
    assert queue.enqueue(_message("d")) == 3
    queue.done()
    queue.take()
    queue.done()
    queue.take()
    assert queue.enqueue(_message("e")) == 2


def test_estimated_wait_skips_the_first_request() -> None:
    assert estimated_wait_minutes(position=1, minutes_each=3) == 0
    assert estimated_wait_minutes(position=2, minutes_each=3) == 3
    assert estimated_wait_minutes(position=4, minutes_each=3) == 9


def test_queue_notice_html_mentions_ahead_and_wait() -> None:
    html = queue_notice_html(ahead=2, wait_minutes=6)
    assert QUEUE_NOTICE_MARKER in html
    assert "Перед вами 2 запроса" in html
    assert "6 мин" in html
    assert is_queue_notice("текст " + QUEUE_NOTICE_MARKER)


def test_enqueue_and_notify_skips_first_and_mails_the_rest() -> None:
    queue = IncomingMailQueue()
    sender = FakeSender()
    first = enqueue_and_notify(queue, sender, _message("a"), minutes_each=3)
    second = enqueue_and_notify(queue, sender, _message("b"), minutes_each=3)
    third = enqueue_and_notify(queue, sender, _message("c"), minutes_each=3)

    assert first == 1
    assert second == 2
    assert third == 3
    assert sender.sent[0].in_reply_to_item_id == "b"
    assert "1 запрос" in sender.sent[0].body
    assert "3 мин" in sender.sent[0].body
    assert sender.sent[1].in_reply_to_item_id == "c"
    assert "2 запроса" in sender.sent[1].body
    assert "6 мин" in sender.sent[1].body


def test_queue_notice_failure_does_not_drop_the_request() -> None:
    queue = IncomingMailQueue()
    sender = FakeSender()
    sender.fail = True
    enqueue_and_notify(queue, sender, _message("a"), minutes_each=3)
    position = enqueue_and_notify(queue, sender, _message("b"), minutes_each=3)
    assert position == 2
    assert queue.take().item_id == "a"
    assert queue.take().item_id == "b"
