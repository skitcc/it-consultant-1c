from mail_gateway.application.handle_incoming_mail import (
    ADMIN_FALLBACK_TEXT,
    HandleIncomingMail,
)
from mail_gateway.domain.models import IncomingMessage, Reply


class FakeAssistant:
    def __init__(self, reply: str | None) -> None:
        self.reply = reply
        self.calls: list[IncomingMessage] = []

    def ask(self, message: IncomingMessage) -> str | None:
        self.calls.append(message)
        return self.reply


class FakeSender:
    def __init__(self) -> None:
        self.sent: list[Reply] = []

    def send_reply(self, reply: Reply) -> None:
        self.sent.append(reply)


def _message() -> IncomingMessage:
    return IncomingMessage(
        conversation_id="conv-1",
        item_id="item-1",
        change_key="ck-1",
        from_address="user@company.ru",
        subject="How to print?",
        body="Please help",
        message_id="<msg-1@company.ru>",
    )


def test_sends_assistant_reply() -> None:
    assistant = FakeAssistant("Answer from AI")
    sender = FakeSender()
    handle = HandleIncomingMail(assistant=assistant, mail_sender=sender)

    handle(_message())

    assert len(sender.sent) == 1
    assert sender.sent[0].body == "Answer from AI"
    assert sender.sent[0].conversation_id == "conv-1"
    assert sender.sent[0].in_reply_to_item_id == "item-1"


def test_uses_admin_fallback_when_no_reply() -> None:
    assistant = FakeAssistant(None)
    sender = FakeSender()
    handle = HandleIncomingMail(assistant=assistant, mail_sender=sender)

    handle(_message())

    assert sender.sent[0].body == ADMIN_FALLBACK_TEXT


def test_uses_admin_fallback_when_empty_reply() -> None:
    assistant = FakeAssistant("   ")
    sender = FakeSender()
    handle = HandleIncomingMail(assistant=assistant, mail_sender=sender)

    handle(_message())

    assert sender.sent[0].body == ADMIN_FALLBACK_TEXT
