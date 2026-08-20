from types import SimpleNamespace
from unittest.mock import MagicMock

from exchangelib import HTMLBody, Message

from mail_gateway.adapters.ews.sender import EwsMailSender
from mail_gateway.domain.models import Reply


def test_ews_sender_uses_html_body() -> None:
    sent: dict = {}

    class Draft:
        def __init__(self) -> None:
            self.body = None

        def send(self) -> None:
            sent["sent"] = True

    item = MagicMock(spec=Message)
    item.subject = "Question"

    def create_reply(subject, body, **kwargs):
        del kwargs
        sent["subject"] = subject
        sent["create_body"] = body
        return Draft()

    item.create_reply.side_effect = create_reply

    class Inbox:
        def get(self, **kwargs):
            sent["get"] = kwargs
            return item

    account = SimpleNamespace(
        inbox=Inbox(),
        protocol=SimpleNamespace(session_pool_size=1, _session_pool_maxsize=2),
    )
    sender = EwsMailSender(account)  # type: ignore[arg-type]
    sender.send_reply(
        Reply(
            conversation_id="c",
            in_reply_to_item_id="id-1",
            in_reply_to_change_key="ck",
            body="<p>hi</p>",
            html=True,
        )
    )
    assert sent["sent"] is True
    assert sent["subject"] == "Re: Question"
    assert isinstance(sent["create_body"], HTMLBody)
    assert "<p>hi</p>" in str(sent["create_body"])


def test_ews_sender_send_mail(monkeypatch) -> None:
    sent: dict = {}

    class FakeMessage:
        def __init__(self, **kwargs) -> None:
            sent["kwargs"] = kwargs

        def send(self) -> None:
            sent["sent"] = True

    monkeypatch.setattr("mail_gateway.adapters.ews.sender.Message", FakeMessage)
    account = SimpleNamespace()
    sender = EwsMailSender(account)  # type: ignore[arg-type]
    sender.send_mail(to="admin@company.ru", subject="alert", body="details")

    assert sent["sent"] is True
    assert sent["kwargs"]["account"] is account
    assert sent["kwargs"]["to_recipients"] == ["admin@company.ru"]
    assert sent["kwargs"]["subject"] == "alert"
    assert sent["kwargs"]["body"] == "details"
