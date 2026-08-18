from mail_gateway.application.handle_incoming_mail import (
    ADMIN_FALLBACK_TEXT,
    HandleIncomingMail,
)
from mail_gateway.domain.models import ConversationTurn, DocumentChunk, IncomingMessage, Reply


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


class FakeHistoryLoader:
    def __init__(self, turns: list[ConversationTurn]) -> None:
        self.turns = turns
        self.calls: list[str] = []

    def load(self, conversation_id: str) -> list[ConversationTurn]:
        self.calls.append(conversation_id)
        return list(self.turns)


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
    assert len(assistant.calls[0].messages) == 1
    assert assistant.calls[0].messages[0].body == "Please help"


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


def test_passes_conversation_history_to_assistant() -> None:
    history = [
        ConversationTurn(
            role="user",
            body="first question",
            from_address="user@company.ru",
            subject="How to print?",
            item_id="item-0",
        ),
        ConversationTurn(
            role="assistant",
            body="first answer",
            from_address="bot@company.ru",
            subject="Re: How to print?",
            item_id="item-bot-0",
        ),
    ]
    assistant = FakeAssistant("follow-up answer")
    sender = FakeSender()
    handle = HandleIncomingMail(
        assistant=assistant,
        mail_sender=sender,
        history_loader=FakeHistoryLoader(history),
    )

    handle(_message())

    assert len(assistant.calls) == 1
    assert len(assistant.calls[0].messages) == 3
    assert assistant.calls[0].messages[0].body == "first question"
    assert assistant.calls[0].messages[1].role == "assistant"
    assert assistant.calls[0].messages[2].item_id == "item-1"
    assert assistant.calls[0].messages[2].body == "Please help"


class FakeRetriever:
    def __init__(self, chunks: list[DocumentChunk]) -> None:
        self.chunks = chunks
        self.calls: list[str] = []

    def retrieve(self, query: str) -> list[DocumentChunk]:
        self.calls.append(query)
        return list(self.chunks)


def test_attaches_rag_context_from_retriever() -> None:
    retriever = FakeRetriever(
        [
            DocumentChunk(
                text="docs say reboot",
                source_path="faq.md",
                chunk_index=0,
            )
        ]
    )
    assistant = FakeAssistant("ok")
    sender = FakeSender()
    handle = HandleIncomingMail(
        assistant=assistant,
        mail_sender=sender,
        document_retriever=retriever,
    )

    handle(_message())

    assert retriever.calls == ["Please help"]
    assert assistant.calls[0].rag_context is not None
    assert "faq.md" in assistant.calls[0].rag_context
    assert "docs say reboot" in assistant.calls[0].rag_context


def test_ignores_own_bot_email_silently() -> None:
    assistant = FakeAssistant("should not run")
    sender = FakeSender()
    handle = HandleIncomingMail(
        assistant=assistant,
        mail_sender=sender,
        bot_email="Assistant@1c-perspective.ru",
    )

    handle(
        IncomingMessage(
            conversation_id="conv-own",
            item_id="item-own",
            change_key="ck",
            from_address="assistant@1c-perspective.ru",
            subject="Re: loop?",
            body="bot talking to itself",
        )
    )

    assert assistant.calls == []
    assert sender.sent == []
