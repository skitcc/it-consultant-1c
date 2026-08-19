from mail_gateway.adapters.assistant.ollama_assistant import OllamaAssistant
from mail_gateway.domain.models import IncomingMessage, with_rag_context


def _message(*, rag: str | None = None) -> IncomingMessage:
    message = IncomingMessage(
        conversation_id="c1",
        item_id="i1",
        change_key="k",
        from_address="user@x.ru",
        subject="s",
        body="Как перейти на K1?",
    )
    if rag is None:
        return message
    return with_rag_context(message, rag)


def test_ollama_assistant_sends_temperature_and_two_layers(monkeypatch) -> None:
    seen: list[dict] = []
    replies = iter(
        [
            {"message": {"thinking": "цитата K0", "content": "черновик K1"}},
            {"message": {"thinking": "цитата K0-2 покупать по K0", "content": "<p>K0</p>"}},
        ]
    )
    rag = "Документ: grades.pdf\nK0-2: PM готов покупать по K0."

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return next(replies)

    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args) -> None:
            return None

        def post(self, url: str, json: dict):
            seen.append({"url": url, "json": json})
            return FakeResponse()

    monkeypatch.setattr(
        "mail_gateway.adapters.assistant.ollama_assistant.httpx.Client",
        FakeClient,
    )

    assistant = OllamaAssistant(
        model="gpt-oss:120b",
        temperature=0.0,
        top_p=0.1,
    )
    answer = assistant.ask(_message(rag=rag))

    assert answer == "<p>K0</p>"
    assert len(seen) == 2
    assert seen[0]["url"].endswith("/api/chat")
    for item in seen:
        assert item["json"]["think"] is True
        assert item["json"]["options"]["temperature"] == 0.0
        assert item["json"]["options"]["top_p"] == 0.1
        assert rag in item["json"]["messages"][0]["content"]
    assert seen[1]["json"]["messages"][1]["content"].count("черновик K1") == 1


def test_ollama_assistant_skips_verifier_without_rag(monkeypatch) -> None:
    seen: list[dict] = []

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"message": {"content": "plain"}}

    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args) -> None:
            return None

        def post(self, url: str, json: dict):
            seen.append(json)
            return FakeResponse()

    monkeypatch.setattr(
        "mail_gateway.adapters.assistant.ollama_assistant.httpx.Client",
        FakeClient,
    )

    answer = OllamaAssistant(model="gpt-oss:120b").ask(_message())
    assert answer == "plain"
    assert len(seen) == 1


def test_ollama_assistant_drops_draft_when_verifier_empty(monkeypatch) -> None:
    replies = iter(
        [
            {"message": {"content": "draft"}},
            {"message": {"content": ""}},
        ]
    )

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return next(replies)

    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args) -> None:
            return None

        def post(self, url: str, json: dict):
            del url, json
            return FakeResponse()

    monkeypatch.setattr(
        "mail_gateway.adapters.assistant.ollama_assistant.httpx.Client",
        FakeClient,
    )

    answer = OllamaAssistant(model="m").ask(_message(rag="Документ: a.md\nфакт"))
    assert answer is None
