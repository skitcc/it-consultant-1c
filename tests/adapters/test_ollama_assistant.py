import json
import logging

import httpx
import pytest

from mail_gateway.adapters.assistant.ollama_assistant import OllamaAssistant
from mail_gateway.domain.models import IncomingMessage, with_rag_context
from mail_gateway.ports import AssistantUnavailableError


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


def _completion(
    answer_html: str,
    *,
    reasoning: str = "internal",
    finish_reason: str = "stop",
) -> dict:
    return {
        "choices": [
            {
                "finish_reason": finish_reason,
                "message": {
                    "reasoning": reasoning,
                    "content": answer_html,
                },
            }
        ],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
        },
    }


def _install_fake_client(monkeypatch, replies: list[dict]) -> list[dict]:
    seen: list[dict] = []
    responses = iter(replies)

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return next(responses)

    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            self.timeout = kwargs.get("timeout")

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
    return seen


def test_ollama_assistant_uses_openai_api_and_two_reasoning_levels(
    monkeypatch,
    caplog,
) -> None:
    caplog.set_level(logging.INFO)
    rag = "Документ: guide.pdf\nТочный факт из документа."
    seen = _install_fake_client(
        monkeypatch,
        [
            _completion(
                "<p>Черновик</p>",
                reasoning="layer one private reasoning",
            ),
            _completion(
                "<p>Проверенный ответ</p>",
                reasoning="layer two private reasoning",
            ),
        ],
    )

    assistant = OllamaAssistant(
        model="gpt-oss:120b",
        temperature=0.0,
        top_p=0.1,
        max_tokens=4096,
        seed=0,
        draft_reasoning_effort="medium",
        verifier_reasoning_effort="high",
    )
    answer = assistant.ask(_message(rag=rag))

    assert answer == "<p>Проверенный ответ</p>"
    chats = [item for item in seen if item["url"].endswith("/v1/chat/completions")]
    assert len(chats) == 2
    assert not any("/api/tokenize" in item["url"] for item in seen)
    assert chats[0]["json"]["reasoning_effort"] == "medium"
    assert chats[1]["json"]["reasoning_effort"] == "high"
    for item in chats:
        assert item["json"]["stream"] is False
        assert item["json"]["temperature"] == 0.0
        assert item["json"]["top_p"] == 0.1
        assert item["json"]["max_tokens"] == 4096
        assert item["json"]["seed"] == 0
        assert "response_format" not in item["json"]
        assert rag in item["json"]["messages"][0]["content"]
    assert "Черновик" in chats[1]["json"]["messages"][1]["content"]
    assert "private reasoning" not in answer
    assert "prompt_tokens=100" in caplog.text
    assert "completion_tokens=50" in caplog.text
    assert "8192" in caplog.text


def test_ollama_assistant_skips_verifier_without_rag(monkeypatch) -> None:
    seen = _install_fake_client(
        monkeypatch,
        [_completion("<p>plain</p>")],
    )

    answer = OllamaAssistant(model="gpt-oss:120b").ask(_message())
    assert answer == "<p>plain</p>"
    assert len([item for item in seen if item["url"].endswith("/v1/chat/completions")]) == 1


def test_ollama_assistant_keeps_draft_when_verifier_is_empty(monkeypatch) -> None:
    rag = "Документ: a.md\nПодтверждённая строка."
    _install_fake_client(
        monkeypatch,
        [
            _completion("<p>Черновик</p>"),
            _completion("   "),
        ],
    )

    answer = OllamaAssistant(model="m").ask(_message(rag=rag))
    assert answer == "<p>Черновик</p>"


def test_ollama_assistant_accepts_legacy_json_answer_html(monkeypatch) -> None:
    rag = "Документ: a.md\nПервая строка."
    payload = {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "content": json.dumps(
                        {
                            "evidence": [{"quote": "not in rag"}],
                            "answer_html": "<p>Ответ</p>",
                        },
                        ensure_ascii=False,
                    )
                },
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }
    _install_fake_client(monkeypatch, [payload, payload])
    assert OllamaAssistant(model="m").ask(_message(rag=rag)) == "<p>Ответ</p>"


def test_ollama_assistant_rejects_reasoning_leaked_into_answer(monkeypatch) -> None:
    rag = "Документ: a.md\nФакт."
    _install_fake_client(
        monkeypatch,
        [_completion("thinking - скрытый текст content Ответ")],
    )

    answer = OllamaAssistant(model="m").ask(_message(rag=rag))
    assert answer is None


def test_ollama_assistant_raises_typed_error_on_timeout(monkeypatch) -> None:
    class TimeoutClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args) -> None:
            return None

        def post(self, url: str, json: dict):
            del url, json
            raise httpx.ReadTimeout("timed out")

    monkeypatch.setattr(
        "mail_gateway.adapters.assistant.ollama_assistant.httpx.Client",
        TimeoutClient,
    )

    with pytest.raises(AssistantUnavailableError):
        OllamaAssistant(model="m").ask(_message())


def test_ollama_assistant_rejects_truncated_completion(monkeypatch) -> None:
    _install_fake_client(
        monkeypatch,
        [_completion("<p>partial</p>", finish_reason="length")],
    )

    with pytest.raises(AssistantUnavailableError):
        OllamaAssistant(model="m").ask(_message())
