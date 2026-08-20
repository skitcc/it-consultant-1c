import json
import logging

import httpx
import pytest

from mail_gateway.adapters.assistant.ollama_assistant import OllamaAssistant
from mail_gateway.domain.models import IncomingMessage, with_rag_context
from mail_gateway.ports import AssistantUnavailableError
from common.timing import begin_request, end_request


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


def _chat_response(
    answer_html: str,
    *,
    thinking: str = "internal",
    done_reason: str = "stop",
) -> dict:
    return {
        "message": {
            "role": "assistant",
            "thinking": thinking,
            "content": answer_html,
        },
        "done": True,
        "done_reason": done_reason,
        "prompt_eval_count": 100,
        "eval_count": 50,
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


def test_ollama_assistant_uses_native_chat_api_and_two_think_levels(
    monkeypatch,
    caplog,
) -> None:
    caplog.set_level(logging.INFO)
    rag = "Документ: guide.pdf\nТочный факт из документа."
    seen = _install_fake_client(
        monkeypatch,
        [
            _chat_response(
                "<p>Черновик</p>",
                thinking="layer one private reasoning",
            ),
            _chat_response(
                "<p>Проверенный ответ</p>",
                thinking="layer two private reasoning",
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
        verifier_enabled=True,
        verifier_reasoning_effort="high",
    )
    timer = begin_request(conversation_id="c1")
    try:
        answer = assistant.ask(_message(rag=rag))
    finally:
        end_request(timer)

    assert answer == "<p>Проверенный ответ</p>"
    chats = [item for item in seen if item["url"].endswith("/api/chat")]
    assert len(chats) == 2
    assert not any("/v1/chat/completions" in item["url"] for item in seen)
    assert chats[0]["json"]["think"] == "medium"
    assert chats[1]["json"]["think"] == "high"
    for item in chats:
        assert item["json"]["stream"] is False
        assert item["json"]["keep_alive"] == -1
        assert "reasoning_effort" not in item["json"]
        assert "max_tokens" not in item["json"]
        assert item["json"]["options"]["temperature"] == 0.0
        assert item["json"]["options"]["top_p"] == 0.1
        assert item["json"]["options"]["num_predict"] == 4096
        assert item["json"]["options"]["seed"] == 0
        assert item["json"]["options"]["num_ctx"] == 8192
        assert item["json"]["options"]["stop"] == ["\nПользователь:", "\nUser:"]
        assert "response_format" not in item["json"]
        assert rag in item["json"]["messages"][0]["content"]
    assert "Черновик" in chats[1]["json"]["messages"][1]["content"]
    assert "private reasoning" not in answer
    assert "prompt_tokens=100" in caplog.text
    assert "completion_tokens=50" in caplog.text
    assert "8192" in caplog.text
    assert [name for name, _elapsed in timer.steps] == ["llm_layer_1", "llm_layer_2"]


def test_ollama_assistant_skips_verifier_without_rag(monkeypatch) -> None:
    seen = _install_fake_client(
        monkeypatch,
        [_chat_response("<p>plain</p>")],
    )

    answer = OllamaAssistant(model="gpt-oss:120b", verifier_enabled=True).ask(
        _message()
    )
    assert answer == "<p>plain</p>"
    assert len([item for item in seen if item["url"].endswith("/api/chat")]) == 1


def test_ollama_assistant_skips_verifier_when_disabled(monkeypatch) -> None:
    rag = "Документ: guide.pdf\nТочный факт из документа."
    seen = _install_fake_client(
        monkeypatch,
        [_chat_response("<p>Черновик</p>")],
    )

    answer = OllamaAssistant(model="gpt-oss:120b").ask(_message(rag=rag))
    assert answer == "<p>Черновик</p>"
    assert len([item for item in seen if item["url"].endswith("/api/chat")]) == 1


def test_ollama_assistant_keeps_draft_when_verifier_is_empty(monkeypatch) -> None:
    rag = "Документ: a.md\nПодтверждённая строка."
    _install_fake_client(
        monkeypatch,
        [
            _chat_response("<p>Черновик</p>"),
            _chat_response("   "),
        ],
    )

    answer = OllamaAssistant(model="m", verifier_enabled=True).ask(_message(rag=rag))
    assert answer == "<p>Черновик</p>"


def test_ollama_assistant_accepts_legacy_json_answer_html(monkeypatch) -> None:
    rag = "Документ: a.md\nПервая строка."
    payload = {
        "message": {
            "role": "assistant",
            "content": json.dumps(
                {
                    "evidence": [{"quote": "not in rag"}],
                    "answer_html": "<p>Ответ</p>",
                },
                ensure_ascii=False,
            ),
        },
        "done": True,
        "done_reason": "stop",
        "prompt_eval_count": 10,
        "eval_count": 5,
    }
    _install_fake_client(monkeypatch, [payload, payload])
    assert (
        OllamaAssistant(model="m", verifier_enabled=True).ask(_message(rag=rag))
        == "<p>Ответ</p>"
    )


def test_ollama_assistant_rejects_reasoning_leaked_into_answer(monkeypatch) -> None:
    rag = "Документ: a.md\nФакт."
    _install_fake_client(
        monkeypatch,
        [_chat_response("thinking - скрытый текст content Ответ")],
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
        [_chat_response("<p>partial</p>", done_reason="length")],
    )

    with pytest.raises(AssistantUnavailableError):
        OllamaAssistant(model="m").ask(_message())
