from mail_gateway.adapters.assistant.openai_assistant import OpenAIAssistant
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


def _chat_response(content: str, *, finish_reason: str = "stop") -> dict:
    return {
        "choices": [
            {
                "finish_reason": finish_reason,
                "message": {
                    "role": "assistant",
                    "content": content,
                    "reasoning": "internal",
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
        "mail_gateway.adapters.assistant.openai_assistant.httpx.Client",
        FakeClient,
    )
    return seen


def test_openai_assistant_uses_chat_completions_and_reasoning_effort(monkeypatch) -> None:
    rag = "Документ: guide.pdf\nТочный факт из документа."
    seen = _install_fake_client(
        monkeypatch,
        [
            _chat_response("<p>Черновик</p>"),
            _chat_response("<p>Проверенный ответ</p>"),
        ],
    )
    assistant = OpenAIAssistant(
        base_url="http://vllm:8001",
        model="openai/gpt-oss-120b",
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
    assert len(seen) == 2
    assert seen[0]["url"] == "http://vllm:8001/v1/chat/completions"
    assert seen[0]["json"]["reasoning_effort"] == "medium"
    assert seen[0]["json"]["chat_template_kwargs"] == {"enable_thinking": True}
    assert seen[1]["json"]["reasoning_effort"] == "high"
    assert seen[0]["json"]["max_tokens"] == 4096
    assert "keep_alive" not in seen[0]["json"]
    assert "think" not in seen[0]["json"]
    assert "options" not in seen[0]["json"]


def test_openai_assistant_disables_thinking_when_none(monkeypatch) -> None:
    seen = _install_fake_client(monkeypatch, [_chat_response("<p>ok</p>")])
    OpenAIAssistant(
        base_url="http://vllm:8001/v1",
        model="m",
        draft_reasoning_effort="none",
    ).ask(_message())
    body = seen[0]["json"]
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert "reasoning_effort" not in body


def test_openai_assistant_rejects_truncated_completion(monkeypatch) -> None:
    _install_fake_client(
        monkeypatch,
        [_chat_response("<p>cut</p>", finish_reason="length")],
    )
    try:
        OpenAIAssistant(base_url="http://vllm:8001", model="m").ask(_message())
    except AssistantUnavailableError as exc:
        assert "truncated" in str(exc)
    else:
        raise AssertionError("expected truncated error")
