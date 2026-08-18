from __future__ import annotations

import httpx

from mail_gateway.adapters.gateway_assistant import GatewayAssistant
from mail_gateway.domain.models import ConversationTurn, IncomingMessage, with_messages


def test_gateway_assistant_sends_clean_openai_thread(monkeypatch) -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["authorization"] = request.headers["authorization"]
        captured["json"] = __import__("json").loads(request.content)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"role": "assistant", "content": "  Готово.  "}}
                ]
            },
        )

    transport = httpx.MockTransport(handler)
    original_client = httpx.Client

    def client_factory(*args, **kwargs):
        kwargs["transport"] = transport
        return original_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "Client", client_factory)
    message = IncomingMessage(
        conversation_id="conv",
        item_id="item",
        change_key="ck",
        from_address="user@example.com",
        subject="Help",
        body="latest",
    )
    message = with_messages(
        message,
        [
            ConversationTurn(role="user", body="first\nSent from my iPhone"),
            ConversationTurn(role="assistant", body="answer"),
            ConversationTurn(role="user", body="latest"),
        ],
    )

    assistant = GatewayAssistant(
        base_url="http://gateway:8000/v1/",
        api_key="secret",
        model="it-consultant",
    )

    assert assistant.ask(message) == "Готово."
    assert captured == {
        "url": "http://gateway:8000/v1/chat/completions",
        "authorization": "Bearer secret",
        "json": {
            "model": "it-consultant",
            "messages": [
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "answer"},
                {"role": "user", "content": "latest"},
            ],
            "stream": False,
        },
    }
