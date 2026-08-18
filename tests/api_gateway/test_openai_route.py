from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from api_gateway.routes.openai import build_openai_router


class Answer:
    def __init__(self) -> None:
        self.calls = []

    def execute(self, question, **kwargs):
        self.calls.append((question, kwargs))
        return "Ответ из RAG"


def _client(answer: Answer) -> TestClient:
    app = FastAPI()
    app.include_router(
        build_openai_router(
            answer_question=answer,
            api_key="chat-secret",
            model="it-consultant",
            knowledge_id="main",
        )
    )
    return TestClient(app)


def test_models_exposes_only_virtual_rag_model() -> None:
    response = _client(Answer()).get(
        "/v1/models",
        headers={"Authorization": "Bearer chat-secret"},
    )

    assert response.status_code == 200
    assert [item["id"] for item in response.json()["data"]] == ["it-consultant"]


def test_chat_uses_last_user_message_as_question() -> None:
    answer = Answer()
    response = _client(answer).post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer chat-secret"},
        json={
            "model": "it-consultant",
            "messages": [
                {"role": "user", "content": "Первый вопрос"},
                {"role": "assistant", "content": "Первый ответ"},
                {"role": "user", "content": "Уточнение"},
            ],
            "stream": False,
        },
    )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "Ответ из RAG"
    question, kwargs = answer.calls[0]
    assert question == "Уточнение"
    assert [item.content for item in kwargs["history"]] == [
        "Первый вопрос",
        "Первый ответ",
    ]
    assert kwargs["knowledge_id"] == "main"
