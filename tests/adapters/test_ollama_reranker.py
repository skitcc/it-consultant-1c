from unittest.mock import MagicMock

import httpx

from mail_gateway.adapters.rag.ollama_reranker import (
    OllamaReranker,
    ScorePassthroughReranker,
    parse_relevance_score,
    score_from_ollama_response,
)
from mail_gateway.domain.models import DocumentChunk

_MODEL = "dengcao/Qwen3-Reranker-8B:Q8_0"


def test_score_passthrough_orders_by_score() -> None:
    chunks = [
        DocumentChunk(text="a", source_path="x", chunk_index=0, score=0.2),
        DocumentChunk(text="b", source_path="x", chunk_index=1, score=0.9),
    ]
    ranked = ScorePassthroughReranker().rerank("q", chunks)
    assert [c.text for c in ranked] == ["b", "a"]


def test_parse_relevance_score_variants() -> None:
    assert parse_relevance_score("0.95") == 0.95
    assert parse_relevance_score('{"score": 0.4}') == 0.4
    assert parse_relevance_score("yes") == 1.0
    assert parse_relevance_score("нет") == 0.0
    assert parse_relevance_score("<think>reason</think>\nyes") == 1.0
    assert parse_relevance_score("score=0.81 extra") == 0.81
    assert parse_relevance_score("") is None
    assert parse_relevance_score("hello world") is None


def test_score_from_logprobs_yes_no() -> None:
    score = score_from_ollama_response(
        {
            "message": {"content": "yes"},
            "logprobs": [
                {
                    "token": "yes",
                    "logprob": -0.1,
                    "top_logprobs": [
                        {"token": "yes", "logprob": -0.1},
                        {"token": "no", "logprob": -2.3},
                    ],
                }
            ],
        }
    )
    assert score is not None
    assert 0.85 < score < 0.95


def test_ollama_reranker_scores_via_chat_yes_no(monkeypatch) -> None:
    chunks = [
        DocumentChunk(text="labels", source_path="f", chunk_index=0, score=0.9),
        DocumentChunk(text="receipt", source_path="f", chunk_index=1, score=0.5),
    ]
    seen: list[dict] = []

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            user = seen[-1]["json"]["messages"][1]["content"]
            answer = "no" if "labels" in user else "yes"
            return {"message": {"role": "assistant", "content": answer}}

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
        "mail_gateway.adapters.rag.ollama_reranker.httpx.Client",
        FakeClient,
    )

    ranked = OllamaReranker(model=_MODEL).rerank("printer", chunks)
    assert [c.text for c in ranked] == ["receipt", "labels"]
    assert ranked[0].score == 1.0
    assert ranked[1].score == 0.0
    assert len(seen) == 2
    assert all(item["url"].endswith("/api/chat") for item in seen)
    first = seen[0]["json"]
    assert first["model"] == _MODEL
    assert first["stream"] is False
    assert first["messages"][0]["role"] == "system"
    user = first["messages"][1]["content"]
    assert user.startswith("<Instruct>:")
    assert "<Query>: printer\n" in user
    assert "<Document>:" in user
    assert first["options"]["temperature"] == 0.0


def test_ollama_reranker_falls_back_on_http_error(monkeypatch) -> None:
    chunks = [
        DocumentChunk(text="a", source_path="f", chunk_index=0, score=0.1),
        DocumentChunk(text="b", source_path="f", chunk_index=1, score=0.8),
    ]

    class FakeResponse:
        def raise_for_status(self) -> None:
            request = MagicMock()
            response = MagicMock()
            response.status_code = 404
            raise httpx.HTTPStatusError(
                "not found",
                request=request,
                response=response,
            )

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
        "mail_gateway.adapters.rag.ollama_reranker.httpx.Client",
        FakeClient,
    )

    ranked = OllamaReranker(model="missing").rerank("q", chunks)
    assert [c.text for c in ranked] == ["b", "a"]


def test_ollama_reranker_falls_back_when_scores_unparseable(monkeypatch) -> None:
    chunks = [
        DocumentChunk(text="a", source_path="f", chunk_index=0, score=0.1),
        DocumentChunk(text="b", source_path="f", chunk_index=1, score=0.8),
    ]

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"message": {"content": "hello world"}}

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
        "mail_gateway.adapters.rag.ollama_reranker.httpx.Client",
        FakeClient,
    )

    ranked = OllamaReranker(model=_MODEL).rerank("q", chunks)
    assert [c.text for c in ranked] == ["b", "a"]
