from unittest.mock import MagicMock

from mail_gateway.adapters.rag.ollama_reranker import (
    OllamaReranker,
    ScorePassthroughReranker,
)
from mail_gateway.domain.models import DocumentChunk


def test_score_passthrough_orders_by_score() -> None:
    chunks = [
        DocumentChunk(text="a", source_path="x", chunk_index=0, score=0.2),
        DocumentChunk(text="b", source_path="x", chunk_index=1, score=0.9),
    ]
    ranked = ScorePassthroughReranker().rerank("q", chunks)
    assert [c.text for c in ranked] == ["b", "a"]


def test_ollama_reranker_parses_api_results(monkeypatch) -> None:
    chunks = [
        DocumentChunk(text="labels", source_path="f", chunk_index=0, score=0.9),
        DocumentChunk(text="receipt", source_path="f", chunk_index=1, score=0.5),
    ]

    class FakeResponse:
        status_code = 200

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "results": [
                    {"index": 1, "relevance_score": 0.95},
                    {"index": 0, "relevance_score": 0.10},
                ]
            }

    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args) -> None:
            return None

        def post(self, url: str, json: dict):
            assert url.endswith("/api/rerank")
            assert json["documents"] == ["labels", "receipt"]
            return FakeResponse()

    monkeypatch.setattr(
        "mail_gateway.adapters.rag.ollama_reranker.httpx.Client",
        FakeClient,
    )

    ranked = OllamaReranker(model="bge-reranker-v2-m3").rerank("printer", chunks)
    assert [c.text for c in ranked] == ["receipt", "labels"]
    assert ranked[0].score == 0.95


def test_ollama_reranker_falls_back_on_404(monkeypatch) -> None:
    chunks = [
        DocumentChunk(text="a", source_path="f", chunk_index=0, score=0.1),
        DocumentChunk(text="b", source_path="f", chunk_index=1, score=0.8),
    ]

    class FakeResponse:
        status_code = 404

        def raise_for_status(self) -> None:
            raise AssertionError("should not raise on 404 branch")

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
