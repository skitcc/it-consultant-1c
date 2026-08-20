from mail_gateway.adapters.rag.vllm_reranker import (
    VllmReranker,
    scores_from_vllm_rerank,
)
from mail_gateway.domain.models import DocumentChunk


def test_scores_from_vllm_rerank_preserves_original_order() -> None:
    scores = scores_from_vllm_rerank(
        {
            "results": [
                {"index": 1, "relevance_score": 0.9},
                {"index": 0, "relevance_score": 0.1},
            ]
        },
        expected=2,
    )
    assert scores == [0.1, 0.9]


def test_vllm_reranker_one_http_call(monkeypatch) -> None:
    seen: list[dict] = []

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "results": [
                    {"index": 0, "relevance_score": 0.2},
                    {"index": 1, "relevance_score": 0.8},
                ]
            }

    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            return None

        def __enter__(self):
            return self

        def __exit__(self, *args) -> None:
            return None

        def post(self, url: str, json: dict):
            seen.append({"url": url, "json": json})
            return FakeResponse()

    monkeypatch.setattr(
        "mail_gateway.adapters.rag.vllm_reranker.httpx.Client",
        FakeClient,
    )
    chunks = [
        DocumentChunk(text="weak", source_path="a.md", chunk_index=0, score=0.9),
        DocumentChunk(text="strong", source_path="b.md", chunk_index=1, score=0.1),
    ]
    ranked = VllmReranker(
        base_url="http://vllm:8002",
        model="Qwen/Qwen3-Reranker-8B",
    ).rerank("printer", chunks)
    assert [chunk.text for chunk in ranked] == ["strong", "weak"]
    assert ranked[0].score == 0.8
    assert len(seen) == 1
    assert seen[0]["url"] == "http://vllm:8002/v1/rerank"
    assert seen[0]["json"]["documents"] == ["weak", "strong"]
    assert "instruction" in seen[0]["json"]


def test_vllm_reranker_falls_back_on_http_error(monkeypatch) -> None:
    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            return None

        def __enter__(self):
            return self

        def __exit__(self, *args) -> None:
            return None

        def post(self, url: str, json: dict):
            raise RuntimeError("down")

    monkeypatch.setattr(
        "mail_gateway.adapters.rag.vllm_reranker.httpx.Client",
        FakeClient,
    )
    chunks = [
        DocumentChunk(text="a", source_path="x", chunk_index=0, score=0.2),
        DocumentChunk(text="b", source_path="x", chunk_index=1, score=0.9),
    ]
    ranked = VllmReranker(base_url="http://x", model="m").rerank("q", chunks)
    assert [chunk.text for chunk in ranked] == ["b", "a"]
