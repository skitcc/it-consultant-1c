from unittest.mock import MagicMock

from common.embeddings import OllamaEmbedder
from mail_gateway.adapters.rag.qdrant_retriever import QdrantRetriever
from mail_gateway.domain.models import DocumentChunk


def test_qdrant_retriever_maps_hits() -> None:
    hit = MagicMock()
    hit.payload = {
        "text": "настройка обмена",
        "source_path": "guide.md",
        "chunk_index": 2,
        "headings": ["Обмен данными", "Настройка"],
    }
    hit.score = 0.87

    response = MagicMock()
    response.points = [hit]

    client = MagicMock()
    client.collection_exists.return_value = True
    client.query_points.return_value = response

    class FakeEmbedder:
        def embed(self, text: str) -> list[float]:
            assert text == "как настроить обмен"
            return [0.1, 0.2]

    retriever = QdrantRetriever(
        qdrant_url="http://127.0.0.1:6333",
        collection="docs",
        embedder=FakeEmbedder(),  # type: ignore[arg-type]
        limit=20,
        score_threshold=0.2,
    )
    retriever._client = client

    chunks = retriever.retrieve("как настроить обмен")

    assert chunks == [
        DocumentChunk(
            text="настройка обмена",
            source_path="guide.md",
            chunk_index=2,
            score=0.87,
            headings=("Обмен данными", "Настройка"),
        )
    ]
    client.query_points.assert_called_once()
    kwargs = client.query_points.call_args.kwargs
    assert kwargs["limit"] == 20
    assert kwargs["score_threshold"] == 0.2
    assert kwargs["query"] == [0.1, 0.2]


def test_qdrant_retriever_empty_query() -> None:
    retriever = QdrantRetriever(
        qdrant_url="http://127.0.0.1:6333",
        collection="docs",
        embedder=MagicMock(spec=OllamaEmbedder),
    )
    assert retriever.retrieve("   ") == []


def test_qdrant_load_neighbors() -> None:
    point = MagicMock()
    point.payload = {
        "text": "neighbor",
        "source_path": "faq.md",
        "chunk_index": 1,
        "headings": ["Принтеры"],
    }
    client = MagicMock()
    client.collection_exists.return_value = True
    client.scroll.return_value = ([point], None)

    retriever = QdrantRetriever(
        qdrant_url="http://127.0.0.1:6333",
        collection="docs",
        embedder=MagicMock(spec=OllamaEmbedder),
    )
    retriever._client = client

    seeds = [
        DocumentChunk(
            text="seed",
            source_path="faq.md",
            chunk_index=1,
            score=0.5,
            headings=("Принтеры",),
        )
    ]
    loaded = retriever.load_neighbors(seeds, window=1)
    assert loaded == [
        DocumentChunk(
            text="neighbor",
            source_path="faq.md",
            chunk_index=1,
            score=None,
            headings=("Принтеры",),
        )
    ]
    client.scroll.assert_called_once()
