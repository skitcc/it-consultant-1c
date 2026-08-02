from pathlib import Path
from unittest.mock import MagicMock

from common.embeddings import OllamaEmbedder
from reindex.qdrant_indexer import QdrantIndexer


class FakeEmbedder:
    def embed(self, text: str) -> list[float]:
        # Deterministic tiny vector for tests.
        return [float(len(text) % 7), 1.0, 0.5]


def test_qdrant_indexer_indexes_markdown(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "guide.md").write_text(
        "Как настроить обмен данными в 1С.\n" * 5,
        encoding="utf-8",
    )

    client = MagicMock()
    client.collection_exists.return_value = False

    indexer = QdrantIndexer(
        qdrant_url="http://127.0.0.1:6333",
        collection="docs",
        embedder=FakeEmbedder(),  # type: ignore[arg-type]
        chunk_size=40,
        chunk_overlap=5,
    )
    indexer._client = client

    indexer.reindex(str(docs))

    client.create_collection.assert_called_once()
    client.upsert.assert_called_once()
    points = client.upsert.call_args.kwargs["points"]
    assert len(points) >= 1
    assert points[0].payload["source_path"] == "guide.md"
    assert "обмен" in points[0].payload["text"]


def test_ollama_embedder_posts_prompt(monkeypatch) -> None:
    captured: dict = {}

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"embedding": [0.1, 0.2, 0.3]}

    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args) -> None:
            return None

        def post(self, url: str, json: dict):
            captured["url"] = url
            captured["json"] = json
            return FakeResponse()

    monkeypatch.setattr("common.embeddings.httpx.Client", FakeClient)

    embedder = OllamaEmbedder(base_url="http://ollama:11434", model="nomic-embed-text")
    vector = embedder.embed("hello docs")

    assert vector == [0.1, 0.2, 0.3]
    assert captured["url"] == "http://ollama:11434/api/embeddings"
    assert captured["json"]["model"] == "nomic-embed-text"
    assert captured["json"]["prompt"] == "hello docs"
