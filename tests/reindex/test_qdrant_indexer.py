from pathlib import Path
from unittest.mock import MagicMock

from common.embeddings import OllamaEmbedder
from reindex.adapters.qdrant_indexer import QdrantIndexer
from reindex.domain.models import DocumentChunk


class FakeEmbedder:
    def embed(self, text: str) -> list[float]:
        return [float(len(text) % 7), 1.0, 0.5]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]


class FakeReader:
    def read(self, path: Path) -> list[DocumentChunk]:
        text = path.read_text(encoding="utf-8")
        return [DocumentChunk(text=text, headings=("Intro",))]


def test_qdrant_indexer_indexes_reader_chunks(tmp_path: Path) -> None:
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
        embedder=FakeEmbedder(),
        document_reader=FakeReader(),
    )
    indexer._client = client

    indexer.reindex(str(docs))

    client.create_collection.assert_called_once()
    client.upsert.assert_called_once()
    points = client.upsert.call_args.kwargs["points"]
    assert len(points) == 1
    assert points[0].payload["source_path"] == "guide.md"
    assert points[0].payload["chunk_index"] == 0
    assert "обмен" in points[0].payload["text"]
    assert points[0].payload["headings"] == ["Intro"]


def test_qdrant_indexer_skips_disallowed_extensions(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "guide.md").write_text("markdown text " * 10, encoding="utf-8")
    (docs / "notes.txt").write_text("plain text " * 10, encoding="utf-8")

    client = MagicMock()
    client.collection_exists.return_value = False

    indexer = QdrantIndexer(
        qdrant_url="http://127.0.0.1:6333",
        collection="docs",
        embedder=FakeEmbedder(),
        document_reader=FakeReader(),
        allowed_extensions=frozenset({".md"}),
    )
    indexer._client = client

    indexer.reindex(str(docs))

    points = client.upsert.call_args.kwargs["points"]
    sources = {point.payload["source_path"] for point in points}
    assert sources == {"guide.md"}


def test_qdrant_indexer_skips_empty_reader_result(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "empty.md").write_text("ignored", encoding="utf-8")

    class EmptyReader:
        def read(self, path: Path) -> list[DocumentChunk]:
            return []

    client = MagicMock()
    indexer = QdrantIndexer(
        qdrant_url="http://127.0.0.1:6333",
        collection="docs",
        embedder=FakeEmbedder(),
        document_reader=EmptyReader(),
    )
    indexer._client = client
    indexer.reindex(str(docs))
    client.upsert.assert_not_called()
    client.create_collection.assert_not_called()


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
