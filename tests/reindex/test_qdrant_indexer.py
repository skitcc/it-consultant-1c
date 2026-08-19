from pathlib import Path
from unittest.mock import MagicMock

from common.embeddings import OllamaEmbedder
from reindex.adapters.qdrant_indexer import QdrantIndexer, file_content_hash
from reindex.domain.models import DocumentChunk


class FakeEmbedder:
    embed_calls = 0

    def embed(self, text: str) -> list[float]:
        type(self).embed_calls += 1
        return [float(len(text) % 7), 1.0, 0.5]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]


class FakeReader:
    def read(self, path: Path) -> list[DocumentChunk]:
        text = path.read_text(encoding="utf-8")
        return [DocumentChunk(text=text, headings=("Intro",))]


def _mock_client(*, exists: bool = True) -> MagicMock:
    client = MagicMock()
    client.collection_exists.return_value = exists
    client.scroll.return_value = ([], None)
    return client


def test_qdrant_indexer_indexes_reader_chunks(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "guide.md").write_text(
        "Как настроить обмен данными в 1С.\n" * 5,
        encoding="utf-8",
    )

    client = _mock_client(exists=False)
    FakeEmbedder.embed_calls = 0

    indexer = QdrantIndexer(
        qdrant_url="http://127.0.0.1:6333",
        collection="docs",
        embedder=FakeEmbedder(),
        document_reader=FakeReader(),
    )
    indexer._client = client

    indexer.reindex(str(docs))

    client.delete_collection.assert_not_called()
    client.create_collection.assert_called_once()
    client.upsert.assert_called_once()
    points = client.upsert.call_args.kwargs["points"]
    assert len(points) == 1
    assert points[0].payload["source_path"] == "guide.md"
    assert points[0].payload["chunk_index"] == 0
    assert "обмен" in points[0].payload["text"]
    assert points[0].payload["headings"] == ["Intro"]
    assert len(points[0].payload["file_hash"]) == 64


def test_qdrant_indexer_skips_disallowed_extensions(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "guide.md").write_text("markdown text " * 10, encoding="utf-8")
    (docs / "notes.txt").write_text("plain text " * 10, encoding="utf-8")

    client = _mock_client(exists=False)
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

    client = _mock_client(exists=False)
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


def test_reindex_skips_unchanged_files_without_wipe(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "guide.md").write_text("stable content", encoding="utf-8")
    content_hash = file_content_hash(docs / "guide.md")

    record = MagicMock(payload={"source_path": "guide.md", "file_hash": content_hash})
    client = _mock_client(exists=True)
    client.scroll.return_value = ([record], None)
    FakeEmbedder.embed_calls = 0

    indexer = QdrantIndexer(
        qdrant_url="http://127.0.0.1:6333",
        collection="docs",
        embedder=FakeEmbedder(),
        document_reader=FakeReader(),
    )
    indexer._client = client
    indexer.reindex(str(docs))

    client.delete_collection.assert_not_called()
    client.upsert.assert_not_called()
    assert FakeEmbedder.embed_calls == 0


def test_reindex_deduplicates_identical_files(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text("same bytes", encoding="utf-8")
    sub = docs / "sub"
    sub.mkdir()
    (sub / "a.md").write_text("same bytes", encoding="utf-8")

    client = _mock_client(exists=False)
    indexer = QdrantIndexer(
        qdrant_url="http://127.0.0.1:6333",
        collection="docs",
        embedder=FakeEmbedder(),
        document_reader=FakeReader(),
    )
    indexer._client = client
    indexer.reindex(str(docs))

    points = client.upsert.call_args.kwargs["points"]
    assert {point.payload["source_path"] for point in points} == {"a.md"}


def test_reindex_removes_stale_qdrant_paths(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "keep.md").write_text("keep", encoding="utf-8")

    stale = MagicMock(payload={"source_path": "gone.md", "file_hash": "dead"})
    client = _mock_client(exists=True)
    client.scroll.side_effect = [
        ([stale], None),
        ([], None),
    ]

    indexer = QdrantIndexer(
        qdrant_url="http://127.0.0.1:6333",
        collection="docs",
        embedder=FakeEmbedder(),
        document_reader=FakeReader(),
    )
    indexer._client = client
    indexer.reindex(str(docs))

    client.delete_collection.assert_not_called()
    delete_filter = client.delete.call_args.kwargs["points_selector"].filter.must[0]
    assert delete_filter.match.value == "gone.md"


def test_ollama_embedder_posts_prompt(monkeypatch) -> None:
    captured: dict = {}

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"embeddings": [[0.1, 0.2, 0.3]]}

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
    assert captured["url"] == "http://ollama:11434/api/embed"
    assert captured["json"]["model"] == "nomic-embed-text"
    assert captured["json"]["input"] == ["hello docs"]


def test_ollama_embedder_accepts_legacy_embedding_field(monkeypatch) -> None:
    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"embedding": [0.4, 0.5]}

    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args) -> None:
            return None

        def post(self, url: str, json: dict):
            return FakeResponse()

    monkeypatch.setattr("common.embeddings.httpx.Client", FakeClient)
    embedder = OllamaEmbedder(base_url="http://ollama:11434")
    assert embedder.embed("hello") == [0.4, 0.5]


def test_ollama_embedder_batches_documents(monkeypatch) -> None:
    captured: dict = {}

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"embeddings": [[0.1, 0.2], [0.3, 0.4]]}

    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args) -> None:
            return None

        def post(self, url: str, json: dict):
            captured["json"] = json
            return FakeResponse()

    monkeypatch.setattr("common.embeddings.httpx.Client", FakeClient)
    embedder = OllamaEmbedder(base_url="http://ollama:11434")
    vectors = embedder.embed_documents(["one", "two"])
    assert vectors == [[0.1, 0.2], [0.3, 0.4]]
    assert captured["json"]["input"] == ["one", "two"]


def test_apply_changes_upserts_one_file_without_recreating_collection(tmp_path: Path) -> None:
    from reindex.domain.changes import FsChange

    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "guide.md").write_text("chunk text " * 10, encoding="utf-8")

    client = _mock_client(exists=True)
    indexer = QdrantIndexer(
        qdrant_url="http://127.0.0.1:6333",
        collection="docs",
        embedder=FakeEmbedder(),
        document_reader=FakeReader(),
    )
    indexer._client = client
    indexer.apply_changes(str(docs), [FsChange("upsert", "guide.md")])

    client.delete_collection.assert_not_called()
    client.delete.assert_called()
    points = client.upsert.call_args.kwargs["points"]
    assert {point.payload["source_path"] for point in points} == {"guide.md"}


def test_apply_changes_skips_unchanged_file(tmp_path: Path) -> None:
    from reindex.domain.changes import FsChange

    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "guide.md").write_text("same", encoding="utf-8")
    content_hash = file_content_hash(docs / "guide.md")

    record = MagicMock(payload={"file_hash": content_hash})
    client = _mock_client(exists=True)
    client.scroll.return_value = ([record], None)
    FakeEmbedder.embed_calls = 0

    indexer = QdrantIndexer(
        qdrant_url="http://127.0.0.1:6333",
        collection="docs",
        embedder=FakeEmbedder(),
        document_reader=FakeReader(),
    )
    indexer._client = client
    indexer.apply_changes(str(docs), [FsChange("upsert", "guide.md")])

    client.upsert.assert_not_called()
    assert FakeEmbedder.embed_calls == 0


def test_apply_changes_skips_duplicate_copy(tmp_path: Path) -> None:
    from reindex.domain.changes import FsChange

    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text("dup", encoding="utf-8")
    sub = docs / "sub"
    sub.mkdir()
    (sub / "a.md").write_text("dup", encoding="utf-8")

    client = _mock_client(exists=True)
    FakeEmbedder.embed_calls = 0
    indexer = QdrantIndexer(
        qdrant_url="http://127.0.0.1:6333",
        collection="docs",
        embedder=FakeEmbedder(),
        document_reader=FakeReader(),
    )
    indexer._client = client
    indexer.apply_changes(str(docs), [FsChange("upsert", "sub/a.md")])

    client.upsert.assert_not_called()
    assert FakeEmbedder.embed_calls == 0


def test_apply_changes_delete_promotes_next_duplicate(tmp_path: Path) -> None:
    from reindex.domain.changes import FsChange

    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text("dup", encoding="utf-8")
    sub = docs / "sub"
    sub.mkdir()
    (sub / "a.md").write_text("dup", encoding="utf-8")
    content_hash = file_content_hash(docs / "a.md")

    record = MagicMock(payload={"source_path": "a.md", "file_hash": content_hash})
    client = _mock_client(exists=True)
    client.scroll.side_effect = [
        ([record], None),
        ([], None),
        ([], None),
    ]

    indexer = QdrantIndexer(
        qdrant_url="http://127.0.0.1:6333",
        collection="docs",
        embedder=FakeEmbedder(),
        document_reader=FakeReader(),
    )
    indexer._client = client
    (docs / "a.md").unlink()
    indexer.apply_changes(str(docs), [FsChange("delete", "a.md")])

    upsert_sources = {
        point.payload["source_path"]
        for call in client.upsert.call_args_list
        for point in call.kwargs["points"]
    }
    assert upsert_sources == {"sub/a.md"}


def test_apply_changes_delete_uses_source_path_filter(tmp_path: Path) -> None:
    from reindex.domain.changes import FsChange

    docs = tmp_path / "docs"
    docs.mkdir()
    client = _mock_client(exists=True)
    indexer = QdrantIndexer(
        qdrant_url="http://127.0.0.1:6333",
        collection="docs",
        embedder=FakeEmbedder(),
        document_reader=FakeReader(),
    )
    indexer._client = client
    indexer.apply_changes(str(docs), [FsChange("delete", "gone.md")])

    client.delete_collection.assert_not_called()
    client.upsert.assert_not_called()
    condition = client.delete.call_args.kwargs["points_selector"].filter.must[0]
    assert condition.match.value == "gone.md"


def test_apply_changes_repeat_upsert_after_content_change(tmp_path: Path) -> None:
    from reindex.domain.changes import FsChange

    docs = tmp_path / "docs"
    docs.mkdir()
    path = docs / "guide.md"
    path.write_text("short", encoding="utf-8")
    old_hash = file_content_hash(path)

    record = MagicMock(payload={"file_hash": old_hash})
    client = _mock_client(exists=True)
    client.scroll.side_effect = [
        ([record], None),
        ([], None),
    ]

    indexer = QdrantIndexer(
        qdrant_url="http://127.0.0.1:6333",
        collection="docs",
        embedder=FakeEmbedder(),
        document_reader=FakeReader(),
    )
    indexer._client = client
    path.write_text("longer text now", encoding="utf-8")
    indexer.apply_changes(str(docs), [FsChange("upsert", "guide.md")])
    assert client.upsert.call_count == 1


def test_apply_changes_prefix_delete_scrolls_matching_sources(tmp_path: Path) -> None:
    from reindex.domain.changes import FsChange

    docs = tmp_path / "docs"
    docs.mkdir()
    keep = MagicMock(id="keep", payload={"source_path": "root.md"})
    nested = MagicMock(id="nested", payload={"source_path": "sub/a.md"})
    client = _mock_client(exists=True)
    client.scroll.return_value = ([keep, nested], None)

    indexer = QdrantIndexer(
        qdrant_url="http://127.0.0.1:6333",
        collection="docs",
        embedder=FakeEmbedder(),
        document_reader=FakeReader(),
    )
    indexer._client = client
    indexer.apply_changes(str(docs), [FsChange("delete", "sub", is_prefix=True)])
    assert client.delete.call_args.kwargs["points_selector"] == ["nested"]
