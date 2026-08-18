from __future__ import annotations

from types import SimpleNamespace

import pytest

from knowledge.adapters.outbound.qdrant_vector_index import QdrantVectorIndex
from knowledge.core.domain import DocumentChunk


class Client:
    def __init__(self, *, exists: bool = True, fail_upsert: bool = False) -> None:
        self.exists = exists
        self.fail_upsert = fail_upsert
        self.events: list[str] = []
        self.points = []
        self.selector = None
        self.created = 0
        self.indexed_fields: list[str] = []
        self.scroll_records = []

    def collection_exists(self, collection: str) -> bool:
        return self.exists

    def create_collection(self, **kwargs) -> None:
        self.events.append("create")
        self.created += 1
        self.exists = True

    def create_payload_index(self, **kwargs) -> None:
        self.indexed_fields.append(kwargs["field_name"])

    def upsert(self, **kwargs) -> None:
        self.events.append("upsert")
        if self.fail_upsert:
            raise RuntimeError("upsert failed")
        self.points = kwargs["points"]

    def delete(self, **kwargs) -> None:
        self.events.append("delete-old")
        self.selector = kwargs["points_selector"]

    def scroll(self, **kwargs):
        return self.scroll_records, None


def chunk(index: int = 0) -> DocumentChunk:
    return DocumentChunk(
        text=f"text-{index}",
        source_path="docs/manual.pdf",
        filename="manual.pdf",
        document_id="doc",
        knowledge_id="main",
        chunk_index=index,
        headings=("Section",),
        content_hash="new-version",
    )


def test_replace_upserts_new_version_before_deleting_only_old_version():
    client = Client()
    index = QdrantVectorIndex(url="unused", collection="docs", client=client)

    index.replace_document(
        document_id="doc",
        knowledge_id="main",
        content_hash="new-version",
        chunks=[chunk()],
        vectors=[[0.1, 0.2]],
    )

    assert client.events == ["upsert", "delete-old"]
    payload = client.points[0].payload
    assert payload == {
        "document_id": "doc",
        "knowledge_id": "main",
        "filename": "manual.pdf",
        "source_path": "docs/manual.pdf",
        "chunk_index": 0,
        "text": "text-0",
        "file_hash": "new-version",
        "document_version": "new-version",
        "headings": ["Section"],
    }
    old_filter = client.selector.filter
    assert {condition.key for condition in old_filter.must} == {
        "document_id",
        "knowledge_id",
    }
    assert old_filter.must_not[0].key == "document_version"
    assert old_filter.must_not[0].match.value == "new-version"


def test_upsert_failure_never_deletes_old_version():
    client = Client(fail_upsert=True)
    index = QdrantVectorIndex(url="unused", collection="docs", client=client)

    with pytest.raises(RuntimeError, match="upsert failed"):
        index.replace_document(
            document_id="doc",
            knowledge_id="main",
            content_hash="new-version",
            chunks=[chunk()],
            vectors=[[0.1, 0.2]],
        )

    assert client.events == ["upsert"]


def test_collection_is_created_only_when_absent_and_payload_fields_are_indexed():
    client = Client(exists=False)
    index = QdrantVectorIndex(url="unused", collection="docs", client=client)

    index.replace_document(
        document_id="doc",
        knowledge_id="main",
        content_hash="new-version",
        chunks=[chunk()],
        vectors=[[0.1, 0.2]],
    )

    assert client.created == 1
    assert {
        "document_id",
        "knowledge_id",
        "filename",
        "source_path",
        "file_hash",
        "document_version",
        "headings",
        "chunk_index",
    } <= set(client.indexed_fields)


def test_heading_neighbors_do_not_cross_into_adjacent_section():
    client = Client()
    client.scroll_records = [
        SimpleNamespace(payload={
            "document_id": "doc",
            "knowledge_id": "main",
            "source_path": "manual.pdf",
            "filename": "manual.pdf",
            "chunk_index": 1,
            "text": "previous",
            "document_version": "new-version",
            "headings": ["Принтеры"],
        }),
        SimpleNamespace(payload={
            "document_id": "doc",
            "knowledge_id": "main",
            "source_path": "manual.pdf",
            "filename": "manual.pdf",
            "chunk_index": 2,
            "text": "selected",
            "document_version": "new-version",
            "headings": ["Принтеры"],
        }),
        SimpleNamespace(payload={
            "document_id": "doc",
            "knowledge_id": "main",
            "source_path": "manual.pdf",
            "filename": "manual.pdf",
            "chunk_index": 3,
            "text": "vpn",
            "document_version": "new-version",
            "headings": ["Сеть"],
        }),
    ]
    index = QdrantVectorIndex(url="unused", collection="docs", client=client)
    seed = DocumentChunk(
        text="selected",
        source_path="manual.pdf",
        filename="manual.pdf",
        document_id="doc",
        knowledge_id="main",
        chunk_index=2,
        headings=("Принтеры",),
        content_hash="new-version",
    )

    neighbors = index.load_neighbors([seed], window=1)

    assert [chunk.text for chunk in neighbors] == ["previous", "selected"]
