from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from knowledge.core.domain import DocumentChunk, DocumentRecord
from knowledge.core.use_cases import IndexDocument


@dataclass
class Registry:
    record: DocumentRecord | None = None
    events: list[str] = field(default_factory=list)

    def get(self, document_id: str, knowledge_id: str = "main"):
        if (
            self.record is not None
            and self.record.document_id == document_id
            and self.record.knowledge_id == knowledge_id
        ):
            return self.record
        return None

    def list(self, knowledge_id: str = "main"):
        return [self.record] if self.record and self.record.knowledge_id == knowledge_id else []

    def save(self, record: DocumentRecord) -> None:
        self.events.append("registry.save")
        self.record = record

    def delete(self, document_id: str, knowledge_id: str = "main") -> None:
        self.record = None


class Parser:
    def __init__(self, chunks=None, error: Exception | None = None) -> None:
        self.chunks = chunks or [DocumentChunk(text=" part one ")]
        self.error = error
        self.calls: list[tuple[bytes, str]] = []

    def parse(self, raw_bytes: bytes, filename: str):
        self.calls.append((raw_bytes, filename))
        if self.error:
            raise self.error
        return self.chunks


class Embedder:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.calls: list[list[str]] = []

    def embed(self, text: str):
        return [1.0]

    def embed_documents(self, texts: list[str]):
        self.calls.append(texts)
        if self.error:
            raise self.error
        return [[float(index), 1.0] for index, _ in enumerate(texts)]


class VectorIndex:
    def __init__(self, events: list[str], error: Exception | None = None) -> None:
        self.events = events
        self.error = error
        self.replacements = []

    def replace_document(self, **kwargs) -> None:
        self.events.append("vector.replace")
        if self.error:
            raise self.error
        self.replacements.append(kwargs)


def make_use_case(*, registry=None, parser=None, embedder=None, vector=None):
    events: list[str] = []
    registry = registry or Registry(events=events)
    parser = parser or Parser()
    embedder = embedder or Embedder()
    vector = vector or VectorIndex(events)
    return (
        IndexDocument(
            parser=parser,
            registry=registry,
            embedder=embedder,
            vector_index=vector,
        ),
        registry,
        parser,
        embedder,
        vector,
        events,
    )


def test_indexes_exact_bytes_and_saves_registry_after_vector_success():
    use_case, registry, parser, embedder, vector, events = make_use_case()
    raw = b"\x00original\r\nbytes\xff"

    result = use_case.execute(
        raw,
        "folder/manual.pdf",
        document_id="doc-1",
        source_path="folder/manual.pdf",
    )

    assert parser.calls == [(raw, "manual.pdf")]
    assert embedder.calls == [["part one"]]
    assert events == ["vector.replace", "registry.save"]
    replacement = vector.replacements[0]
    assert replacement["document_id"] == "doc-1"
    assert replacement["chunks"][0].source_path == "folder/manual.pdf"
    assert replacement["chunks"][0].content_hash == result.content_hash
    assert result.status == "indexed"
    assert result.chunk_count == 1
    assert not hasattr(result, "text")
    assert registry.record.content_hash == result.content_hash


def test_skips_unchanged_document_without_parsing_or_embedding():
    raw = b"same"
    use_case, registry, parser, embedder, vector, events = make_use_case()
    first = use_case(raw, "same.txt", document_id="doc")

    second = use_case(raw, "same.txt", document_id="doc")

    assert first.status == "indexed"
    assert second.status == "unchanged"
    assert len(parser.calls) == 1
    assert len(embedder.calls) == 1
    assert len(vector.replacements) == 1
    assert events == ["vector.replace", "registry.save"]
    assert registry.record.content_hash == first.content_hash


@pytest.mark.parametrize("failure_stage", ["parse", "embed", "vector"])
def test_failure_preserves_previous_registry_version(failure_stage: str):
    old = DocumentRecord(
        document_id="doc",
        knowledge_id="main",
        filename="manual.pdf",
        content_hash="old-hash",
        status="indexed",
        chunk_count=3,
    )
    events: list[str] = []
    registry = Registry(record=old, events=events)
    parser = Parser(error=RuntimeError("parse")) if failure_stage == "parse" else Parser()
    embedder = Embedder(error=RuntimeError("embed")) if failure_stage == "embed" else Embedder()
    vector = VectorIndex(
        events,
        error=RuntimeError("vector") if failure_stage == "vector" else None,
    )
    use_case, *_ = make_use_case(
        registry=registry,
        parser=parser,
        embedder=embedder,
        vector=vector,
    )

    with pytest.raises(RuntimeError, match=failure_stage):
        use_case(b"new bytes", "manual.pdf", document_id="doc")

    assert registry.record == old
    assert "registry.save" not in events
    if failure_stage in {"parse", "embed"}:
        assert "vector.replace" not in events
