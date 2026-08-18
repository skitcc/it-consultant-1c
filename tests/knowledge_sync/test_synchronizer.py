from __future__ import annotations

from dataclasses import replace

from knowledge.core.domain import DocumentRecord
from knowledge_sync.open_webui_client import KnowledgeFile
from knowledge_sync.synchronizer import KnowledgeSynchronizer


class Registry:
    def __init__(self, records=()) -> None:
        self.records = {(r.knowledge_id, r.document_id): r for r in records}

    def get(self, document_id, knowledge_id="main"):
        return self.records.get((knowledge_id, document_id))

    def list(self, knowledge_id="main"):
        return [r for (kid, _), r in self.records.items() if kid == knowledge_id]

    def save(self, record):
        self.records[(record.knowledge_id, record.document_id)] = record

    def delete(self, document_id, knowledge_id="main"):
        self.records.pop((knowledge_id, document_id), None)


class Client:
    def __init__(self, files, contents=None) -> None:
        self.files = files
        self.contents = contents or {}
        self.downloads = []

    def list_knowledge_files(self, knowledge_id):
        assert knowledge_id == "main"
        return self.files

    def download_file(self, file_id):
        self.downloads.append(file_id)
        return self.contents[file_id]


class Index:
    def __init__(self, registry) -> None:
        self.registry = registry
        self.calls = []

    def execute(self, raw_bytes, filename, **kwargs):
        self.calls.append((raw_bytes, filename, kwargs))
        self.registry.save(
            DocumentRecord(
                document_id=kwargs["document_id"],
                knowledge_id=kwargs["knowledge_id"],
                filename=filename,
                content_hash="new-hash",
                status="indexed",
                chunk_count=1,
                source_updated_at=kwargs["source_updated_at"],
                last_seen_at=kwargs["last_seen_at"],
            )
        )


class Update:
    def __init__(self, registry) -> None:
        self.registry = registry

    def execute(self, document_id, *, knowledge_id="main", **kwargs):
        current = self.registry.get(document_id, knowledge_id)
        values = {key: value for key, value in kwargs.items() if value is not None}
        updated = replace(current, **values)
        self.registry.save(updated)
        return updated


class Remove:
    def __init__(self, registry) -> None:
        self.registry = registry
        self.calls = []

    def execute(self, document_id, *, knowledge_id="main"):
        self.calls.append(document_id)
        self.registry.delete(document_id, knowledge_id)
        return True


def _sync(client, registry, *, grace=3):
    index = Index(registry)
    remove = Remove(registry)
    synchronizer = KnowledgeSynchronizer(
        client=client,
        registry=registry,
        index_document=index,
        remove_document=remove,
        update_metadata=Update(registry),
        delete_grace_snapshots=grace,
    )
    return synchronizer, index, remove


def test_sync_downloads_only_new_document() -> None:
    existing = DocumentRecord(
        document_id="old",
        knowledge_id="main",
        filename="old.pdf",
        content_hash="old-hash",
        status="active",
        chunk_count=2,
    )
    registry = Registry([existing])
    client = Client(
        [
            KnowledgeFile("old", "old.pdf", "old-hash"),
            KnowledgeFile("new", "new.pdf", "new-hash"),
        ],
        {"new": b"%PDF-new"},
    )
    synchronizer, index, _ = _sync(client, registry)

    synchronizer.synchronize_once()

    assert client.downloads == ["new"]
    assert [call[2]["document_id"] for call in index.calls] == ["new"]


def test_sync_deletes_only_after_successful_grace_snapshots() -> None:
    record = DocumentRecord(
        document_id="gone",
        knowledge_id="main",
        filename="gone.pdf",
        content_hash="hash",
        status="active",
        chunk_count=2,
    )
    registry = Registry([record])
    synchronizer, _, remove = _sync(Client([]), registry, grace=2)

    synchronizer.synchronize_once()
    assert remove.calls == []
    synchronizer.synchronize_once()
    assert remove.calls == ["gone"]
