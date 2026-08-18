"""Remove one document without affecting other indexed documents."""

from __future__ import annotations

from knowledge.core.ports import DocumentRegistry, VectorIndex


class RemoveDocument:
    def __init__(self, *, registry: DocumentRegistry, vector_index: VectorIndex) -> None:
        self._registry = registry
        self._vector_index = vector_index

    def execute(self, document_id: str, *, knowledge_id: str = "main") -> bool:
        if self._registry.get(document_id, knowledge_id) is None:
            return False
        self._vector_index.remove_document(document_id, knowledge_id)
        self._registry.delete(document_id, knowledge_id)
        return True

    __call__ = execute
