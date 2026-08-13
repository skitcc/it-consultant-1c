"""Port: extract embeddable chunks from a document file."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, runtime_checkable

from reindex.domain.models import DocumentChunk


@runtime_checkable
class DocumentReader(Protocol):
    """Turn a document file into domain chunks (no infrastructure types)."""

    def read(self, path: Path) -> Sequence[DocumentChunk]:
        """Return chunks extracted from ``path``, preserving semantic structure."""
        ...
