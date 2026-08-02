"""Ports for the reindex service."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class DocumentReader(Protocol):
    """Extract plain text from a document file."""

    def read(self, path: Path) -> str:
        """Return UTF-8 text extracted from ``path``."""
        ...
