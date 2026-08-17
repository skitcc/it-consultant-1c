"""Port: build or refresh an index over the watched directory."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from reindex.domain.changes import FsChange


@runtime_checkable
class Indexer(Protocol):
    """Builds or refreshes an index over the watched database directory."""

    def reindex(self, watch_path: str) -> None:
        """Rebuild the whole index for ``watch_path`` (startup / ``--once``)."""
        ...

    def apply_changes(self, watch_path: str, changes: Sequence[FsChange]) -> None:
        """Apply incremental create/modify/delete/move operations."""
        ...
