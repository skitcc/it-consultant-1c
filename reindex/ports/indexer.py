"""Port: build or refresh an index over the watched directory."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Indexer(Protocol):
    """Builds or refreshes an index over the watched database directory."""

    def reindex(self, watch_path: str) -> None:
        """Reindex the database located at ``watch_path``."""
        ...
