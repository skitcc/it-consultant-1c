"""Stub indexer that only logs (tests / dry-run)."""

from __future__ import annotations

import logging
from collections.abc import Sequence

from reindex.domain.changes import FsChange

logger = logging.getLogger(__name__)


class LoggingIndexer:
    """Stub indexer that only logs a debug message (no real indexing)."""

    def reindex(self, watch_path: str) -> None:
        logger.debug("Reindex requested for watch_path=%s (stub, no indexing)", watch_path)

    def apply_changes(self, watch_path: str, changes: Sequence[FsChange]) -> None:
        logger.debug(
            "apply_changes watch_path=%s changes=%s (stub, no indexing)",
            watch_path,
            list(changes),
        )
