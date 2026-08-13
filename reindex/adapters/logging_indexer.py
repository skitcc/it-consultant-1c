"""Stub indexer that only logs (tests / dry-run)."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class LoggingIndexer:
    """Stub indexer that only logs a debug message (no real indexing)."""

    def reindex(self, watch_path: str) -> None:
        logger.debug("Reindex requested for watch_path=%s (stub, no indexing)", watch_path)
