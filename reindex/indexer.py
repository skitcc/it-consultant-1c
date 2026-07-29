"""Indexer abstractions for the file-based database."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class Indexer(ABC):
    """Builds or refreshes an index over the watched database directory."""

    @abstractmethod
    def reindex(self, watch_path: str) -> None:
        """Reindex the database located at ``watch_path``."""


class LoggingIndexer(Indexer):
    """Stub indexer that only logs a debug message (no real indexing yet)."""

    def reindex(self, watch_path: str) -> None:
        logger.debug("Reindex requested for watch_path=%s (stub, no indexing)", watch_path)
