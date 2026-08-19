"""Tests for shared logging setup."""

from __future__ import annotations

import logging

from common.logging_config import configure_logging


def test_info_level_quiets_httpx() -> None:
    configure_logging("INFO")
    assert logging.getLogger().level == logging.INFO
    assert logging.getLogger("httpx").level == logging.WARNING
    assert logging.getLogger("transformers").level == logging.WARNING


def test_debug_level_shows_httpx() -> None:
    configure_logging("DEBUG")
    assert logging.getLogger().level == logging.DEBUG
    assert logging.getLogger("httpx").level == logging.DEBUG
    configure_logging("INFO")
