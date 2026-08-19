"""Supported document suffixes for discovery."""

from __future__ import annotations

TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".rst", ".log"}

DOCLING_SUFFIXES = {
    ".pdf",
    ".docx",
    ".pptx",
    ".xlsx",
    ".xls",
    ".html",
    ".htm",
    ".csv",
}

SUPPORTED_SUFFIXES = TEXT_SUFFIXES | DOCLING_SUFFIXES
