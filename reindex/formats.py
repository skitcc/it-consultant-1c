"""Supported document suffixes for discovery and readers."""

from __future__ import annotations

# Plain text: read as-is (no Docling).
TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".rst", ".log"}

# Structured / office formats converted to Markdown via Docling.
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
