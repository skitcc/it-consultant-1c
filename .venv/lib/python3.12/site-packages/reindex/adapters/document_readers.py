"""Document reader adapters (text / PDF / DOCX)."""

from __future__ import annotations

import logging
from pathlib import Path

from reindex.ports import DocumentReader

logger = logging.getLogger(__name__)

TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".rst", ".log", ".csv"}
PDF_SUFFIXES = {".pdf"}
DOCX_SUFFIXES = {".docx"}
SUPPORTED_SUFFIXES = TEXT_SUFFIXES | PDF_SUFFIXES | DOCX_SUFFIXES


class TextDocumentReader:
    """Read UTF-8 text files as-is."""

    def read(self, path: Path) -> str:
        return path.read_text(encoding="utf-8", errors="replace")


class PdfDocumentReader:
    """Extract text from PDF via pypdf."""

    def read(self, path: Path) -> str:
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise RuntimeError(
                "Reading PDF requires pypdf; install with: pip install -e '.[reindex]'"
            ) from exc

        reader = PdfReader(str(path))
        parts: list[str] = []
        for page in reader.pages:
            text = page.extract_text() or ""
            if text.strip():
                parts.append(text)
        return "\n\n".join(parts)


class DocxDocumentReader:
    """Extract text from DOCX via python-docx."""

    def read(self, path: Path) -> str:
        try:
            from docx import Document
        except ImportError as exc:
            raise RuntimeError(
                "Reading DOCX requires python-docx; install with: pip install -e '.[reindex]'"
            ) from exc

        document = Document(str(path))
        return "\n".join(
            paragraph.text for paragraph in document.paragraphs if paragraph.text
        )


class CompositeDocumentReader:
    """Dispatch to a reader by file suffix."""

    def __init__(
        self,
        readers: dict[str, DocumentReader] | None = None,
    ) -> None:
        if readers is not None:
            self._readers = {suffix.lower(): reader for suffix, reader in readers.items()}
            return

        text = TextDocumentReader()
        pdf = PdfDocumentReader()
        docx = DocxDocumentReader()
        self._readers = {
            **{suffix: text for suffix in TEXT_SUFFIXES},
            **{suffix: pdf for suffix in PDF_SUFFIXES},
            **{suffix: docx for suffix in DOCX_SUFFIXES},
        }

    def read(self, path: Path) -> str:
        suffix = path.suffix.lower()
        reader = self._readers.get(suffix)
        if reader is None:
            raise ValueError(f"Unsupported document type: {path}")
        return reader.read(path)


def build_default_document_reader() -> DocumentReader:
    return CompositeDocumentReader()
