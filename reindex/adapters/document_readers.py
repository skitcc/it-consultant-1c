"""Document reader adapters (plain text / Docling)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from reindex.formats import DOCLING_SUFFIXES, TEXT_SUFFIXES
from reindex.ports import DocumentReader

logger = logging.getLogger(__name__)


class TextDocumentReader:
    """Read UTF-8 text files as-is."""

    def read(self, path: Path) -> str:
        return path.read_text(encoding="utf-8", errors="replace")


class DoclingDocumentReader:
    """Convert office/PDF/HTML/CSV documents to Markdown via Docling."""

    def __init__(self, converter: Any | None = None) -> None:
        self._converter = converter

    def _get_converter(self) -> Any:
        if self._converter is not None:
            return self._converter
        try:
            from docling.document_converter import DocumentConverter
        except ImportError as exc:
            raise RuntimeError(
                "Reading documents requires docling; install with: pip install -e '.[reindex]'"
            ) from exc
        self._converter = DocumentConverter()
        return self._converter

    def read(self, path: Path) -> str:
        converter = self._get_converter()
        result = converter.convert(str(path))
        return result.document.export_to_markdown()


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
        docling = DoclingDocumentReader()
        self._readers = {
            **{suffix: text for suffix in TEXT_SUFFIXES},
            **{suffix: docling for suffix in DOCLING_SUFFIXES},
        }

    def read(self, path: Path) -> str:
        suffix = path.suffix.lower()
        reader = self._readers.get(suffix)
        if reader is None:
            raise ValueError(f"Unsupported document type: {path}")
        return reader.read(path)


def build_default_document_reader() -> DocumentReader:
    return CompositeDocumentReader()
