"""Load plain text from watched document files."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".rst", ".log", ".csv"}
_SUPPORTED_SUFFIXES = _TEXT_SUFFIXES | {".pdf", ".docx"}


def iter_document_files(watch_path: str | Path) -> list[Path]:
    root = Path(watch_path)
    files: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.name.startswith("."):
            continue
        if path.suffix.lower() in _SUPPORTED_SUFFIXES:
            files.append(path)
    return files


def read_document_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in _TEXT_SUFFIXES:
        return path.read_text(encoding="utf-8", errors="replace")
    if suffix == ".pdf":
        return _read_pdf(path)
    if suffix == ".docx":
        return _read_docx(path)
    raise ValueError(f"Unsupported document type: {path}")


def _read_pdf(path: Path) -> str:
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


def _read_docx(path: Path) -> str:
    try:
        from docx import Document
    except ImportError as exc:
        raise RuntimeError(
            "Reading DOCX requires python-docx; install with: pip install -e '.[reindex]'"
        ) from exc

    document = Document(str(path))
    return "\n".join(paragraph.text for paragraph in document.paragraphs if paragraph.text)
