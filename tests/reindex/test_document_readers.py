from pathlib import Path
from unittest.mock import MagicMock

from reindex.adapters.document_readers import (
    CompositeDocumentReader,
    DocxDocumentReader,
    PdfDocumentReader,
    TextDocumentReader,
)


def test_text_document_reader(tmp_path: Path) -> None:
    path = tmp_path / "note.md"
    path.write_text("привет docs", encoding="utf-8")
    assert TextDocumentReader().read(path) == "привет docs"


def test_composite_dispatches_by_suffix(tmp_path: Path) -> None:
    path = tmp_path / "a.txt"
    path.write_text("plain", encoding="utf-8")
    reader = CompositeDocumentReader()
    assert reader.read(path) == "plain"


def test_composite_unsupported_suffix(tmp_path: Path) -> None:
    path = tmp_path / "a.bin"
    path.write_bytes(b"\x00\x01")
    reader = CompositeDocumentReader()
    try:
        reader.read(path)
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "Unsupported" in str(exc)


def test_pdf_reader_uses_pypdf(monkeypatch, tmp_path: Path) -> None:
    path = tmp_path / "a.pdf"
    path.write_bytes(b"%PDF")

    page = MagicMock()
    page.extract_text.return_value = "pdf text"
    reader_obj = MagicMock()
    reader_obj.pages = [page]

    import sys

    fake_pypdf = MagicMock()
    fake_pypdf.PdfReader.return_value = reader_obj
    monkeypatch.setitem(sys.modules, "pypdf", fake_pypdf)

    text = PdfDocumentReader().read(path)
    assert text == "pdf text"
    fake_pypdf.PdfReader.assert_called_once_with(str(path))


def test_docx_reader_uses_python_docx(monkeypatch, tmp_path: Path) -> None:
    path = tmp_path / "a.docx"
    path.write_bytes(b"PK")

    para = MagicMock()
    para.text = "docx line"
    document = MagicMock()
    document.paragraphs = [para]

    import sys
    fake_docx = MagicMock()
    fake_docx.Document.return_value = document
    monkeypatch.setitem(sys.modules, "docx", fake_docx)

    text = DocxDocumentReader().read(path)
    assert text == "docx line"
    fake_docx.Document.assert_called_once_with(str(path))
