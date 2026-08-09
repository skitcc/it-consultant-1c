from pathlib import Path
from unittest.mock import MagicMock

from reindex.adapters.document_readers import (
    CompositeDocumentReader,
    DoclingDocumentReader,
    TextDocumentReader,
)


def test_text_document_reader(tmp_path: Path) -> None:
    path = tmp_path / "note.md"
    path.write_text("привет docs", encoding="utf-8")
    assert TextDocumentReader().read(path) == "привет docs"


def test_composite_dispatches_text_by_suffix(tmp_path: Path) -> None:
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


def test_docling_reader_exports_markdown(tmp_path: Path) -> None:
    path = tmp_path / "a.pdf"
    path.write_bytes(b"%PDF")

    document = MagicMock()
    document.export_to_markdown.return_value = "# Title\n\npdf body"
    result = MagicMock()
    result.document = document
    converter = MagicMock()
    converter.convert.return_value = result

    text = DoclingDocumentReader(converter=converter).read(path)
    assert text == "# Title\n\npdf body"
    converter.convert.assert_called_once_with(str(path))
    document.export_to_markdown.assert_called_once_with()


def test_composite_dispatches_docling_suffix(tmp_path: Path) -> None:
    path = tmp_path / "sheet.xlsx"
    path.write_bytes(b"PK")

    document = MagicMock()
    document.export_to_markdown.return_value = "| a | b |\n|---|---|\n| 1 | 2 |"
    result = MagicMock()
    result.document = document
    converter = MagicMock()
    converter.convert.return_value = result

    reader = CompositeDocumentReader(
        readers={".xlsx": DoclingDocumentReader(converter=converter)}
    )
    assert "1 | 2" in reader.read(path)
    converter.convert.assert_called_once_with(str(path))


def test_docling_reader_missing_package(monkeypatch, tmp_path: Path) -> None:
    path = tmp_path / "a.docx"
    path.write_bytes(b"PK")

    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "docling.document_converter" or name.startswith("docling"):
            raise ImportError("no docling")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    try:
        DoclingDocumentReader().read(path)
        assert False, "expected RuntimeError"
    except RuntimeError as exc:
        assert "docling" in str(exc)
