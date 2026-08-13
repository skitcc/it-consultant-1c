from pathlib import Path
from unittest.mock import MagicMock

from reindex.adapters.document_readers import (
    CompositeDocumentReader,
    DoclingDocumentReader,
    TextDocumentReader,
)
from reindex.domain.models import DocumentChunk


def test_text_document_reader(tmp_path: Path) -> None:
    path = tmp_path / "note.md"
    path.write_text("привет docs", encoding="utf-8")
    chunks = TextDocumentReader().read(path)
    assert chunks == [DocumentChunk(text="привет docs")]


def test_text_document_reader_empty(tmp_path: Path) -> None:
    path = tmp_path / "empty.txt"
    path.write_text("  \n", encoding="utf-8")
    assert TextDocumentReader().read(path) == []


def test_composite_unsupported_suffix(tmp_path: Path) -> None:
    path = tmp_path / "a.bin"
    path.write_bytes(b"\x00\x01")
    reader = CompositeDocumentReader(readers={})
    try:
        reader.read(path)
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "Unsupported" in str(exc)


def test_docling_reader_hybrid_chunks(tmp_path: Path) -> None:
    path = tmp_path / "a.pdf"
    path.write_bytes(b"%PDF")

    raw_chunk = MagicMock()
    raw_chunk.meta.headings = ["Раздел 1", "Подраздел"]
    result = MagicMock()
    result.document = object()
    converter = MagicMock()
    converter.convert.return_value = result
    chunker = MagicMock()
    chunker.chunk.return_value = [raw_chunk]
    chunker.contextualize.return_value = "Раздел 1\n\nтекст чанка"

    chunks = DoclingDocumentReader(converter=converter, chunker=chunker).read(path)

    assert chunks == [
        DocumentChunk(text="Раздел 1\n\nтекст чанка", headings=("Раздел 1", "Подраздел")),
    ]
    converter.convert.assert_called_once_with(str(path))
    chunker.chunk.assert_called_once_with(dl_doc=result.document)
    chunker.contextualize.assert_called_once_with(raw_chunk)


def test_composite_dispatches_all_supported_suffixes(tmp_path: Path) -> None:
    path = tmp_path / "guide.md"
    path.write_text("# Title", encoding="utf-8")

    raw_chunk = MagicMock()
    raw_chunk.meta.headings = ["Title"]
    result = MagicMock()
    result.document = object()
    converter = MagicMock()
    converter.convert.return_value = result
    chunker = MagicMock()
    chunker.chunk.return_value = [raw_chunk]
    chunker.contextualize.return_value = "# Title\n\nbody"

    reader = CompositeDocumentReader(converter=converter, chunker=chunker)
    chunks = list(reader.read(path))
    assert chunks[0].text == "# Title\n\nbody"
    assert chunks[0].headings == ("Title",)
    converter.convert.assert_called_once_with(str(path))


def test_composite_dispatches_xlsx(tmp_path: Path) -> None:
    path = tmp_path / "sheet.xlsx"
    path.write_bytes(b"PK")

    raw_chunk = MagicMock()
    raw_chunk.meta.headings = []
    result = MagicMock()
    result.document = object()
    converter = MagicMock()
    converter.convert.return_value = result
    chunker = MagicMock()
    chunker.chunk.return_value = [raw_chunk]
    chunker.contextualize.return_value = "| a | b |\n|---|---|\n| 1 | 2 |"

    reader = CompositeDocumentReader(
        readers={".xlsx": DoclingDocumentReader(converter=converter, chunker=chunker)}
    )
    chunks = list(reader.read(path))
    assert "1 | 2" in chunks[0].text
    converter.convert.assert_called_once_with(str(path))


def test_docling_reader_skips_empty_contextualized_chunks(tmp_path: Path) -> None:
    path = tmp_path / "a.docx"
    path.write_bytes(b"PK")

    empty = MagicMock()
    empty.meta.headings = []
    filled = MagicMock()
    filled.meta.headings = ["H"]
    result = MagicMock()
    result.document = object()
    converter = MagicMock()
    converter.convert.return_value = result
    chunker = MagicMock()
    chunker.chunk.return_value = [empty, filled]
    chunker.contextualize.side_effect = ["  ", "kept"]

    chunks = DoclingDocumentReader(converter=converter, chunker=chunker).read(path)
    assert chunks == [DocumentChunk(text="kept", headings=("H",))]


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
