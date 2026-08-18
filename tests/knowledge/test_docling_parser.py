from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from knowledge.adapters.outbound.docling_parser import DoclingDocumentParser


class Converter:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.path: Path | None = None
        self.received = b""

    def convert(self, path: Path):
        self.path = Path(path)
        assert self.path.exists()
        self.received = self.path.read_bytes()
        if self.fail:
            raise RuntimeError("conversion failed")
        return SimpleNamespace(document=object())


class Chunker:
    def chunk(self, *, dl_doc: object):
        del dl_doc
        return [
            SimpleNamespace(meta=SimpleNamespace(headings=["Setup", "1C"])),
            SimpleNamespace(meta=SimpleNamespace(headings=[])),
        ]

    def contextualize(self, chunk):
        return "First" if chunk.meta.headings else "  "


def test_parser_preserves_bytes_suffix_and_removes_temporary_file():
    converter = Converter()
    raw = b"\x00\xffnot-normalized\r\n"
    parser = DoclingDocumentParser(converter=converter, chunker=Chunker())

    chunks = parser.parse(raw, "manual.DOCX")

    assert converter.received == raw
    assert converter.path is not None
    assert converter.path.suffix == ".DOCX"
    assert not converter.path.exists()
    assert [chunk.text for chunk in chunks] == ["First"]
    assert chunks[0].headings == ("Setup", "1C")


def test_parser_removes_temporary_file_when_docling_fails():
    converter = Converter(fail=True)
    parser = DoclingDocumentParser(converter=converter, chunker=Chunker())

    with pytest.raises(RuntimeError, match="conversion failed"):
        parser.parse(b"source", "manual.pdf")

    assert converter.path is not None
    assert not converter.path.exists()
