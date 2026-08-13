from pathlib import Path

from reindex.domain.documents import (
    iter_document_files,
    parse_index_extensions,
    resolve_index_extensions,
)


def test_parse_index_extensions() -> None:
    assert parse_index_extensions("md, .PDF; txt") == frozenset({".md", ".pdf", ".txt"})
    assert parse_index_extensions("") == frozenset()


def test_resolve_drops_unknown_extensions() -> None:
    resolved = resolve_index_extensions(".md,.xlsx,.pdf,.bin")
    assert resolved == frozenset({".md", ".xlsx", ".pdf"})


def test_iter_document_files_filters_by_extension(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text("md", encoding="utf-8")
    (tmp_path / "b.txt").write_text("txt", encoding="utf-8")
    (tmp_path / "c.pdf").write_bytes(b"%PDF")
    (tmp_path / "d.docx").write_bytes(b"PK")

    files = iter_document_files(tmp_path, allowed_extensions={".md", ".pdf"})
    names = {path.name for path in files}
    assert names == {"a.md", "c.pdf"}
