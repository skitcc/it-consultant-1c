from pathlib import Path

from reindex.domain.documents import (
    iter_document_files,
    parse_index_extensions,
    resolve_index_extensions,
)
from reindex.domain.upload_names import owui_upload_file_id, parse_watched_upload
from knowledge.core.use_cases.index_document import stable_document_id


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


def test_owui_upload_name_uses_file_uuid_and_original_filename() -> None:
    parsed = parse_watched_upload(
        "38ec13c1-3127-4a81-b301-f0e2b6f72baa_manual.pdf"
    )
    assert parsed.document_id == "38ec13c1-3127-4a81-b301-f0e2b6f72baa"
    assert parsed.filename == "manual.pdf"
    assert parsed.source_path == "manual.pdf"


def test_owui_upload_name_keeps_subdirectory_in_source_path() -> None:
    parsed = parse_watched_upload(
        "guides/38ec13c1-3127-4a81-b301-f0e2b6f72baa_manual.pdf"
    )
    assert parsed.filename == "manual.pdf"
    assert parsed.source_path == "guides/manual.pdf"


def test_plain_filename_falls_back_to_stable_document_id() -> None:
    parsed = parse_watched_upload("notes/guide.md", knowledge_id="main")
    assert parsed.filename == "guide.md"
    assert parsed.source_path == "notes/guide.md"
    assert parsed.document_id == stable_document_id("main", "notes/guide.md")


def test_owui_upload_file_id_ignores_manual_copies() -> None:
    assert owui_upload_file_id("faq.md") is None
    assert (
        owui_upload_file_id("38ec13c1-3127-4a81-b301-f0e2b6f72baa_manual.pdf")
        == "38ec13c1-3127-4a81-b301-f0e2b6f72baa"
    )
