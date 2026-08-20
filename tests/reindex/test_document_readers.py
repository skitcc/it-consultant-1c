from pathlib import Path
from unittest.mock import MagicMock

import pytest

from reindex.adapters.document_readers import (
    CompositeDocumentReader,
    DoclingDocumentReader,
    PictureDescriptionConfig,
    TextDocumentReader,
    chat_completions_url,
    format_picture_block,
    merge_split_table_chunks,
    picture_pipeline_flags,
    split_oversized_text,
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


def test_format_picture_block_includes_description_and_caption() -> None:
    text = format_picture_block(
        description="Форма настройки обмена, кнопка Записать",
        caption="Рис. 1. Обмен",
    )
    assert text.startswith("[Изображение]")
    assert "Описание: Форма настройки обмена, кнопка Записать" in text
    assert "Подпись: Рис. 1. Обмен" in text
    assert "<!-- image -->" not in text


def test_format_picture_block_empty_description() -> None:
    assert format_picture_block(description="  ", caption="Рис. 1") == ""
    assert "<!-- image -->" not in format_picture_block(description="")


def test_picture_pipeline_flags_disabled_keeps_ocr_off_only() -> None:
    flags = picture_pipeline_flags(PictureDescriptionConfig(enabled=False))
    assert flags == {"do_ocr": False}
    assert "enable_remote_services" not in flags
    assert "do_picture_description" not in flags


def test_picture_pipeline_flags_enabled_uses_remote_vlm() -> None:
    flags = picture_pipeline_flags(PictureDescriptionConfig(enabled=True))
    assert flags["do_ocr"] is False
    assert flags["do_picture_description"] is True
    assert flags["enable_remote_services"] is True
    assert flags["generate_picture_images"] is True


def test_chat_completions_url_uses_ollama_base() -> None:
    assert (
        chat_completions_url("http://127.0.0.1:11434")
        == "http://127.0.0.1:11434/v1/chat/completions"
    )
    assert (
        chat_completions_url("http://ollama:11434/")
        == "http://ollama:11434/v1/chat/completions"
    )


def test_picture_serializer_uses_annotation_and_caption() -> None:
    from reindex.adapters.document_readers import _PictureDescriptionSerializer

    item = MagicMock()
    annotation = MagicMock()
    annotation.text = "кнопка Провести"
    item.annotations = [annotation]
    item.caption_text = lambda doc=None: "Скрин формы"
    item.self_ref = "#/pictures/0"
    result = _PictureDescriptionSerializer().serialize(item=item, doc=None)
    text = result if isinstance(result, str) else getattr(result, "text", "")
    assert "[Изображение]" in text
    assert "Описание: кнопка Провести" in text
    assert "Подпись: Скрин формы" in text


def test_picture_serializer_is_accepted_by_chunking_doc_serializer() -> None:
    pytest.importorskip("docling_core")
    from docling_core.transforms.chunker.hierarchical_chunker import ChunkingDocSerializer
    from docling_core.types.doc.document import DoclingDocument

    from reindex.adapters.document_readers import _as_base_picture_serializer

    serializer = ChunkingDocSerializer(
        doc=DoclingDocument(name="probe"),
        picture_serializer=_as_base_picture_serializer(),
    )
    assert serializer.picture_serializer is not None
    from reindex.adapters.document_readers import _PictureDescriptionSerializer

    item = MagicMock()
    annotation = MagicMock()
    annotation.text = "кнопка Провести"
    item.annotations = [annotation]
    item.caption_text = lambda doc=None: "Скрин формы"
    item.self_ref = "#/pictures/0"
    result = _PictureDescriptionSerializer().serialize(item=item, doc=None)
    text = result if isinstance(result, str) else getattr(result, "text", "")
    assert "[Изображение]" in text
    assert "Описание: кнопка Провести" in text
    assert "Подпись: Скрин формы" in text


class _CountingTokenizer:
    def count_tokens(self, text: str) -> int:
        return max(1, (len(text) + 3) // 4)


def test_split_oversized_text_keeps_short_chunk() -> None:
    assert split_oversized_text("hello", max_tokens=32) == ["hello"]


def test_split_oversized_text_splits_paragraphs() -> None:
    part_a = "a" * 40
    part_b = "b" * 40
    pieces = split_oversized_text(
        f"{part_a}\n\n{part_b}",
        max_tokens=12,
        tokenizer=_CountingTokenizer(),
    )
    assert len(pieces) == 2
    assert part_a in pieces[0]
    assert part_b in pieces[1]


def test_split_oversized_text_keeps_whole_table_as_one_chunk() -> None:
    table = (
        "| Этап | Условие |\n"
        "|------|---------|\n"
        "| Intern 1 | адаптация |\n"
        "| K0-2 | PM готов покупать по K0 |\n"
        "| K1 | после K0-2 |"
    )
    pieces = split_oversized_text(
        table,
        max_tokens=20,
        tokenizer=_CountingTokenizer(),
    )
    assert pieces == [table]
    assert "K0-2 | PM готов покупать по K0" in pieces[0]
    assert "K1 | после K0-2" in pieces[0]


def test_split_oversized_text_extracts_table_from_surrounding_prose() -> None:
    table = (
        "| Этап | Условие |\n"
        "|------|---------|\n"
        "| K0-2 | PM готов покупать по K0 |\n"
        "| K1 | после K0-2 |"
    )
    text = f"{'a' * 40}\n\n{table}\n\n{'b' * 40}"
    pieces = split_oversized_text(
        text,
        max_tokens=12,
        tokenizer=_CountingTokenizer(),
    )
    assert len(pieces) == 3
    assert pieces[0] == "a" * 40
    assert pieces[1] == table
    assert pieces[2] == "b" * 40


def test_split_oversized_text_keeps_html_table_as_one_chunk() -> None:
    table = (
        "<table><thead><tr><th>Этап</th><th>Условие</th></tr></thead>"
        "<tbody><tr><td>K0-2</td><td>PM готов покупать по K0</td></tr>"
        "<tr><td>K1</td><td>после K0-2</td></tr></tbody></table>"
    )
    pieces = split_oversized_text(
        f"{'a' * 40}\n\n{table}\n\n{'b' * 40}",
        max_tokens=12,
        tokenizer=_CountingTokenizer(),
    )
    assert pieces[1] == table
    assert "K0-2" in pieces[1] and "K1" in pieces[1]


def test_merge_split_table_chunks_joins_repeated_header_fragments() -> None:
    headings = ("Критерии перехода",)
    first = DocumentChunk(
        text=(
            "| Этап | Условие |\n"
            "|------|---------|\n"
            "| Intern 1 | адаптация |"
        ),
        headings=headings,
    )
    echo = DocumentChunk(text="Критерии перехода", headings=headings)
    second = DocumentChunk(
        text=(
            "| Этап | Условие |\n"
            "|------|---------|\n"
            "| K0-2 | PM готов покупать по K0 |\n"
            "| K1 | после K0-2 |"
        ),
        headings=headings,
    )

    merged = merge_split_table_chunks([first, echo, second])
    assert len(merged) == 1
    assert "Intern 1 | адаптация" in merged[0].text
    assert "K0-2 | PM готов покупать по K0" in merged[0].text
    assert "K1 | после K0-2" in merged[0].text
    assert merged[0].text.count("| Этап | Условие |") == 1


def test_merge_split_table_chunks_keeps_different_tables_apart() -> None:
    headings = ("Раздел",)
    first = DocumentChunk(
        text="| Этап | Условие |\n|------|---------|\n| K0 | адаптация |",
        headings=headings,
    )
    second = DocumentChunk(
        text="| Роль | Требование |\n|------|------------|\n| PM | покупает |",
        headings=headings,
    )
    merged = merge_split_table_chunks([first, second])
    assert [chunk.text for chunk in merged] == [first.text, second.text]


def test_merge_split_table_chunks_does_not_extend_atomic_table() -> None:
    headings = ("Раздел",)
    full = DocumentChunk(
        text="| Этап | Условие |\n|------|---------|\n| K0 | адаптация |\n| K1 | переход |",
        headings=headings,
        atomic=True,
    )
    fragment = DocumentChunk(
        text="| Этап | Условие |\n|------|---------|\n| K1 | переход |",
        headings=headings,
    )
    merged = merge_split_table_chunks([full, fragment])
    assert merged[0] is full
    assert merged[1].text == fragment.text


def test_docling_reader_renders_full_table_item_once(tmp_path: Path) -> None:
    path = tmp_path / "grades.pptx"
    path.write_bytes(b"PK")

    full_table = (
        "| Этап | Условие |\n"
        "|------|---------|\n"
        "| Intern 1 | адаптация |\n"
        "| K0-2 | PM готов покупать по K0 |\n"
        "| K1 | после K0-2 |"
    )
    table = MagicMock()
    table.self_ref = "#/tables/0"
    table.export_to_markdown.return_value = full_table
    table.data.grid = []

    first = MagicMock()
    first.meta.headings = ["Критерии"]
    first.meta.doc_items = [table]
    second = MagicMock()
    second.meta.headings = ["Критерии"]
    second.meta.doc_items = [table]
    result = MagicMock()
    result.document = object()
    converter = MagicMock()
    converter.convert.return_value = result
    chunker = MagicMock()
    chunker.chunk.return_value = [first, second]
    chunker.contextualize.side_effect = [
        "| Этап | Условие |\n|------|---------|\n| Intern 1 | адаптация |",
        "| Этап | Условие |\n|------|---------|\n| K1 | после K0-2 |",
    ]
    chunker.serializer_provider = None

    chunks = DoclingDocumentReader(
        converter=converter,
        chunker=chunker,
        max_tokens=8,
    ).read(path)

    tables = [chunk for chunk in chunks if chunk.atomic]
    assert len(tables) == 1
    assert tables[0].text == full_table
    assert "Intern 1 | адаптация" in tables[0].text
    assert "K1 | после K0-2" in tables[0].text
    assert tables[0].headings == ("Критерии",)


def test_docling_reader_renders_table_from_grid_without_size_limit(
    tmp_path: Path,
) -> None:
    path = tmp_path / "sheet.xlsx"
    path.write_bytes(b"PK")

    class _Cell:
        def __init__(self, text: str) -> None:
            self.text = text

    class _Table:
        self_ref = "#/tables/3"
        data = type(
            "Data",
            (),
            {
                "grid": [
                    [_Cell("Этап"), _Cell("Условие")],
                    [_Cell("K0-2"), _Cell("PM готов покупать по K0")],
                    [_Cell("K1"), _Cell("после K0-2")],
                ]
            },
        )()

    raw = MagicMock()
    raw.meta.headings = ["Лист"]
    raw.meta.doc_items = [_Table()]
    result = MagicMock()
    result.document = object()
    converter = MagicMock()
    converter.convert.return_value = result
    chunker = MagicMock()
    chunker.chunk.return_value = [raw]
    chunker.contextualize.return_value = "should not be used"
    chunker.serializer_provider = None

    chunks = DoclingDocumentReader(
        converter=converter,
        chunker=chunker,
        max_tokens=4,
    ).read(path)

    assert len(chunks) == 1
    assert chunks[0].atomic is True
    assert chunks[0].text.startswith("| Этап | Условие |")
    assert "K0-2" in chunks[0].text
    assert "K1" in chunks[0].text


def test_vlm_http_logging_records_chat_completions(monkeypatch, caplog) -> None:
    import logging as logging_mod

    import requests

    class DummyResponse:
        ok = True
        status_code = 200

        def json(self) -> dict:
            return {"choices": [{"message": {"content": "скрин формы обмена"}}]}

    def fake_post(self, url, *args, **kwargs):
        return DummyResponse()

    monkeypatch.setattr(requests.Session, "post", fake_post)
    from reindex.adapters.document_readers import _log_vlm_http

    caplog.set_level(logging_mod.INFO)
    with _log_vlm_http():
        session = requests.Session()
        session.post("https://huggingface.co/api/models/x")
        session.post(
            "http://spark.pers.local:11434/v1/chat/completions",
            json={"model": "qwen3-vl:8b"},
        )

    assert "VLM request sent" in caplog.text
    assert "qwen3-vl:8b" in caplog.text
    assert "huggingface.co" not in caplog.text
