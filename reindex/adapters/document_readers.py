"""Document reader adapters (plain text / Docling HybridChunker)."""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from common.chunking import chunk_text
from reindex.domain.formats import SUPPORTED_SUFFIXES
from reindex.domain.models import DocumentChunk
from reindex.ports import DocumentReader

logger = logging.getLogger(__name__)

_DEFAULT_MAX_TOKENS = 1024
# nomic-embed-text context is ~8192; keep tokenizer encoding uncapped vs 512.
_TOKENIZER_MODEL_MAX_LENGTH = 8192
_VLM_URL_MARKERS = ("/v1/chat/completions", "/api/chat")
_VLM_PROMPT = (
    "Опиши изображение из документации 1С для поиска. "
    "Кратко и точно: что изображено (скрин формы, схема, таблица), "
    "какой видимый текст на кнопках, полях и заголовках. "
    "Не выдумывай то, чего нет на картинке."
)


@dataclass(frozen=True, slots=True)
class PictureDescriptionConfig:
    """Adapter config for Docling picture-description enrichment via Ollama."""

    enabled: bool = False
    ollama_base_url: str = "http://127.0.0.1:11434"
    model: str = "qwen2.5vl:3b"
    timeout_sec: float = 90.0
    concurrency: int = 2
    area_threshold: float = 0.02


class TextDocumentReader:
    """Read UTF-8 text files as a single chunk (no Docling)."""

    def read(self, path: Path) -> list[DocumentChunk]:
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        if not text:
            return []
        return [DocumentChunk(text=text)]


class DoclingDocumentReader:
    """Convert a file with Docling and split it via HybridChunker."""

    def __init__(
        self,
        *,
        converter: Any | None = None,
        chunker: Any | None = None,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
        picture: PictureDescriptionConfig | None = None,
    ) -> None:
        self._converter = converter
        self._chunker = chunker
        self._max_tokens = max_tokens
        self._picture = picture or PictureDescriptionConfig()
        self._tokenizer: Any | None = None

    def _get_converter(self) -> Any:
        if self._converter is not None:
            return self._converter
        try:
            self._converter = _build_converter(self._picture)
        except ImportError as exc:
            raise RuntimeError(
                "Reading documents requires docling; install with: pip install -e '.[reindex]'"
            ) from exc
        return self._converter

    def _get_chunker(self) -> Any:
        if self._chunker is not None:
            return self._chunker
        try:
            tokenizer = _tokenizer_with_max_tokens(self._max_tokens)
            self._tokenizer = tokenizer
            self._chunker = _build_chunker(self._max_tokens, tokenizer=tokenizer)
        except ImportError as exc:
            raise RuntimeError(
                "Chunking documents requires docling; install with: pip install -e '.[reindex]'"
            ) from exc
        return self._chunker

    def read(self, path: Path) -> list[DocumentChunk]:
        converter = self._get_converter()
        chunker = self._get_chunker()
        picture = self._picture
        logger.debug(
            "Convert start path=%s vlm=%s url=%s model=%s concurrency=%s "
            "timeout=%ss area_threshold=%.3f max_tokens=%s",
            path.name,
            picture.enabled,
            chat_completions_url(picture.ollama_base_url) if picture.enabled else "-",
            picture.model if picture.enabled else "-",
            picture.concurrency if picture.enabled else 0,
            picture.timeout_sec if picture.enabled else 0,
            picture.area_threshold,
            self._max_tokens,
        )
        logger.info(
            "Convert start path=%s vlm=%s model=%s",
            path.name,
            picture.enabled,
            picture.model if picture.enabled else "-",
        )
        started = time.perf_counter()
        with _log_vlm_http():
            result = converter.convert(str(path))
        convert_sec = time.perf_counter() - started
        document = result.document
        described, skipped = _log_picture_outcomes(document, vlm_enabled=picture.enabled)
        logger.info(
            "Convert done path=%s elapsed=%.1fs pictures_described=%s pictures_skipped=%s",
            path.name,
            convert_sec,
            described,
            skipped,
        )

        started = time.perf_counter()
        chunks = _chunk_document(
            document,
            chunker=chunker,
            max_tokens=self._max_tokens,
            tokenizer=self._tokenizer,
            path_name=path.name,
        )
        logger.info(
            "Chunked path=%s stored=%s elapsed=%.2fs",
            path.name,
            len(chunks),
            time.perf_counter() - started,
        )
        return chunks


class CompositeDocumentReader:
    """Dispatch to a reader by file suffix."""

    def __init__(
        self,
        readers: dict[str, DocumentReader] | None = None,
        *,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
        converter: Any | None = None,
        chunker: Any | None = None,
        picture: PictureDescriptionConfig | None = None,
    ) -> None:
        if readers is not None:
            self._readers = {suffix.lower(): reader for suffix, reader in readers.items()}
            return

        docling = DoclingDocumentReader(
            converter=converter,
            chunker=chunker,
            max_tokens=max_tokens,
            picture=picture,
        )
        self._readers = {suffix: docling for suffix in SUPPORTED_SUFFIXES}

    def read(self, path: Path) -> Sequence[DocumentChunk]:
        suffix = path.suffix.lower()
        reader = self._readers.get(suffix)
        if reader is None:
            raise ValueError(f"Unsupported document type: {path}")
        return reader.read(path)


def build_default_document_reader(
    *,
    max_tokens: int = _DEFAULT_MAX_TOKENS,
    picture: PictureDescriptionConfig | None = None,
) -> DocumentReader:
    return CompositeDocumentReader(max_tokens=max_tokens, picture=picture)


def chat_completions_url(base_url: str) -> str:
    return f"{base_url.rstrip('/')}/v1/chat/completions"


def picture_pipeline_flags(config: PictureDescriptionConfig) -> dict[str, Any]:
    """Keyword flags applied to Docling pipeline options (no Docling types)."""
    flags: dict[str, Any] = {"do_ocr": False}
    if config.enabled:
        flags.update(
            do_picture_description=True,
            enable_remote_services=True,
            generate_picture_images=True,
            images_scale=2,
        )
    return flags


def format_picture_block(*, description: str, caption: str = "") -> str:
    """Render a picture as searchable text, or empty if VLM returned nothing."""
    desc = description.strip()
    cap = caption.strip()
    if not desc:
        return ""
    lines = ["[Изображение]", f"Описание: {desc}"]
    if cap:
        lines.append(f"Подпись: {cap}")
    return "\n".join(lines)


def _build_converter(config: PictureDescriptionConfig) -> Any:
    from docling.document_converter import DocumentConverter

    try:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import PdfFormatOption
    except ImportError:
        return DocumentConverter()

    pdf_options = _apply_pipeline_flags(PdfPipelineOptions(), config)
    format_options: dict[Any, Any] = {
        InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_options),
    }
    format_options.update(_office_format_options(config, InputFormat))
    return DocumentConverter(format_options=format_options)


def _apply_pipeline_flags(options: Any, config: PictureDescriptionConfig) -> Any:
    for key, value in picture_pipeline_flags(config).items():
        if hasattr(options, key):
            setattr(options, key, value)
    if config.enabled:
        desc_opts = _build_picture_description_options(config)
        if desc_opts is not None and hasattr(options, "picture_description_options"):
            options.picture_description_options = desc_opts
    return options


def _build_picture_description_options(config: PictureDescriptionConfig) -> Any | None:
    try:
        from docling.datamodel.pipeline_options import PictureDescriptionApiOptions
    except ImportError:
        logger.warning("PictureDescriptionApiOptions is unavailable; skipping VLM enrichment")
        return None
    kwargs: dict[str, Any] = {
        "url": chat_completions_url(config.ollama_base_url),
        "params": {"model": config.model, "max_completion_tokens": 400},
        "prompt": _VLM_PROMPT,
        "timeout": config.timeout_sec,
        "concurrency": max(1, int(config.concurrency)),
    }
    try:
        return PictureDescriptionApiOptions(
            **kwargs,
            picture_area_threshold=config.area_threshold,
        )
    except TypeError:
        return PictureDescriptionApiOptions(**kwargs)


def _office_format_options(config: PictureDescriptionConfig, input_format: Any) -> dict[Any, Any]:
    if not config.enabled:
        return {}
    try:
        from docling.datamodel.pipeline_options import ConvertPipelineOptions
    except ImportError:
        logger.warning("ConvertPipelineOptions is unavailable; office VLM enrichment skipped")
        return {}

    convert_options = _apply_pipeline_flags(ConvertPipelineOptions(), config)
    mapping: dict[str, tuple[str, str]] = {
        "DOCX": ("docling.document_converter", "WordFormatOption"),
        "PPTX": ("docling.document_converter", "PowerpointFormatOption"),
        "HTML": ("docling.document_converter", "HTMLFormatOption"),
        "XLSX": ("docling.document_converter", "ExcelFormatOption"),
    }
    result: dict[Any, Any] = {}
    for format_name, (module_name, class_name) in mapping.items():
        fmt = getattr(input_format, format_name, None)
        if fmt is None:
            continue
        option_cls = _optional_import_attr(module_name, class_name)
        if option_cls is None and format_name == "XLSX":
            option_cls = _optional_import_attr(module_name, "MSExcelFormatOption")
        if option_cls is None:
            continue
        try:
            result[fmt] = option_cls(pipeline_options=convert_options)
        except TypeError:
            logger.warning("Cannot attach picture description to %s", format_name)
    return result


def _optional_import_attr(module_name: str, attr: str) -> Any | None:
    try:
        module = __import__(module_name, fromlist=[attr])
    except ImportError:
        return None
    return getattr(module, attr, None)


def _build_chunker(max_tokens: int, *, tokenizer: Any | None = None) -> Any:
    from docling.chunking import HybridChunker

    kwargs: dict[str, Any] = {
        "merge_peers": True,
        "repeat_table_header": True,
    }
    tok = tokenizer if tokenizer is not None else _tokenizer_with_max_tokens(max_tokens)
    if tok is not None:
        kwargs["tokenizer"] = tok
    else:
        kwargs["max_tokens"] = max_tokens
    serializer_provider = _chunking_serializer_provider()
    if serializer_provider is not None:
        kwargs["serializer_provider"] = serializer_provider
    return HybridChunker(**kwargs)


def _tokenizer_with_max_tokens(max_tokens: int) -> Any | None:
    try:
        from docling_core.transforms.chunker.tokenizer.huggingface import (
            HuggingFaceTokenizer,
            get_default_tokenizer,
        )
    except ImportError:
        return None
    default = get_default_tokenizer()
    hf_tokenizer = default.tokenizer
    # Default pretrained cap is 512; HybridChunker then warns on tables (1742 > 512)
    # and can truncate token counts used for splitting.
    model_max = max(_TOKENIZER_MODEL_MAX_LENGTH, max_tokens * 8)
    try:
        hf_tokenizer.model_max_length = model_max
    except (AttributeError, TypeError):
        pass
    logger.debug(
        "HybridChunker tokenizer max_tokens=%s model_max_length=%s",
        max_tokens,
        getattr(hf_tokenizer, "model_max_length", "?"),
    )
    return HuggingFaceTokenizer(tokenizer=hf_tokenizer, max_tokens=max_tokens)


def _chunking_serializer_provider() -> Any | None:
    try:
        from docling_core.transforms.chunker.hierarchical_chunker import (
            ChunkingDocSerializer,
            ChunkingSerializerProvider,
        )
        from docling_core.transforms.serializer.markdown import (
            MarkdownParams,
            MarkdownTableSerializer,
        )
    except ImportError:
        return None

    picture_serializer = _as_base_picture_serializer()

    class MarkdownChunkSerializerProvider(ChunkingSerializerProvider):
        def get_serializer(self, doc: Any) -> Any:
            kwargs: dict[str, Any] = {
                "doc": doc,
                "table_serializer": MarkdownTableSerializer(),
                "params": MarkdownParams(compact_tables=False),
            }
            try:
                return ChunkingDocSerializer(
                    **kwargs,
                    picture_serializer=picture_serializer,
                )
            except TypeError:
                return ChunkingDocSerializer(**kwargs)
            except Exception as exc:
                # Pydantic v2 raises ValidationError (not TypeError) when the
                # serializer is not a BasePictureSerializer instance.
                if type(exc).__name__ != "ValidationError":
                    raise
                logger.warning(
                    "ChunkingDocSerializer rejected picture_serializer (%s); "
                    "falling back to default",
                    exc,
                )
                return ChunkingDocSerializer(**kwargs)

    return MarkdownChunkSerializerProvider()


def _as_base_picture_serializer() -> Any:
    """Wrap picture serializer so Pydantic accepts it as BasePictureSerializer."""
    try:
        from docling_core.transforms.serializer.base import BasePictureSerializer
    except ImportError:
        return _PictureDescriptionSerializer()

    class PictureDescriptionSerializer(BasePictureSerializer):
        def serialize(
            self,
            *,
            item: Any,
            doc_serializer: Any = None,
            doc: Any = None,
            **kwargs: Any,
        ) -> Any:
            return _PictureDescriptionSerializer().serialize(
                item=item,
                doc_serializer=doc_serializer,
                doc=doc,
                **kwargs,
            )

    return PictureDescriptionSerializer()


class _PictureDescriptionSerializer:
    """Serialize PictureItem as captioned VLM text instead of an image tag."""

    def serialize(self, *, item: Any, doc_serializer: Any = None, doc: Any = None, **kwargs: Any) -> Any:
        del doc_serializer, kwargs
        text = format_picture_block(
            description=_picture_description(item),
            caption=_picture_caption(item, doc),
        )
        if not text:
            logger.debug(
                "Skipping picture without VLM description: %s",
                getattr(item, "self_ref", "?"),
            )
        return _serialization_result(text=text, item=item)


def _serialization_result(*, text: str, item: Any) -> Any:
    try:
        from docling_core.transforms.serializer.common import create_ser_result
    except ImportError:
        return text
    try:
        return create_ser_result(text=text, span_source=item)
    except TypeError:
        try:
            return create_ser_result(text=text, orig=item)
        except TypeError:
            return create_ser_result(text=text)


def _picture_description(item: Any) -> str:
    annotations = getattr(item, "annotations", None) or ()
    for annotation in annotations:
        text = getattr(annotation, "text", None)
        if text and str(text).strip():
            return str(text).strip()
        nested = getattr(annotation, "description", None)
        if nested and str(nested).strip() and not callable(nested):
            return str(nested).strip()
    meta = getattr(item, "meta", None)
    description = getattr(meta, "description", None) if meta is not None else None
    if description is None:
        return ""
    text = getattr(description, "text", None)
    if text and str(text).strip():
        return str(text).strip()
    if isinstance(description, str) and description.strip():
        return description.strip()
    return ""


def _picture_caption(item: Any, doc: Any) -> str:
    caption_text = getattr(item, "caption_text", None)
    if callable(caption_text):
        try:
            value = caption_text(doc=doc) if doc is not None else caption_text()
        except TypeError:
            try:
                value = caption_text()
            except Exception:
                value = ""
        except Exception:
            value = ""
        return str(value or "").strip()
    captions = getattr(item, "captions", None) or ()
    parts = [str(part).strip() for part in captions if str(part).strip()]
    return " ".join(parts)


def _headings_from_chunk(chunk: Any) -> tuple[str, ...]:
    meta = getattr(chunk, "meta", None)
    headings = getattr(meta, "headings", None) if meta is not None else None
    if not headings:
        return ()
    return tuple(str(item) for item in headings if item)


@contextmanager
def _log_vlm_http() -> Iterator[None]:
    """Log Docling VLM POSTs (they use ``requests``, not httpx)."""
    try:
        import requests
    except ImportError:
        yield
        return

    original = requests.Session.post

    def _logged(self: Any, url: Any, *args: Any, **kwargs: Any) -> Any:
        url_text = str(url)
        if not any(marker in url_text for marker in _VLM_URL_MARKERS):
            return original(self, url, *args, **kwargs)
        payload = kwargs.get("json") if isinstance(kwargs.get("json"), dict) else {}
        model = payload.get("model", "?") if isinstance(payload, dict) else "?"
        started = time.perf_counter()
        logger.info("VLM request sent url=%s model=%s", url_text, model)
        try:
            response = original(self, url, *args, **kwargs)
        except Exception:
            logger.exception(
                "VLM request failed url=%s model=%s elapsed=%.1fs",
                url_text,
                model,
                time.perf_counter() - started,
            )
            raise
        description_chars = _vlm_response_chars(response)
        logger.info(
            "VLM response url=%s model=%s status=%s elapsed=%.1fs description_chars=%s",
            url_text,
            model,
            getattr(response, "status_code", "?"),
            time.perf_counter() - started,
            description_chars,
        )
        return response

    requests.Session.post = _logged  # type: ignore[method-assign]
    try:
        yield
    finally:
        requests.Session.post = original  # type: ignore[method-assign]


def _vlm_response_chars(response: Any) -> int:
    try:
        if not getattr(response, "ok", False):
            return 0
        data = response.json()
        if not isinstance(data, dict):
            return 0
        choices = data.get("choices") or []
        if not choices:
            return 0
        message = (choices[0] or {}).get("message") or {}
        content = message.get("content") or ""
        return len(str(content))
    except Exception:
        return 0


def _log_picture_outcomes(document: Any, *, vlm_enabled: bool) -> tuple[int, int]:
    pictures = _iter_pictures(document)
    described = skipped = 0
    for picture in pictures:
        ref = getattr(picture, "self_ref", "?")
        text = _picture_description(picture)
        if text:
            described += 1
            logger.debug(
                "VLM picture described ref=%s chars=%s",
                ref,
                len(text),
            )
            continue
        skipped += 1
        reason = (
            "below area threshold or API returned empty"
            if vlm_enabled
            else "VLM disabled"
        )
        logger.debug("VLM picture skipped ref=%s reason=%s", ref, reason)
    if not pictures:
        logger.debug("VLM pictures found=0")
    return described, skipped


def _iter_pictures(document: Any) -> list[Any]:
    pictures = getattr(document, "pictures", None)
    if pictures is None:
        return []
    try:
        return list(pictures)
    except TypeError:
        return []


def _chunk_document(
    document: Any,
    *,
    chunker: Any,
    max_tokens: int,
    tokenizer: Any | None,
    path_name: str,
) -> list[DocumentChunk]:
    """HybridChunker for prose; each Docling TableItem becomes one unbounded MD chunk."""
    serializer = _document_serializer(chunker, document)
    emitted_tables: set[str] = set()
    chunks: list[DocumentChunk] = []
    raw_count = 0
    for raw in chunker.chunk(dl_doc=document):
        raw_count += 1
        headings = _headings_from_chunk(raw)
        table_items = _table_items_from_chunk(raw)
        other_items = _non_table_items_from_chunk(raw)
        emitted_table = False
        for table in table_items:
            ref = _table_ref(table)
            if ref in emitted_tables:
                continue
            markdown = _render_full_table_markdown(
                table,
                document=document,
                serializer=serializer,
            )
            if not markdown:
                logger.warning(
                    "Docling table rendered empty path=%s ref=%s",
                    path_name,
                    ref,
                )
                continue
            emitted_tables.add(ref)
            emitted_table = True
            logger.info(
                "Table chunk path=%s ref=%s chars=%s unbounded=true",
                path_name,
                ref,
                len(markdown),
            )
            chunks.append(
                DocumentChunk(text=markdown, headings=headings, atomic=True)
            )

        if emitted_table and not other_items:
            continue
        if emitted_table and other_items:
            prose = _serialize_items(
                other_items,
                document=document,
                serializer=serializer,
            )
            if not prose:
                contextualized = str(chunker.contextualize(raw)).strip()
                prose = _prose_without_tables(contextualized)
            for piece in split_oversized_text(
                prose,
                max_tokens=max_tokens,
                tokenizer=tokenizer,
            ):
                chunks.append(DocumentChunk(text=piece, headings=headings))
            continue

        text = str(chunker.contextualize(raw)).strip()
        if not text:
            continue
        pieces = split_oversized_text(
            text,
            max_tokens=max_tokens,
            tokenizer=tokenizer,
        )
        if len(pieces) > 1:
            logger.debug(
                "Split oversized chunk path=%s headings=%s pieces=%s chars=%s",
                path_name,
                " / ".join(headings) if headings else "-",
                len(pieces),
                len(text),
            )
        for piece in pieces:
            chunks.append(DocumentChunk(text=piece, headings=headings))
    logger.debug("HybridChunker raw chunks path=%s raw=%s", path_name, raw_count)
    return merge_split_table_chunks(chunks)


def _document_serializer(chunker: Any, document: Any) -> Any | None:
    provider = getattr(chunker, "serializer_provider", None)
    getter = getattr(provider, "get_serializer", None) if provider is not None else None
    if callable(getter):
        try:
            serializer = getter(document)
        except TypeError:
            try:
                serializer = getter(doc=document)
            except Exception:
                serializer = None
        except Exception:
            serializer = None
        if serializer is not None:
            return serializer
    return _make_markdown_serializer(document)


def _make_markdown_serializer(document: Any) -> Any | None:
    provider = _chunking_serializer_provider()
    getter = getattr(provider, "get_serializer", None) if provider is not None else None
    if not callable(getter):
        return None
    try:
        return getter(document)
    except TypeError:
        try:
            return getter(doc=document)
        except Exception:
            return None
    except Exception:
        logger.debug("Cannot build Docling markdown serializer", exc_info=True)
        return None


def _table_items_from_chunk(chunk: Any) -> list[Any]:
    return [item for item in _doc_items(chunk) if _is_table_item(item)]


def _non_table_items_from_chunk(chunk: Any) -> list[Any]:
    return [item for item in _doc_items(chunk) if not _is_table_item(item)]


def _doc_items(chunk: Any) -> list[Any]:
    meta = getattr(chunk, "meta", None)
    items = getattr(meta, "doc_items", None) if meta is not None else None
    if items is None:
        return []
    try:
        len(items)
    except TypeError:
        return []
    try:
        return list(items)
    except TypeError:
        return []


def _is_table_item(item: Any) -> bool:
    if item is None:
        return False
    if type(item).__name__ == "TableItem":
        return True
    label = getattr(item, "label", None)
    if label is not None:
        value = getattr(label, "value", label)
        if str(value).lower() == "table":
            return True
    ref = str(getattr(item, "self_ref", "") or "")
    return "#/tables/" in ref or "/tables/" in ref


def _table_ref(table: Any) -> str:
    ref = getattr(table, "self_ref", None)
    if ref:
        return str(ref)
    return f"id:{id(table)}"


def _render_full_table_markdown(
    table: Any,
    *,
    document: Any,
    serializer: Any | None,
) -> str:
    if serializer is not None:
        text = _text_from_serialize(serializer, table)
        if text:
            return text
    export = getattr(table, "export_to_markdown", None)
    if callable(export):
        for kwargs in ({"doc": document}, {}):
            try:
                rendered = export(**kwargs) if kwargs else export()
            except TypeError:
                continue
            except Exception:
                logger.debug("table.export_to_markdown failed", exc_info=True)
                break
            if isinstance(rendered, str) and rendered.strip():
                return rendered.strip()
    return _markdown_from_table_grid(table)


def _text_from_serialize(serializer: Any, item: Any) -> str:
    serialize = getattr(serializer, "serialize", None)
    if not callable(serialize):
        return ""
    try:
        result = serialize(item=item)
    except TypeError:
        try:
            result = serialize(item)
        except Exception:
            return ""
    except Exception:
        logger.debug("Docling serialize failed", exc_info=True)
        return ""
    text = getattr(result, "text", result)
    if isinstance(text, str) and text.strip():
        return text.strip()
    return ""


def _serialize_items(
    items: list[Any],
    *,
    document: Any,
    serializer: Any | None,
) -> str:
    del document
    if serializer is None:
        return ""
    parts: list[str] = []
    for item in items:
        text = _text_from_serialize(serializer, item)
        if text:
            parts.append(text)
    return "\n\n".join(parts)


def _prose_without_tables(text: str) -> str:
    return "\n\n".join(
        block for kind, block in _iter_content_blocks(text) if kind == "prose"
    )


def _markdown_from_table_grid(table: Any) -> str:
    data = getattr(table, "data", None)
    grid = getattr(data, "grid", None)
    if not grid:
        return ""
    rows: list[list[str]] = []
    for row in grid:
        cells: list[str] = []
        for cell in row:
            value = getattr(cell, "text", cell)
            text = "" if value is None else str(value)
            cells.append(text.replace("\n", " ").replace("|", "\\|"))
        if cells:
            rows.append(cells)
    if not rows:
        return ""
    width = max(len(row) for row in rows)
    normalized = [row + [""] * (width - len(row)) for row in rows]
    header = normalized[0]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for row in normalized[1:]:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


_TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")
_TABLE_SEP_RE = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)+\|?\s*$")
_HTML_TABLE_RE = re.compile(r"<table\b.*?</table>", re.IGNORECASE | re.DOTALL)
_HEADING_MARKDOWN_RE = re.compile(r"^#{1,6}\s+")


def split_oversized_text(
    text: str,
    *,
    max_tokens: int,
    tokenizer: Any | None = None,
) -> list[str]:
    """Split prose that exceeds ``max_tokens``. Tables stay one piece each."""
    stripped = text.strip()
    if not stripped:
        return []
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")

    pieces: list[str] = []
    for kind, block in _iter_content_blocks(stripped):
        if kind == "table":
            tokens = _token_count(block, tokenizer)
            if tokens > max_tokens:
                logger.warning(
                    "Keeping oversized table as one chunk tokens=%s max_tokens=%s chars=%s",
                    tokens,
                    max_tokens,
                    len(block),
                )
            pieces.append(block)
            continue
        if _token_count(block, tokenizer) <= max_tokens:
            pieces.append(block)
            continue
        pieces.extend(
            _split_prose(block, max_tokens=max_tokens, tokenizer=tokenizer)
        )
    return pieces


def merge_split_table_chunks(chunks: list[DocumentChunk]) -> list[DocumentChunk]:
    """Join HybridChunker table fragments that repeat the same header."""
    if len(chunks) < 2:
        return chunks

    merged: list[DocumentChunk] = []
    index = 0
    while index < len(chunks):
        chunk = chunks[index]
        if not _is_pure_table(chunk.text) or chunk.atomic:
            merged.append(chunk)
            index += 1
            continue

        acc = chunk
        cursor = index + 1
        skipped: list[DocumentChunk] = []
        while cursor < len(chunks):
            nxt = chunks[cursor]
            if _is_heading_echo(nxt.text, acc.headings):
                skipped.append(nxt)
                cursor += 1
                continue
            if (
                not nxt.atomic
                and nxt.headings == acc.headings
                and _is_pure_table(nxt.text)
            ):
                joined = _join_table_text(acc.text, nxt.text)
                if joined is not None:
                    acc = DocumentChunk(text=joined, headings=acc.headings)
                    skipped.clear()
                    cursor += 1
                    continue
            break
        merged.append(acc)
        merged.extend(skipped)
        index = cursor
    return merged


def _iter_content_blocks(text: str) -> list[tuple[str, str]]:
    spans = _table_spans(text)
    if not spans:
        return [("prose", text.strip())] if text.strip() else []

    blocks: list[tuple[str, str]] = []
    cursor = 0
    for start, end in spans:
        if start > cursor:
            prose = text[cursor:start].strip()
            if prose:
                blocks.append(("prose", prose))
        table = text[start:end].strip()
        if table:
            blocks.append(("table", table))
        cursor = end
    if cursor < len(text):
        prose = text[cursor:].strip()
        if prose:
            blocks.append(("prose", prose))
    return blocks


def _table_spans(text: str) -> list[tuple[int, int]]:
    html_spans = [(match.start(), match.end()) for match in _HTML_TABLE_RE.finditer(text)]
    spans = list(html_spans)
    for start, end in _markdown_table_spans(text):
        if any(start < html_end and end > html_start for html_start, html_end in html_spans):
            continue
        spans.append((start, end))
    spans.sort()
    return spans


def _markdown_table_spans(text: str) -> list[tuple[int, int]]:
    lines = text.splitlines(keepends=True)
    if not lines:
        return []
    stripped = [line.strip() for line in lines]
    starts: list[int] = []
    offset = 0
    for line in lines:
        starts.append(offset)
        offset += len(line)

    spans: list[tuple[int, int]] = []
    index = 0
    while index < len(lines):
        if not _is_markdown_table_start(stripped, index):
            index += 1
            continue
        end = index + 1
        while end < len(lines):
            current = stripped[end]
            if not current:
                break
            if _is_table_row(current) or _is_table_sep(current):
                end += 1
                continue
            break
        last = end - 1
        spans.append((starts[index], starts[last] + len(lines[last])))
        index = end
    return spans


def _is_markdown_table_start(stripped_lines: list[str], index: int) -> bool:
    if index >= len(stripped_lines) or not _is_table_row(stripped_lines[index]):
        return False
    if index + 1 >= len(stripped_lines):
        return False
    nxt = stripped_lines[index + 1]
    return _is_table_sep(nxt) or _is_table_row(nxt)


def _is_pure_table(text: str) -> bool:
    blocks = _iter_content_blocks(text)
    return len(blocks) == 1 and blocks[0][0] == "table"


def _is_heading_echo(text: str, headings: tuple[str, ...]) -> bool:
    if not headings or _is_pure_table(text):
        return False
    leftover = text
    for heading in headings:
        leftover = leftover.replace(heading, "\n")
    leftover = _HEADING_MARKDOWN_RE.sub("", leftover)
    return not leftover.strip()


def _join_table_text(first: str, second: str) -> str | None:
    parsed_first = _parse_pipe_table(first)
    parsed_second = _parse_pipe_table(second)
    if parsed_first is None or parsed_second is None:
        return None
    header, sep, body = parsed_first
    other_header, other_sep, other_body = parsed_second
    if _normalize_table_row(header) != _normalize_table_row(other_header):
        return None
    marker = sep or other_sep
    lines = [header]
    if marker:
        lines.append(marker)
    lines.extend(body)
    lines.extend(other_body)
    return "\n".join(lines)


def _parse_pipe_table(text: str) -> tuple[str, str | None, list[str]] | None:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 2:
        return None
    for index, line in enumerate(lines[:-1]):
        if _is_table_row(line) and _is_table_sep(lines[index + 1]):
            header = line
            sep = lines[index + 1]
            body = [item for item in lines[index + 2 :] if _is_table_row(item)]
            return header, sep, body
    if all(_is_table_row(line) or _is_table_sep(line) for line in lines):
        header = lines[0]
        rest = [line for line in lines[1:] if _is_table_row(line)]
        if not rest:
            return None
        return header, None, rest
    html_match = _HTML_TABLE_RE.search(text)
    if html_match and html_match.group(0).strip() == text.strip():
        return text.strip(), None, []
    return None


def _normalize_table_row(row: str) -> str:
    cells = [cell.strip() for cell in row.strip().strip("|").split("|")]
    return "|".join(cells)


def _split_prose(
    text: str,
    *,
    max_tokens: int,
    tokenizer: Any | None = None,
) -> list[str]:
    separator = "\n\n" if "\n\n" in text else "\n"
    parts = [part.strip() for part in text.split(separator) if part.strip()]
    if len(parts) <= 1:
        return _hard_split_text(text, max_tokens=max_tokens, tokenizer=tokenizer)

    packed: list[str] = []
    buffer: list[str] = []

    def flush() -> None:
        if buffer:
            packed.append(separator.join(buffer))
            buffer.clear()

    for part in parts:
        if _token_count(part, tokenizer) > max_tokens:
            flush()
            packed.extend(
                _hard_split_text(part, max_tokens=max_tokens, tokenizer=tokenizer)
            )
            continue
        trial = separator.join(buffer + [part]) if buffer else part
        if buffer and _token_count(trial, tokenizer) > max_tokens:
            flush()
        buffer.append(part)
    flush()
    return packed or [text]


def _token_count(text: str, tokenizer: Any | None) -> int:
    count = getattr(tokenizer, "count_tokens", None) if tokenizer is not None else None
    if callable(count):
        try:
            return int(count(text))
        except Exception:
            logger.debug("tokenizer.count_tokens failed; using char heuristic", exc_info=True)
    return max(1, (len(text) + 3) // 4)


def _hard_split_text(
    text: str,
    *,
    max_tokens: int,
    tokenizer: Any | None = None,
) -> list[str]:
    del tokenizer
    window = max(64, max_tokens * 4)
    overlap = min(150, max(0, window // 6))
    return chunk_text(text, chunk_size=window, overlap=overlap)


def _is_table_row(line: str) -> bool:
    stripped = line.strip()
    return bool(_TABLE_ROW_RE.match(stripped)) and stripped.count("|") >= 2


def _is_table_sep(line: str) -> bool:
    return bool(_TABLE_SEP_RE.match(line.strip()))
