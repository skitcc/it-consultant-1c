"""Byte-preserving Docling parser with HybridChunker and optional VLM enrichment."""

from __future__ import annotations

import hashlib
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePath
from typing import Any

from knowledge.core.domain import DocumentChunk

logger = logging.getLogger(__name__)

_DEFAULT_MAX_TOKENS = 512
_VLM_PROMPT = (
    "Опиши изображение из документации 1С для поиска. "
    "Кратко и точно: что изображено (скрин формы, схема, таблица), "
    "какой видимый текст на кнопках, полях и заголовках. "
    "Не выдумывай то, чего нет на картинке."
)


@dataclass(frozen=True, slots=True)
class PictureDescriptionConfig:
    enabled: bool = False
    ollama_base_url: str = "http://127.0.0.1:11434"
    model: str = "qwen2.5vl:3b"
    timeout_sec: float = 90.0
    area_threshold: float = 0.02


class DoclingDocumentParser:
    """Convert the exact supplied bytes from a suffix-preserving temporary file."""

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

    def parse(self, raw_bytes: bytes, filename: str) -> list[DocumentChunk]:
        if not isinstance(raw_bytes, bytes):
            raise TypeError("raw_bytes must be bytes")
        suffix = PurePath(filename).suffix
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
                temporary_path = Path(handle.name)
                written = handle.write(raw_bytes)
                if written != len(raw_bytes):
                    raise OSError("Could not write all source bytes")
                handle.flush()
                os.fsync(handle.fileno())

            temp_hash = hashlib.sha256(temporary_path.read_bytes()).digest()
            input_hash = hashlib.sha256(raw_bytes).digest()
            if temp_hash != input_hash:
                raise OSError("Temporary document differs from supplied bytes")

            result = self._get_converter().convert(temporary_path)
            document = result.document
            chunker = self._get_chunker()
            chunks: list[DocumentChunk] = []
            for raw_chunk in chunker.chunk(dl_doc=document):
                text = str(chunker.contextualize(raw_chunk)).strip()
                if text:
                    chunks.append(
                        DocumentChunk(
                            text=text,
                            headings=headings_from_chunk(raw_chunk),
                        )
                    )
            return chunks
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    def _get_converter(self) -> Any:
        if self._converter is None:
            try:
                self._converter = build_converter(self._picture)
            except ImportError as exc:
                raise RuntimeError("Docling is required to parse documents") from exc
        return self._converter

    def _get_chunker(self) -> Any:
        if self._chunker is None:
            try:
                self._chunker = build_chunker(self._max_tokens)
            except ImportError as exc:
                raise RuntimeError("Docling is required to chunk documents") from exc
        return self._chunker


def chat_completions_url(base_url: str) -> str:
    return f"{base_url.rstrip('/')}/v1/chat/completions"


def picture_pipeline_flags(config: PictureDescriptionConfig) -> dict[str, Any]:
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
    description = description.strip()
    caption = caption.strip()
    if not description:
        return ""
    lines = ["[Изображение]", f"Описание: {description}"]
    if caption:
        lines.append(f"Подпись: {caption}")
    return "\n".join(lines)


def build_converter(config: PictureDescriptionConfig) -> Any:
    from docling.document_converter import DocumentConverter

    try:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import PdfFormatOption
    except ImportError:
        return DocumentConverter()

    pdf_options = apply_pipeline_flags(PdfPipelineOptions(), config)
    format_options: dict[Any, Any] = {
        InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_options),
    }
    format_options.update(office_format_options(config, InputFormat))
    return DocumentConverter(format_options=format_options)


def apply_pipeline_flags(options: Any, config: PictureDescriptionConfig) -> Any:
    for key, value in picture_pipeline_flags(config).items():
        if hasattr(options, key):
            setattr(options, key, value)
    if config.enabled:
        description_options = build_picture_description_options(config)
        if description_options is not None and hasattr(
            options, "picture_description_options"
        ):
            options.picture_description_options = description_options
    return options


def build_picture_description_options(
    config: PictureDescriptionConfig,
) -> Any | None:
    try:
        from docling.datamodel.pipeline_options import PictureDescriptionApiOptions
    except ImportError:
        logger.warning("Docling picture-description API options are unavailable")
        return None
    kwargs: dict[str, Any] = {
        "url": chat_completions_url(config.ollama_base_url),
        "params": {"model": config.model, "max_completion_tokens": 400},
        "prompt": _VLM_PROMPT,
        "timeout": config.timeout_sec,
    }
    try:
        return PictureDescriptionApiOptions(
            **kwargs,
            picture_area_threshold=config.area_threshold,
        )
    except TypeError:
        return PictureDescriptionApiOptions(**kwargs)


def office_format_options(config: PictureDescriptionConfig, input_format: Any) -> dict[Any, Any]:
    if not config.enabled:
        return {}
    try:
        from docling.datamodel.pipeline_options import ConvertPipelineOptions
    except ImportError:
        return {}
    options = apply_pipeline_flags(ConvertPipelineOptions(), config)
    mapping = {
        "DOCX": ("docling.document_converter", "WordFormatOption"),
        "PPTX": ("docling.document_converter", "PowerpointFormatOption"),
        "HTML": ("docling.document_converter", "HTMLFormatOption"),
        "XLSX": ("docling.document_converter", "ExcelFormatOption"),
    }
    result: dict[Any, Any] = {}
    for format_name, (module_name, class_name) in mapping.items():
        input_value = getattr(input_format, format_name, None)
        option_class = optional_import_attr(module_name, class_name)
        if option_class is None and format_name == "XLSX":
            option_class = optional_import_attr(module_name, "MSExcelFormatOption")
        if input_value is None or option_class is None:
            continue
        try:
            result[input_value] = option_class(pipeline_options=options)
        except TypeError:
            logger.warning("Cannot enable picture descriptions for %s", format_name)
    return result


def optional_import_attr(module_name: str, attribute: str) -> Any | None:
    try:
        module = __import__(module_name, fromlist=[attribute])
    except ImportError:
        return None
    return getattr(module, attribute, None)


def build_chunker(max_tokens: int) -> Any:
    from docling.chunking import HybridChunker

    kwargs: dict[str, Any] = {
        "merge_peers": True,
        "repeat_table_header": True,
    }
    tokenizer = tokenizer_with_max_tokens(max_tokens)
    if tokenizer is None:
        kwargs["max_tokens"] = max_tokens
    else:
        kwargs["tokenizer"] = tokenizer
    serializer_provider = chunking_serializer_provider()
    if serializer_provider is not None:
        kwargs["serializer_provider"] = serializer_provider
    return HybridChunker(**kwargs)


def tokenizer_with_max_tokens(max_tokens: int) -> Any | None:
    try:
        from docling_core.transforms.chunker.tokenizer.huggingface import (
            HuggingFaceTokenizer,
            get_default_tokenizer,
        )
    except ImportError:
        return None
    default = get_default_tokenizer()
    return HuggingFaceTokenizer(tokenizer=default.tokenizer, max_tokens=max_tokens)


def chunking_serializer_provider() -> Any | None:
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

    picture_serializer = as_base_picture_serializer()

    class MarkdownChunkSerializerProvider(ChunkingSerializerProvider):
        def get_serializer(self, doc: Any) -> Any:
            kwargs = {
                "doc": doc,
                "table_serializer": MarkdownTableSerializer(),
                "params": MarkdownParams(compact_tables=True),
            }
            try:
                return ChunkingDocSerializer(
                    **kwargs,
                    picture_serializer=picture_serializer,
                )
            except (TypeError, ValueError):
                return ChunkingDocSerializer(**kwargs)

    return MarkdownChunkSerializerProvider()


def as_base_picture_serializer() -> Any:
    try:
        from docling_core.transforms.serializer.base import BasePictureSerializer
    except ImportError:
        return PictureDescriptionSerializer()

    class TypedPictureDescriptionSerializer(BasePictureSerializer):
        def serialize(
            self,
            *,
            item: Any,
            doc_serializer: Any = None,
            doc: Any = None,
            **kwargs: Any,
        ) -> Any:
            return PictureDescriptionSerializer().serialize(
                item=item,
                doc_serializer=doc_serializer,
                doc=doc,
                **kwargs,
            )

    return TypedPictureDescriptionSerializer()


class PictureDescriptionSerializer:
    def serialize(
        self,
        *,
        item: Any,
        doc_serializer: Any = None,
        doc: Any = None,
        **kwargs: Any,
    ) -> Any:
        del doc_serializer, kwargs
        text = format_picture_block(
            description=picture_description(item),
            caption=picture_caption(item, doc),
        )
        return serialization_result(text=text, item=item)


def serialization_result(*, text: str, item: Any) -> Any:
    try:
        from docling_core.transforms.serializer.common import create_ser_result
    except ImportError:
        return text
    for keyword in ("span_source", "orig"):
        try:
            return create_ser_result(text=text, **{keyword: item})
        except TypeError:
            continue
    return create_ser_result(text=text)


def picture_description(item: Any) -> str:
    for annotation in getattr(item, "annotations", None) or ():
        for name in ("text", "description"):
            value = getattr(annotation, name, None)
            if value and not callable(value) and str(value).strip():
                return str(value).strip()
    meta = getattr(item, "meta", None)
    description = getattr(meta, "description", None) if meta is not None else None
    text = getattr(description, "text", None)
    return str(text or description or "").strip()


def picture_caption(item: Any, doc: Any) -> str:
    caption_text = getattr(item, "caption_text", None)
    if callable(caption_text):
        try:
            return str(caption_text(doc=doc) if doc is not None else caption_text()).strip()
        except Exception:
            return ""
    return " ".join(
        str(caption).strip()
        for caption in (getattr(item, "captions", None) or ())
        if str(caption).strip()
    )


def headings_from_chunk(chunk: Any) -> tuple[str, ...]:
    meta = getattr(chunk, "meta", None)
    headings = getattr(meta, "headings", None) if meta is not None else None
    return tuple(str(heading) for heading in (headings or ()) if heading)
