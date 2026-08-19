"""Document reader adapters (plain text / Docling HybridChunker)."""

from __future__ import annotations

import logging
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
        chunks: list[DocumentChunk] = []
        raw_count = 0
        for raw in chunker.chunk(dl_doc=document):
            raw_count += 1
            text = str(chunker.contextualize(raw)).strip()
            if not text:
                continue
            headings = _headings_from_chunk(raw)
            pieces = split_oversized_text(
                text,
                max_tokens=self._max_tokens,
                tokenizer=self._tokenizer,
            )
            if len(pieces) > 1:
                logger.debug(
                    "Split oversized chunk path=%s headings=%s pieces=%s chars=%s",
                    path.name,
                    " / ".join(headings) if headings else "-",
                    len(pieces),
                    len(text),
                )
            for piece in pieces:
                chunks.append(DocumentChunk(text=piece, headings=headings))
        logger.info(
            "Chunked path=%s raw=%s stored=%s elapsed=%.2fs",
            path.name,
            raw_count,
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
                "params": MarkdownParams(compact_tables=True),
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


def split_oversized_text(
    text: str,
    *,
    max_tokens: int,
    tokenizer: Any | None = None,
) -> list[str]:
    """Split HybridChunker output that still exceeds ``max_tokens`` (wide tables)."""
    stripped = text.strip()
    if not stripped:
        return []
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if _token_count(stripped, tokenizer) <= max_tokens:
        return [stripped]

    separator = "\n\n" if "\n\n" in stripped else "\n"
    parts = [part.strip() for part in stripped.split(separator) if part.strip()]
    if len(parts) <= 1:
        return _hard_split_text(stripped, max_tokens=max_tokens)

    packed: list[str] = []
    buffer: list[str] = []

    def flush() -> None:
        if buffer:
            packed.append(separator.join(buffer))
            buffer.clear()

    for part in parts:
        if _token_count(part, tokenizer) > max_tokens:
            flush()
            packed.extend(_hard_split_text(part, max_tokens=max_tokens))
            continue
        trial = separator.join(buffer + [part]) if buffer else part
        if buffer and _token_count(trial, tokenizer) > max_tokens:
            flush()
        buffer.append(part)
    flush()
    return packed or [stripped]


def _token_count(text: str, tokenizer: Any | None) -> int:
    count = getattr(tokenizer, "count_tokens", None) if tokenizer is not None else None
    if callable(count):
        try:
            return int(count(text))
        except Exception:
            logger.debug("tokenizer.count_tokens failed; using char heuristic", exc_info=True)
    return max(1, (len(text) + 3) // 4)


def _hard_split_text(text: str, *, max_tokens: int) -> list[str]:
    window = max(64, max_tokens * 4)
    overlap = min(150, max(0, window // 6))
    return chunk_text(text, chunk_size=window, overlap=overlap)
