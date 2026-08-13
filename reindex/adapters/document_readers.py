"""Document reader adapters (plain text / Docling HybridChunker)."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from reindex.domain.formats import SUPPORTED_SUFFIXES
from reindex.domain.models import DocumentChunk
from reindex.ports import DocumentReader

logger = logging.getLogger(__name__)

_DEFAULT_MAX_TOKENS = 512


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
    ) -> None:
        self._converter = converter
        self._chunker = chunker
        self._max_tokens = max_tokens

    def _get_converter(self) -> Any:
        if self._converter is not None:
            return self._converter
        try:
            self._converter = _build_converter()
        except ImportError as exc:
            raise RuntimeError(
                "Reading documents requires docling; install with: pip install -e '.[reindex]'"
            ) from exc
        return self._converter

    def _get_chunker(self) -> Any:
        if self._chunker is not None:
            return self._chunker
        try:
            self._chunker = _build_chunker(self._max_tokens)
        except ImportError as exc:
            raise RuntimeError(
                "Chunking documents requires docling; install with: pip install -e '.[reindex]'"
            ) from exc
        return self._chunker

    def read(self, path: Path) -> list[DocumentChunk]:
        converter = self._get_converter()
        chunker = self._get_chunker()
        result = converter.convert(str(path))
        document = result.document
        chunks: list[DocumentChunk] = []
        for raw in chunker.chunk(dl_doc=document):
            text = str(chunker.contextualize(raw)).strip()
            if not text:
                continue
            chunks.append(
                DocumentChunk(text=text, headings=_headings_from_chunk(raw)),
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
    ) -> None:
        if readers is not None:
            self._readers = {suffix.lower(): reader for suffix, reader in readers.items()}
            return

        docling = DoclingDocumentReader(
            converter=converter,
            chunker=chunker,
            max_tokens=max_tokens,
        )
        self._readers = {suffix: docling for suffix in SUPPORTED_SUFFIXES}

    def read(self, path: Path) -> Sequence[DocumentChunk]:
        suffix = path.suffix.lower()
        reader = self._readers.get(suffix)
        if reader is None:
            raise ValueError(f"Unsupported document type: {path}")
        return reader.read(path)


def build_default_document_reader(*, max_tokens: int = _DEFAULT_MAX_TOKENS) -> DocumentReader:
    return CompositeDocumentReader(max_tokens=max_tokens)


def _build_converter() -> Any:
    from docling.document_converter import DocumentConverter

    try:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import PdfFormatOption
    except ImportError:
        return DocumentConverter()

    pdf_options = PdfPipelineOptions(do_ocr=False)
    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_options),
        }
    )


def _build_chunker(max_tokens: int) -> Any:
    from docling.chunking import HybridChunker

    kwargs: dict[str, Any] = {
        "merge_peers": True,
        "repeat_table_header": True,
    }
    tokenizer = _tokenizer_with_max_tokens(max_tokens)
    if tokenizer is not None:
        kwargs["tokenizer"] = tokenizer
    else:
        kwargs["max_tokens"] = max_tokens
    serializer_provider = _markdown_table_serializer_provider()
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
    return HuggingFaceTokenizer(tokenizer=default.tokenizer, max_tokens=max_tokens)


def _markdown_table_serializer_provider() -> Any | None:
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

    class MDTableSerializerProvider(ChunkingSerializerProvider):
        def get_serializer(self, doc: Any) -> Any:
            return ChunkingDocSerializer(
                doc=doc,
                table_serializer=MarkdownTableSerializer(),
                params=MarkdownParams(compact_tables=True),
            )

    return MDTableSerializerProvider()


def _headings_from_chunk(chunk: Any) -> tuple[str, ...]:
    meta = getattr(chunk, "meta", None)
    headings = getattr(meta, "headings", None) if meta is not None else None
    if not headings:
        return ()
    return tuple(str(item) for item in headings if item)
