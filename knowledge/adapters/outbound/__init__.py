"""Outbound adapters implementing knowledge core ports."""

from knowledge.adapters.outbound.docling_parser import DoclingDocumentParser
from knowledge.adapters.outbound.ollama_chat import OllamaChatModel
from knowledge.adapters.outbound.ollama_embedder import OllamaEmbedder
from knowledge.adapters.outbound.ollama_reranker import OllamaQwen3Reranker
from knowledge.adapters.outbound.qdrant_vector_index import QdrantVectorIndex
from knowledge.adapters.outbound.sqlite_registry import SQLiteDocumentRegistry

__all__ = [
    "DoclingDocumentParser",
    "OllamaChatModel",
    "OllamaEmbedder",
    "OllamaQwen3Reranker",
    "QdrantVectorIndex",
    "SQLiteDocumentRegistry",
]
