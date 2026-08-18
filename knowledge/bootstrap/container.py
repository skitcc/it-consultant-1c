"""Composition root for knowledge core ports and use cases."""

from __future__ import annotations

from dataclasses import dataclass

import httpx

from knowledge.adapters.outbound import (
    DoclingDocumentParser,
    OllamaChatModel,
    OllamaEmbedder,
    OllamaQwen3Reranker,
    QdrantVectorIndex,
    SQLiteDocumentRegistry,
)
from knowledge.adapters.outbound.docling_parser import PictureDescriptionConfig
from knowledge.adapters.outbound.ollama_reranker import ScorePassthroughReranker
from knowledge.core.use_cases import (
    AnswerQuestion,
    IndexDocument,
    RemoveDocument,
    RetrieveKnowledge,
    UpdateDocumentMetadata,
)
from knowledge.settings import KnowledgeSettings


@dataclass(slots=True)
class KnowledgeContainer:
    settings: KnowledgeSettings
    registry: SQLiteDocumentRegistry
    vector_index: QdrantVectorIndex
    index_document: IndexDocument
    remove_document: RemoveDocument
    update_metadata: UpdateDocumentMetadata
    retrieve_knowledge: RetrieveKnowledge
    answer_question: AnswerQuestion

    def readiness(self) -> None:
        self.registry.list(self.settings.default_knowledge_id)
        timeout = min(self.settings.ollama_timeout_sec, 5.0)
        with httpx.Client(timeout=timeout) as client:
            client.get(
                f"{self.settings.ollama_base_url.rstrip('/')}/api/tags"
            ).raise_for_status()
            client.get(
                f"{self.settings.qdrant_url.rstrip('/')}/collections"
            ).raise_for_status()


def build_container(settings: KnowledgeSettings | None = None) -> KnowledgeContainer:
    config = settings or KnowledgeSettings()
    registry = SQLiteDocumentRegistry(config.document_registry_path)
    embedder = OllamaEmbedder(
        base_url=config.ollama_base_url,
        model=config.embedding_model,
        timeout_sec=config.embedding_timeout_sec,
    )
    vector_index = QdrantVectorIndex(
        url=config.qdrant_url,
        collection=config.qdrant_collection,
    )
    parser = DoclingDocumentParser(
        max_tokens=config.chunk_size,
        picture=PictureDescriptionConfig(
            enabled=config.picture_description_enabled,
            ollama_base_url=config.ollama_base_url,
            model=config.vlm_model,
            timeout_sec=config.vlm_timeout_sec,
            area_threshold=config.picture_area_threshold,
        ),
    )
    reranker = (
        OllamaQwen3Reranker(
            base_url=config.rerank_base_url or config.ollama_base_url,
            model=config.rerank_model,
            timeout_sec=config.rerank_timeout_sec,
        )
        if config.rerank_enabled
        else ScorePassthroughReranker()
    )
    chat_model = OllamaChatModel(
        base_url=config.ollama_base_url,
        model=config.ollama_model,
        timeout_sec=config.ollama_timeout_sec,
    )
    retrieve = RetrieveKnowledge(
        embedder=embedder,
        vector_index=vector_index,
        reranker=reranker,
        candidate_limit=config.rag_candidates,
        score_threshold=config.rag_score_threshold,
    )
    index_document = IndexDocument(
        parser=parser,
        registry=registry,
        embedder=embedder,
        vector_index=vector_index,
    )
    return KnowledgeContainer(
        settings=config,
        registry=registry,
        vector_index=vector_index,
        index_document=index_document,
        remove_document=RemoveDocument(
            registry=registry,
            vector_index=vector_index,
        ),
        update_metadata=UpdateDocumentMetadata(
            registry=registry,
            vector_index=vector_index,
        ),
        retrieve_knowledge=retrieve,
        answer_question=AnswerQuestion(
            retriever=retrieve,
            chat_model=chat_model,
            top_k=config.rag_top_k,
            neighbor_window=config.rag_neighbor_window,
        ),
    )
