"""Composition root and reconnect loop."""

from __future__ import annotations

import logging
import time

from common import Settings
from common.embeddings import OllamaEmbedder
from common.logging_config import configure_logging
from mail_gateway.adapters.assistant.ollama_assistant import OllamaAssistant
from mail_gateway.adapters.ews.account import build_account
from mail_gateway.adapters.ews.conversation_history import EwsConversationHistoryLoader
from mail_gateway.adapters.ews.listener import EwsMailListener
from mail_gateway.adapters.ews.sender import EwsMailSender
from mail_gateway.adapters.rag.ollama_reranker import (
    OllamaReranker,
    ScorePassthroughReranker,
)
from mail_gateway.adapters.rag.qdrant_retriever import QdrantRetriever
from mail_gateway.adapters.rag.reranking_retriever import RerankingRetriever
from mail_gateway.application.handle_incoming_mail import HandleIncomingMail
from mail_gateway.ports import DocumentRetriever, Reranker

logger = logging.getLogger(__name__)


def _account_from_settings(settings: Settings):
    return build_account(
        server=settings.ews_server,
        email=settings.ews_email,
        password=settings.ews_password,
        auth=settings.ews_auth,
        verify_ssl=settings.ews_verify_ssl,
        username=settings.ews_username,
        session_pool_size=settings.ews_session_pool_size,
    )


def build_reranker(settings: Settings) -> Reranker:
    if not settings.rerank_enabled:
        logger.info("RERANK_ENABLED=false — using vector-score order")
        return ScorePassthroughReranker()
    base_url = settings.rerank_base_url or settings.ollama_base_url
    logger.info(
        "Reranker enabled base_url=%s model=%s",
        base_url,
        settings.rerank_model,
    )
    return OllamaReranker(
        base_url=base_url,
        model=settings.rerank_model,
        timeout_sec=settings.rerank_timeout_sec,
        num_predict=settings.rerank_num_predict,
    )


def build_document_retriever(settings: Settings) -> DocumentRetriever:
    embedder = OllamaEmbedder(
        base_url=settings.ollama_base_url,
        model=settings.embedding_model,
        timeout_sec=settings.embedding_timeout_sec,
    )
    base = QdrantRetriever(
        qdrant_url=settings.qdrant_url,
        collection=settings.qdrant_collection,
        embedder=embedder,
        limit=settings.rag_candidates,
        score_threshold=settings.rag_score_threshold,
    )
    return RerankingRetriever(
        base=base,
        reranker=build_reranker(settings),
        top_k=settings.rag_top_k,
        neighbor_window=settings.rag_neighbor_window,
    )


def run(settings: Settings | None = None) -> None:
    settings = settings or Settings()
    configure_logging(settings.log_level)

    login = settings.ews_username or settings.ews_email
    logger.info(
        "Connecting to EWS server=%s auth=%s login=%s mailbox=%s",
        settings.ews_server,
        settings.ews_auth,
        login,
        settings.ews_email,
    )
    # Two EWS sessions: streaming blocks one HTTP connection; fetch/reply need another.
    stream_account = _account_from_settings(settings)
    work_account = _account_from_settings(settings)
    logger.info(
        "EWS transport initialized shared_protocol=%s session_pool_max=%s",
        stream_account.protocol is work_account.protocol,
        stream_account.protocol._session_pool_maxsize,
    )

    listener = EwsMailListener(
        stream_account,
        work_account,
        connection_timeout_minutes=settings.ews_streaming_timeout_minutes,
        ignore_own_mail=settings.ews_ignore_own_mail,
        catchup_minutes=settings.ews_catchup_minutes,
        own_addresses={settings.ews_email},
    )
    sender = EwsMailSender(work_account)
    history_loader = EwsConversationHistoryLoader(
        work_account,
        bot_email=settings.ews_email,
    )
    retriever = build_document_retriever(settings)
    assistant = OllamaAssistant(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        timeout_sec=settings.ollama_timeout_sec,
        temperature=settings.ollama_temperature,
        top_p=settings.ollama_top_p,
        max_tokens=settings.ollama_max_tokens,
        seed=settings.ollama_seed,
        draft_reasoning_effort=settings.ollama_draft_reasoning_effort,
        verifier_reasoning_effort=settings.ollama_verifier_reasoning_effort,
        system_prompt=settings.ai_system_prompt,
    )
    handle = HandleIncomingMail(
        assistant=assistant,
        mail_sender=sender,
        history_loader=history_loader,
        document_retriever=retriever,
        bot_email=settings.ews_email,
    )

    logger.info(
        "Mail gateway started mailbox=%s ollama=%s model=%s qdrant=%s "
        "collection=%s rag_candidates=%s rag_top_k=%s rerank=%s",
        settings.ews_email,
        settings.ollama_base_url,
        settings.ollama_model,
        settings.qdrant_url,
        settings.qdrant_collection,
        settings.rag_candidates,
        settings.rag_top_k,
        settings.rerank_enabled,
    )

    while True:
        try:
            for message in listener.listen():
                try:
                    started_at = time.perf_counter()
                    logger.info(
                        "Pipeline start conversation_id=%s item_id=%s",
                        message.conversation_id,
                        message.item_id,
                    )
                    handle(message)
                    logger.info(
                        "Pipeline complete conversation_id=%s elapsed=%.3fs",
                        message.conversation_id,
                        time.perf_counter() - started_at,
                    )
                except Exception:
                    logger.exception(
                        "Failed to handle message conversation_id=%s item_id=%s",
                        message.conversation_id,
                        message.item_id,
                    )
            logger.warning(
                "EWS streaming ended; reconnecting in %s sec",
                settings.reconnect_delay_sec,
            )
        except Exception:
            logger.exception(
                "EWS listener error; reconnecting in %s sec",
                settings.reconnect_delay_sec,
            )
        time.sleep(settings.reconnect_delay_sec)
