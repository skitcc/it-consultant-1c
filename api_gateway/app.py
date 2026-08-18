"""FastAPI composition root."""

from __future__ import annotations

from fastapi import FastAPI

from api_gateway.middleware.request_id import RequestIdMiddleware
from api_gateway.routes.health import build_health_router
from api_gateway.routes.openai import build_openai_router
from api_gateway.routes.owui import build_owui_router
from api_gateway.settings import GatewaySettings
from common.logging_config import configure_logging
from knowledge.bootstrap import KnowledgeContainer, build_container


def create_app(
    settings: GatewaySettings | None = None,
    container: KnowledgeContainer | None = None,
) -> FastAPI:
    cfg = settings or GatewaySettings()
    configure_logging(cfg.log_level)
    services = container or build_container(cfg)

    app = FastAPI(
        title="IT Consultant API Gateway",
        version="1.0.0",
        docs_url=None,
        redoc_url=None,
    )
    app.state.settings = cfg
    app.state.knowledge = services
    app.add_middleware(RequestIdMiddleware)
    app.include_router(
        build_owui_router(
            index_document=services.index_document,
            update_metadata=services.update_metadata,
            loader_key=cfg.owui_loader_key,
            knowledge_id=cfg.default_knowledge_id,
            max_upload_bytes=cfg.max_upload_bytes,
            allowed_extensions=cfg.allowed_extensions,
        )
    )
    app.include_router(
        build_openai_router(
            answer_question=services.answer_question,
            api_key=cfg.api_gateway_api_key,
            model=cfg.api_gateway_model,
            knowledge_id=cfg.default_knowledge_id,
        )
    )
    app.include_router(build_health_router(services.readiness))
    return app
