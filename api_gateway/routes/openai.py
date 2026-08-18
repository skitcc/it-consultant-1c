"""Minimal OpenAI-compatible model discovery and chat completions."""

from __future__ import annotations

import json
import time
import uuid

from fastapi import APIRouter, Header, HTTPException, status
from fastapi.responses import StreamingResponse
from starlette.concurrency import run_in_threadpool

from api_gateway.auth import require_bearer
from api_gateway.schemas import ChatCompletionRequest
from knowledge.core.domain import ConversationMessage
from knowledge.core.use_cases import AnswerQuestion


def build_openai_router(
    *,
    answer_question: AnswerQuestion,
    api_key: str,
    model: str,
    knowledge_id: str,
) -> APIRouter:
    router = APIRouter(prefix="/v1")
    authenticate = require_bearer(api_key)

    @router.get("/models")
    async def list_models(
        authorization: str | None = Header(default=None),
    ) -> dict:
        await authenticate(authorization)
        return {
            "object": "list",
            "data": [
                {
                    "id": model,
                    "object": "model",
                    "created": 0,
                    "owned_by": "it-consultant",
                }
            ],
        }

    @router.post("/chat/completions")
    async def chat_completions(
        body: ChatCompletionRequest,
        authorization: str | None = Header(default=None),
    ):
        await authenticate(authorization)
        if body.model != model:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Unknown model: {body.model}",
            )
        question_index = _last_user_index(body)
        question = body.messages[question_index].content.strip()
        history = [
            ConversationMessage(role=item.role, content=item.content)
            for item in body.messages[:question_index]
            if item.role in {"user", "assistant"} and item.content.strip()
        ]
        content = await run_in_threadpool(
            answer_question.execute,
            question,
            history=history,
            knowledge_id=knowledge_id,
        )
        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        if body.stream:
            return StreamingResponse(
                _stream_completion(
                    completion_id=completion_id,
                    created=created,
                    model=model,
                    content=content,
                ),
                media_type="text/event-stream",
            )
        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
        }

    return router


def _last_user_index(body: ChatCompletionRequest) -> int:
    for index in range(len(body.messages) - 1, -1, -1):
        item = body.messages[index]
        if item.role == "user" and item.content.strip():
            return index
    raise HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        detail="At least one non-empty user message is required",
    )


async def _stream_completion(
    *,
    completion_id: str,
    created: int,
    model: str,
    content: str,
):
    chunks = [
        {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": content},
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }
            ],
        },
    ]
    for chunk in chunks:
        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"
