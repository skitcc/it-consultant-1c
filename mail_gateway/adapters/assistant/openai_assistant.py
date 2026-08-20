"""OpenAI-compatible chat backend (vLLM) via ``/v1/chat/completions``."""

from __future__ import annotations

import logging
import time

import httpx

from common.timing import record
from mail_gateway.adapters.assistant.ollama_assistant import (
    _REASONING_EFFORTS,
    _STOP_SEQUENCES,
    _parse_answer_html,
    _reasoning_chars,
)
from mail_gateway.adapters.assistant.payload import (
    build_assistant_payload,
    build_verifier_payload,
    log_assistant_payload,
)
from mail_gateway.domain.models import IncomingMessage
from mail_gateway.ports import Assistant, AssistantUnavailableError

logger = logging.getLogger(__name__)


class OpenAIAssistant(Assistant):
    """Draft chat against an OpenAI-compatible server, optional verifier pass."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        timeout_sec: float = 420.0,
        temperature: float = 0.0,
        top_p: float = 0.1,
        max_tokens: int = 4096,
        context_length: int = 8192,
        seed: int = 0,
        draft_reasoning_effort: str = "medium",
        verifier_enabled: bool = False,
        verifier_reasoning_effort: str = "high",
        system_prompt: str | None = None,
    ) -> None:
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        if context_length < 1:
            raise ValueError("context_length must be positive")
        if draft_reasoning_effort not in _REASONING_EFFORTS:
            raise ValueError("unsupported draft_reasoning_effort")
        if verifier_reasoning_effort not in _REASONING_EFFORTS:
            raise ValueError("unsupported verifier_reasoning_effort")
        self._base_url = _openai_base_url(base_url)
        self._model = model
        self._timeout = timeout_sec
        self._temperature = temperature
        self._top_p = top_p
        self._max_tokens = max_tokens
        self._context_length = context_length
        self._seed = seed
        self._draft_reasoning_effort = draft_reasoning_effort
        self._verifier_enabled = verifier_enabled
        self._verifier_reasoning_effort = verifier_reasoning_effort
        self._system_prompt = system_prompt

    def ask(self, message: IncomingMessage) -> str | None:
        payload = build_assistant_payload(
            message,
            system_prompt=self._system_prompt,
        )
        log_assistant_payload(
            payload,
            destination=f"vllm:{self._base_url} model={self._model} layer=1",
        )
        rag_context = (message.rag_context or "").strip()
        draft = self._chat(
            payload,
            layer=1,
            conversation_id=message.conversation_id,
            rag_context=rag_context,
        )
        if not draft:
            return None

        if not rag_context or not self._verifier_enabled:
            return draft

        verifier = build_verifier_payload(
            conversation_id=message.conversation_id,
            draft=draft,
            rag_context=rag_context,
        )
        log_assistant_payload(
            verifier,
            destination=f"vllm:{self._base_url} model={self._model} layer=2",
        )
        verified = self._chat(
            verifier,
            layer=2,
            conversation_id=message.conversation_id,
            rag_context=rag_context,
        )
        if not verified:
            logger.warning(
                "Verifier returned empty conversation_id=%s; using draft",
                message.conversation_id,
            )
            return draft
        return verified

    def _chat(
        self,
        payload: dict,
        *,
        layer: int,
        conversation_id: str,
        rag_context: str,
    ) -> str | None:
        messages: list[dict[str, str]] = [
            {"role": "system", "content": payload["system_prompt"]},
        ]
        for item in payload["messages"]:
            role = item["role"]
            if role not in {"user", "assistant"}:
                role = "user"
            messages.append({"role": role, "content": item["body"]})

        reasoning_effort = (
            self._draft_reasoning_effort
            if layer == 1
            else self._verifier_reasoning_effort
        )
        request_body = {
            "model": self._model,
            "messages": messages,
            "stream": False,
            "temperature": self._temperature,
            "top_p": self._top_p,
            "seed": self._seed,
            "max_tokens": self._max_tokens,
            "stop": list(_STOP_SEQUENCES),
            **_reasoning_fields(reasoning_effort),
        }
        started_at = time.perf_counter()
        history_chars = sum(
            len(str(item.get("body") or "")) for item in payload["messages"]
        )
        try:
            try:
                logger.info(
                    "Calling vLLM layer=%s model=%s conversation_id=%s messages=%s "
                    "system_chars=%s history_chars=%s rag_chars=%s "
                    "max_tokens=%s reasoning=%s temperature=%s top_p=%s seed=%s",
                    layer,
                    self._model,
                    conversation_id,
                    len(payload["messages"]),
                    len(str(payload["system_prompt"])),
                    history_chars,
                    len(rag_context),
                    self._max_tokens,
                    reasoning_effort,
                    self._temperature,
                    self._top_p,
                    self._seed,
                )
                with httpx.Client(timeout=self._timeout) as client:
                    response = client.post(
                        f"{self._base_url}/chat/completions",
                        json=request_body,
                    )
                    response.raise_for_status()
                    data = response.json()
            except (httpx.HTTPError, ValueError) as exc:
                logger.exception(
                    "vLLM request failed layer=%s conversation_id=%s",
                    layer,
                    conversation_id,
                )
                raise AssistantUnavailableError(
                    f"vLLM layer {layer} request failed"
                ) from exc

            elapsed = time.perf_counter() - started_at
            if not isinstance(data, dict):
                raise AssistantUnavailableError(
                    f"vLLM layer {layer} returned no chat response"
                )
            choice = _first_choice(data)
            finish_reason = str(choice.get("finish_reason") or "")
            if finish_reason == "length":
                logger.warning(
                    "vLLM output truncated layer=%s conversation_id=%s",
                    layer,
                    conversation_id,
                )
                raise AssistantUnavailableError(f"vLLM layer {layer} output truncated")
            if finish_reason and finish_reason not in {"stop", "eos"}:
                logger.warning(
                    "vLLM unexpected finish_reason layer=%s conversation_id=%s "
                    "finish_reason=%s",
                    layer,
                    conversation_id,
                    finish_reason,
                )
                raise AssistantUnavailableError(
                    f"vLLM layer {layer} stopped with {finish_reason}"
                )

            message = choice.get("message")
            if not isinstance(message, dict):
                raise AssistantUnavailableError(
                    f"vLLM layer {layer} returned no message"
                )
            reasoning_chars = _reasoning_chars(message)
            content = _message_text(message)
            usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
            prompt_tokens = usage.get("prompt_tokens")
            completion_tokens = usage.get("completion_tokens")
            total_tokens = usage.get("total_tokens")
            logger.info(
                "vLLM layer=%s done conversation_id=%s elapsed=%.3fs "
                "finish_reason=%s prompt_tokens=%s/%s completion_tokens=%s "
                "total_tokens=%s reasoning_chars=%s content_chars=%s",
                layer,
                conversation_id,
                elapsed,
                finish_reason or None,
                prompt_tokens,
                self._context_length,
                completion_tokens,
                total_tokens,
                reasoning_chars,
                len(content),
            )
            if not content.strip():
                logger.warning(
                    "vLLM layer=%s returned empty message content conversation_id=%s",
                    layer,
                    conversation_id,
                )
                return None
            return _parse_answer_html(
                content,
                layer=layer,
                conversation_id=conversation_id,
            )
        finally:
            record(f"llm_layer_{layer}", time.perf_counter() - started_at)


def _openai_base_url(base_url: str) -> str:
    cleaned = base_url.rstrip("/")
    if cleaned.endswith("/v1"):
        return cleaned
    return f"{cleaned}/v1"


def _reasoning_fields(reasoning_effort: str) -> dict:
    if reasoning_effort == "none":
        return {"chat_template_kwargs": {"enable_thinking": False}}
    effort = "high" if reasoning_effort == "max" else reasoning_effort
    return {
        "reasoning_effort": effort,
        "chat_template_kwargs": {"enable_thinking": True},
    }


def _first_choice(data: dict) -> dict:
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise AssistantUnavailableError("vLLM returned no choices")
    choice = choices[0]
    if not isinstance(choice, dict):
        raise AssistantUnavailableError("vLLM returned an invalid choice")
    return choice


def _message_text(message: dict) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str) and item:
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str) and text:
                    parts.append(text)
        return "".join(parts)
    return ""
