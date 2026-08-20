"""Two-layer Ollama backend via OpenAI-compatible chat completions."""

from __future__ import annotations

import json
import logging
import re
import time

import httpx

from mail_gateway.adapters.assistant.payload import (
    build_assistant_payload,
    build_verifier_payload,
    log_assistant_payload,
)
from mail_gateway.domain.models import IncomingMessage
from mail_gateway.ports import Assistant, AssistantUnavailableError

logger = logging.getLogger(__name__)

_REASONING_EFFORTS = frozenset({"low", "medium", "high", "max", "none"})
_INTERNAL_REASONING_PREFIX = re.compile(
    r"^\s*(?:<[^>]+>\s*)*(?:thinking|analysis|reasoning|content)\b",
    re.IGNORECASE,
)


class OllamaAssistant(Assistant):
    """Two-layer chat: draft, then a second pass over the same chunks."""

    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:11434",
        model: str,
        timeout_sec: float = 420.0,
        temperature: float = 0.0,
        top_p: float = 0.1,
        max_tokens: int = 4096,
        context_length: int = 8192,
        seed: int = 0,
        draft_reasoning_effort: str = "medium",
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
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout_sec
        self._temperature = temperature
        self._top_p = top_p
        self._max_tokens = max_tokens
        self._context_length = context_length
        self._seed = seed
        self._draft_reasoning_effort = draft_reasoning_effort
        self._verifier_reasoning_effort = verifier_reasoning_effort
        self._system_prompt = system_prompt

    def ask(self, message: IncomingMessage) -> str | None:
        payload = build_assistant_payload(
            message,
            system_prompt=self._system_prompt,
        )
        log_assistant_payload(
            payload,
            destination=f"ollama:{self._base_url} model={self._model} layer=1",
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

        if not rag_context:
            return draft

        verifier = build_verifier_payload(
            conversation_id=message.conversation_id,
            draft=draft,
            rag_context=rag_context,
        )
        log_assistant_payload(
            verifier,
            destination=f"ollama:{self._base_url} model={self._model} layer=2",
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
        ollama_messages: list[dict[str, str]] = [
            {"role": "system", "content": payload["system_prompt"]},
        ]
        for item in payload["messages"]:
            role = item["role"]
            if role not in {"user", "assistant"}:
                role = "user"
            ollama_messages.append({"role": role, "content": item["body"]})

        reasoning_effort = (
            self._draft_reasoning_effort
            if layer == 1
            else self._verifier_reasoning_effort
        )
        request_body = {
            "model": self._model,
            "messages": ollama_messages,
            "stream": False,
            "reasoning_effort": reasoning_effort,
            "temperature": self._temperature,
            "top_p": self._top_p,
            "seed": self._seed,
            "max_tokens": self._max_tokens,
        }
        started_at = time.perf_counter()
        history_chars = sum(
            len(str(item.get("body") or "")) for item in payload["messages"]
        )
        try:
            with httpx.Client(timeout=self._timeout) as client:
                prompt_tokens = _count_prompt_tokens(
                    client,
                    base_url=self._base_url,
                    model=self._model,
                    messages=ollama_messages,
                )
                logger.info(
                    "Calling Ollama layer=%s model=%s conversation_id=%s messages=%s "
                    "system_chars=%s history_chars=%s rag_chars=%s "
                    "prompt_tokens=%s/%s (%s) max_tokens=%s remaining_tokens=%s "
                    "reasoning_effort=%s temperature=%s top_p=%s seed=%s",
                    layer,
                    self._model,
                    conversation_id,
                    len(payload["messages"]),
                    len(str(payload["system_prompt"])),
                    history_chars,
                    len(rag_context),
                    prompt_tokens if prompt_tokens is not None else "unknown",
                    self._context_length,
                    _ratio_label(prompt_tokens, self._context_length),
                    self._max_tokens,
                    _remaining_tokens(prompt_tokens, self._context_length),
                    reasoning_effort,
                    self._temperature,
                    self._top_p,
                    self._seed,
                )
                if (
                    prompt_tokens is not None
                    and prompt_tokens >= self._context_length
                ):
                    logger.warning(
                        "Prompt fills the context window layer=%s conversation_id=%s "
                        "prompt_tokens=%s context_length=%s",
                        layer,
                        conversation_id,
                        prompt_tokens,
                        self._context_length,
                    )
                response = client.post(
                    f"{self._base_url}/v1/chat/completions",
                    json=request_body,
                )
                response.raise_for_status()
                data = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            logger.exception(
                "Ollama request failed layer=%s conversation_id=%s",
                layer,
                conversation_id,
            )
            raise AssistantUnavailableError(
                f"Ollama layer {layer} request failed"
            ) from exc

        elapsed = time.perf_counter() - started_at
        choice = _first_choice(data)
        if choice is None:
            raise AssistantUnavailableError(
                f"Ollama layer {layer} returned no completion choice"
            )
        finish_reason = str(choice.get("finish_reason") or "")
        if finish_reason == "length":
            logger.warning(
                "Ollama output truncated layer=%s conversation_id=%s",
                layer,
                conversation_id,
            )
            raise AssistantUnavailableError(f"Ollama layer {layer} output truncated")
        if finish_reason and finish_reason != "stop":
            logger.warning(
                "Ollama unexpected finish_reason layer=%s conversation_id=%s "
                "finish_reason=%s",
                layer,
                conversation_id,
                finish_reason,
            )
            raise AssistantUnavailableError(
                f"Ollama layer {layer} stopped with {finish_reason}"
            )

        message = choice.get("message")
        if not isinstance(message, dict):
            raise AssistantUnavailableError(
                f"Ollama layer {layer} returned no message"
            )
        reasoning_chars = _reasoning_chars(message)
        content = message.get("content")
        usage = data.get("usage") if isinstance(data, dict) else None
        if not isinstance(usage, dict):
            usage = {}
        prompt_tokens = usage.get("prompt_tokens")
        total_tokens = usage.get("total_tokens")
        logger.info(
            "Ollama layer=%s done conversation_id=%s elapsed=%.3fs "
            "finish_reason=%s prompt_tokens=%s/%s (%s) "
            "completion_tokens=%s total_tokens=%s/%s (%s) "
            "reasoning_chars=%s content_chars=%s",
            layer,
            conversation_id,
            elapsed,
            finish_reason or None,
            prompt_tokens,
            self._context_length,
            _ratio_label(prompt_tokens, self._context_length),
            usage.get("completion_tokens"),
            total_tokens,
            self._context_length,
            _ratio_label(total_tokens, self._context_length),
            reasoning_chars,
            len(str(content or "")),
        )
        if not isinstance(content, str) or not content.strip():
            logger.warning(
                "Ollama layer=%s returned empty message content conversation_id=%s",
                layer,
                conversation_id,
            )
            return None
        return _parse_answer_html(
            content,
            layer=layer,
            conversation_id=conversation_id,
        )


def _count_prompt_tokens(
    client: httpx.Client,
    *,
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
) -> int | None:
    prompt = "\n\n".join(
        f"{item.get('role', '')}\n{item.get('content', '')}" for item in messages
    )
    try:
        response = client.post(
            f"{base_url}/api/tokenize",
            json={"model": model, "content": prompt, "prompt": prompt},
        )
        response.raise_for_status()
        data = response.json()
    except (httpx.HTTPError, ValueError):
        logger.debug("Ollama tokenize failed model=%s; using character estimate", model)
        return _estimate_tokens(prompt)
    tokens = data.get("tokens") if isinstance(data, dict) else None
    if isinstance(tokens, list):
        return len(tokens)
    count = data.get("count") if isinstance(data, dict) else None
    if isinstance(count, int) and count >= 0:
        return count
    return _estimate_tokens(prompt)


def _estimate_tokens(text: str) -> int:
    # Cyrillic and mixed HTML usually take more tokens than English (~3 chars).
    return max(1, (len(text) + 2) // 3)


def _ratio_label(used: object, total: int) -> str:
    if not isinstance(used, int) or total <= 0:
        return "unknown"
    return f"{100.0 * used / total:.1f}%"


def _remaining_tokens(used: object, total: int) -> str:
    if not isinstance(used, int):
        return "unknown"
    return str(max(0, total - used))


def _first_choice(data: object) -> dict | None:
    if not isinstance(data, dict):
        return None
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    choice = choices[0]
    return choice if isinstance(choice, dict) else None


def _reasoning_chars(message: dict) -> int:
    for key in ("reasoning", "reasoning_content", "thinking", "analysis"):
        value = message.get(key)
        if isinstance(value, str) and value:
            return len(value)
    return 0


def _parse_answer_html(
    content: str,
    *,
    layer: int,
    conversation_id: str,
) -> str | None:
    stripped = content.strip()
    try:
        raw = json.loads(stripped)
    except (json.JSONDecodeError, TypeError):
        raw = None
    if isinstance(raw, dict):
        answer_html = raw.get("answer_html")
        if isinstance(answer_html, str) and answer_html.strip():
            stripped = answer_html.strip()
        elif isinstance(raw.get("content"), str) and str(raw["content"]).strip():
            stripped = str(raw["content"]).strip()
        else:
            logger.warning(
                "Ollama structured content has no answer_html layer=%s "
                "conversation_id=%s",
                layer,
                conversation_id,
            )
            return None
    elif isinstance(raw, str) and raw.strip():
        stripped = raw.strip()

    if _INTERNAL_REASONING_PREFIX.match(stripped):
        logger.warning(
            "Ollama leaked internal reasoning layer=%s conversation_id=%s",
            layer,
            conversation_id,
        )
        return None
    return stripped
