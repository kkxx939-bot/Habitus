"""OpenAI Chat Completions 兼容协议的同步与异步 Provider。"""

from __future__ import annotations

import asyncio
import ipaddress
import json
import threading
import time
import weakref
from collections.abc import AsyncIterator, Iterator, Mapping
from urllib.parse import urlsplit

import httpx

from ModelClient.config import ChatModelConfig
from ModelClient.contracts import (
    ChatMessage,
    ChatRequest,
    ModelConfigurationError,
    ModelResponse,
    ModelResponseError,
    ModelStreamEvent,
    ProviderCapabilities,
    TokenUsage,
    ToolCall,
    ToolDefinition,
)
from ModelClient.json_parser import parse_json_response

_RESERVED_BODY_FIELDS = {
    "max_completion_tokens",
    "max_tokens",
    "messages",
    "model",
    "reasoning_effort",
    "response_format",
    "stream",
    "temperature",
    "tool_choice",
    "tools",
}


class OpenAICompatibleChatProvider:
    """执行一次 Chat Completions 请求并转换为统一模型契约。"""

    def __init__(self, config: ChatModelConfig, *, api_key: str = "") -> None:
        if not isinstance(config, ChatModelConfig):
            raise TypeError("config must be ChatModelConfig")
        if config.route.adapter != "openai_compatible_chat":
            raise ModelConfigurationError(
                "OpenAICompatibleChatProvider requires adapter='openai_compatible_chat'"
            )
        if not config.route.base_url:
            raise ModelConfigurationError(
                "openai_compatible_chat adapter requires an explicit base_url"
            )
        overlap = _RESERVED_BODY_FIELDS & set(config.route.extra_body)
        if overlap:
            raise ModelConfigurationError(
                "openai_compatible_chat extra_body cannot override request fields: "
                f"{sorted(overlap)}"
            )

        self.config = config
        self.provider_name = config.route.provider
        self.model = config.route.model
        self.is_remote = _is_remote_url(config.route.base_url)
        self.capabilities = ProviderCapabilities(
            async_completion=True,
            streaming=True,
            tools=True,
            structured_output_mode=config.structured_output_mode,
            reasoning=config.reasoning,
        )
        self._api_key = api_key.strip()
        self._endpoint = f"{config.route.base_url}/chat/completions"
        self._models_endpoint = f"{config.route.base_url}/models"
        concurrency = config.route.max_concurrent
        limits = httpx.Limits(
            max_connections=concurrency,
            max_keepalive_connections=concurrency,
        )
        timeout = httpx.Timeout(config.route.timeout_seconds)
        self._sync_client = httpx.Client(
            follow_redirects=False,
            limits=limits,
            timeout=timeout,
        )
        self._async_clients: weakref.WeakKeyDictionary[
            asyncio.AbstractEventLoop,
            httpx.AsyncClient,
        ] = weakref.WeakKeyDictionary()
        self._async_clients_lock = threading.Lock()

    def complete(self, request: ChatRequest) -> ModelResponse:
        started = time.monotonic()
        payload = self._payload(request, stream=False)
        with self._sync_client.stream(
            "POST",
            self._endpoint,
            headers=self._headers(),
            json=payload,
        ) as response:
            content = _read_limited(response.iter_bytes(), self.config.route.max_response_bytes)
            self._require_success(response, content)
        return _normalize_response(
            _decode_json_object(content),
            provider=self.provider_name,
            configured_model=self.model,
            prompt_version=request.prompt_version,
            started=started,
        )

    async def complete_async(self, request: ChatRequest) -> ModelResponse:
        started = time.monotonic()
        payload = self._payload(request, stream=False)
        client = self._async_client()
        async with client.stream(
            "POST",
            self._endpoint,
            headers=self._headers(),
            json=payload,
        ) as response:
            content = await _read_limited_async(
                response.aiter_bytes(),
                self.config.route.max_response_bytes,
            )
            self._require_success(response, content)
        return _normalize_response(
            _decode_json_object(content),
            provider=self.provider_name,
            configured_model=self.model,
            prompt_version=request.prompt_version,
            started=started,
        )

    def stream(self, request: ChatRequest) -> Iterator[ModelStreamEvent]:
        payload = self._payload(request, stream=True)
        with self._sync_client.stream(
            "POST",
            self._endpoint,
            headers=self._headers(accept="text/event-stream"),
            json=payload,
        ) as response:
            if response.status_code >= 400:
                content = _read_limited(
                    response.iter_bytes(),
                    self.config.route.max_response_bytes,
                )
                self._require_success(response, content)
            decoder = _SSEDecoder()
            total = 0
            for chunk in response.iter_bytes():
                total += len(chunk)
                if total > self.config.route.max_response_bytes:
                    raise ModelResponseError(
                        "model stream exceeds the configured byte bound"
                    )
                yield from decoder.feed(chunk)
            yield from decoder.finish()

    async def stream_async(self, request: ChatRequest) -> AsyncIterator[ModelStreamEvent]:
        payload = self._payload(request, stream=True)
        client = self._async_client()
        async with client.stream(
            "POST",
            self._endpoint,
            headers=self._headers(accept="text/event-stream"),
            json=payload,
        ) as response:
            if response.status_code >= 400:
                content = await _read_limited_async(
                    response.aiter_bytes(),
                    self.config.route.max_response_bytes,
                )
                self._require_success(response, content)
            decoder = _SSEDecoder()
            total = 0
            async for chunk in response.aiter_bytes():
                total += len(chunk)
                if total > self.config.route.max_response_bytes:
                    raise ModelResponseError(
                        "model stream exceeds the configured byte bound"
                    )
                for event in decoder.feed(chunk):
                    yield event
            for event in decoder.finish():
                yield event

    def health_check(self) -> Mapping[str, object]:
        with self._sync_client.stream(
            "GET",
            self._models_endpoint,
            headers=self._headers(),
        ) as response:
            content = _read_limited(response.iter_bytes(), self.config.route.max_response_bytes)
            self._require_success(response, content)
            _decode_json_object(content)
        return {
            "ok": True,
            "provider": self.provider_name,
            "model": self.model,
        }

    def _payload(self, request: ChatRequest, *, stream: bool) -> dict[str, object]:
        if not isinstance(request, ChatRequest):
            raise TypeError("request must be ChatRequest")
        payload: dict[str, object] = {
            "model": self.model,
            "messages": [_message_payload(message) for message in request.messages],
            **dict(self.config.route.extra_body),
        }
        if not self.config.reasoning and request.temperature is not None:
            payload["temperature"] = float(request.temperature)
        if request.reasoning is not None:
            if not self.capabilities.reasoning:
                raise ModelConfigurationError(
                    "selected chat route does not enable reasoning parameters"
                )
            if request.reasoning.effort:
                payload["reasoning_effort"] = request.reasoning.effort

        max_tokens = (
            request.max_output_tokens
            if request.max_output_tokens is not None
            else self.config.max_output_tokens
        )
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if request.tools:
            payload["tools"] = [_tool_payload(tool) for tool in request.tools]
            payload["tool_choice"] = request.tool_choice or "auto"
        if request.response_format is not None:
            mode = self.config.structured_output_mode
            if mode == "json_object":
                payload["response_format"] = {"type": "json_object"}
            elif mode == "json_schema":
                payload["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": request.response_format.name,
                        "strict": request.response_format.strict,
                        "schema": dict(request.response_format.schema),
                    },
                }
        if stream:
            payload["stream"] = True
            payload.setdefault("stream_options", {"include_usage": True})
        return payload

    def _headers(self, *, accept: str = "application/json") -> dict[str, str]:
        headers = {
            "Accept": accept,
            "Content-Type": "application/json",
            **dict(self.config.route.extra_headers),
        }
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def _async_client(self) -> httpx.AsyncClient:
        loop = asyncio.get_running_loop()
        with self._async_clients_lock:
            client = self._async_clients.get(loop)
            if client is not None:
                return client
            concurrency = self.config.route.max_concurrent
            client = httpx.AsyncClient(
                follow_redirects=False,
                limits=httpx.Limits(
                    max_connections=concurrency,
                    max_keepalive_connections=concurrency,
                ),
                timeout=httpx.Timeout(self.config.route.timeout_seconds),
            )
            self._async_clients[loop] = client
            return client

    @staticmethod
    def _require_success(response: httpx.Response, content: bytes) -> None:
        if 200 <= response.status_code < 300:
            return
        raise _HTTPStatusError(
            _error_message(content, response.status_code),
            status_code=response.status_code,
            retry_after=_retry_after(response.headers),
        )


class _HTTPStatusError(RuntimeError):
    def __init__(self, message: str, *, status_code: int, retry_after: float | None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retry_after = retry_after


class _SSEDecoder:
    """增量解析 data-only SSE，并确保只产生一个终止事件。"""

    def __init__(self) -> None:
        self._buffer = bytearray()
        self._data_lines: list[str] = []
        self._finish_reason: str | None = None
        self._terminated = False
        self._finished = False

    def feed(self, chunk: bytes) -> tuple[ModelStreamEvent, ...]:
        if not isinstance(chunk, bytes):
            raise ModelResponseError("model stream yielded non-byte content")
        if self._finished:
            raise ModelResponseError("model stream received data after finalization")
        self._buffer.extend(chunk)
        events: list[ModelStreamEvent] = []
        while True:
            newline = self._buffer.find(b"\n")
            if newline < 0:
                break
            raw_line = bytes(self._buffer[:newline])
            del self._buffer[: newline + 1]
            events.extend(self._consume_line(raw_line.rstrip(b"\r")))
        return tuple(events)

    def finish(self) -> tuple[ModelStreamEvent, ...]:
        if self._finished:
            return ()
        events: list[ModelStreamEvent] = []
        if self._buffer:
            events.extend(self._consume_line(bytes(self._buffer).rstrip(b"\r")))
            self._buffer.clear()
        events.extend(self._flush_event())
        self._finished = True
        events.append(
            ModelStreamEvent(
                kind="done",
                finish_reason=self._finish_reason or "stop",
            )
        )
        return tuple(events)

    def _consume_line(self, raw_line: bytes) -> tuple[ModelStreamEvent, ...]:
        try:
            line = raw_line.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ModelResponseError("model stream returned non-UTF-8 data") from exc
        if not line:
            return self._flush_event()
        if line.startswith(":") or self._terminated:
            return ()
        if line.startswith("data:"):
            self._data_lines.append(line[5:].lstrip())
        elif line.startswith("{"):
            self._data_lines.append(line)
        return ()

    def _flush_event(self) -> tuple[ModelStreamEvent, ...]:
        if not self._data_lines:
            return ()
        data = "\n".join(self._data_lines).strip()
        self._data_lines.clear()
        if data == "[DONE]":
            self._terminated = True
            return ()
        source = _decode_json_object(data.encode("utf-8"))
        result: list[ModelStreamEvent] = []
        for event in _normalize_stream_chunk(source):
            if event.kind == "done":
                self._finish_reason = event.finish_reason or self._finish_reason
            else:
                result.append(event)
        return tuple(result)


def _message_payload(message: ChatMessage) -> dict[str, object]:
    payload: dict[str, object] = {
        "role": message.role,
        "content": message.content,
    }
    if message.name:
        payload["name"] = message.name
    if message.tool_call_id:
        payload["tool_call_id"] = message.tool_call_id
    if message.tool_calls:
        payload["tool_calls"] = [
            {
                "id": tool_call.id,
                "type": "function",
                "function": {
                    "name": tool_call.name,
                    "arguments": json.dumps(
                        tool_call.arguments,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                },
            }
            for tool_call in message.tool_calls
        ]
    return payload


def _tool_payload(tool: ToolDefinition) -> dict[str, object]:
    function: dict[str, object] = {
        "name": tool.name,
        "description": tool.description,
        "parameters": dict(tool.parameters),
    }
    if tool.strict:
        function["strict"] = True
    return {"type": "function", "function": function}


def _normalize_response(
    source: Mapping[str, object],
    *,
    provider: str,
    configured_model: str,
    prompt_version: str | None,
    started: float,
) -> ModelResponse:
    choices = source.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], Mapping):
        raise ModelResponseError("model response has no choices")
    first = choices[0]
    message = first.get("message")
    if not isinstance(message, Mapping):
        raise ModelResponseError("model response has no assistant message")
    content = _message_text(message.get("content"))
    tool_calls = _parse_tool_calls(message.get("tool_calls"))
    if not content and not tool_calls:
        raise ModelResponseError("model response has neither content nor tool calls")
    return ModelResponse(
        content=content,
        model=str(source.get("model") or configured_model),
        provider=provider,
        tool_calls=tool_calls,
        finish_reason=str(first.get("finish_reason") or "stop"),
        reasoning_content=_optional_text(
            message.get("reasoning_content", message.get("reasoning"))
        ),
        usage=_normalize_usage(source.get("usage")),
        prompt_version=prompt_version,
        latency_ms=max(0, round((time.monotonic() - started) * 1000)),
        raw=source,
    )


def _normalize_stream_chunk(
    source: Mapping[str, object],
) -> tuple[ModelStreamEvent, ...]:
    events: list[ModelStreamEvent] = []
    choices = source.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], Mapping):
        first = choices[0]
        delta = first.get("delta")
        if isinstance(delta, Mapping):
            content = _optional_text(delta.get("content"))
            if content:
                events.append(
                    ModelStreamEvent(
                        kind="content_delta",
                        content_delta=content,
                        raw=source,
                    )
                )
            reasoning = _optional_text(
                delta.get("reasoning_content", delta.get("reasoning"))
            )
            if reasoning:
                events.append(
                    ModelStreamEvent(
                        kind="reasoning_delta",
                        reasoning_delta=reasoning,
                        raw=source,
                    )
                )
            raw_tool_calls = delta.get("tool_calls")
            if isinstance(raw_tool_calls, list):
                for fallback_index, raw_tool_call in enumerate(raw_tool_calls):
                    if not isinstance(raw_tool_call, Mapping):
                        continue
                    function = raw_tool_call.get("function")
                    function = function if isinstance(function, Mapping) else {}
                    raw_index = raw_tool_call.get("index", fallback_index)
                    index = raw_index if isinstance(raw_index, int) else fallback_index
                    events.append(
                        ModelStreamEvent(
                            kind="tool_call_delta",
                            tool_call_index=index,
                            tool_call_id=_optional_text(raw_tool_call.get("id")),
                            tool_name=_optional_text(function.get("name")),
                            tool_arguments_delta=_optional_text(function.get("arguments")),
                            raw=source,
                        )
                    )
        finish_reason = _optional_text(first.get("finish_reason"))
        if finish_reason:
            events.append(
                ModelStreamEvent(
                    kind="done",
                    finish_reason=finish_reason,
                    raw=source,
                )
            )
    if isinstance(source.get("usage"), Mapping):
        events.append(
            ModelStreamEvent(
                kind="usage",
                usage=_normalize_usage(source["usage"]),
                raw=source,
            )
        )
    return tuple(events)


def _parse_tool_calls(source: object) -> tuple[ToolCall, ...]:
    if source is None:
        return ()
    if not isinstance(source, list):
        raise ModelResponseError("model tool_calls must be an array")
    result: list[ToolCall] = []
    for index, raw in enumerate(source):
        if not isinstance(raw, Mapping):
            raise ModelResponseError(f"model tool_calls[{index}] must be an object")
        function = raw.get("function")
        if not isinstance(function, Mapping):
            raise ModelResponseError(f"model tool_calls[{index}] has no function")
        arguments = _tool_arguments(function.get("arguments", "{}"), index=index)
        call_id = raw.get("id")
        name = function.get("name")
        if not isinstance(call_id, str) or not call_id.strip():
            raise ModelResponseError(f"model tool_calls[{index}] id must be non-empty")
        if not isinstance(name, str) or not name.strip():
            raise ModelResponseError(f"model tool_calls[{index}] name must be non-empty")
        result.append(ToolCall(id=call_id, name=name, arguments=arguments))
    return tuple(result)


def _tool_arguments(source: object, *, index: int) -> dict[str, object]:
    if isinstance(source, Mapping):
        return dict(source)
    if not isinstance(source, str):
        raise ModelResponseError(
            f"model tool_calls[{index}] arguments must be JSON text"
        )
    try:
        parsed = parse_json_response(source).value
    except ValueError as exc:
        raise ModelResponseError(
            f"model tool_calls[{index}] arguments are not valid JSON"
        ) from exc
    if not isinstance(parsed, Mapping):
        raise ModelResponseError(
            f"model tool_calls[{index}] arguments must be an object"
        )
    return dict(parsed)


def _normalize_usage(source: object) -> TokenUsage:
    if not isinstance(source, Mapping):
        return TokenUsage()
    input_tokens = _non_negative_int(source.get("prompt_tokens", source.get("input_tokens")))
    output_tokens = _non_negative_int(
        source.get("completion_tokens", source.get("output_tokens"))
    )
    total_tokens = _non_negative_int(source.get("total_tokens"))
    if not total_tokens:
        total_tokens = input_tokens + output_tokens
    prompt_details = source.get("prompt_tokens_details")
    completion_details = source.get("completion_tokens_details")
    cached_tokens = _mapping_int(prompt_details, "cached_tokens") or _non_negative_int(
        source.get("prompt_cache_hit_tokens")
    )
    reasoning_tokens = _mapping_int(completion_details, "reasoning_tokens")
    return TokenUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        cached_input_tokens=cached_tokens,
        reasoning_tokens=reasoning_tokens,
        details=dict(source),
    )


def _message_text(source: object) -> str | None:
    if source is None:
        return None
    if isinstance(source, str):
        return source or None
    if isinstance(source, list):
        parts: list[str] = []
        for item in source:
            if isinstance(item, Mapping) and item.get("type") in {"text", "output_text"}:
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts) or None
    raise ModelResponseError("model response content must be text or text parts")


def _decode_json_object(content: bytes) -> dict[str, object]:
    try:
        value = json.loads(content, parse_constant=_reject_non_finite)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ModelResponseError("model provider returned malformed JSON") from exc
    if not isinstance(value, dict):
        raise ModelResponseError("model provider response must be a JSON object")
    return value


def _read_limited(chunks: Iterator[bytes], maximum: int) -> bytes:
    result: list[bytes] = []
    total = 0
    for chunk in chunks:
        if not isinstance(chunk, bytes):
            raise ModelResponseError("model response yielded non-byte content")
        total += len(chunk)
        if total > maximum:
            raise ModelResponseError("model response exceeds the configured byte bound")
        result.append(chunk)
    return b"".join(result)


async def _read_limited_async(chunks: AsyncIterator[bytes], maximum: int) -> bytes:
    result: list[bytes] = []
    total = 0
    async for chunk in chunks:
        if not isinstance(chunk, bytes):
            raise ModelResponseError("model response yielded non-byte content")
        total += len(chunk)
        if total > maximum:
            raise ModelResponseError("model response exceeds the configured byte bound")
        result.append(chunk)
    return b"".join(result)


def _error_message(content: bytes, status_code: int) -> str:
    fallback = f"model provider request failed with HTTP {status_code}"
    try:
        payload = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return fallback
    if not isinstance(payload, Mapping):
        return fallback
    error = payload.get("error")
    message = error.get("message") if isinstance(error, Mapping) else None
    if not isinstance(message, str) or not message.strip():
        message = payload.get("message")
    return message.strip()[:2048] if isinstance(message, str) and message.strip() else fallback


def _retry_after(headers: Mapping[str, str]) -> float | None:
    raw = headers.get("Retry-After")
    if raw is None:
        return None
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return None


def _optional_text(source: object) -> str | None:
    return source if isinstance(source, str) and source else None


def _non_negative_int(source: object) -> int:
    return source if isinstance(source, int) and not isinstance(source, bool) and source >= 0 else 0


def _mapping_int(source: object, key: str) -> int:
    return _non_negative_int(source.get(key)) if isinstance(source, Mapping) else 0


def _reject_non_finite(value: str) -> None:
    raise ValueError(f"non-finite JSON number is not allowed: {value}")


def _is_remote_url(url: str) -> bool:
    hostname = str(urlsplit(url).hostname or "").casefold()
    if hostname == "localhost":
        return False
    try:
        return not ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return True


__all__ = ["OpenAICompatibleChatProvider"]
