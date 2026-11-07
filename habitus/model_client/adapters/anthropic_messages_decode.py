"""Anthropic Messages 协议的最终响应与 SSE 流解码，换成供应商无关的 ``ModelResponse`` / ``ModelStreamEvent``。

只做协议转换：内容块按顺序归并（``text`` 拼成正文、``thinking`` 拼成推理文本、``tool_use`` 成工具调用，其余块类型
不携带本契约能表达的内容，跳过）；``stop_reason`` 换成运行层认的终止原因；用量把缓存写入、缓存命中与未缓存三段输入
加总成输入 Token。流里的 ``error`` 事件按协议文档的错误类型换回对应 HTTP 状态码，交给运行层统一分类与重试。
"""

from __future__ import annotations

import time
from collections.abc import Mapping

from habitus.foundation.validation import is_int, is_nonblank_str
from habitus.model_client.adapters._http import HTTPStatusError
from habitus.model_client.contracts import (
    ModelContentSafetyError,
    ModelResponse,
    ModelResponseError,
    ModelStreamEvent,
    TokenUsage,
    ToolCall,
)
from habitus.model_client.json_parser import parse_json_response

# Messages 的 stop_reason → 运行层（structured.py 认 length / content_filter）使用的终止原因；未列出的原样传
_FINISH_REASONS = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "max_tokens": "length",
    # 这一轮没写完：服务端暂停（pause_turn）或上下文窗口用满，与 max_tokens 截断同类
    "pause_turn": "length",
    "model_context_window_exceeded": "length",
    "tool_use": "tool_calls",
    "refusal": "content_filter",
}
# 协议文档里错误类型与 HTTP 状态码的对应；流开始后服务端只能用 error 事件报错，换回状态码好让运行层同样分类
_STATUS_BY_ERROR_TYPE = {
    "invalid_request_error": 400,
    "authentication_error": 401,
    "billing_error": 402,
    "permission_error": 403,
    "not_found_error": 404,
    "request_too_large": 413,
    "rate_limit_error": 429,
    "api_error": 500,
    "timeout_error": 504,
    "overloaded_error": 529,
}


def finish_reason(stop_reason: object) -> str:
    if not is_nonblank_str(stop_reason):
        raise ModelResponseError("anthropic_messages response has no stop_reason")
    return _FINISH_REASONS.get(stop_reason, stop_reason)


def normalize_message(
    source: Mapping[str, object],
    *,
    provider: str,
    configured_model: str,
    started: float,
) -> ModelResponse:
    """把一次非流式 ``/v1/messages`` 成功响应换成 ``ModelResponse``。"""

    if source.get("type") == "error":
        raise ModelResponseError(f"anthropic_messages returned an error payload: {_error_text(source)}")
    blocks = source.get("content")
    if not isinstance(blocks, list):
        raise ModelResponseError("anthropic_messages response content must be an array")
    texts: list[str] = []
    thoughts: list[str] = []
    tool_calls: list[ToolCall] = []
    for index, block in enumerate(blocks):
        if not isinstance(block, Mapping):
            raise ModelResponseError(f"anthropic_messages content[{index}] must be an object")
        kind = block.get("type")
        if kind == "text":
            texts.append(_text(block.get("text"), f"content[{index}].text"))
        elif kind == "thinking":
            thoughts.append(_text(block.get("thinking", ""), f"content[{index}].thinking"))
        elif kind == "tool_use":
            tool_calls.append(_tool_call(block, index))
    reason = finish_reason(source.get("stop_reason"))
    content = "".join(texts)
    if not content.strip() and not tool_calls:
        if reason == "content_filter":
            raise ModelContentSafetyError("anthropic_messages model refused the request")
        raise ModelResponseError("anthropic_messages response has neither text nor tool calls")
    reasoning = "".join(thoughts)
    model = source.get("model")
    return ModelResponse(
        content=content or None,
        model=model if isinstance(model, str) and model.strip() else configured_model,
        provider=provider,
        tool_calls=tuple(tool_calls),
        finish_reason=reason,
        reasoning_content=reasoning or None,
        usage=normalize_usage(source.get("usage")),
        latency_ms=max(0, round((time.monotonic() - started) * 1000)),
        raw=source,
    )


def normalize_usage(source: object, *, output_tokens: object = None) -> TokenUsage:
    """``input_tokens`` 只是未缓存部分；输入 Token 按三段之和算，缓存命中单列。"""

    if not isinstance(source, Mapping):
        return TokenUsage()
    uncached = _count(source.get("input_tokens"))
    cache_write = _count(source.get("cache_creation_input_tokens"))
    cache_read = _count(source.get("cache_read_input_tokens"))
    output = _count(source.get("output_tokens") if output_tokens is None else output_tokens)
    input_tokens = uncached + cache_write + cache_read
    return TokenUsage(
        input_tokens=input_tokens,
        output_tokens=output,
        total_tokens=input_tokens + output,
        cached_input_tokens=cache_read,
        details=dict(source),
    )


def stream_error(event: Mapping[str, object]) -> Exception:
    """流里的 ``error`` 事件：已知错误类型换回 HTTP 状态码，未知的按服务端答坏处理。"""

    error = event.get("error")
    kind = error.get("type") if isinstance(error, Mapping) else None
    status = _STATUS_BY_ERROR_TYPE.get(kind) if isinstance(kind, str) else None
    message = f"anthropic_messages stream error: {_error_text(event)}"
    if status is None:
        return ModelResponseError(message)
    return HTTPStatusError(message, status_code=status, retry_after=None)


class MessagesStreamDecoder:
    """增量解析 Messages SSE：只产生一个 done，没有 ``message_stop`` 就结束算截断。"""

    def __init__(self) -> None:
        self._buffer = bytearray()
        self._data_lines: list[str] = []
        self._usage: Mapping[str, object] | None = None
        self._blocks: dict[int, tuple[str, int]] = {}
        self._closed: set[int] = set()
        self._tools: list[dict[str, str]] = []
        self._finish_reason: str | None = None
        self._stopped = False
        self._finished = False
        self._saw_output = False

    def feed(self, chunk: bytes) -> tuple[ModelStreamEvent, ...]:
        if self._finished:
            raise ModelResponseError("anthropic_messages stream received data after finalization")
        self._buffer.extend(chunk)
        events: list[ModelStreamEvent] = []
        while (newline := self._buffer.find(b"\n")) >= 0:
            raw_line = bytes(self._buffer[:newline]).rstrip(b"\r")
            del self._buffer[: newline + 1]
            events.extend(self._consume_line(raw_line))
        return tuple(events)

    def finish(self) -> tuple[ModelStreamEvent, ...]:
        if self._finished:
            return ()
        events = list(self._consume_line(bytes(self._buffer).rstrip(b"\r"))) if self._buffer else []
        self._buffer.clear()
        events.extend(self._flush())
        self._finished = True
        if not self._stopped or self._finish_reason is None:
            raise ModelResponseError("anthropic_messages stream ended without message_delta and message_stop")
        self._validate_tool_calls()
        if not self._saw_output:
            if self._finish_reason == "content_filter":
                raise ModelContentSafetyError("anthropic_messages model refused the request")
            raise ModelResponseError("anthropic_messages stream ended without semantic output")
        events.append(ModelStreamEvent(kind="done", finish_reason=self._finish_reason))
        return tuple(events)

    def _consume_line(self, raw_line: bytes) -> tuple[ModelStreamEvent, ...]:
        try:
            line = raw_line.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ModelResponseError("anthropic_messages stream returned non-UTF-8 data") from exc
        if not line:
            return self._flush()
        if line.startswith("data:"):
            self._data_lines.append(line[5:].lstrip())
        return ()

    def _flush(self) -> tuple[ModelStreamEvent, ...]:
        if not self._data_lines:
            return ()
        data = "\n".join(self._data_lines)
        self._data_lines.clear()
        try:
            event = parse_json_response(data, allow_repair=False).value
        except ValueError as exc:
            raise ModelResponseError("anthropic_messages stream event is not valid JSON") from exc
        if not isinstance(event, Mapping):
            raise ModelResponseError("anthropic_messages stream event must be a JSON object")
        if self._stopped:
            raise ModelResponseError("anthropic_messages stream received an event after message_stop")
        return self._dispatch(event)

    def _dispatch(self, event: Mapping[str, object]) -> tuple[ModelStreamEvent, ...]:
        kind = event.get("type")
        if kind == "error":
            raise stream_error(event)
        if self._finish_reason is not None and kind not in {"message_stop", "ping"}:
            raise ModelResponseError("anthropic_messages stream emitted content after message_delta")
        if kind == "message_start":
            message = event.get("message")
            self._usage = message.get("usage") if isinstance(message, Mapping) else None
        elif kind == "content_block_start":
            return self._start_block(event)
        elif kind == "content_block_delta":
            return self._delta(event)
        elif kind == "content_block_stop":
            self._close_block(event)
        elif kind == "message_delta":
            return self._message_delta(event)
        elif kind == "message_stop":
            self._stopped = True
        return ()

    def _start_block(self, event: Mapping[str, object]) -> tuple[ModelStreamEvent, ...]:
        index = _block_index(event)
        block = event.get("content_block")
        if index in self._blocks or not isinstance(block, Mapping):
            raise ModelResponseError("anthropic_messages stream started an invalid or duplicate content block")
        kind = block.get("type")
        if kind != "tool_use":
            self._blocks[index] = (str(kind), -1)
            return ()
        identity = (block.get("id"), block.get("name"))
        if not all(isinstance(item, str) and item.strip() for item in identity):
            raise ModelResponseError("anthropic_messages stream tool_use block needs an id and a name")
        ordinal = len(self._tools)
        self._blocks[index] = ("tool_use", ordinal)
        self._tools.append({"id": str(identity[0]), "name": str(identity[1]), "arguments": ""})
        self._saw_output = True
        return (
            ModelStreamEvent(
                kind="tool_call_delta",
                tool_call_index=ordinal,
                tool_call_id=str(identity[0]),
                tool_name=str(identity[1]),
                raw=event,
            ),
        )

    def _delta(self, event: Mapping[str, object]) -> tuple[ModelStreamEvent, ...]:
        index = _block_index(event)
        if index not in self._blocks or index in self._closed:
            raise ModelResponseError("anthropic_messages stream delta refers to an unknown or closed content block")
        delta = event.get("delta")
        if not isinstance(delta, Mapping):
            raise ModelResponseError("anthropic_messages stream delta must be an object")
        kind = delta.get("type")
        if kind == "text_delta":
            text = _text(delta.get("text"), "text_delta.text")
            self._saw_output = self._saw_output or bool(text.strip())
            return (ModelStreamEvent(kind="content_delta", content_delta=text, raw=event),) if text else ()
        if kind == "thinking_delta":
            thought = _text(delta.get("thinking"), "thinking_delta.thinking")
            return (ModelStreamEvent(kind="reasoning_delta", reasoning_delta=thought, raw=event),) if thought else ()
        if kind == "input_json_delta":
            block_kind, ordinal = self._blocks[index]
            fragment = _text(delta.get("partial_json"), "input_json_delta.partial_json")
            if block_kind != "tool_use":
                raise ModelResponseError("anthropic_messages stream sent tool input to a non-tool block")
            self._tools[ordinal]["arguments"] += fragment
            if not fragment:
                return ()
            return (
                ModelStreamEvent(
                    kind="tool_call_delta", tool_call_index=ordinal, tool_arguments_delta=fragment, raw=event
                ),
            )
        return ()

    def _close_block(self, event: Mapping[str, object]) -> None:
        index = _block_index(event)
        if index not in self._blocks or index in self._closed:
            raise ModelResponseError("anthropic_messages stream stopped an unknown or already closed content block")
        self._closed.add(index)

    def _message_delta(self, event: Mapping[str, object]) -> tuple[ModelStreamEvent, ...]:
        delta = event.get("delta")
        if not isinstance(delta, Mapping):
            raise ModelResponseError("anthropic_messages message_delta must carry a delta object")
        self._finish_reason = finish_reason(delta.get("stop_reason"))
        usage = event.get("usage")
        if not isinstance(usage, Mapping):
            return ()
        merged = normalize_usage(self._usage or {}, output_tokens=usage.get("output_tokens"))
        return (ModelStreamEvent(kind="usage", usage=merged, raw=event),)

    def _validate_tool_calls(self) -> None:
        for state in self._tools:
            text = state["arguments"] or "{}"
            try:
                arguments = parse_json_response(text, allow_repair=False).value
            except ValueError as exc:
                raise ModelResponseError("anthropic_messages stream ended with invalid tool input") from exc
            if not isinstance(arguments, Mapping):
                raise ModelResponseError("anthropic_messages stream tool input must form a JSON object")


def _tool_call(block: Mapping[str, object], index: int) -> ToolCall:
    call_id = block.get("id")
    name = block.get("name")
    arguments = block.get("input")
    if not is_nonblank_str(call_id) or not is_nonblank_str(name):
        raise ModelResponseError(f"anthropic_messages content[{index}] tool_use needs an id and a name")
    if not isinstance(arguments, Mapping):
        raise ModelResponseError(f"anthropic_messages content[{index}] tool_use input must be an object")
    return ToolCall(id=call_id, name=name, arguments=arguments)


def _block_index(event: Mapping[str, object]) -> int:
    index = event.get("index")
    if not is_int(index) or index < 0:
        raise ModelResponseError("anthropic_messages stream content block index must be a non-negative integer")
    return index


def _text(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise ModelResponseError(f"anthropic_messages {label} must be text")
    return value


def _count(value: object) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


def _error_text(source: Mapping[str, object]) -> str:
    error = source.get("error")
    message = error.get("message") if isinstance(error, Mapping) else None
    return message.strip()[:2048] if isinstance(message, str) and message.strip() else "no message"


__all__ = ["MessagesStreamDecoder", "finish_reason", "normalize_message", "normalize_usage", "stream_error"]
