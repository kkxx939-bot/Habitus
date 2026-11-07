"""Anthropic Messages 协议（``POST {base_url}/v1/messages``）的同步与异步 Chat Provider。

与 ``openai_compatible_chat`` 平级、实现同一个 ``ChatProvider`` 契约，只做协议转换和单次 HTTP 调用；重试、并发、
结构化输出的失败反馈都在 Chat 运行层。不按 provider、base_url 或模型名分支：说这套协议的官方服务、网关与兼容实现
只换配置。协议差异全在这里消化：

- 认证头 ``x-api-key``，协议版本头 ``anthropic-version``（固定 ``2026-10-21``）；两者都不许从 ``extra_headers`` 覆盖；
- 开头连续的 system/developer 消息成为顶层 ``system`` 文本块；对话开始后再出现的 system 消息直接报配置错，不挪位、不丢；
- ``max_tokens`` 必填：取请求的 ``max_output_tokens``，否则取路由配置；路由没配就在构造时报错；
- ``2026-10-21`` 起请求带 ``temperature``/``top_p``/``top_k`` 一律 400，所以 ``ChatRequest.temperature`` 不发送；
- 结构化输出 ``json_schema`` → ``output_config.format``（Schema 先换成协议子集，见 ``anthropic_messages_schema``）；
  协议没有不带 Schema 的 JSON 模式，``json_object`` 在构造时报错；``none`` 不加约束；
- 推理意图 ``ReasoningOptions.effort`` → ``output_config.effort``（协议没有 ``minimal``）。
"""

from __future__ import annotations

import asyncio
import threading
import time
import weakref
from collections.abc import AsyncIterator, Iterator, Mapping

import httpx

from habitus.foundation.integrity import canonical_json
from habitus.foundation.network import OutboundSettings
from habitus.model_client.adapters._http import (
    is_remote_url,
    raise_for_status,
    read_bounded,
    read_bounded_sync,
)
from habitus.model_client.adapters.anthropic_messages_decode import MessagesStreamDecoder, normalize_message
from habitus.model_client.adapters.anthropic_messages_schema import messages_output_schema
from habitus.model_client.config import ChatModelConfig
from habitus.model_client.contracts import (
    ChatMessage,
    ChatRequest,
    ModelConfigurationError,
    ModelResponse,
    ModelResponseError,
    ModelStreamEvent,
    PreparedChatRequest,
    ProviderCapabilities,
    ToolDefinition,
)
from habitus.model_client.http_clients import async_http_client, sync_http_client
from habitus.model_client.json_parser import parse_json_response

ANTHROPIC_VERSION = "2026-10-21"
_LABEL = "anthropic_messages"
_RESERVED_BODY_FIELDS = frozenset({"max_tokens", "messages", "model", "stream", "system", "tool_choice", "tools"})
# 2026-10-21 起带任一采样字段的请求一律 400；在构造时拒绝，免得每次请求都失败
_SAMPLING_FIELDS = frozenset({"temperature", "top_p", "top_k"})
_PROTOCOL_HEADERS = frozenset({"x-api-key", "anthropic-version"})
_EFFORTS = frozenset({"low", "medium", "high"})
_TOOL_CHOICES: Mapping[str, Mapping[str, str]] = {
    "auto": {"type": "auto"},
    "none": {"type": "none"},
    "required": {"type": "any"},
}
_SYSTEM_ROLES = frozenset({"system", "developer"})
# 健康探测只要确认端点、密钥与模型可用；输出上限给到很小，成本可以忽略
_HEALTH_MAX_TOKENS = 8


class AnthropicMessagesProvider:
    """执行一次 Messages 请求并转换为统一模型契约。"""

    def __init__(self, config: ChatModelConfig, *, api_key: str = "", outbound: OutboundSettings | None = None) -> None:
        validate_messages_route(config)
        self.config = config
        self.provider_name = config.route.provider
        self.model = config.route.model
        self.is_remote = is_remote_url(config.route.base_url)
        self.capabilities = ProviderCapabilities(
            async_completion=True,
            streaming=True,
            tools=True,
            structured_output_mode=config.structured_output_mode,
            reasoning=config.reasoning,
        )
        self._api_key = api_key.strip()
        self._endpoint = messages_endpoint(config.route.base_url)
        self._outbound = OutboundSettings() if outbound is None else outbound
        self._sync_client = sync_http_client(config.route, self._outbound)
        self._async_clients: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, httpx.AsyncClient] = (
            weakref.WeakKeyDictionary()
        )
        self._async_clients_lock = threading.Lock()
        self._closed = False

    def prepare(self, request: ChatRequest, *, stream: bool) -> PreparedChatRequest:
        """一次生成最终请求正文及其模型可见输入预算。"""

        if self._closed:
            raise ModelConfigurationError("chat provider is closed")
        visible = _visible_payload(request)
        response_format = _response_format(request, self.config)
        if response_format is not None:
            visible["output_config"] = {"format": response_format}
        max_tokens = request.max_output_tokens or self.config.max_output_tokens
        assert max_tokens is not None  # 构造时已要求路由配 max_output_tokens
        transport: dict[str, object] = {
            "model": self.model,
            **dict(self.config.route.extra_body),
            "max_tokens": max_tokens,
        }
        if request.tool_choice is not None:
            transport["tool_choice"] = _tool_choice(request.tool_choice)
        if stream:
            transport["stream"] = True
        payload = {**transport, **visible}
        output_config = _output_config(request, self.config, response_format)
        if output_config:
            payload["output_config"] = output_config
        return PreparedChatRequest(
            request=request,
            body=canonical_json(payload).encode("utf-8"),
            model_visible_body=canonical_json(visible).encode("utf-8"),
            reserved_output_tokens=max_tokens,
            stream=stream,
        )

    def complete(self, request: PreparedChatRequest) -> ModelResponse:
        self._require_prepared(request, stream=False)
        started = time.monotonic()
        with self._sync_client.stream("POST", self._endpoint, headers=self._headers(), content=request.body) as reply:
            content = read_bounded_sync(reply, self.config.route.max_response_bytes, label=_LABEL)
            raise_for_status(reply, content, label=_LABEL)
        return normalize_message(
            _json_object(content), provider=self.provider_name, configured_model=self.model, started=started
        )

    async def complete_async(self, request: PreparedChatRequest) -> ModelResponse:
        self._require_prepared(request, stream=False)
        started = time.monotonic()
        async with self._async_client().stream(
            "POST", self._endpoint, headers=self._headers(), content=request.body
        ) as reply:
            content = await read_bounded(reply, self.config.route.max_response_bytes, label=_LABEL)
            raise_for_status(reply, content, label=_LABEL)
        return normalize_message(
            _json_object(content), provider=self.provider_name, configured_model=self.model, started=started
        )

    def stream(self, request: PreparedChatRequest) -> Iterator[ModelStreamEvent]:
        self._require_prepared(request, stream=True)
        headers = self._headers(accept="text/event-stream")
        with self._sync_client.stream("POST", self._endpoint, headers=headers, content=request.body) as reply:
            if reply.status_code >= 400:
                content = read_bounded_sync(reply, self.config.route.max_response_bytes, label=_LABEL)
                raise_for_status(reply, content, label=_LABEL)
            decoder = MessagesStreamDecoder()
            total = 0
            for chunk in reply.iter_bytes():
                total += len(chunk)
                if total > self.config.route.max_response_bytes:
                    raise ModelResponseError("anthropic_messages stream exceeds the configured byte bound")
                yield from decoder.feed(chunk)
            yield from decoder.finish()

    async def stream_async(self, request: PreparedChatRequest) -> AsyncIterator[ModelStreamEvent]:
        self._require_prepared(request, stream=True)
        headers = self._headers(accept="text/event-stream")
        async with self._async_client().stream("POST", self._endpoint, headers=headers, content=request.body) as reply:
            if reply.status_code >= 400:
                content = await read_bounded(reply, self.config.route.max_response_bytes, label=_LABEL)
                raise_for_status(reply, content, label=_LABEL)
            decoder = MessagesStreamDecoder()
            total = 0
            async for chunk in reply.aiter_bytes():
                total += len(chunk)
                if total > self.config.route.max_response_bytes:
                    raise ModelResponseError("anthropic_messages stream exceeds the configured byte bound")
                for event in decoder.feed(chunk):
                    yield event
            for event in decoder.finish():
                yield event

    def health_check(self) -> Mapping[str, object]:
        """一次最小的真实 ``/v1/messages`` 调用：端点、密钥与模型都通才算健康。"""

        with self._sync_client.stream("POST", self._endpoint, headers=self._headers(), content=self._probe()) as reply:
            content = read_bounded_sync(reply, self.config.route.max_response_bytes, label=_LABEL)
            raise_for_status(reply, content, label=_LABEL)
        return self._probe_result(content)

    async def health_check_async(self) -> Mapping[str, object]:
        async with self._async_client().stream(
            "POST", self._endpoint, headers=self._headers(), content=self._probe()
        ) as reply:
            content = await read_bounded(reply, self.config.route.max_response_bytes, label=_LABEL)
            raise_for_status(reply, content, label=_LABEL)
        return self._probe_result(content)

    async def aclose(self) -> None:
        """幂等关闭同步与所有事件循环对应的异步 HTTP 连接池。"""

        with self._async_clients_lock:
            if self._closed:
                return
            self._closed = True
            clients = tuple(self._async_clients.values())
            self._async_clients.clear()
        first_error: BaseException | None = None
        try:
            await asyncio.to_thread(self._sync_client.close)
        except BaseException as exc:
            first_error = exc
        results = await asyncio.gather(*(client.aclose() for client in clients), return_exceptions=True)
        if first_error is None:
            first_error = next((result for result in results if isinstance(result, BaseException)), None)
        if first_error is not None:
            raise first_error

    def _probe(self) -> bytes:
        payload = {"model": self.model, "max_tokens": _HEALTH_MAX_TOKENS, "messages": [_text_turn("user", "ping")]}
        return canonical_json(payload).encode("utf-8")

    def _probe_result(self, content: bytes) -> Mapping[str, object]:
        if _json_object(content).get("type") != "message":
            raise ModelResponseError("anthropic_messages health probe did not return a message")
        return {"ok": True, "provider": self.provider_name, "model": self.model}

    def _require_prepared(self, request: PreparedChatRequest, *, stream: bool) -> None:
        if self._closed:
            raise ModelConfigurationError("chat provider is closed")
        if request.stream is not stream:
            raise ModelConfigurationError("prepared chat request stream mode does not match the operation")

    def _headers(self, *, accept: str = "application/json") -> dict[str, str]:
        headers = {
            "Accept": accept,
            "Content-Type": "application/json",
            **dict(self.config.route.extra_headers),
            "anthropic-version": ANTHROPIC_VERSION,
        }
        if self._api_key:
            headers["x-api-key"] = self._api_key
        return headers

    def _async_client(self) -> httpx.AsyncClient:
        loop = asyncio.get_running_loop()
        with self._async_clients_lock:
            client = self._async_clients.get(loop)
            if client is None:
                client = async_http_client(self.config.route, self._outbound)
                self._async_clients[loop] = client
            return client


def messages_endpoint(base_url: str) -> str:
    """``base_url`` 已去掉末尾斜杠；已经以 ``/v1`` 结尾就只补 ``/messages``，否则补 ``/v1/messages``。"""

    return f"{base_url}/messages" if base_url.endswith("/v1") else f"{base_url}/v1/messages"


def validate_messages_route(config: ChatModelConfig) -> None:
    """路由里协议做不到的设置一律在发请求前报配置错；Provider 构造与产品层（doctor、启动前检查）共用这一份规则。"""

    route = config.route
    if route.adapter != "anthropic_messages":
        raise ModelConfigurationError("AnthropicMessagesProvider requires adapter='anthropic_messages'")
    if not route.base_url:
        raise ModelConfigurationError("anthropic_messages adapter requires an explicit base_url")
    if config.max_output_tokens is None:
        raise ModelConfigurationError(
            "anthropic_messages adapter requires chat max_output_tokens (the protocol's max_tokens is mandatory)"
        )
    if config.structured_output_mode == "json_object":
        raise ModelConfigurationError(
            "anthropic_messages has no schema-less json_object mode; use structured_output_mode json_schema or none"
        )
    overlap = _RESERVED_BODY_FIELDS & set(route.extra_body)
    if overlap:
        raise ModelConfigurationError(
            f"anthropic_messages extra_body cannot override request fields: {sorted(overlap)}"
        )
    sampling = sorted(_SAMPLING_FIELDS & set(route.extra_body))
    if sampling:
        raise ModelConfigurationError(
            f"anthropic_messages extra_body cannot set sampling fields (rejected by the protocol): {sampling}"
        )
    output_config = route.extra_body.get("output_config", {})
    if not isinstance(output_config, Mapping):
        raise ModelConfigurationError("anthropic_messages extra_body output_config must be an object")
    if "format" in output_config:
        raise ModelConfigurationError(
            "anthropic_messages extra_body cannot set output_config.format; use structured_output_mode"
        )
    protected = sorted(key for key in route.extra_headers if key.casefold() in _PROTOCOL_HEADERS)
    if protected:
        raise ModelConfigurationError(
            f"anthropic_messages extra_headers cannot set protocol or credential headers "
            f"(x-api-key, anthropic-version): {protected}"
        )


def _visible_payload(request: ChatRequest) -> dict[str, object]:
    system, turns = _split_system(request.messages)
    visible: dict[str, object] = {}
    if system:
        visible["system"] = system
    visible["messages"] = turns
    if request.tools:
        visible["tools"] = [_tool_payload(tool) for tool in request.tools]
    return visible


def _response_format(request: ChatRequest, config: ChatModelConfig) -> dict[str, object] | None:
    """只有 ``json_schema`` 路由才把 Schema 交给协议约束；``none`` 时 Schema 只在运行层的提示词里。"""

    if request.response_format is None or config.structured_output_mode != "json_schema":
        return None
    return {"type": "json_schema", "schema": messages_output_schema(request.response_format.schema)}


def _output_config(
    request: ChatRequest, config: ChatModelConfig, response_format: dict[str, object] | None
) -> dict[str, object]:
    """路由 ``extra_body.output_config``（如 effort）+ 请求的推理意图 + 结构化输出格式，合成一个对象。"""

    configured = config.route.extra_body.get("output_config")
    result: dict[str, object] = dict(configured) if isinstance(configured, Mapping) else {}
    if request.reasoning is not None and request.reasoning.effort is not None:
        if not config.reasoning:
            raise ModelConfigurationError("selected chat route does not enable reasoning parameters")
        if request.reasoning.effort not in _EFFORTS:
            raise ModelConfigurationError(
                f"anthropic_messages has no reasoning effort '{request.reasoning.effort}'; use low, medium or high"
            )
        if "effort" in result:
            raise ModelConfigurationError("reasoning effort conflicts with extra_body output_config.effort")
        result["effort"] = request.reasoning.effort
    if response_format is not None:
        result["format"] = response_format
    return result


def _split_system(messages: tuple[ChatMessage, ...]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """开头连续的 system/developer 成为顶层 system；之后合并同角色相邻消息（工具结果必须同在一条 user 消息里）。"""

    if any(message.name is not None for message in messages):
        raise ModelConfigurationError("anthropic_messages has no per-message name field")
    leading = 0
    while leading < len(messages) and messages[leading].role in _SYSTEM_ROLES:
        leading += 1
    system: list[dict[str, object]] = [{"type": "text", "text": message.content} for message in messages[:leading]]
    turns: list[dict[str, object]] = []
    for message in messages[leading:]:
        if message.role in _SYSTEM_ROLES:
            raise ModelConfigurationError(
                "anthropic_messages only accepts system/developer messages before the conversation starts"
            )
        role, blocks = _turn_blocks(message)
        if turns and turns[-1]["role"] == role:
            previous = turns[-1]["content"]
            assert isinstance(previous, list)
            previous.extend(blocks)
        else:
            turns.append({"role": role, "content": blocks})
    return system, turns


def _turn_blocks(message: ChatMessage) -> tuple[str, list[dict[str, object]]]:
    if message.role == "tool":
        return "user", [{"type": "tool_result", "tool_use_id": message.tool_call_id, "content": message.content}]
    blocks: list[dict[str, object]] = []
    if message.content:
        blocks.append({"type": "text", "text": message.content})
    blocks.extend(
        {"type": "tool_use", "id": call.id, "name": call.name, "input": dict(call.arguments)}
        for call in message.tool_calls
    )
    return message.role, blocks


def _text_turn(role: str, text: str) -> dict[str, object]:
    return {"role": role, "content": [{"type": "text", "text": text}]}


def _tool_payload(tool: ToolDefinition) -> dict[str, object]:
    if not tool.strict:
        return {"name": tool.name, "description": tool.description, "input_schema": dict(tool.parameters)}
    return {
        "name": tool.name,
        "description": tool.description,
        "input_schema": messages_output_schema(tool.parameters),
        "strict": True,
    }


def _tool_choice(choice: str | Mapping[str, object]) -> Mapping[str, object]:
    if isinstance(choice, str):
        if choice in _TOOL_CHOICES:
            return dict(_TOOL_CHOICES[choice])
    else:
        function = choice.get("function")
        name = function.get("name") if isinstance(function, Mapping) else None
        if choice.get("type") == "function" and isinstance(name, str) and name.strip():
            return {"type": "tool", "name": name}
    raise ModelConfigurationError("anthropic_messages tool_choice must be auto, none, required or a function selection")


def _json_object(content: bytes) -> dict[str, object]:
    try:
        value = parse_json_response(content.decode("utf-8"), allow_repair=False).value
    except (UnicodeDecodeError, ValueError) as exc:
        raise ModelResponseError("anthropic_messages provider returned malformed JSON") from exc
    if not isinstance(value, dict):
        raise ModelResponseError("anthropic_messages provider response must be a JSON object")
    return value


__all__ = ["ANTHROPIC_VERSION", "AnthropicMessagesProvider", "messages_endpoint", "validate_messages_route"]
