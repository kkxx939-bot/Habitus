"""文本、工具调用、结构化输出和流式输出的供应商无关契约。"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Literal, Protocol

from habitus.model_client.config import ChatStructuredOutputMode

MessageRole = Literal["system", "developer", "user", "assistant", "tool"]
StreamEventKind = Literal[
    "content_delta",
    "reasoning_delta",
    "tool_call_delta",
    "usage",
    "done",
]


@dataclass(frozen=True)
class ToolCall:
    """模型返回的一次规范化函数调用。"""

    id: str
    name: str
    arguments: Mapping[str, object]

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("tool call id must be non-empty")
        if not self.name.strip():
            raise ValueError("tool call name must be non-empty")
        object.__setattr__(self, "arguments", dict(self.arguments))


@dataclass(frozen=True)
class ToolDefinition:
    """与供应商无关的函数工具声明。"""

    name: str
    description: str
    parameters: Mapping[str, object]
    strict: bool = False

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("tool name must be non-empty")
        if not self.description.strip():
            raise ValueError("tool description must be non-empty")
        object.__setattr__(self, "parameters", dict(self.parameters))


@dataclass(frozen=True)
class ChatMessage:
    """保留助手工具调用和工具结果的规范化消息。"""

    role: MessageRole
    content: str | None = None
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()

    def __post_init__(self) -> None:
        if self.role not in {"system", "developer", "user", "assistant", "tool"}:
            raise ValueError(f"unsupported chat message role: {self.role}")
        if self.name is not None and not self.name.strip():
            raise ValueError("message name must be non-empty when provided")
        if self.tool_call_id is not None and not self.tool_call_id.strip():
            raise ValueError("tool_call_id must be non-empty when provided")
        object.__setattr__(self, "tool_calls", tuple(self.tool_calls))
        if self.role == "tool":
            if not self.tool_call_id:
                raise ValueError("tool result messages require tool_call_id")
            if self.content is None:
                raise ValueError("tool result messages require content")
            if self.tool_calls:
                raise ValueError("tool result messages cannot contain tool calls")
        elif self.tool_call_id is not None:
            raise ValueError("tool_call_id is only valid for tool result messages")
        if self.tool_calls and self.role != "assistant":
            raise ValueError("only assistant messages may contain tool calls")
        if self.content is None and not self.tool_calls:
            raise ValueError("messages require content or assistant tool calls")


@dataclass(frozen=True)
class ResponseFormat:
    """向原生支持结构化输出的供应商请求使用的 JSON Schema。"""

    name: str
    schema: Mapping[str, object]
    strict: bool = True

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("response format name must be non-empty")
        if not self.schema:
            raise ValueError("response format schema must be a non-empty object")
        object.__setattr__(self, "schema", dict(self.schema))


@dataclass(frozen=True)
class ReasoningOptions:
    """可移植的推理意图；供应商专用开关应放入路由的 extra_body。"""

    effort: Literal["minimal", "low", "medium", "high"] | None = None

    def __post_init__(self) -> None:
        if self.effort not in {None, "minimal", "low", "medium", "high"}:
            raise ValueError("unsupported reasoning effort")


@dataclass(frozen=True)
class ChatRequest:
    """不依赖具体供应商或模型路由的一次逻辑模型请求。"""

    messages: tuple[ChatMessage, ...]
    temperature: float | None = 0.0
    max_output_tokens: int | None = None
    tools: tuple[ToolDefinition, ...] = ()
    tool_choice: str | Mapping[str, object] | None = None
    response_format: ResponseFormat | None = None
    reasoning: ReasoningOptions | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "messages", tuple(self.messages))
        object.__setattr__(self, "tools", tuple(self.tools))
        if not self.messages:
            raise ValueError("chat request requires normalized messages")
        if self.temperature is not None and not 0 <= float(self.temperature) <= 2:
            raise ValueError("temperature must be between zero and two")
        if self.max_output_tokens is not None and self.max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be a positive integer")
        if self.tool_choice is not None:
            if isinstance(self.tool_choice, str) and not self.tool_choice.strip():
                raise ValueError("tool_choice text must be non-empty")
            if isinstance(self.tool_choice, Mapping):
                object.__setattr__(self, "tool_choice", dict(self.tool_choice))
        if self.tool_choice is not None and not self.tools:
            raise ValueError("tool_choice requires at least one tool")


@dataclass(frozen=True)
class ChatCallContext:
    """不进入模型请求的调用侧内部信息和本次输入预算上限。"""

    prompt_version: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)
    input_token_limit: int | None = None

    def __post_init__(self) -> None:
        if self.prompt_version is not None and not self.prompt_version.strip():
            raise ValueError("chat call prompt_version must be non-empty when provided")
        object.__setattr__(self, "metadata", dict(self.metadata))
        if self.input_token_limit is not None and self.input_token_limit <= 0:
            raise ValueError("chat call input_token_limit must be a positive integer")


@dataclass(frozen=True)
class PreparedChatRequest:
    """Adapter 一次构造、预算检查和网络发送共同消费的最终请求。"""

    request: ChatRequest
    body: bytes
    model_visible_body: bytes
    reserved_output_tokens: int
    stream: bool

    def __post_init__(self) -> None:
        if not self.body:
            raise ValueError("prepared chat request body must be non-empty bytes")
        if not self.model_visible_body:
            raise ValueError("prepared chat request model_visible_body must be non-empty bytes")
        if self.reserved_output_tokens < 0:
            raise ValueError("prepared chat request reserved_output_tokens must be a non-negative integer")

    @property
    def estimated_input_tokens(self) -> int:
        """只从冻结的模型可见投影派生，杜绝调用方另传估算值。"""

        from habitus.model_client.token_budget import estimate_model_visible_bytes

        return estimate_model_visible_bytes(self.model_visible_body)


@dataclass(frozen=True)
class TokenUsage:
    """规范化的 Token 用量统计，可附带供应商明细。"""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cached_input_tokens: int = 0
    reasoning_tokens: int = 0
    details: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "cached_input_tokens",
            "reasoning_tokens",
        ):
            value = getattr(self, name)
            if value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        object.__setattr__(self, "details", dict(self.details))


@dataclass(frozen=True)
class ModelResponse:
    """一次规范化最终响应，包含工具调用和终止元数据。"""

    content: str | None
    model: str
    provider: str
    tool_calls: tuple[ToolCall, ...] = ()
    finish_reason: str = "stop"
    reasoning_content: str | None = None
    usage: TokenUsage = field(default_factory=TokenUsage)
    prompt_version: str | None = None
    latency_ms: int | None = None
    raw: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "tool_calls", tuple(self.tool_calls))
        if not self.content and not self.tool_calls:
            raise ValueError("model response requires content or tool calls")
        if not self.model.strip():
            raise ValueError("model response model must be non-empty")
        if not self.provider.strip():
            raise ValueError("model response provider must be non-empty")
        if not self.finish_reason.strip():
            raise ValueError("model response finish_reason must be non-empty")
        if self.raw is not None:
            object.__setattr__(self, "raw", dict(self.raw))


@dataclass(frozen=True)
class ModelStreamEvent:
    """一次规范化流事件；工具参数在结束事件前始终保持增量形式。"""

    kind: StreamEventKind
    content_delta: str | None = None
    reasoning_delta: str | None = None
    tool_call_index: int | None = None
    tool_call_id: str | None = None
    tool_name: str | None = None
    tool_arguments_delta: str | None = None
    usage: TokenUsage | None = None
    finish_reason: str | None = None
    raw: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        allowed = {"content_delta", "reasoning_delta", "tool_call_delta", "usage", "done"}
        if self.kind not in allowed:
            raise ValueError(f"unsupported stream event kind: {self.kind}")
        if self.raw is not None:
            object.__setattr__(self, "raw", dict(self.raw))
        payload_fields = {
            "content_delta",
            "reasoning_delta",
            "tool_call_index",
            "tool_call_id",
            "tool_name",
            "tool_arguments_delta",
            "usage",
            "finish_reason",
        }
        expected_fields = {
            "content_delta": {"content_delta"},
            "reasoning_delta": {"reasoning_delta"},
            "tool_call_delta": {
                "tool_call_index",
                "tool_call_id",
                "tool_name",
                "tool_arguments_delta",
            },
            "usage": {"usage"},
            "done": {"finish_reason"},
        }[self.kind]
        unexpected = sorted(name for name in payload_fields - expected_fields if getattr(self, name) is not None)
        if unexpected:
            raise ValueError(f"stream event {self.kind} contains payload for another kind: {unexpected}")
        if self.kind == "content_delta":
            if not self.content_delta:
                raise ValueError("content_delta events require non-empty text")
        elif self.kind == "reasoning_delta":
            if not self.reasoning_delta:
                raise ValueError("reasoning_delta events require non-empty text")
        elif self.kind == "tool_call_delta":
            if self.tool_call_index is None or self.tool_call_index < 0:
                raise ValueError("tool_call_delta events require a non-negative integer index")
            for name in ("tool_call_id", "tool_name", "tool_arguments_delta"):
                value = getattr(self, name)
                if value == "":
                    raise ValueError(f"{name} must be non-empty text when provided")
            if all(value is None for value in (self.tool_call_id, self.tool_name, self.tool_arguments_delta)):
                raise ValueError("tool_call_delta events require at least one tool field")
        elif self.kind == "usage":
            if self.usage is None:
                raise TypeError("usage events require TokenUsage")
        elif self.finish_reason is None or not self.finish_reason.strip():
            raise ValueError("done events require a non-empty finish_reason")


@dataclass(frozen=True)
class ProviderCapabilities:
    """用于适配请求的能力声明，避免按供应商名称编写条件分支。"""

    async_completion: bool = True
    streaming: bool = True
    tools: bool = True
    structured_output_mode: ChatStructuredOutputMode = "none"
    reasoning: bool = False

    def __post_init__(self) -> None:
        if self.structured_output_mode not in {"none", "json_object", "json_schema"}:
            raise ValueError("provider structured_output_mode must be none, json_object or json_schema")


class ChatProvider(Protocol):
    """具体供应商在此实现统一的模型边界。"""

    provider_name: str
    model: str
    is_remote: bool
    capabilities: ProviderCapabilities

    def prepare(self, request: ChatRequest, *, stream: bool) -> PreparedChatRequest: ...

    def complete(self, request: PreparedChatRequest) -> ModelResponse: ...

    async def complete_async(self, request: PreparedChatRequest) -> ModelResponse: ...

    def stream(self, request: PreparedChatRequest) -> Iterator[ModelStreamEvent]: ...

    def stream_async(self, request: PreparedChatRequest) -> AsyncIterator[ModelStreamEvent]: ...

    def health_check(self) -> Mapping[str, object]: ...

    async def aclose(self) -> None: ...


class ModelClientError(RuntimeError):
    """携带重试和故障转移策略元数据的基础异常。"""

    retryable = False
    failover_eligible = False
    code = "MODEL_CLIENT_ERROR"

    def __init__(self, message: str, *, retry_after_seconds: float | None = None) -> None:
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


class ModelConfigurationError(ModelClientError):
    code = "MODEL_CONFIGURATION"


class ModelDependencyError(ModelConfigurationError):
    failover_eligible = True
    code = "MODEL_DEPENDENCY"


class ModelAuthenticationError(ModelClientError):
    failover_eligible = True
    code = "MODEL_AUTHENTICATION"


class ModelPermissionError(ModelClientError):
    failover_eligible = True
    code = "MODEL_PERMISSION"


class ModelTransportError(ModelClientError):
    retryable = True
    failover_eligible = True
    code = "MODEL_TRANSPORT"


class ModelRateLimitError(ModelTransportError):
    code = "MODEL_RATE_LIMIT"


#: 证书不被信任时告诉用户配哪个字段（报错与 doctor 共用这一句）。
TLS_UNTRUSTED_HINT = (
    "the model server's TLS certificate is not trusted; if Habitus runs behind a corporate proxy that inspects TLS, "
    "set network.ca_bundle to the corporate root certificate (PEM file)"
)
#: 代理对 CONNECT 回 407 时的去处（用户 11-06 定：NTLM / Kerberos / PAC 不实现，只提示本机转发代理）。
PROXY_AUTH_HINT = (
    "the proxy requires authentication (HTTP 407). Basic authentication (user name and password) can go into the "
    "HTTPS_PROXY environment variable; Windows domain sign-in (NTLM / Kerberos) and PAC auto-configuration are not "
    "supported directly: run a local forwarding proxy such as cntlm or px on this machine and point network.proxy "
    "at it (for example http://127.0.0.1:3128)"
)


class ModelTLSError(ModelClientError):
    """模型服务的证书不被信任（含主机名不匹配）：同一请求重发多少遍都一样，改了证书配置才能恢复，所以不重试、等模型恢复。"""

    failover_eligible = True
    code = "MODEL_TLS"


class ModelProxyAuthError(ModelClientError):
    """代理要求认证（407）：配置不变就永远过不去，不重试；改了代理配置才能恢复，所以等模型恢复。"""

    failover_eligible = True
    code = "MODEL_PROXY_AUTH"


class ModelQuotaError(ModelClientError):
    failover_eligible = True
    code = "MODEL_QUOTA"


class ModelInputTooLargeError(ModelClientError):
    code = "MODEL_INPUT_TOO_LARGE"


class ModelContentSafetyError(ModelClientError):
    code = "MODEL_CONTENT_SAFETY"


class ModelResponseError(ModelClientError):
    failover_eligible = True
    code = "MODEL_RESPONSE"


class ModelStructuredOutputError(ModelClientError):
    code = "MODEL_STRUCTURED_OUTPUT"


#: 模型此刻用不了（连不上、限流、额度满或余额不足、鉴权或权限失效、服务端答坏了、证书不被信任、代理要认证）：换一份输入也一样，过一阵再来。
#: 服务端答坏了（``ModelResponseError``，带 ``failover_eligible``）问题在服务端、不在这份输入。各处"等模型恢复"的分类
#: 只从这一组取（记忆 Job 的长退避、会话 lane 的重试堆、未封口判断的桥）。
MODEL_UNAVAILABLE: tuple[type[ModelClientError], ...] = (
    ModelTransportError,
    ModelQuotaError,
    ModelAuthenticationError,
    ModelPermissionError,
    ModelResponseError,
    ModelTLSError,
    ModelProxyAuthError,
)
#: 同一份输入再问几遍也一样（输入超长、触发内容安全）：只有输入或处理口径变了才值得再问。
MODEL_DETERMINISTIC: tuple[type[ModelClientError], ...] = (ModelInputTooLargeError, ModelContentSafetyError)
#: ``MODEL_DETERMINISTIC`` 里"这一条触发了内容安全"的那一种：只这一条记「错误」、不重试、健康面单列（裁定 94 错误分类）；
#: 输入超长是材料上限与模型窗口没对上，按我们自己处理不了的 bug 算，不在这里。
MODEL_REFUSED: tuple[type[ModelClientError], ...] = (ModelContentSafetyError,)
#: 模型整个用不了：此刻用不了的那一组加上没配好（``ModelConfigurationError``，含缺依赖）。换代里任何一处调模型碰到它
#: （``ModelUnreachable``）是那一步走退路、换代照常往下做；只有必须成的步骤失败才不写完成标记、进退避（裁定 94 第 8 条）。
MODEL_UNUSABLE: tuple[type[ModelClientError], ...] = (*MODEL_UNAVAILABLE, ModelConfigurationError)


class ModelUnreachable(RuntimeError):
    """模型整个调不通（连不上 / 限流、额度或余额、鉴权、权限、服务端答坏、没配好）：再多重试也处理不了。换代碰到它：调模型的
    那一步走退路（关系沿用上一代、结算留到下次……）、健康面写"模型调不通"，换代照常往下做；只有必须成的步骤失败才不写完成标记、
    按退避重做（裁定 94 第 4、8 条）。不是 ``ModelClientError``：它不是某一次调用的
    失败分类，而是"这一处调模型的结论是整个用不了"，由各触点（``model_client.asking.ask``、换代的探针）抛出。"""


__all__ = [
    "MODEL_DETERMINISTIC",
    "MODEL_REFUSED",
    "MODEL_UNAVAILABLE",
    "MODEL_UNUSABLE",
    "PROXY_AUTH_HINT",
    "TLS_UNTRUSTED_HINT",
    "ChatMessage",
    "ChatRequest",
    "MessageRole",
    "ModelAuthenticationError",
    "ModelClientError",
    "ModelConfigurationError",
    "ModelContentSafetyError",
    "ModelDependencyError",
    "ModelInputTooLargeError",
    "ModelPermissionError",
    "ModelProxyAuthError",
    "ChatProvider",
    "ModelQuotaError",
    "ModelRateLimitError",
    "ModelResponse",
    "ModelResponseError",
    "ModelStreamEvent",
    "ModelStructuredOutputError",
    "ModelTLSError",
    "ModelTransportError",
    "ModelUnreachable",
    "ProviderCapabilities",
    "ReasoningOptions",
    "ResponseFormat",
    "StreamEventKind",
    "TokenUsage",
    "ToolCall",
    "ToolDefinition",
]
