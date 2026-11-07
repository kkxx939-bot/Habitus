"""供应商错误分类和有界重试计时。"""

from __future__ import annotations

import random
import ssl
from collections.abc import Callable

import httpx

from habitus.model_client.contracts import (
    PROXY_AUTH_HINT,
    TLS_UNTRUSTED_HINT,
    ModelAuthenticationError,
    ModelClientError,
    ModelContentSafetyError,
    ModelInputTooLargeError,
    ModelPermissionError,
    ModelProxyAuthError,
    ModelQuotaError,
    ModelRateLimitError,
    ModelResponseError,
    ModelTLSError,
    ModelTransportError,
)

_INPUT_TOO_LARGE = (
    "payload too large",
    "request entity too large",
    "context window exceeded",
    "context_length_exceeded",
    "maximum context length",
    "too many input tokens",
    "input length exceeds",
    "prompt is too long",
)
_CONTENT_SAFETY = (
    "content policy",
    "content_filter",
    "contentfilter",
    "moderation",
    "sensitive content",
    "内容安全",
    "敏感内容",
)
_QUOTA = (
    "quota exceeded",
    "quota_exceeded",
    "insufficient_quota",
    "account quota",
    "accountoverdue",
    "billing hard limit",
    "insufficient balance",
)
_RATE_LIMIT = (
    "rate limit",
    "rate_limit",
    "too many requests",
    "toomanyrequests",
    "tpm limit",
    "rpm limit",
)
_TRANSIENT = (
    "timeout",
    "timed out",
    "connection refused",
    "connection reset",
    "temporarily unavailable",
    "service unavailable",
)


def normalize_provider_error(error: Exception) -> ModelClientError:
    """把 SDK 专用异常和传输异常映射到稳定的客户端异常体系。"""

    if isinstance(error, ModelClientError):
        return error
    chain = _exception_chain(error)
    status = _status_code(chain)
    retry_after = _retry_after(chain)
    message = str(error).strip() or type(error).__name__
    lowered = "\n".join(str(item) for item in chain).casefold()

    # 证书不被信任（含主机名不匹配）与代理要认证（407）按异常类型 / 状态认，放最前：报错里带的主机名可能碰上下面的
    # 关键词（例如主机名含 moderation）；而且它们都是 OSError / 传输错误的近亲，重发多少遍都一样，不能落进"连不上就重试"（W18）
    if any(isinstance(item, ssl.SSLCertVerificationError) for item in chain):
        return ModelTLSError(f"{message}; {TLS_UNTRUSTED_HINT}")
    if status == 407 or any(_is_proxy_auth_rejection(item) for item in chain):
        return ModelProxyAuthError(f"{message}; {PROXY_AUTH_HINT}")
    if _contains(lowered, _CONTENT_SAFETY):
        return ModelContentSafetyError(message)
    if status == 413 or _contains(lowered, _INPUT_TOO_LARGE):
        return ModelInputTooLargeError(message)
    # 402 Payment Required：DeepSeek 余额不足（"Insufficient Balance"）走这个状态码
    if status == 402 or _contains(lowered, _QUOTA):
        return ModelQuotaError(message)
    if status == 401:
        return ModelAuthenticationError(message)
    if status == 403:
        return ModelPermissionError(message)
    if status == 429 or _contains(lowered, _RATE_LIMIT):
        return ModelRateLimitError(message, retry_after_seconds=retry_after)
    # 529：服务过载（Anthropic Messages 协议用它，另有部分网关沿用），与 503 同类、过一阵可恢复
    if status in {408, 409, 425, 500, 502, 503, 504, 529}:
        return ModelTransportError(message, retry_after_seconds=retry_after)
    if any(isinstance(item, (TimeoutError, ConnectionError, OSError)) for item in chain) or _contains(
        lowered,
        _TRANSIENT,
    ):
        return ModelTransportError(message, retry_after_seconds=retry_after)
    return ModelResponseError(message)


def retry_delay(
    attempt: int,
    *,
    base_delay: float,
    max_delay: float,
    error: ModelClientError,
    uniform: Callable[[float, float], float] = random.uniform,
) -> float:
    """返回带随机抖动的指数退避时间，并遵守有界的 Retry-After 值。"""

    if error.retry_after_seconds is not None:
        return min(max(0.0, error.retry_after_seconds), max_delay)
    delay = min(base_delay * (2**attempt), max_delay)
    return min(max_delay, delay * uniform(0.8, 1.2))


def _is_proxy_auth_rejection(error: BaseException) -> bool:
    """httpx 在代理对 CONNECT 回非 2xx 时抛 ``ProxyError("<状态码> <原因>")``，拿不到响应头，只能从状态码前缀认 407。"""

    return isinstance(error, httpx.ProxyError) and str(error).startswith("407")


def _contains(text: str, patterns: tuple[str, ...]) -> bool:
    return any(pattern in text for pattern in patterns)


def _exception_chain(error: Exception) -> tuple[BaseException, ...]:
    """沿显式原因或异常上下文展开有限异常链，识别传输库包装的底层错误。"""

    chain: list[BaseException] = []
    seen: set[int] = set()
    candidate: BaseException | None = error
    while candidate is not None and id(candidate) not in seen and len(chain) < 16:
        seen.add(id(candidate))
        chain.append(candidate)
        cause = candidate.__cause__
        candidate = cause if cause is not None else candidate.__context__
    return tuple(chain)


def _status_code(chain: tuple[BaseException, ...]) -> int | None:
    for candidate in chain:
        value = getattr(candidate, "status_code", None)
        if value is None:
            value = getattr(candidate, "code", None)
        if isinstance(value, bool) or not isinstance(value, str | int):
            continue
        try:
            status = int(value)
        except (TypeError, ValueError):
            # 外部异常上的状态码字段解析不了，就当这一层没有状态码。
            continue
        if 100 <= status <= 599:
            return status
    return None


def _retry_after(chain: tuple[BaseException, ...]) -> float | None:
    for candidate in chain:
        direct = getattr(candidate, "retry_after", None)
        if direct is not None:
            try:
                return max(0.0, float(direct))
            except (TypeError, ValueError):
                # 外部异常的 retry_after 解析不了，就当没给，继续看响应头。
                pass
        response = getattr(candidate, "response", None)
        headers = getattr(response, "headers", None)
        if headers is None:
            continue
        try:
            value = headers.get("Retry-After")
        except (AttributeError, TypeError):
            # 外部异常的 response.headers 不是可读映射，就当没有响应头。
            continue
        try:
            return max(0.0, float(value))
        except (TypeError, ValueError):
            # Retry-After 不是秒数（缺失或 HTTP 日期），就当没给。
            continue
    return None


__all__ = ["normalize_provider_error", "retry_delay"]
