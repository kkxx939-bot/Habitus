"""模型调用失败的唯一归类处（裁定 108），以及有界重试的计时。

先看协议的错误代码字段（OpenAI 兼容协议的 ``error.code`` / ``error.type``、Anthropic Messages 的 ``error.type``，
适配器从响应体或流里的错误事件取出来放在 ``provider_code`` 上），再看 HTTP 状态码，再看传输层异常；都对不上是"认不出"。
不按厂商、模型名或报错文字判断（OpenViking 按 SDK 异常类名与报错文字识别的那部分不借）。
"""

from __future__ import annotations

import random
import ssl
from collections.abc import Callable

import httpx

from habitus.model_client.config import CallFailureConfig, ProviderConfig
from habitus.model_client.errors import (
    PROXY_AUTH_HINT,
    TLS_UNTRUSTED_HINT,
    ModelAuthenticationError,
    ModelClientError,
    ModelConfigurationError,
    ModelContentSafetyError,
    ModelErrorKind,
    ModelInputTooLargeError,
    ModelInvalidRequestError,
    ModelNotFoundError,
    ModelPermissionError,
    ModelProxyAuthError,
    ModelQuotaError,
    ModelRateLimitError,
    ModelServiceError,
    ModelTLSError,
    ModelTransportError,
    ModelUnknownError,
)

#: 协议错误代码（去掉非字母数字、转大写）→ 类别。只收具体的代码；``invalid_request_error`` 这类笼统的类型不收，交给状态码。
_CODE_KINDS: dict[str, ModelErrorKind] = {
    # 认证、权限
    "INVALIDAPIKEY": ModelErrorKind.AUTHENTICATION,
    "APIKEYINVALID": ModelErrorKind.AUTHENTICATION,
    "AUTHENTICATIONERROR": ModelErrorKind.AUTHENTICATION,
    "UNAUTHENTICATED": ModelErrorKind.AUTHENTICATION,
    "PERMISSIONERROR": ModelErrorKind.PERMISSION,
    "PERMISSIONDENIED": ModelErrorKind.PERMISSION,
    "ACCESSDENIED": ModelErrorKind.PERMISSION,
    # 额度、余额
    "INSUFFICIENTQUOTA": ModelErrorKind.QUOTA,
    "BILLINGERROR": ModelErrorKind.QUOTA,
    "BILLINGHARDLIMITREACHED": ModelErrorKind.QUOTA,
    "QUOTAEXCEEDED": ModelErrorKind.QUOTA,
    "ACCOUNTQUOTAEXCEEDED": ModelErrorKind.QUOTA,
    "ACCOUNTOVERDUE": ModelErrorKind.QUOTA,
    "ARREARAGE": ModelErrorKind.QUOTA,
    "INSUFFICIENTBALANCE": ModelErrorKind.QUOTA,
    # 限流
    "RATELIMITERROR": ModelErrorKind.RATE_LIMIT,
    "RATELIMITEXCEEDED": ModelErrorKind.RATE_LIMIT,
    "TOOMANYREQUESTS": ModelErrorKind.RATE_LIMIT,
    "REQUESTLIMITEXCEEDED": ModelErrorKind.RATE_LIMIT,
    "THROTTLING": ModelErrorKind.RATE_LIMIT,
    # 模型不存在、不支持
    "MODELNOTFOUND": ModelErrorKind.MODEL_NOT_FOUND,
    "NOTFOUNDERROR": ModelErrorKind.MODEL_NOT_FOUND,
    "INVALIDMODEL": ModelErrorKind.MODEL_NOT_FOUND,
    "MODELNOTEXIST": ModelErrorKind.MODEL_NOT_FOUND,
    "UNSUPPORTEDPARAMETER": ModelErrorKind.UNSUPPORTED,
    "UNSUPPORTEDVALUE": ModelErrorKind.UNSUPPORTED,
    # 这一条输入有问题
    "CONTEXTLENGTHEXCEEDED": ModelErrorKind.INPUT_TOO_LARGE,
    "REQUESTTOOLARGE": ModelErrorKind.INPUT_TOO_LARGE,
    "CONTENTTOOLARGE": ModelErrorKind.INPUT_TOO_LARGE,
    "INPUTTEXTEXCEEDSLIMIT": ModelErrorKind.INPUT_TOO_LARGE,
    "CONTENTFILTER": ModelErrorKind.CONTENT_SAFETY,
    "CONTENTPOLICYVIOLATION": ModelErrorKind.CONTENT_SAFETY,
    "DATAINSPECTIONFAILED": ModelErrorKind.CONTENT_SAFETY,
    "SAFETY": ModelErrorKind.CONTENT_SAFETY,
    # 服务、网络
    "APIERROR": ModelErrorKind.SERVICE,
    "OVERLOADEDERROR": ModelErrorKind.SERVICE,
    "SERVERERROR": ModelErrorKind.SERVICE,
    "INTERNALERROR": ModelErrorKind.SERVICE,
    "SERVICEUNAVAILABLE": ModelErrorKind.SERVICE,
    "TIMEOUTERROR": ModelErrorKind.NETWORK,
}

_STATUS_KINDS: dict[int, ModelErrorKind] = {
    400: ModelErrorKind.INVALID_REQUEST,
    422: ModelErrorKind.INVALID_REQUEST,
    401: ModelErrorKind.AUTHENTICATION,
    402: ModelErrorKind.QUOTA,
    403: ModelErrorKind.PERMISSION,
    404: ModelErrorKind.MODEL_NOT_FOUND,
    405: ModelErrorKind.UNSUPPORTED,
    501: ModelErrorKind.UNSUPPORTED,
    407: ModelErrorKind.NETWORK_SETUP,
    408: ModelErrorKind.NETWORK,
    504: ModelErrorKind.NETWORK,
    413: ModelErrorKind.INPUT_TOO_LARGE,
    429: ModelErrorKind.RATE_LIMIT,
    409: ModelErrorKind.SERVICE,
    425: ModelErrorKind.SERVICE,
}

_KIND_TYPES: dict[ModelErrorKind, type[ModelClientError]] = {
    ModelErrorKind.SERVICE: ModelServiceError,
    ModelErrorKind.NETWORK: ModelTransportError,
    ModelErrorKind.RATE_LIMIT: ModelRateLimitError,
    ModelErrorKind.UNKNOWN: ModelUnknownError,
    ModelErrorKind.AUTHENTICATION: ModelAuthenticationError,
    ModelErrorKind.PERMISSION: ModelPermissionError,
    ModelErrorKind.QUOTA: ModelQuotaError,
    ModelErrorKind.MODEL_NOT_FOUND: ModelNotFoundError,
    ModelErrorKind.UNSUPPORTED: ModelConfigurationError,
    ModelErrorKind.NETWORK_SETUP: ModelProxyAuthError,
    ModelErrorKind.INVALID_REQUEST: ModelInvalidRequestError,
    ModelErrorKind.INPUT_TOO_LARGE: ModelInputTooLargeError,
    ModelErrorKind.CONTENT_SAFETY: ModelContentSafetyError,
}

#: 结构化输出的终止原因（``finish_reason``）里表示"被内容安全拦下"的那几个（两种协议的适配器已换成同一套写法）。
CONTENT_BLOCKED_FINISH_REASONS = frozenset({"content_filter", "safety"})


def classify_failure(error: Exception) -> ModelClientError:
    """把一次模型调用的异常归到唯一一套类别里；已经归过类的原样返回。"""

    if isinstance(error, ModelClientError):
        return error
    chain = _exception_chain(error)
    status = _first_int(chain, "status_code")
    code = _first_code(chain)
    message = str(error).strip() or type(error).__name__
    retry_after = _retry_after(chain)
    # 证书不被信任与代理要认证按异常类型 / 状态认：都是传输错误的近亲，但重发多少遍都一样（W18）
    if any(isinstance(item, ssl.SSLCertVerificationError) for item in chain):
        return ModelTLSError(f"{message}; {TLS_UNTRUSTED_HINT}", status_code=status, provider_code=code)
    if status == 407 or any(_is_proxy_auth_rejection(item) for item in chain):
        return ModelProxyAuthError(f"{message}; {PROXY_AUTH_HINT}", status_code=status, provider_code=code)
    kind = _kind_of(code, status, chain)
    return _KIND_TYPES[kind](message, retry_after_seconds=retry_after, status_code=status, provider_code=code)


def _kind_of(code: str | None, status: int | None, chain: tuple[BaseException, ...]) -> ModelErrorKind:
    normalized = None if code is None else "".join(ch for ch in code.upper() if ch.isalnum())
    if normalized in _CODE_KINDS:
        return _CODE_KINDS[normalized]
    if status is not None:
        if status in _STATUS_KINDS:
            return _STATUS_KINDS[status]
        return ModelErrorKind.SERVICE if status >= 500 else ModelErrorKind.UNKNOWN
    if any(isinstance(item, (TimeoutError, ConnectionError, OSError, httpx.TransportError)) for item in chain):
        return ModelErrorKind.NETWORK
    return ModelErrorKind.UNKNOWN


def retry_delay(
    attempt: int,
    *,
    route: ProviderConfig,
    failures: CallFailureConfig,
    error: ModelClientError,
    uniform: Callable[[float, float], float] = random.uniform,
) -> float:
    """第 ``attempt`` 次重试前等多久：带抖动的指数退避，限流用限流的间隔；服务给了 ``Retry-After`` 就按它（同样封顶）。"""

    if isinstance(error, ModelRateLimitError):
        base, cap = failures.rate_limit_retry_base_delay_seconds, failures.rate_limit_retry_max_delay_seconds
    else:
        base, cap = route.retry_base_delay_seconds, route.retry_max_delay_seconds
    if error.retry_after_seconds is not None:
        return min(max(0.0, error.retry_after_seconds), cap)
    return min(cap, min(base * (2**attempt), cap) * uniform(0.8, 1.2))


def _is_proxy_auth_rejection(error: BaseException) -> bool:
    """httpx 在代理对 CONNECT 回非 2xx 时抛 ``ProxyError("<状态码> <原因>")``，拿不到响应头，只能从状态码前缀认 407。"""

    return isinstance(error, httpx.ProxyError) and str(error).startswith("407")


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


def _first_int(chain: tuple[BaseException, ...], name: str) -> int | None:
    for candidate in chain:
        value = getattr(candidate, name, None)
        if isinstance(value, int) and not isinstance(value, bool) and 100 <= value <= 599:
            return value
    return None


def _first_code(chain: tuple[BaseException, ...]) -> str | None:
    for candidate in chain:
        value = getattr(candidate, "provider_code", None)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _retry_after(chain: tuple[BaseException, ...]) -> float | None:
    for candidate in chain:
        value = getattr(candidate, "retry_after", None)
        if isinstance(value, int | float) and not isinstance(value, bool):
            return max(0.0, float(value))
    return None


__all__ = ["CONTENT_BLOCKED_FINISH_REASONS", "classify_failure", "retry_delay"]
