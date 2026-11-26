"""模型适配器（anthropic_messages）用的单次 JSON POST 与 HTTP 错误换算。

只负责一次请求：带上限地读响应、把 4xx/5xx 换成带 ``status_code`` 与 ``retry_after`` 的异常（交给
``model_client.failures`` 归类）、把成功响应解析成 JSON 对象。重试、并发与业务字段校验都不在这里。
"""

from __future__ import annotations

import ipaddress
import json
from collections.abc import Mapping
from urllib.parse import urlsplit

import httpx

from habitus.foundation.validation import is_nonblank_str
from habitus.model_client.errors import ModelResponseError


class HTTPStatusError(RuntimeError):
    """服务返回 4xx/5xx（或流里的错误事件）；``failures.classify_failure`` 按状态码与协议错误代码归类。
    流里的错误事件没有状态码，``status_code`` 为空。"""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None,
        retry_after: float | None,
        provider_code: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retry_after = retry_after
        self.provider_code = provider_code


async def read_bounded(response: httpx.Response, maximum: int, *, label: str) -> bytes:
    """逐块读响应体，累计超过 ``maximum`` 字节立即报错，不先整份读进内存。"""

    chunks: list[bytes] = []
    total = 0
    async for chunk in response.aiter_bytes():
        if not isinstance(chunk, bytes):
            raise ModelResponseError(f"{label} response yielded non-byte content")
        total += len(chunk)
        if total > maximum:
            raise ModelResponseError(f"{label} response exceeds the configured byte bound")
        chunks.append(chunk)
    return b"".join(chunks)


def read_bounded_sync(response: httpx.Response, maximum: int, *, label: str) -> bytes:
    """``read_bounded`` 的同步版本，给同步调用路径用。"""

    chunks: list[bytes] = []
    total = 0
    for chunk in response.iter_bytes():
        total += len(chunk)
        if total > maximum:
            raise ModelResponseError(f"{label} response exceeds the configured byte bound")
        chunks.append(chunk)
    return b"".join(chunks)


def raise_for_status(response: httpx.Response, content: bytes, *, label: str) -> None:
    """4xx/5xx 换成带 ``status_code`` 与 ``retry_after`` 的 ``HTTPStatusError``；2xx/3xx 什么都不做。"""

    if response.status_code >= 400:
        raise HTTPStatusError(
            error_message(content, response.status_code, label=label),
            status_code=response.status_code,
            retry_after=retry_after(response.headers),
            provider_code=error_code(content),
        )


def error_message(content: bytes, status_code: int, *, label: str) -> str:
    """取 ``error.message``，没有就取顶层 ``message``，都没有给一句带状态码的说明。"""

    fallback = f"{label} provider request failed with HTTP {status_code}"
    try:
        payload = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return fallback
    if not isinstance(payload, Mapping):
        return fallback
    error = payload.get("error")
    message = error.get("message") if isinstance(error, Mapping) else None
    if not is_nonblank_str(message):
        message = payload.get("message")
    return message.strip()[:2048] if isinstance(message, str) and message.strip() else fallback


def error_code(content: bytes | Mapping[str, object]) -> str | None:
    """协议的错误代码：``error.code``、``error.type``、顶层 ``code``、顶层 ``type``（Anthropic 顶层固定是 ``error``，跳过）里
    第一个非空文本；响应体不是 JSON 对象就没有。"""

    payload: object = content
    if isinstance(content, bytes):
        try:
            payload = json.loads(content)
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None
    if not isinstance(payload, Mapping):
        return None
    error = payload.get("error")
    nested: Mapping[str, object] = error if isinstance(error, Mapping) else {}
    for value in (nested.get("code"), nested.get("type"), payload.get("code"), payload.get("type")):
        if isinstance(value, str) and value.strip() and value != "error":
            return value.strip()[:128]
    return None


def retry_after(headers: object) -> float | None:
    """``Retry-After`` 的秒数；缺省或不是数字时为 None，负数按 0。"""

    getter = getattr(headers, "get", None)
    if not callable(getter):
        return None
    raw = getter("Retry-After")
    if raw is None:
        return None
    try:
        return max(0.0, float(str(raw)))
    except (TypeError, ValueError):
        return None


def is_remote_url(url: str) -> bool:
    """``localhost`` 与回环地址算本地，其余都算远程。"""

    hostname = str(urlsplit(url).hostname or "").casefold()
    if hostname == "localhost":
        return False
    try:
        return not ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return True


__all__ = [
    "HTTPStatusError",
    "error_code",
    "error_message",
    "is_remote_url",
    "raise_for_status",
    "read_bounded",
    "read_bounded_sync",
    "retry_after",
]
