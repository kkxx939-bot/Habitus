"""模型适配器（OpenAI 兼容的 rerank、embedding，以及 anthropic_messages）共用的单次 JSON POST 与 HTTP 错误换算。

只负责一次请求：带上限地读响应、把 4xx/5xx 换成带 ``status_code`` 与 ``retry_after`` 的异常（交给
``model_client.retry`` 分类）、把成功响应解析成 JSON 对象。重试、并发与业务字段校验都不在这里。
"""

from __future__ import annotations

import ipaddress
import json
from collections.abc import Mapping
from urllib.parse import urlsplit

import httpx

from habitus.model_client.contracts import ModelResponseError
from habitus.model_client.json_parser import parse_json_response


class HTTPStatusError(RuntimeError):
    """服务返回 4xx/5xx；``retry.normalize_provider_error`` 按这两个属性分类。"""

    def __init__(self, message: str, *, status_code: int, retry_after: float | None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retry_after = retry_after


async def post_json_object(
    client: httpx.AsyncClient,
    endpoint: str,
    *,
    headers: Mapping[str, str],
    payload: Mapping[str, object],
    max_response_bytes: int,
    label: str,
) -> Mapping[str, object]:
    """发一次 JSON POST，返回成功响应里的 JSON 对象；``label`` 只用来写报错。"""

    async with client.stream("POST", endpoint, headers=dict(headers), json=dict(payload)) as response:
        content = await read_bounded(response, max_response_bytes, label=label)
        status_code = getattr(response, "status_code", None)
        if not isinstance(status_code, int):
            raise ModelResponseError(f"{label} provider response has no HTTP status")
        if status_code >= 400:
            raise HTTPStatusError(
                error_message(content, status_code, label=label),
                status_code=status_code,
                retry_after=retry_after(getattr(response, "headers", None)),
            )
    try:
        value = parse_json_response(content.decode("utf-8"), allow_repair=False).value
    except (UnicodeDecodeError, ValueError) as exc:
        raise ModelResponseError(f"{label} provider returned malformed JSON") from exc
    if not isinstance(value, Mapping):
        raise ModelResponseError(f"{label} provider response must be a JSON object")
    return value


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
    if not isinstance(message, str) or not message.strip():
        message = payload.get("message")
    return message.strip()[:2048] if isinstance(message, str) and message.strip() else fallback


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
    "error_message",
    "is_remote_url",
    "post_json_object",
    "raise_for_status",
    "read_bounded",
    "read_bounded_sync",
    "retry_after",
]
