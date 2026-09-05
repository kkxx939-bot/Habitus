"""OpenAI-compatible 文本重排 REST 协议适配器。"""

from __future__ import annotations

import asyncio
import ipaddress
import json
import math
import threading
import weakref
from collections.abc import Mapping, Sequence
from urllib.parse import urlsplit

import httpx

from habitus.model_client.config import RerankModelConfig
from habitus.model_client.contracts import ModelConfigurationError, ModelResponseError
from habitus.model_client.json_parser import parse_json_response

_RESERVED_REQUEST_FIELDS = {
    "api_key",
    "base_url",
    "documents",
    "model",
    "query",
    "top_n",
}


class OpenAICompatibleRerankProvider:
    """执行一次 `/reranks` 请求并按输入文档顺序返回相关性分数。"""

    def __init__(self, config: RerankModelConfig, *, api_key: str) -> None:
        if not isinstance(config, RerankModelConfig):
            raise TypeError("config must be RerankModelConfig")
        if config.route.adapter != "openai_compatible_rerank":
            raise ModelConfigurationError(
                "OpenAICompatibleRerankProvider requires adapter='openai_compatible_rerank'"
            )
        if not config.route.base_url:
            raise ModelConfigurationError(
                "openai_compatible_rerank adapter requires an explicit base_url"
            )
        if not isinstance(api_key, str) or not api_key.strip():
            raise ModelConfigurationError(
                "openai_compatible_rerank adapter requires an API key"
            )
        overlap = _RESERVED_REQUEST_FIELDS & set(config.route.extra_body)
        if overlap:
            raise ModelConfigurationError(
                "openai_compatible_rerank extra_body cannot override request fields: "
                f"{sorted(overlap)}"
            )

        self.config = config
        self.provider_name = config.route.provider
        self.model = config.route.model
        self.is_remote = _is_remote_url(config.route.base_url)
        self._api_key = api_key.strip()
        self._endpoint = f"{config.route.base_url}/reranks"
        self._clients: weakref.WeakKeyDictionary[
            asyncio.AbstractEventLoop,
            httpx.AsyncClient,
        ] = weakref.WeakKeyDictionary()
        self._clients_lock = threading.Lock()
        self._closed = False

    async def rerank(
        self,
        query: str,
        documents: Sequence[str],
    ) -> tuple[float, ...]:
        if self._closed:
            raise ModelConfigurationError("rerank provider is closed")
        payload: dict[str, object] = {
            **dict(self.config.route.extra_body),
            "model": self.model,
            "query": query,
            "documents": list(documents),
        }
        response = await self._post(payload)
        return self._scores(response, expected=len(documents))

    async def aclose(self) -> None:
        """幂等关闭所有事件循环对应的异步 HTTP 连接池。"""

        with self._clients_lock:
            if self._closed:
                return
            self._closed = True
            clients = tuple(self._clients.values())
            self._clients.clear()
        results = await asyncio.gather(
            *(client.aclose() for client in clients),
            return_exceptions=True,
        )
        first_error = next(
            (result for result in results if isinstance(result, BaseException)),
            None,
        )
        if first_error is not None:
            raise first_error

    async def _post(self, payload: Mapping[str, object]) -> Mapping[str, object]:
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            **dict(self.config.route.extra_headers),
        }
        client = self._client()
        async with client.stream(
            "POST",
            self._endpoint,
            headers=headers,
            json=dict(payload),
        ) as response:
            content = await self._read_response(response)
            status_code = getattr(response, "status_code", None)
            if not isinstance(status_code, int):
                raise ModelResponseError("rerank provider response has no HTTP status")
            if status_code >= 400:
                raise _HTTPStatusError(
                    _error_message(content, status_code),
                    status_code=status_code,
                    retry_after=_retry_after(getattr(response, "headers", None)),
                )
        try:
            value = parse_json_response(
                content.decode("utf-8"),
                allow_repair=False,
            ).value
        except (UnicodeDecodeError, ValueError) as exc:
            raise ModelResponseError("rerank provider returned malformed JSON") from exc
        if not isinstance(value, Mapping):
            raise ModelResponseError("rerank provider response must be a JSON object")
        return value

    async def _read_response(self, response: httpx.Response) -> bytes:
        maximum = self.config.route.max_response_bytes
        chunks: list[bytes] = []
        total = 0
        async for chunk in response.aiter_bytes():
            if not isinstance(chunk, bytes):
                raise ModelResponseError("rerank response yielded non-byte content")
            total += len(chunk)
            if total > maximum:
                raise ModelResponseError("rerank response exceeds the configured byte bound")
            chunks.append(chunk)
        return b"".join(chunks)

    def _client(self) -> httpx.AsyncClient:
        loop = asyncio.get_running_loop()
        with self._clients_lock:
            client = self._clients.get(loop)
            if client is not None:
                return client
            concurrency = self.config.route.max_concurrent
            client = httpx.AsyncClient(
                follow_redirects=False,
                limits=httpx.Limits(
                    max_connections=concurrency,
                    max_keepalive_connections=concurrency,
                ),
                timeout=httpx.Timeout(float(self.config.route.timeout_seconds)),
            )
            self._clients[loop] = client
            return client

    @staticmethod
    def _scores(response: Mapping[str, object], *, expected: int) -> tuple[float, ...]:
        results = response.get("results")
        if not isinstance(results, Sequence) or isinstance(results, str):
            raise ModelResponseError("rerank provider response has no results array")
        if len(results) != expected:
            raise ModelResponseError("rerank provider returned an unexpected result count")

        scores: list[float | None] = [None] * expected
        for position, item in enumerate(results):
            if not isinstance(item, Mapping):
                raise ModelResponseError(f"rerank result[{position}] must be an object")
            index = item.get("index")
            if isinstance(index, bool) or not isinstance(index, int):
                raise ModelResponseError(f"rerank result[{position}] index must be an integer")
            if not 0 <= index < expected:
                raise ModelResponseError(f"rerank result[{position}] index is out of bounds")
            if scores[index] is not None:
                raise ModelResponseError("rerank provider returned a duplicate document index")
            raw_score = item.get("relevance_score")
            if isinstance(raw_score, bool) or not isinstance(raw_score, int | float):
                raise ModelResponseError(
                    f"rerank result[{position}] relevance_score must be numeric"
                )
            score = float(raw_score)
            if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                raise ModelResponseError(
                    f"rerank result[{position}] relevance_score must be finite and between zero and one"
                )
            scores[index] = score
        if any(score is None for score in scores):
            raise ModelResponseError("rerank provider omitted a document score")
        return tuple(score for score in scores if score is not None)


class _HTTPStatusError(RuntimeError):
    def __init__(self, message: str, *, status_code: int, retry_after: float | None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retry_after = retry_after


def _error_message(content: bytes, status_code: int) -> str:
    fallback = f"rerank provider request failed with HTTP {status_code}"
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


def _retry_after(headers: object) -> float | None:
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


def _is_remote_url(url: str) -> bool:
    hostname = str(urlsplit(url).hostname or "").casefold()
    if hostname == "localhost":
        return False
    try:
        return not ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return True


__all__ = ["OpenAICompatibleRerankProvider"]
