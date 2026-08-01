"""火山方舟图文向量 REST 协议适配器。"""

from __future__ import annotations

import asyncio
import json
import threading
import weakref
from collections.abc import Mapping, Sequence

import httpx

from ModelClient.config import EmbeddingModelConfig
from ModelClient.contracts import (
    ModelConfigurationError,
    ModelResponseError,
)
from ModelClient.embedding import EmbeddingVector
from ModelClient.json_parser import parse_json_response

_RESERVED_REQUEST_FIELDS = {
    "api_key",
    "base_url",
    "encoding_format",
    "extra_body",
    "extra_headers",
    "input",
    "model",
    "timeout",
}


class ArkMultimodalEmbeddingProvider:
    """完成 Ark `/embeddings/multimodal` 的单次请求和响应转换。"""

    def __init__(self, config: EmbeddingModelConfig, *, api_key: str) -> None:
        if not isinstance(config, EmbeddingModelConfig):
            raise TypeError("config must be EmbeddingModelConfig")
        if config.route.adapter != "ark_multimodal":
            raise ModelConfigurationError(
                "ArkMultimodalEmbeddingProvider requires adapter='ark_multimodal'"
            )
        if config.input_mode != "multimodal":
            raise ModelConfigurationError("ark_multimodal adapter requires input_mode='multimodal'")
        if not config.route.base_url:
            raise ModelConfigurationError("ark_multimodal adapter requires an explicit base_url")
        if not isinstance(api_key, str) or not api_key.strip():
            raise ModelConfigurationError("ark_multimodal adapter requires an API key")
        self._validate_parameters(config)
        self.config = config
        self.provider_name = config.route.provider
        self.model = config.route.model
        self.is_remote = True
        self._api_key = api_key.strip()
        self._endpoint = f"{config.route.base_url}/embeddings/multimodal"
        self._clients: weakref.WeakKeyDictionary[
            asyncio.AbstractEventLoop,
            httpx.AsyncClient,
        ] = weakref.WeakKeyDictionary()
        self._cache_lock = threading.Lock()
        self._closed = False

    async def embed(self, text: str, *, is_query: bool) -> EmbeddingVector:
        if self._closed:
            raise ModelConfigurationError("embedding provider is closed")
        parameters = self.config.query_parameters if is_query else self.config.document_parameters
        payload: dict[str, object] = {
            **dict(self.config.route.extra_body),
            **dict(parameters),
            "encoding_format": "float",
            "input": [{"type": "text", "text": text}],
            "model": self.model,
        }
        response_payload = await self._post(payload)
        return self._vector(response_payload)

    async def aclose(self) -> None:
        """幂等关闭所有事件循环对应的异步 HTTP 连接池。"""

        with self._cache_lock:
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
        client = self._client()
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            **dict(self.config.route.extra_headers),
        }
        async with client.stream("POST", self._endpoint, headers=headers, json=dict(payload)) as response:
            content = await self._read_response(response)
            status_code = getattr(response, "status_code", None)
            if not isinstance(status_code, int):
                raise ModelResponseError("Ark embedding response has no HTTP status")
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
            raise ModelResponseError("Ark embedding response is not valid JSON") from exc
        if not isinstance(value, Mapping):
            raise ModelResponseError("Ark embedding response must be a JSON object")
        return value

    async def _read_response(self, response: httpx.Response) -> bytes:
        maximum = self.config.route.max_response_bytes
        chunks: list[bytes] = []
        total = 0
        async for chunk in response.aiter_bytes():
            if not isinstance(chunk, bytes):
                raise ModelResponseError("Ark embedding response yielded non-byte content")
            total += len(chunk)
            if total > maximum:
                raise ModelResponseError("Ark embedding response exceeds the configured byte bound")
            chunks.append(chunk)
        return b"".join(chunks)

    def _client(self) -> httpx.AsyncClient:
        loop = asyncio.get_running_loop()
        with self._cache_lock:
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

    def _vector(self, response: Mapping[str, object]) -> EmbeddingVector:
        data = response.get("data")
        if isinstance(data, Sequence) and not isinstance(data, str):
            if len(data) != 1:
                raise ModelResponseError("Ark multimodal embedding returned an unexpected item count")
            data = data[0]
        raw_vector = _field(data, "embedding")
        if not isinstance(raw_vector, Sequence) or isinstance(raw_vector, str):
            raise ModelResponseError("Ark multimodal embedding returned a malformed vector")
        values = tuple(raw_vector)
        if len(values) < self.config.dimension:
            raise ModelResponseError("Ark multimodal embedding vector is shorter than configured dimension")
        try:
            return EmbeddingVector(values[: self.config.dimension])
        except (TypeError, ValueError) as exc:
            raise ModelResponseError("Ark multimodal embedding returned invalid numeric values") from exc

    @staticmethod
    def _validate_parameters(config: EmbeddingModelConfig) -> None:
        for label, values in (
            ("extra_body", config.route.extra_body),
            ("query_parameters", config.query_parameters),
            ("document_parameters", config.document_parameters),
        ):
            overlap = _RESERVED_REQUEST_FIELDS & set(values)
            if overlap:
                raise ModelConfigurationError(
                    f"ark_multimodal {label} cannot override request fields: {sorted(overlap)}"
                )

class _HTTPStatusError(RuntimeError):
    def __init__(self, message: str, *, status_code: int, retry_after: float | None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retry_after = retry_after


def _field(source: object, name: str) -> object:
    if isinstance(source, Mapping):
        return source.get(name)
    return getattr(source, name, None)


def _error_message(content: bytes, status_code: int) -> str:
    try:
        payload = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return f"Ark embedding request failed with HTTP {status_code}"
    if isinstance(payload, Mapping):
        error = payload.get("error")
        message = _field(error, "message")
        if isinstance(message, str) and message.strip():
            return message.strip()[:1024]
    return f"Ark embedding request failed with HTTP {status_code}"


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


__all__ = ["ArkMultimodalEmbeddingProvider"]
