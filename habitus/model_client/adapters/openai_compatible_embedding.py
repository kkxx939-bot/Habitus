"""OpenAI 兼容 Embeddings REST 协议适配器（``POST {base_url}/embeddings``）。

按 OpenAI 官方 OpenAPI 规格（openai/openai-openapi 2.3.0）的 ``CreateEmbeddingRequest`` / ``CreateEmbeddingResponse``：
请求体 ``{"model", "input", "encoding_format": "float"}``，响应 ``data[].embedding`` 为浮点数组。一次只发一条文本；
不按厂商或模型名特判任何参数，厂商需要的额外字段（例如 ``dimensions``）由用户写进 ``extra_body``。
"""

from __future__ import annotations

import asyncio
import threading
import weakref
from collections.abc import Mapping, Sequence

import httpx

from habitus.foundation.embedding import EmbeddingVector
from habitus.foundation.network import OutboundSettings
from habitus.model_client.adapters._http import is_remote_url, post_json_object
from habitus.model_client.config import EmbeddingModelConfig
from habitus.model_client.contracts import ModelConfigurationError, ModelResponseError
from habitus.model_client.http_clients import async_http_client

ADAPTER_NAME = "openai_compatible_embedding"

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


class OpenAICompatibleEmbeddingProvider:
    """执行一次 ``/embeddings`` 请求，把唯一一条结果转成配置维度的向量。"""

    def __init__(
        self, config: EmbeddingModelConfig, *, api_key: str = "", outbound: OutboundSettings | None = None
    ) -> None:
        if config.route.adapter != ADAPTER_NAME:
            raise ModelConfigurationError(f"OpenAICompatibleEmbeddingProvider requires adapter='{ADAPTER_NAME}'")
        if not config.route.base_url:
            raise ModelConfigurationError(f"{ADAPTER_NAME} adapter requires an explicit base_url")
        _validate_parameters(config)
        self.config = config
        self.provider_name = config.route.provider
        self.model = config.route.model
        self.is_remote = is_remote_url(config.route.base_url)
        # 自部署服务（Ollama、TEI、vLLM 默认）可以不要密钥：没配凭据就不发 Authorization
        self._api_key = api_key.strip()
        self._endpoint = f"{config.route.base_url}/embeddings"
        self._outbound = OutboundSettings() if outbound is None else outbound
        self._clients: weakref.WeakKeyDictionary[
            asyncio.AbstractEventLoop,
            httpx.AsyncClient,
        ] = weakref.WeakKeyDictionary()
        self._clients_lock = threading.Lock()
        self._closed = False

    async def embed(self, text: str, *, is_query: bool) -> EmbeddingVector:
        if self._closed:
            raise ModelConfigurationError("embedding provider is closed")
        parameters = self.config.query_parameters if is_query else self.config.document_parameters
        payload: dict[str, object] = {
            **dict(self.config.route.extra_body),
            **dict(parameters),
            "model": self.model,
            "input": text,
            "encoding_format": "float",
        }
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            **dict(self.config.route.extra_headers),
        }
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        response = await post_json_object(
            self._client(),
            self._endpoint,
            headers=headers,
            payload=payload,
            max_response_bytes=self.config.route.max_response_bytes,
            label="embedding",
        )
        return self._vector(response)

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

    def _client(self) -> httpx.AsyncClient:
        loop = asyncio.get_running_loop()
        with self._clients_lock:
            client = self._clients.get(loop)
            if client is not None:
                return client
            client = async_http_client(self.config.route, self._outbound)
            self._clients[loop] = client
            return client

    def _vector(self, response: Mapping[str, object]) -> EmbeddingVector:
        data = response.get("data")
        if not isinstance(data, Sequence) or isinstance(data, str) or len(data) != 1:
            raise ModelResponseError("embedding response must contain exactly one data item")
        item = data[0]
        raw_vector = item.get("embedding") if isinstance(item, Mapping) else None
        if not isinstance(raw_vector, Sequence) or isinstance(raw_vector, str):
            raise ModelResponseError("embedding response item has no float embedding array")
        expected = self.config.dimension
        if len(raw_vector) != expected:
            # 不截取：只有按 Matryoshka 训练的模型截短后仍可用，一般模型截掉一段会悄悄让召回变差
            raise ModelResponseError(
                f"embedding service returned {len(raw_vector)} dimensions but the configuration expects "
                f"{expected}; set models.embedding.dimension to {len(raw_vector)}, or add "
                f'"dimensions": {expected} to models.embedding.route.extra_body if the service supports it'
            )
        try:
            return EmbeddingVector(tuple(raw_vector))
        except (TypeError, ValueError) as exc:
            raise ModelResponseError("embedding response contains invalid numeric values") from exc


def _validate_parameters(config: EmbeddingModelConfig) -> None:
    for label, values in (
        ("extra_body", config.route.extra_body),
        ("query_parameters", config.query_parameters),
        ("document_parameters", config.document_parameters),
    ):
        overlap = _RESERVED_REQUEST_FIELDS & set(values)
        if overlap:
            raise ModelConfigurationError(f"{ADAPTER_NAME} {label} cannot override request fields: {sorted(overlap)}")


__all__ = ["ADAPTER_NAME", "OpenAICompatibleEmbeddingProvider"]
