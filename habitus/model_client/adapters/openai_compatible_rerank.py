"""OpenAI-compatible 文本重排 REST 协议适配器。"""

from __future__ import annotations

import asyncio
import math
import threading
import weakref
from collections.abc import Mapping, Sequence

import httpx

from habitus.foundation.network import OutboundSettings
from habitus.foundation.validation import is_int, is_number
from habitus.model_client.adapters._http import is_remote_url, post_json_object
from habitus.model_client.config import RerankModelConfig
from habitus.model_client.contracts import ModelConfigurationError, ModelResponseError
from habitus.model_client.http_clients import async_http_client

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

    def __init__(self, config: RerankModelConfig, *, api_key: str, outbound: OutboundSettings | None = None) -> None:
        if config.route.adapter != "openai_compatible_rerank":
            raise ModelConfigurationError("OpenAICompatibleRerankProvider requires adapter='openai_compatible_rerank'")
        if not config.route.base_url:
            raise ModelConfigurationError("openai_compatible_rerank adapter requires an explicit base_url")
        if not api_key.strip():
            raise ModelConfigurationError("openai_compatible_rerank adapter requires an API key")
        overlap = _RESERVED_REQUEST_FIELDS & set(config.route.extra_body)
        if overlap:
            raise ModelConfigurationError(
                f"openai_compatible_rerank extra_body cannot override request fields: {sorted(overlap)}"
            )

        self.config = config
        self.provider_name = config.route.provider
        self.model = config.route.model
        self.is_remote = is_remote_url(config.route.base_url)
        self._api_key = api_key.strip()
        self._endpoint = f"{config.route.base_url}/reranks"
        self._outbound = OutboundSettings() if outbound is None else outbound
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
        return await post_json_object(
            self._client(),
            self._endpoint,
            headers=headers,
            payload=payload,
            max_response_bytes=self.config.route.max_response_bytes,
            label="rerank",
        )

    def _client(self) -> httpx.AsyncClient:
        loop = asyncio.get_running_loop()
        with self._clients_lock:
            client = self._clients.get(loop)
            if client is not None:
                return client
            client = async_http_client(self.config.route, self._outbound)
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
            if not is_int(index):
                raise ModelResponseError(f"rerank result[{position}] index must be an integer")
            if not 0 <= index < expected:
                raise ModelResponseError(f"rerank result[{position}] index is out of bounds")
            if scores[index] is not None:
                raise ModelResponseError("rerank provider returned a duplicate document index")
            raw_score = item.get("relevance_score")
            if not is_number(raw_score):
                raise ModelResponseError(f"rerank result[{position}] relevance_score must be numeric")
            score = float(raw_score)
            if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                raise ModelResponseError(
                    f"rerank result[{position}] relevance_score must be finite and between zero and one"
                )
            scores[index] = score
        if any(score is None for score in scores):
            raise ModelResponseError("rerank provider omitted a document score")
        return tuple(score for score in scores if score is not None)


__all__ = ["OpenAICompatibleRerankProvider"]
