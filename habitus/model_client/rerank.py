"""供应商无关的文本重排 Provider 协议和统一异步运行层。"""

from __future__ import annotations

import asyncio
import math
import threading
import time
import weakref
from collections.abc import Awaitable, Callable, Sequence
from typing import Protocol

from habitus.foundation.observability import NullObserver, ObservationEvent, ObservationStatus, Observer
from habitus.model_client.config import RerankModelConfig
from habitus.model_client.contracts import ModelClientError, ModelResponseError
from habitus.model_client.retry import normalize_provider_error, retry_delay

_ASYNC_RERANK_SEMAPHORES: weakref.WeakKeyDictionary[
    asyncio.AbstractEventLoop,
    dict[int, asyncio.Semaphore],
] = weakref.WeakKeyDictionary()
_ASYNC_RERANK_LOCK = threading.Lock()


class RerankProvider(Protocol):
    """一种具体重排协议的单次异步调用边界。"""

    @property
    def provider_name(self) -> str: ...

    @property
    def model(self) -> str: ...

    @property
    def is_remote(self) -> bool: ...

    async def rerank(self, query: str, documents: Sequence[str]) -> tuple[float, ...]:
        """执行一次请求；不得在 Provider 内重试或取得全局并发槽。"""

        ...

    async def aclose(self) -> None: ...


class Reranker(Protocol):
    """按照输入原顺序返回相关性分数的异步重排接口。"""

    @property
    def provider_name(self) -> str: ...

    @property
    def model(self) -> str: ...

    @property
    def is_remote(self) -> bool: ...

    async def rerank(self, query: str, documents: Sequence[str]) -> tuple[float, ...]: ...

    async def aclose(self) -> None: ...


class RerankClient:
    """统一执行输入校验、共享并发限制、超时和有界重试。"""

    def __init__(
        self,
        config: RerankModelConfig,
        provider: RerankProvider,
        *,
        async_sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        if not isinstance(config, RerankModelConfig):
            raise TypeError("config must be RerankModelConfig")
        self.config = config
        self.provider = provider
        self._async_sleep = async_sleep

    @property
    def provider_name(self) -> str:
        return self.provider.provider_name

    @property
    def model(self) -> str:
        return self.provider.model

    @property
    def is_remote(self) -> bool:
        return self.provider.is_remote

    async def rerank(
        self,
        query: str,
        documents: Sequence[str],
    ) -> tuple[float, ...]:
        normalized_query = self._text(
            query,
            "rerank query",
            maximum=self.config.max_query_chars,
        )
        if isinstance(documents, str) or not isinstance(documents, Sequence):
            raise TypeError("rerank documents must be a sequence of strings")
        normalized_documents = tuple(
            self._text(
                document,
                f"rerank document[{index}]",
                maximum=self.config.max_document_chars,
            )
            for index, document in enumerate(documents)
        )
        if not normalized_documents:
            return ()
        if len(normalized_documents) > self.config.max_documents:
            raise ValueError("rerank documents exceed the configured count bound")

        for attempt in range(self.config.route.max_retries + 1):
            failure: ModelClientError | None = None
            source_error: Exception | None = None
            semaphore = _async_rerank_semaphore(self.config.route.max_concurrent)
            await semaphore.acquire()
            try:
                scores = await asyncio.wait_for(
                    self.provider.rerank(normalized_query, normalized_documents),
                    timeout=self.config.route.timeout_seconds,
                )
                return self._scores(scores, expected=len(normalized_documents))
            except Exception as exc:
                failure = normalize_provider_error(exc)
                source_error = exc
            finally:
                semaphore.release()

            if failure is None or source_error is None:  # pragma: no cover
                raise AssertionError("rerank provider failure was not captured")
            if failure.retryable and attempt < self.config.route.max_retries:
                await self._async_sleep(self._retry_delay(attempt, failure))
                continue
            raise failure from source_error
        raise AssertionError("rerank retry loop exhausted without a result")  # pragma: no cover

    async def aclose(self) -> None:
        """释放 Provider 持有的连接池；重复调用由 Provider 保证幂等。"""

        await self.provider.aclose()

    @staticmethod
    def _scores(source: object, *, expected: int) -> tuple[float, ...]:
        if not isinstance(source, Sequence) or isinstance(source, str):
            raise ModelResponseError("rerank provider returned invalid scores")
        if len(source) != expected:
            raise ModelResponseError("rerank provider returned an unexpected score count")
        scores: list[float] = []
        for value in source:
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise ModelResponseError("rerank provider returned a non-numeric score")
            score = float(value)
            if not math.isfinite(score):
                raise ModelResponseError("rerank provider returned a non-finite score")
            scores.append(score)
        return tuple(scores)

    @staticmethod
    def _text(value: object, label: str, *, maximum: int) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{label} must be non-empty text")
        normalized = value.strip()
        if len(normalized) > maximum:
            raise ValueError(f"{label} exceeds the configured character bound")
        return normalized

    def _retry_delay(self, attempt: int, failure: ModelClientError) -> float:
        return retry_delay(
            attempt,
            base_delay=self.config.route.retry_base_delay_seconds,
            max_delay=self.config.route.retry_max_delay_seconds,
            error=failure,
        )


class ObservedReranker:
    """只包装统一调用边界，不改变供应商实现和排序语义。"""

    def __init__(self, reranker: Reranker, *, observer: Observer | None = None) -> None:
        self.reranker = reranker
        self.observer = observer or NullObserver()

    @property
    def provider_name(self) -> str:
        return self.reranker.provider_name

    @property
    def model(self) -> str:
        return self.reranker.model

    @property
    def is_remote(self) -> bool:
        return self.reranker.is_remote

    async def rerank(self, query: str, documents: Sequence[str]) -> tuple[float, ...]:
        started = time.monotonic()
        try:
            result = await self.reranker.rerank(query, documents)
        except Exception as exc:
            self._observe(ObservationStatus.FAILURE, started, error_type=type(exc).__name__)
            raise
        self._observe(ObservationStatus.SUCCESS, started)
        return result

    async def aclose(self) -> None:
        await self.reranker.aclose()

    def _observe(
        self,
        status: ObservationStatus,
        started: float,
        *,
        error_type: str | None = None,
    ) -> None:
        attributes = {"provider": self.provider_name, "model": self.model}
        if error_type is not None:
            attributes["error_type"] = error_type
        try:
            self.observer.record(
                ObservationEvent(
                    category="model",
                    operation="rerank",
                    status=status,
                    duration_seconds=max(0.0, time.monotonic() - started),
                    attributes=attributes,
                )
            )
        except Exception:
            pass


def _async_rerank_semaphore(limit: int) -> asyncio.Semaphore:
    """同一事件循环和并发上限共享槽位，避免多实例绕过总并发限制。"""

    loop = asyncio.get_running_loop()
    with _ASYNC_RERANK_LOCK:
        semaphores_by_limit = _ASYNC_RERANK_SEMAPHORES.setdefault(loop, {})
        semaphore = semaphores_by_limit.get(limit)
        if semaphore is None:
            semaphore = asyncio.Semaphore(limit)
            semaphores_by_limit[limit] = semaphore
        return semaphore


__all__ = [
    "ObservedReranker",
    "RerankClient",
    "Reranker",
    "RerankProvider",
]
