"""供应商无关的向量值对象、Provider 协议和统一异步运行层。"""

from __future__ import annotations

import asyncio
import logging
import math
import threading
import time
import weakref
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

from ModelClient.config import EmbeddingModelConfig
from ModelClient.contracts import ModelClientError, ModelResponseError
from ModelClient.retry import normalize_provider_error, retry_delay

logger = logging.getLogger(__name__)

_ASYNC_EMBED_SEMAPHORES: weakref.WeakKeyDictionary[
    asyncio.AbstractEventLoop,
    dict[int, asyncio.Semaphore],
] = weakref.WeakKeyDictionary()
_ASYNC_EMBED_LOCK = threading.Lock()
_SLOW_CALL_SECONDS = 3.0


@dataclass(frozen=True)
class EmbeddingVector:
    """经过有限值、非零范数和 L2 归一化的稠密向量。"""

    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if isinstance(self.values, list):
            values = tuple(self.values)
        elif isinstance(self.values, tuple):
            values = self.values
        else:
            raise TypeError("embedding vector values must be a tuple or list")
        if not values:
            raise ValueError("embedding vector must not be empty")
        normalized: list[float] = []
        for value in values:
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise TypeError("embedding vector values must be numeric")
            number = float(value)
            if not math.isfinite(number):
                raise ValueError("embedding vector values must be finite")
            normalized.append(number)
        norm = math.sqrt(sum(value * value for value in normalized))
        if norm == 0:
            raise ValueError("embedding vector must have a non-zero norm")
        object.__setattr__(self, "values", tuple(value / norm for value in normalized))

    @property
    def dimension(self) -> int:
        return len(self.values)


class EmbeddingProvider(Protocol):
    """一种具体向量协议的单次异步调用边界。"""

    @property
    def provider_name(self) -> str: ...

    @property
    def model(self) -> str: ...

    @property
    def is_remote(self) -> bool: ...

    async def embed(self, text: str, *, is_query: bool) -> EmbeddingVector:
        """执行一次请求；不得在 Provider 内重试或取得全局并发槽。"""

        ...


class Embedder(Protocol):
    """供检索领域使用的查询与文档向量接口。"""

    @property
    def provider_name(self) -> str: ...

    @property
    def model(self) -> str: ...

    @property
    def is_remote(self) -> bool: ...

    async def embed_query(self, text: str) -> EmbeddingVector: ...

    async def embed_documents(self, texts: Sequence[str]) -> tuple[EmbeddingVector, ...]: ...


class EmbeddingClient:
    """统一执行输入校验、批量调度、共享并发限制和有界重试。"""

    def __init__(
        self,
        config: EmbeddingModelConfig,
        provider: EmbeddingProvider,
        *,
        async_sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if not isinstance(config, EmbeddingModelConfig):
            raise TypeError("config must be EmbeddingModelConfig")
        self.config = config
        self.provider = provider
        self._async_sleep = async_sleep
        self._monotonic = monotonic

    @property
    def provider_name(self) -> str:
        return self.provider.provider_name

    @property
    def model(self) -> str:
        return self.provider.model

    @property
    def is_remote(self) -> bool:
        return self.provider.is_remote

    async def embed_query(self, text: str) -> EmbeddingVector:
        normalized = self._text(text, "embedding query")
        return await self._embed_one(normalized, is_query=True)

    async def embed_documents(self, texts: Sequence[str]) -> tuple[EmbeddingVector, ...]:
        if isinstance(texts, str) or not isinstance(texts, Sequence):
            raise TypeError("embedding documents must be a sequence of strings")
        normalized = tuple(self._text(text, "embedding document") for text in texts)
        if not normalized:
            return ()

        result: list[EmbeddingVector] = []
        size = self.config.max_batch_size
        for offset in range(0, len(normalized), size):
            batch = normalized[offset : offset + size]
            result.extend(
                await asyncio.gather(
                    *(self._embed_one(text, is_query=False) for text in batch)
                )
            )
        return tuple(result)

    async def _embed_one(self, text: str, *, is_query: bool) -> EmbeddingVector:
        operation = "embed_query" if is_query else "embed_document"
        for attempt in range(self.config.route.max_retries + 1):
            failure: ModelClientError | None = None
            source_error: Exception | None = None
            semaphore = _async_embed_semaphore(self.config.route.max_concurrent)
            wait_started = self._monotonic()
            await semaphore.acquire()
            wait_seconds = self._monotonic() - wait_started
            started = self._monotonic()
            try:
                vector = await asyncio.wait_for(
                    self.provider.embed(text, is_query=is_query),
                    timeout=self.config.route.timeout_seconds,
                )
                self._validate_vector(vector)
                return vector
            except Exception as exc:
                failure = normalize_provider_error(exc)
                source_error = exc
            finally:
                duration_seconds = self._monotonic() - started
                semaphore.release()
                self._observe_call(
                    operation=operation,
                    attempt=attempt,
                    wait_seconds=wait_seconds,
                    duration_seconds=duration_seconds,
                )

            if failure is None or source_error is None:  # pragma: no cover
                raise AssertionError("embedding provider failure was not captured")
            if failure.retryable and attempt < self.config.route.max_retries:
                await self._async_sleep(self._retry_delay(attempt, failure))
                continue
            raise failure from source_error
        raise AssertionError("embedding retry loop exhausted without a result")  # pragma: no cover

    def _validate_vector(self, vector: object) -> None:
        if not isinstance(vector, EmbeddingVector):
            raise ModelResponseError("embedding provider returned an invalid vector")
        if vector.dimension != self.config.dimension:
            raise ModelResponseError("embedding provider returned an unexpected vector dimension")

    def _text(self, value: object, label: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{label} must be non-empty text")
        normalized = value.strip()
        if len(normalized) > self.config.max_input_chars:
            raise ValueError(f"{label} exceeds the configured character bound")
        return normalized

    def _retry_delay(self, attempt: int, failure: ModelClientError) -> float:
        return retry_delay(
            attempt,
            base_delay=self.config.route.retry_base_delay_seconds,
            max_delay=self.config.route.retry_max_delay_seconds,
            error=failure,
        )

    def _observe_call(
        self,
        *,
        operation: str,
        attempt: int,
        wait_seconds: float,
        duration_seconds: float,
    ) -> None:
        log = logger.warning if duration_seconds >= _SLOW_CALL_SECONDS else logger.debug
        log(
            "%s provider=%s model=%s attempt=%d wait_ms=%.2f duration_ms=%.2f",
            operation,
            self.provider_name,
            self.model,
            attempt + 1,
            wait_seconds * 1000,
            duration_seconds * 1000,
        )


def _async_embed_semaphore(limit: int) -> asyncio.Semaphore:
    """同一事件循环和并发上限共享槽位，避免多实例绕过总并发限制。"""

    loop = asyncio.get_running_loop()
    with _ASYNC_EMBED_LOCK:
        semaphores_by_limit = _ASYNC_EMBED_SEMAPHORES.setdefault(loop, {})
        semaphore = semaphores_by_limit.get(limit)
        if semaphore is None:
            semaphore = asyncio.Semaphore(limit)
            semaphores_by_limit[limit] = semaphore
        return semaphore


__all__ = [
    "Embedder",
    "EmbeddingClient",
    "EmbeddingProvider",
    "EmbeddingVector",
]
