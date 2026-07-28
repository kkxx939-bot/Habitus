"""在一个已构造 ChatProvider 上执行并发控制、重试和流式调用。"""

from __future__ import annotations

import asyncio
import threading
import time
import weakref
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from dataclasses import replace

from foundation.observability import NullObserver, ObservationEvent, ObservationStatus, Observer
from ModelClient.config import ChatModelConfig
from ModelClient.contracts import (
    ChatMessage,
    ChatProvider,
    ChatRequest,
    ModelClientError,
    ModelResponse,
    ModelResponseError,
    ModelStreamEvent,
)
from ModelClient.retry import normalize_provider_error, retry_delay


class ChatClient:
    """供应商无关的对话调用入口；Provider 的创建只属于工厂。"""

    def __init__(
        self,
        config: ChatModelConfig,
        provider: ChatProvider,
        *,
        sleep: Callable[[float], None] = time.sleep,
        async_sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        observer: Observer | None = None,
    ) -> None:
        if not isinstance(config, ChatModelConfig):
            raise TypeError("config must be ChatModelConfig")
        self.config = config
        self.provider = provider
        self._sleep = sleep
        self._async_sleep = async_sleep
        self._sync_slots = threading.BoundedSemaphore(config.route.max_concurrent)
        self._async_slots: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Semaphore] = (
            weakref.WeakKeyDictionary()
        )
        self._async_slots_lock = threading.Lock()
        self._observer = observer or NullObserver()

    @property
    def provider_name(self) -> str:
        return self.provider.provider_name

    @property
    def model(self) -> str:
        return self.provider.model

    @property
    def is_remote(self) -> bool:
        return self.provider.is_remote

    def complete(self, request: ChatRequest | str) -> ModelResponse:
        normalized = self._request(request)
        with self._sync_slots:
            for attempt in range(self.config.route.max_retries + 1):
                try:
                    return self._require_response(self.provider.complete(normalized))
                except Exception as exc:
                    failure = normalize_provider_error(exc)
                    if failure.retryable and attempt < self.config.route.max_retries:
                        self._sleep(self._retry_delay(attempt, failure))
                        continue
                    raise failure from exc
        raise AssertionError("chat retry loop exhausted without a result")  # pragma: no cover

    async def complete_async(self, request: ChatRequest | str) -> ModelResponse:
        normalized = self._request(request)
        started = time.monotonic()
        async with self._async_slot():
            for attempt in range(self.config.route.max_retries + 1):
                try:
                    response = self._require_response(await self.provider.complete_async(normalized))
                    self._observe("complete_async", ObservationStatus.SUCCESS, started, response=response)
                    return response
                except Exception as exc:
                    failure = normalize_provider_error(exc)
                    if failure.retryable and attempt < self.config.route.max_retries:
                        await self._async_sleep(self._retry_delay(attempt, failure))
                        continue
                    self._observe("complete_async", ObservationStatus.FAILURE, started, error=failure)
                    raise failure from exc
        raise AssertionError("chat retry loop exhausted without a result")  # pragma: no cover

    def stream(self, request: ChatRequest | str) -> Iterator[ModelStreamEvent]:
        normalized = self._request(request)
        return self._stream_sync(normalized)

    async def stream_async(self, request: ChatRequest | str) -> AsyncIterator[ModelStreamEvent]:
        normalized = self._request(request)
        async with self._async_slot():
            for attempt in range(self.config.route.max_retries + 1):
                emitted = False
                try:
                    async for event in self.provider.stream_async(normalized):
                        if not isinstance(event, ModelStreamEvent):
                            raise ModelResponseError("provider returned an invalid stream event")
                        emitted = True
                        yield event
                    if not emitted:
                        raise ModelResponseError("provider returned an empty stream")
                    return
                except Exception as exc:
                    failure = normalize_provider_error(exc)
                    if emitted:
                        raise failure from exc
                    if failure.retryable and attempt < self.config.route.max_retries:
                        await self._async_sleep(self._retry_delay(attempt, failure))
                        continue
                    raise failure from exc

    def health_check(self) -> dict[str, object]:
        try:
            result = self.provider.health_check()
            return dict(result)
        except Exception as exc:
            failure = normalize_provider_error(exc)
            return {
                "ok": False,
                "provider": self.provider.provider_name,
                "model": self.provider.model,
                "error_code": failure.code,
            }

    async def aclose(self) -> None:
        """释放 Chat Provider 的同步和异步传输资源。"""

        await self.provider.aclose()

    def _stream_sync(self, request: ChatRequest) -> Iterator[ModelStreamEvent]:
        with self._sync_slots:
            for attempt in range(self.config.route.max_retries + 1):
                emitted = False
                try:
                    for event in self.provider.stream(request):
                        if not isinstance(event, ModelStreamEvent):
                            raise ModelResponseError("provider returned an invalid stream event")
                        emitted = True
                        yield event
                    if not emitted:
                        raise ModelResponseError("provider returned an empty stream")
                    return
                except Exception as exc:
                    failure = normalize_provider_error(exc)
                    if emitted:
                        raise failure from exc
                    if failure.retryable and attempt < self.config.route.max_retries:
                        self._sleep(self._retry_delay(attempt, failure))
                        continue
                    raise failure from exc

    def _request(self, request: ChatRequest | str) -> ChatRequest:
        if isinstance(request, str):
            if not request.strip():
                raise ValueError("model prompt cannot be empty")
            request = ChatRequest(messages=(ChatMessage(role="user", content=request),))
        if not isinstance(request, ChatRequest):
            raise TypeError("model request must be ChatRequest or non-empty text")
        if request.max_output_tokens is None and self.config.max_output_tokens is not None:
            return replace(request, max_output_tokens=self.config.max_output_tokens)
        return request

    @staticmethod
    def _require_response(response: object) -> ModelResponse:
        if not isinstance(response, ModelResponse):
            raise ModelResponseError("provider returned an invalid normalized response")
        return response

    def _retry_delay(self, attempt: int, failure: ModelClientError) -> float:
        return retry_delay(
            attempt,
            base_delay=self.config.route.retry_base_delay_seconds,
            max_delay=self.config.route.retry_max_delay_seconds,
            error=failure,
        )

    def _async_slot(self) -> asyncio.Semaphore:
        loop = asyncio.get_running_loop()
        with self._async_slots_lock:
            semaphore = self._async_slots.get(loop)
            if semaphore is None:
                semaphore = asyncio.Semaphore(self.config.route.max_concurrent)
                self._async_slots[loop] = semaphore
            return semaphore

    def _observe(
        self,
        operation: str,
        status: ObservationStatus,
        started: float,
        *,
        response: ModelResponse | None = None,
        error: ModelClientError | None = None,
    ) -> None:
        attributes: dict[str, str | int | float | bool] = {
            "provider": self.provider_name,
            "model": self.model,
        }
        if response is not None:
            attributes.update(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                total_tokens=response.usage.total_tokens,
            )
        if error is not None:
            attributes["error_code"] = error.code
        self._observer.record(
            ObservationEvent(
                category="model",
                operation=operation,
                status=status,
                duration_seconds=max(0.0, time.monotonic() - started),
                attributes=attributes,
            )
        )


__all__ = ["ChatClient"]
