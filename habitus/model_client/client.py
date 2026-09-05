"""在一个已构造 ChatProvider 上执行并发控制、重试和流式调用。"""

from __future__ import annotations

import asyncio
import threading
import time
import weakref
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator, Mapping
from dataclasses import replace
from typing import cast

from habitus.foundation.observability import NullObserver, ObservationEvent, ObservationStatus, Observer
from habitus.model_client.config import ChatModelConfig
from habitus.model_client.contracts import (
    ChatCallContext,
    ChatMessage,
    ChatProvider,
    ChatRequest,
    ModelClientError,
    ModelInputTooLargeError,
    ModelResponse,
    ModelResponseError,
    ModelStreamEvent,
    PreparedChatRequest,
    TokenUsage,
)
from habitus.model_client.retry import normalize_provider_error, retry_delay


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

    def complete(
        self,
        request: ChatRequest | str,
        *,
        context: ChatCallContext | None = None,
    ) -> ModelResponse:
        call_context = self._context(context)
        prepared = self._prepare(request, stream=False, context=call_context)
        started = time.monotonic()
        with self._sync_slots:
            for attempt in range(self.config.route.max_retries + 1):
                try:
                    response = self._response_context(
                        self._require_response(self.provider.complete(prepared)),
                        call_context,
                    )
                    self._observe(
                        "complete",
                        ObservationStatus.SUCCESS,
                        started,
                        response=response,
                        retry_count=attempt,
                    )
                    return response
                except Exception as exc:
                    failure = normalize_provider_error(exc)
                    if failure.retryable and attempt < self.config.route.max_retries:
                        self._sleep(self._retry_delay(attempt, failure))
                        continue
                    self._observe(
                        "complete",
                        ObservationStatus.FAILURE,
                        started,
                        error=failure,
                        retry_count=attempt,
                    )
                    raise failure from exc
        raise AssertionError("chat retry loop exhausted without a result")  # pragma: no cover

    async def complete_async(
        self,
        request: ChatRequest | str,
        *,
        context: ChatCallContext | None = None,
    ) -> ModelResponse:
        call_context = self._context(context)
        prepared = self._prepare(request, stream=False, context=call_context)
        started = time.monotonic()
        async with self._async_slot():
            for attempt in range(self.config.route.max_retries + 1):
                try:
                    response = self._response_context(
                        self._require_response(await self.provider.complete_async(prepared)),
                        call_context,
                    )
                    self._observe(
                        "complete_async",
                        ObservationStatus.SUCCESS,
                        started,
                        response=response,
                        retry_count=attempt,
                    )
                    return response
                except Exception as exc:
                    failure = normalize_provider_error(exc)
                    if failure.retryable and attempt < self.config.route.max_retries:
                        await self._async_sleep(self._retry_delay(attempt, failure))
                        continue
                    self._observe(
                        "complete_async",
                        ObservationStatus.FAILURE,
                        started,
                        error=failure,
                        retry_count=attempt,
                    )
                    raise failure from exc
        raise AssertionError("chat retry loop exhausted without a result")  # pragma: no cover

    def stream(
        self,
        request: ChatRequest | str,
        *,
        context: ChatCallContext | None = None,
    ) -> Iterator[ModelStreamEvent]:
        prepared = self._prepare(request, stream=True, context=self._context(context))
        return self._stream_sync(prepared)

    async def stream_async(
        self,
        request: ChatRequest | str,
        *,
        context: ChatCallContext | None = None,
    ) -> AsyncIterator[ModelStreamEvent]:
        prepared = self._prepare(request, stream=True, context=self._context(context))
        started = time.monotonic()
        async with self._async_slot():
            for attempt in range(self.config.route.max_retries + 1):
                emitted = False
                usage: TokenUsage | None = None
                try:
                    async for event in self.provider.stream_async(prepared):
                        if not isinstance(event, ModelStreamEvent):
                            raise ModelResponseError("provider returned an invalid stream event")
                        emitted = True
                        if event.usage is not None:
                            usage = event.usage
                        yield event
                    if not emitted:
                        raise ModelResponseError("provider returned an empty stream")
                    self._observe(
                        "stream_async",
                        ObservationStatus.SUCCESS,
                        started,
                        usage=usage,
                        retry_count=attempt,
                    )
                    return
                except Exception as exc:
                    failure = normalize_provider_error(exc)
                    if emitted:
                        self._observe(
                            "stream_async",
                            ObservationStatus.FAILURE,
                            started,
                            error=failure,
                            retry_count=attempt,
                        )
                        raise failure from exc
                    if failure.retryable and attempt < self.config.route.max_retries:
                        await self._async_sleep(self._retry_delay(attempt, failure))
                        continue
                    self._observe(
                        "stream_async",
                        ObservationStatus.FAILURE,
                        started,
                        error=failure,
                        retry_count=attempt,
                    )
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

    async def health_check_async(self) -> dict[str, object]:
        """只调用 Provider 显式提供的异步探针，保证调用方可真正取消。"""

        probe = getattr(self.provider, "health_check_async", None)
        if not callable(probe):
            return {
                "ok": False,
                "provider": self.provider.provider_name,
                "model": self.provider.model,
                "error_code": "async_health_check_unsupported",
            }
        try:
            async_probe = cast(
                Callable[[], Awaitable[Mapping[str, object]]],
                probe,
            )
            result = await async_probe()
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

    def _stream_sync(self, request: PreparedChatRequest) -> Iterator[ModelStreamEvent]:
        started = time.monotonic()
        with self._sync_slots:
            for attempt in range(self.config.route.max_retries + 1):
                emitted = False
                usage: TokenUsage | None = None
                try:
                    for event in self.provider.stream(request):
                        if not isinstance(event, ModelStreamEvent):
                            raise ModelResponseError("provider returned an invalid stream event")
                        emitted = True
                        if event.usage is not None:
                            usage = event.usage
                        yield event
                    if not emitted:
                        raise ModelResponseError("provider returned an empty stream")
                    self._observe(
                        "stream",
                        ObservationStatus.SUCCESS,
                        started,
                        usage=usage,
                        retry_count=attempt,
                    )
                    return
                except Exception as exc:
                    failure = normalize_provider_error(exc)
                    if emitted:
                        self._observe(
                            "stream",
                            ObservationStatus.FAILURE,
                            started,
                            error=failure,
                            retry_count=attempt,
                        )
                        raise failure from exc
                    if failure.retryable and attempt < self.config.route.max_retries:
                        self._sleep(self._retry_delay(attempt, failure))
                        continue
                    self._observe(
                        "stream",
                        ObservationStatus.FAILURE,
                        started,
                        error=failure,
                        retry_count=attempt,
                    )
                    raise failure from exc

    def _request(self, request: ChatRequest | str) -> ChatRequest:
        if isinstance(request, str):
            if not request.strip():
                raise ValueError("model prompt cannot be empty")
            request = ChatRequest(messages=(ChatMessage(role="user", content=request),))
        if not isinstance(request, ChatRequest):
            raise TypeError("model request must be ChatRequest or non-empty text")
        if request.max_output_tokens is None and self.config.max_output_tokens is not None:
            request = replace(request, max_output_tokens=self.config.max_output_tokens)
        return request

    def _prepare(
        self,
        request: ChatRequest | str,
        *,
        stream: bool,
        context: ChatCallContext,
    ) -> PreparedChatRequest:
        normalized = self._request(request)
        prepared = self.provider.prepare(normalized, stream=stream)
        if not isinstance(prepared, PreparedChatRequest):
            raise ModelResponseError("provider returned an invalid prepared chat request")
        if prepared.request != normalized or prepared.stream is not stream:
            raise ModelResponseError("provider prepared a request for another logical call")
        if (
            prepared.estimated_input_tokens + prepared.reserved_output_tokens
            > self.config.context_window_tokens
        ):
            raise ModelInputTooLargeError(
                "estimated chat input and reserved output exceed the configured context window"
            )
        if (
            context.input_token_limit is not None
            and prepared.estimated_input_tokens > context.input_token_limit
        ):
            raise ModelInputTooLargeError("estimated chat input exceeds the call-specific input-token limit")
        return prepared

    @staticmethod
    def _context(context: ChatCallContext | None) -> ChatCallContext:
        if context is None:
            return ChatCallContext()
        if not isinstance(context, ChatCallContext):
            raise TypeError("context must be ChatCallContext or None")
        return context

    @staticmethod
    def _response_context(response: ModelResponse, context: ChatCallContext) -> ModelResponse:
        if response.prompt_version == context.prompt_version:
            return response
        return replace(response, prompt_version=context.prompt_version)

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
        usage: TokenUsage | None = None,
        error: ModelClientError | None = None,
        retry_count: int = 0,
    ) -> None:
        attributes: dict[str, str | int | float | bool] = {
            "provider": self.provider_name,
            "model": self.model,
            "retry_count": retry_count,
        }
        resolved_usage = response.usage if response is not None else usage
        if resolved_usage is not None:
            attributes.update(
                input_tokens=resolved_usage.input_tokens,
                output_tokens=resolved_usage.output_tokens,
                total_tokens=resolved_usage.total_tokens,
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
