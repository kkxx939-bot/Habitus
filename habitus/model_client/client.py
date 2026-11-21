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
from habitus.model_client.breaker import CircuitBreaker
from habitus.model_client.config import CallFailureConfig, ChatModelConfig
from habitus.model_client.contracts import (
    ChatCallContext,
    ChatMessage,
    ChatProvider,
    ChatRequest,
    ModelResponse,
    ModelStreamEvent,
    PreparedChatRequest,
    TokenUsage,
)
from habitus.model_client.errors import (
    ModelCircuitOpenError,
    ModelClientError,
    ModelErrorRule,
    ModelInputTooLargeError,
    ModelResponseError,
)
from habitus.model_client.failures import classify_failure, retry_delay


class ChatClient:
    """供应商无关的对话调用入口；Provider 的创建只属于工厂。失败按 ``model_client.errors`` 的三条规则处理：只有"等一等可能好"
    的按路由有界重试，用尽交回调用方；每次逻辑调用的结局记进这条能力的熔断（``model_client.breaker``）。"""

    def __init__(
        self,
        config: ChatModelConfig,
        provider: ChatProvider,
        *,
        failures: CallFailureConfig | None = None,
        sleep: Callable[[float], None] = time.sleep,
        async_sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        observer: Observer | None = None,
    ) -> None:
        self.config = config
        self.provider = provider
        self.failures = failures or CallFailureConfig()
        #: 对话这一种能力的熔断：同一个客户端的所有调用（会话、选词、判官、记忆、摘要）共用。
        self.breaker = CircuitBreaker(self.failures, name="chat")
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
        call, started = (prepared.request, call_context), time.monotonic()
        probe = self._admit("complete", started, call)
        try:
            with self._sync_slots:
                for attempt in range(self.config.route.max_retries + 1):
                    try:
                        raw = self.provider.complete(prepared)
                        response = self._response_context(self._require_response(raw), call_context)
                    except Exception as exc:
                        failure, delay = self._settle(attempt, exc)
                        if delay is None:
                            self._observe(
                                "complete", ObservationStatus.FAILURE, started, call, error=failure, retry_count=attempt
                            )
                            raise failure from exc
                        self._sleep(delay)
                        continue
                    self.breaker.record_success()
                    self._observe(
                        "complete", ObservationStatus.SUCCESS, started, call, response=response, retry_count=attempt
                    )
                    return response
        finally:
            self._release(probe)
        raise AssertionError("chat retry loop exhausted without a result")  # pragma: no cover

    async def complete_async(
        self,
        request: ChatRequest | str,
        *,
        context: ChatCallContext | None = None,
    ) -> ModelResponse:
        call_context = self._context(context)
        prepared = self._prepare(request, stream=False, context=call_context)
        call, started = (prepared.request, call_context), time.monotonic()
        probe = self._admit("complete_async", started, call)
        try:
            async with self._async_slot():
                for attempt in range(self.config.route.max_retries + 1):
                    try:
                        raw = await self.provider.complete_async(prepared)
                        response = self._response_context(self._require_response(raw), call_context)
                    except Exception as exc:
                        failure, delay = self._settle(attempt, exc)
                        if delay is None:
                            self._observe(
                                "complete_async",
                                ObservationStatus.FAILURE,
                                started,
                                call,
                                error=failure,
                                retry_count=attempt,
                            )
                            raise failure from exc
                        await self._async_sleep(delay)
                        continue
                    self.breaker.record_success()
                    self._observe(
                        "complete_async",
                        ObservationStatus.SUCCESS,
                        started,
                        call,
                        response=response,
                        retry_count=attempt,
                    )
                    return response
        finally:
            self._release(probe)
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
        probe = self._admit("stream_async", started)
        try:
            async with self._async_slot():
                for attempt in range(self.config.route.max_retries + 1):
                    events = _StreamTally()
                    try:
                        async for event in self.provider.stream_async(prepared):
                            yield events.take(event)
                        events.finish()
                    except Exception as exc:
                        failure, delay = self._settle(attempt, exc, retry=not events.emitted)
                        if delay is None:
                            self._observe(
                                "stream_async", ObservationStatus.FAILURE, started, error=failure, retry_count=attempt
                            )
                            raise failure from exc
                        await self._async_sleep(delay)
                        continue
                    self.breaker.record_success()
                    self._observe(
                        "stream_async", ObservationStatus.SUCCESS, started, usage=events.usage, retry_count=attempt
                    )
                    return
        finally:
            self._release(probe)

    def health_check(self) -> dict[str, object]:
        """探针不经过熔断、不计入熔断：要看到服务此刻真实的样子。"""

        try:
            result = self.provider.health_check()
            return dict(result)
        except Exception as exc:
            return self._probe_failed(classify_failure(exc))

    async def health_check_async(self) -> dict[str, object]:
        """只调用 Provider 显式提供的异步探针，保证调用方可真正取消。探针不经过熔断；"等一等可能好"的错按路由的次数与退避
        再探，一次超时不当成整个用不了（裁定 94 第 8 条）。"""

        probe = getattr(self.provider, "health_check_async", None)
        if not callable(probe):
            return {
                "ok": False,
                "provider": self.provider.provider_name,
                "model": self.provider.model,
                "error_code": "async_health_check_unsupported",
            }
        async_probe = cast(Callable[[], Awaitable[Mapping[str, object]]], probe)
        for attempt in range(self.config.route.max_retries + 1):
            try:
                return dict(await async_probe())
            except Exception as exc:
                failure = classify_failure(exc)
                if failure.rule is ModelErrorRule.WAIT and attempt < self.config.route.max_retries:
                    await self._async_sleep(self._retry_delay(attempt, failure))
                    continue
                return self._probe_failed(failure)
        raise AssertionError("health probe retry loop exhausted without a result")  # pragma: no cover

    async def aclose(self) -> None:
        """释放 Chat Provider 的同步和异步传输资源。"""

        await self.provider.aclose()

    def _stream_sync(self, request: PreparedChatRequest) -> Iterator[ModelStreamEvent]:
        started = time.monotonic()
        probe = self._admit("stream", started)
        try:
            with self._sync_slots:
                for attempt in range(self.config.route.max_retries + 1):
                    events = _StreamTally()
                    try:
                        for event in self.provider.stream(request):
                            yield events.take(event)
                        events.finish()
                    except Exception as exc:
                        failure, delay = self._settle(attempt, exc, retry=not events.emitted)
                        if delay is None:
                            self._observe(
                                "stream", ObservationStatus.FAILURE, started, error=failure, retry_count=attempt
                            )
                            raise failure from exc
                        self._sleep(delay)
                        continue
                    self.breaker.record_success()
                    self._observe("stream", ObservationStatus.SUCCESS, started, usage=events.usage, retry_count=attempt)
                    return
        finally:
            self._release(probe)

    def _admit(self, operation: str, started: float, call: tuple[ChatRequest, ChatCallContext] | None = None) -> bool:
        """问熔断能不能发；熔断开着就记一笔失败观测、原样抛 ``ModelCircuitOpenError``。返回这次是不是半开时的试探。"""

        try:
            return self.breaker.before_call()
        except ModelCircuitOpenError as exc:
            self._observe(operation, ObservationStatus.FAILURE, started, call, error=exc)
            raise

    def _settle(self, attempt: int, exc: Exception, *, retry: bool = True) -> tuple[ModelClientError, float | None]:
        """一次尝试失败：归类；"等一等可能好"且还有次数就返回等多久，否则把这次逻辑调用的失败记进熔断、返回 None。"""

        failure = classify_failure(exc)
        if retry and failure.rule is ModelErrorRule.WAIT and attempt < self.config.route.max_retries:
            return failure, self._retry_delay(attempt, failure)
        self.breaker.record_failure(failure)
        return failure, None

    def _release(self, probe: bool) -> None:
        if probe:
            self.breaker.release_probe()

    def _probe_failed(self, failure: ModelClientError) -> dict[str, object]:
        return {
            "ok": False,
            "provider": self.provider.provider_name,
            "model": self.provider.model,
            "error_code": failure.code,
        }

    def estimate_input_tokens(self, request: ChatRequest | str) -> int:
        """按供应商真实的请求投影估算模型可见输入 token，不发请求；与每次调用前的预算检查（``_prepare``）同一个估法。"""

        prepared = self.provider.prepare(self._request(request), stream=False)
        if not isinstance(prepared, PreparedChatRequest):
            raise ModelResponseError("provider returned an invalid prepared chat request")
        return prepared.estimated_input_tokens

    def _request(self, request: ChatRequest | str) -> ChatRequest:
        if isinstance(request, str):
            if not request.strip():
                raise ValueError("model prompt cannot be empty")
            request = ChatRequest(messages=(ChatMessage(role="user", content=request),))
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
        if prepared.estimated_input_tokens + prepared.reserved_output_tokens > self.config.context_window_tokens:
            raise ModelInputTooLargeError(
                "estimated chat input and reserved output exceed the configured context window"
            )
        if context.input_token_limit is not None and prepared.estimated_input_tokens > context.input_token_limit:
            raise ModelInputTooLargeError("estimated chat input exceeds the call-specific input-token limit")
        return prepared

    @staticmethod
    def _context(context: ChatCallContext | None) -> ChatCallContext:
        return ChatCallContext() if context is None else context

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
        return retry_delay(attempt, route=self.config.route, failures=self.failures, error=failure)

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
        call: tuple[ChatRequest, ChatCallContext] | None = None,
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
        if call is not None:
            attributes.update(_call_attributes(*call, response))
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


class _StreamTally:
    """一次流式尝试里收到的事件：有没有吐出过事件（吐出过就不能再重试）、最后的用量。"""

    def __init__(self) -> None:
        self.emitted = False
        self.usage: TokenUsage | None = None

    def take(self, event: object) -> ModelStreamEvent:
        if not isinstance(event, ModelStreamEvent):
            raise ModelResponseError("provider returned an invalid stream event")
        self.emitted = True
        if event.usage is not None:
            self.usage = event.usage
        return event

    def finish(self) -> None:
        if not self.emitted:
            raise ModelResponseError("provider returned an empty stream")


def _call_attributes(
    request: ChatRequest, context: ChatCallContext, response: ModelResponse | None
) -> dict[str, str | int | float | bool]:
    """一次调用的读数，给 trace 的 ``model.call`` 用：触点（结构化输出的名字，没有就看提示词版本）、提示词与答复的字符数，以及
    结构化校验的第几遍（``StructuredChatClient`` 记在调用侧元数据里）。只有长度与名字，不含内容。"""

    format_name = request.response_format.name if request.response_format is not None else None
    attributes: dict[str, str | int | float | bool] = {
        "touch": (format_name or context.prompt_version or "chat")[:128],
        "prompt_chars": sum(len(message.content) for message in request.messages if isinstance(message.content, str)),
    }
    if response is not None:
        attributes["response_chars"] = len(response.content or "")
        # 实际作答的后端与模型（适配器可能回退到别的后端）；``provider`` / ``model`` 仍是路由配置的那一对
        attributes["response_provider"] = response.provider[:128]
        attributes["response_model"] = response.model[:128]
    attempt = context.metadata.get("structured_attempt")
    if isinstance(attempt, int) and not isinstance(attempt, bool):
        attributes["validation_attempt"] = attempt
    return attributes


__all__ = ["ChatClient"]
