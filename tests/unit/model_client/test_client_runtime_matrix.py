"""ChatClient 的重试、并发、流式边界与请求规范化场景矩阵。"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping

import pytest

from ModelClient import (
    ChatClient,
    ChatMessage,
    ChatModelConfig,
    ChatRequest,
    ModelAuthenticationError,
    ModelInputTooLargeError,
    ModelResponse,
    ModelResponseError,
    ModelStreamEvent,
    ModelTransportError,
    ProviderCapabilities,
    ProviderConfig,
)


def config(*, retries: int = 1, concurrent: int = 2, output_tokens: int | None = 128) -> ChatModelConfig:
    return ChatModelConfig(
        ProviderConfig(
            provider="scripted",
            adapter="scripted",
            model="scripted-model",
            base_url="https://example.com/v1",
            max_retries=retries,
            retry_base_delay_seconds=0.01,
            retry_max_delay_seconds=0.1,
            max_concurrent=concurrent,
        ),
        max_output_tokens=output_tokens,
    )


def response(content: str = "ok") -> ModelResponse:
    return ModelResponse(content, "scripted-model", "scripted")


def test_chat_client_rejects_estimated_context_overflow_before_provider_call() -> None:
    provider = ScriptedProvider()
    selected = ChatModelConfig(
        config().route,
        context_window_tokens=1_024,
        max_output_tokens=128,
    )
    client = ChatClient(selected, provider)
    with pytest.raises(ModelInputTooLargeError):
        client.complete(ChatRequest(messages=(ChatMessage(role="user", content="x" * 10_000),)))
    assert provider.requests == []


def stream_events() -> tuple[ModelStreamEvent, ...]:
    return (
        ModelStreamEvent(kind="content_delta", content_delta="ok"),
        ModelStreamEvent(kind="done", finish_reason="stop"),
    )


class ScriptedProvider:
    """为同步和异步接口分别消费显式结果脚本。"""

    provider_name = "scripted"
    model = "scripted-model"
    is_remote = True
    capabilities = ProviderCapabilities()

    def __init__(
        self,
        *,
        completions: list[object] | None = None,
        async_completions: list[object] | None = None,
        streams: list[object] | None = None,
        async_streams: list[object] | None = None,
        health: object = None,
    ) -> None:
        self.completions = completions or [response()]
        self.async_completions = async_completions or [response()]
        self.streams = streams or [stream_events()]
        self.async_streams = async_streams or [stream_events()]
        self.health = {"ok": True} if health is None else health
        self.requests: list[ChatRequest] = []

    @staticmethod
    def _take(queue: list[object]) -> object:
        current = queue.pop(0)
        if isinstance(current, BaseException):
            raise current
        return current

    def complete(self, request: ChatRequest):
        self.requests.append(request)
        return self._take(self.completions)

    async def complete_async(self, request: ChatRequest):
        self.requests.append(request)
        return self._take(self.async_completions)

    def stream(self, request: ChatRequest):
        self.requests.append(request)
        current = self._take(self.streams)
        yield from current

    async def stream_async(self, request: ChatRequest):
        self.requests.append(request)
        current = self._take(self.async_streams)
        for event in current:
            yield event

    def health_check(self) -> Mapping[str, object]:
        if isinstance(self.health, BaseException):
            raise self.health
        return self.health


def test_client_constructor_requires_chat_model_config() -> None:
    with pytest.raises(TypeError, match="config"):
        ChatClient(object(), ScriptedProvider())  # type: ignore[arg-type]


def test_client_exposes_provider_identity_without_copying_route_values() -> None:
    client = ChatClient(config(), ScriptedProvider())
    assert client.provider_name == "scripted"
    assert client.model == "scripted-model"
    assert client.is_remote is True


@pytest.mark.parametrize("invalid", [None, object(), 0, 1.5, (), [], {}])
def test_complete_rejects_non_request_non_text_input(invalid: object) -> None:
    client = ChatClient(config(), ScriptedProvider())
    with pytest.raises(TypeError, match="ChatRequest"):
        client.complete(invalid)  # type: ignore[arg-type]


@pytest.mark.parametrize("prompt", ["", " ", "\t\n"])
def test_complete_rejects_blank_text_prompt(prompt: str) -> None:
    client = ChatClient(config(), ScriptedProvider())
    with pytest.raises(ValueError, match="empty"):
        client.complete(prompt)


def test_text_prompt_becomes_one_user_message_and_inherits_output_limit() -> None:
    provider = ScriptedProvider()
    client = ChatClient(config(output_tokens=321), provider)

    assert client.complete("hello") == response()
    assert provider.requests == [
        ChatRequest(
            messages=(ChatMessage(role="user", content="hello"),),
            max_output_tokens=321,
        )
    ]


def test_explicit_request_output_limit_is_never_overridden() -> None:
    provider = ScriptedProvider()
    client = ChatClient(config(output_tokens=321), provider)
    request = ChatRequest(
        messages=(ChatMessage(role="user", content="hello"),),
        max_output_tokens=12,
    )

    client.complete(request)
    assert provider.requests[0] is request
    assert provider.requests[0].max_output_tokens == 12


def test_sync_completion_retries_transport_error_and_records_delay() -> None:
    provider = ScriptedProvider(completions=[TimeoutError("slow"), response()])
    delays: list[float] = []
    client = ChatClient(config(retries=1), provider, sleep=delays.append)

    assert client.complete("hello") == response()
    assert len(provider.requests) == 2
    assert len(delays) == 1
    assert 0 <= delays[0] <= 0.1


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (ModelAuthenticationError("bad key"), ModelAuthenticationError),
        (ModelResponseError("bad response"), ModelResponseError),
        (ValueError("invalid payload"), ModelResponseError),
    ],
)
def test_sync_completion_does_not_retry_nonretryable_failure(
    error: Exception,
    expected: type[Exception],
) -> None:
    provider = ScriptedProvider(completions=[error, response()])
    delays: list[float] = []
    client = ChatClient(config(retries=3), provider, sleep=delays.append)

    with pytest.raises(expected):
        client.complete("hello")
    assert len(provider.requests) == 1
    assert delays == []


def test_sync_completion_rejects_invalid_provider_response_after_retry_boundary() -> None:
    provider = ScriptedProvider(completions=[object()])
    client = ChatClient(config(retries=3), provider)

    with pytest.raises(ModelResponseError, match="invalid normalized"):
        client.complete("hello")
    assert len(provider.requests) == 1


def test_async_completion_retries_with_async_sleep_and_returns_response() -> None:
    provider = ScriptedProvider(async_completions=[ConnectionError("lost"), response()])
    delays: list[float] = []

    async def record(delay: float) -> None:
        delays.append(delay)

    client = ChatClient(config(retries=1), provider, async_sleep=record)
    assert asyncio.run(client.complete_async("hello")) == response()
    assert len(provider.requests) == 2
    assert len(delays) == 1


def test_async_completion_rejects_invalid_normalized_response() -> None:
    provider = ScriptedProvider(async_completions=["invalid"])
    client = ChatClient(config(retries=0), provider)

    with pytest.raises(ModelResponseError, match="invalid normalized"):
        asyncio.run(client.complete_async("hello"))


def test_client_can_be_reused_across_sequential_event_loops() -> None:
    provider = ScriptedProvider(async_completions=[response("one"), response("two")])
    client = ChatClient(config(retries=0), provider)

    assert asyncio.run(client.complete_async("one")).content == "one"
    assert asyncio.run(client.complete_async("two")).content == "two"


def test_async_concurrency_never_exceeds_route_limit() -> None:
    class BlockingProvider(ScriptedProvider):
        def __init__(self) -> None:
            super().__init__()
            self.active = 0
            self.peak = 0

        async def complete_async(self, request: ChatRequest):
            self.requests.append(request)
            self.active += 1
            self.peak = max(self.peak, self.active)
            await asyncio.sleep(0.01)
            self.active -= 1
            return response()

    provider = BlockingProvider()
    client = ChatClient(config(retries=0, concurrent=2), provider)

    async def scenario() -> None:
        await asyncio.gather(*(client.complete_async(str(index)) for index in range(6)))

    asyncio.run(scenario())
    assert provider.peak == 2


def test_sync_stream_retries_only_before_first_event() -> None:
    provider = ScriptedProvider(streams=[TimeoutError("connect"), stream_events()])
    delays: list[float] = []
    client = ChatClient(config(retries=1), provider, sleep=delays.append)

    assert tuple(client.stream("hello")) == stream_events()
    assert len(provider.requests) == 2
    assert len(delays) == 1


def test_sync_stream_empty_result_is_nonretryable_response_failure() -> None:
    provider = ScriptedProvider(streams=[(), stream_events()])
    client = ChatClient(config(retries=1), provider, sleep=lambda _delay: None)
    with pytest.raises(ModelResponseError, match="empty stream"):
        tuple(client.stream("hello"))
    assert len(provider.requests) == 1


@pytest.mark.parametrize("stream", [(), (object(),)])
def test_sync_stream_rejects_final_empty_or_invalid_event(stream: tuple[object, ...]) -> None:
    provider = ScriptedProvider(streams=[stream])
    client = ChatClient(config(retries=0), provider)
    with pytest.raises(ModelResponseError):
        tuple(client.stream("hello"))


def test_sync_stream_never_retries_after_emitting_event() -> None:
    class PartialProvider(ScriptedProvider):
        def stream(self, request: ChatRequest):
            self.requests.append(request)
            yield stream_events()[0]
            raise TimeoutError("disconnected")

    provider = PartialProvider()
    client = ChatClient(config(retries=3), provider)
    iterator = client.stream("hello")

    assert next(iterator) == stream_events()[0]
    with pytest.raises(ModelTransportError):
        next(iterator)
    assert len(provider.requests) == 1


def test_async_stream_retries_before_first_event() -> None:
    provider = ScriptedProvider(
        async_streams=[TimeoutError("connect"), stream_events()]
    )
    delays: list[float] = []

    async def record(delay: float) -> None:
        delays.append(delay)

    client = ChatClient(config(retries=1), provider, async_sleep=record)

    async def collect() -> tuple[ModelStreamEvent, ...]:
        return tuple([event async for event in client.stream_async("hello")])

    assert asyncio.run(collect()) == stream_events()
    assert len(provider.requests) == 2
    assert len(delays) == 1


@pytest.mark.parametrize("stream", [(), (object(),)])
def test_async_stream_rejects_final_empty_or_invalid_event(stream: tuple[object, ...]) -> None:
    provider = ScriptedProvider(async_streams=[stream])
    client = ChatClient(config(retries=0), provider)

    async def collect() -> None:
        async for _event in client.stream_async("hello"):
            pass

    with pytest.raises(ModelResponseError):
        asyncio.run(collect())


def test_async_stream_never_retries_after_emitting_event() -> None:
    class PartialProvider(ScriptedProvider):
        async def stream_async(self, request: ChatRequest):
            self.requests.append(request)
            yield stream_events()[0]
            raise TimeoutError("disconnected")

    provider = PartialProvider()
    client = ChatClient(config(retries=3), provider)

    async def collect() -> None:
        async for _event in client.stream_async("hello"):
            pass

    with pytest.raises(ModelTransportError):
        asyncio.run(collect())
    assert len(provider.requests) == 1


def test_health_check_returns_detached_success_mapping() -> None:
    health = {"ok": True, "details": "ready"}
    client = ChatClient(config(), ScriptedProvider(health=health))

    result = client.health_check()
    health["ok"] = False
    assert result == {"ok": True, "details": "ready"}


@pytest.mark.parametrize(
    ("error", "code"),
    [
        (TimeoutError("offline"), "MODEL_TRANSPORT"),
        (PermissionError("forbidden"), "MODEL_TRANSPORT"),
        (ValueError("invalid"), "MODEL_RESPONSE"),
    ],
)
def test_health_check_normalizes_failure_without_raising(error: Exception, code: str) -> None:
    client = ChatClient(config(), ScriptedProvider(health=error))
    assert client.health_check() == {
        "ok": False,
        "provider": "scripted",
        "model": "scripted-model",
        "error_code": code,
    }
