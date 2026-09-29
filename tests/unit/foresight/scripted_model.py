"""脚本化的模型后端：按脚本回放结构化输出，记录调用次数、提示词与 schema。

预测层的判断测试与组合根接线测试共用。它以前住在旧语义树的测试夹具里，随那一包删掉后搬到这里——
它与语义树无关，只是一个假 Provider。
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterator, Mapping
from typing import Any

from habitus.model_client import (
    ChatClient,
    ChatModelConfig,
    ChatRequest,
    ModelResponse,
    PreparedChatRequest,
    ProviderCapabilities,
    ProviderConfig,
    StructuredChatClient,
)


class ScriptedProvider:
    """按脚本回放结构化输出的假 Provider；记录调用次数与提示词。"""

    provider_name = "fake"
    model = "fake-1"
    is_remote = False
    capabilities = ProviderCapabilities(
        async_completion=True, streaming=False, tools=False, structured_output_mode="json_schema", reasoning=False
    )

    def __init__(self, bodies: list[dict[str, object]]) -> None:
        self.bodies = bodies
        self.calls = 0
        self.prompts: list[str] = []

    def prepare(self, request: ChatRequest, *, stream: bool) -> PreparedChatRequest:
        return PreparedChatRequest(
            request=request, body=b"{}", model_visible_body=b"{}", reserved_output_tokens=1_000, stream=stream
        )

    async def complete_async(self, request: PreparedChatRequest) -> ModelResponse:
        self.prompts.append(request.request.messages[-1].content or "")
        body = self.bodies[min(self.calls, len(self.bodies) - 1)]
        self.calls += 1
        return ModelResponse(
            content=json.dumps(body, ensure_ascii=False),
            model=self.model,
            provider=self.provider_name,
            finish_reason="stop",
        )

    def complete(self, request: PreparedChatRequest) -> ModelResponse:  # pragma: no cover
        raise NotImplementedError

    def stream(self, request: PreparedChatRequest) -> Iterator[Any]:  # pragma: no cover
        raise NotImplementedError

    def stream_async(self, request: PreparedChatRequest) -> AsyncIterator[Any]:  # pragma: no cover
        raise NotImplementedError

    def health_check(self) -> Mapping[str, object]:  # pragma: no cover
        return {}

    async def aclose(self) -> None:  # pragma: no cover
        return None


class RecordingClient(StructuredChatClient):
    """记下每次调用带的 schema 与名字——服务层"把条数钉死"这条设计要能被断言。"""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.schemas: list[Any] = []
        self.names: list[Any] = []

    async def complete_json_async(self, request, **kwargs):  # type: ignore[no-untyped-def, override]
        self.schemas.append(kwargs.get("schema"))
        self.names.append(kwargs.get("name"))
        return await super().complete_json_async(request, **kwargs)


def model_config() -> ChatModelConfig:
    return ChatModelConfig(
        route=ProviderConfig(
            provider="fake",
            adapter="openai_compatible_chat",
            model="fake-1",
            base_url="https://example.invalid",
            credential_ref="FAKE_KEY",
        ),
        context_window_tokens=128_000,
        max_output_tokens=8_000,
        structured_output_mode="json_schema",
    )


def recording_client(bodies: list[dict[str, object]]) -> tuple[RecordingClient, ScriptedProvider]:
    """走**真构造器**的结构化客户端：装配错误在这条路径上会变成结构层的纠正重试，与生产一致。"""

    provider = ScriptedProvider(bodies)
    return RecordingClient(ChatClient(model_config(), provider), validation_retries=1), provider


__all__ = ["RecordingClient", "ScriptedProvider", "model_config", "recording_client"]
