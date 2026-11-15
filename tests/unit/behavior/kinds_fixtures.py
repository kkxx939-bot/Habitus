"""词表测试的共用现场：脚本化模型、按名字作答的选词替身、归约侧的选词一步（词表方案 2026-11-14）。"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterator, Mapping
from datetime import UTC, datetime
from typing import Any

from habitus.behavior.kinds.calls import KindModelCaller
from habitus.behavior.kinds.config import BehaviorKindConfig
from habitus.behavior.kinds.ids import ClassId, Lane
from habitus.behavior.kinds.model import Vocabulary, Word
from habitus.behavior.kinds.pick.picker import WordPicker
from habitus.behavior.reduction.kinds_step import KindStamping
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

NOW = datetime(2026, 10, 5, 12, 0, 0, tzinfo=UTC)


class ScriptedProvider:
    """按脚本回放结构化输出；记录调用次数与每次的用户消息。"""

    provider_name = "fake"
    model = "fake-1"
    is_remote = False
    capabilities = ProviderCapabilities(
        async_completion=True,
        streaming=False,
        tools=False,
        structured_output_mode="json_schema",
        reasoning=False,
    )

    def __init__(self, bodies: list[Any]) -> None:
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
        if isinstance(body, Exception):
            raise body
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


def scripted_client(bodies: list[Any]) -> tuple[StructuredChatClient, ScriptedProvider]:
    provider = ScriptedProvider(bodies)
    return client_for(provider), provider


def client_for(provider: ScriptedProvider) -> StructuredChatClient:
    config = ChatModelConfig(
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
    return StructuredChatClient(ChatClient(config, provider), validation_retries=0)


def scripted_caller(
    bodies: list[Any], *, config: BehaviorKindConfig | None = None
) -> tuple[KindModelCaller, ScriptedProvider]:
    client, provider = scripted_client(bodies)
    return KindModelCaller(client, config=config or BehaviorKindConfig()), provider


def named_words(names: tuple[str, ...], *, lane: Lane = Lane.SESSION) -> Vocabulary:
    """一份只有这几个第 1 层词的词表（编号按先后从 1 起），代替随代码发布的初始词表给归约侧的测试用。"""

    return Vocabulary(
        lane,
        tuple(
            Word(ClassId(lane, number), 1, name, None, f"{name}这件事", "测试")
            for number, name in enumerate(names, start=1)
        ),
    )


class NameMatchingProvider(ScriptedProvider):
    """选词的确定性假模型（物理 lane 那种一条记录一问）：记录的名字与词表里某个词名逐字相同就答那个词的编号，否则答
    ``missing``。只认选词提示词的形状（系统消息里【词表】一段的 "编号 名字：定义"，用户消息里【这一段】打头的名字）。"""

    def __init__(self) -> None:
        super().__init__([])

    async def complete_async(self, request: PreparedChatRequest) -> ModelResponse:
        system = "\n".join(message.content or "" for message in request.request.messages[:-1])
        text = request.request.messages[-1].content or ""
        self.prompts.append(text)
        self.calls += 1
        words = {}
        for line in system.partition("【词表】")[2].splitlines():
            number, _, rest = line.strip().partition(" ")
            if number.isdigit():
                words[rest.split("：", 1)[0].split("（", 1)[0]] = int(number)
        name = text.partition("【这一段】\n")[2].split("　｜", 1)[0]
        pick: object = words.get(name, "missing")
        return ModelResponse(
            content=json.dumps({"pick": pick}, ensure_ascii=False),
            model=self.model,
            provider=self.provider_name,
            finish_reason="stop",
        )


def kind_components(
    *,
    names: tuple[str, ...],
    provider: ScriptedProvider,
    config: BehaviorKindConfig | None = None,
) -> KindStamping:
    """归约侧的选词一步：在 ``names`` 这几个词里选（判断上没带选词结果的才问模型）。"""

    resolved = config or BehaviorKindConfig()
    picker = WordPicker(KindModelCaller(client_for(provider), config=resolved), named_words(names))
    return KindStamping(picker, lane=Lane.SESSION)
