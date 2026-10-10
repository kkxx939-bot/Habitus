"""词表测试的共用现场：脚本化模型、两个最小行为类、一条请求。"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterator, Mapping
from datetime import UTC, datetime
from typing import Any

from habitus.behavior.kinds.calls import KindModelCaller
from habitus.behavior.kinds.classify import ClassifyRequest, OccurrenceContent
from habitus.behavior.kinds.config import BehaviorKindConfig
from habitus.behavior.kinds.ids import ClassId, Lane
from habitus.behavior.kinds.model import BehaviorClass, ClassOrigin, Exclusion
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
EDIT = ClassId(Lane.SESSION, 1)
RESEARCH = ClassId(Lane.SESSION, 2)


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


def edit_class() -> BehaviorClass:
    return BehaviorClass(
        EDIT,
        "修改代码",
        "动手改代码、加功能、重构",
        "该改代码了",
        examples=("为 Tagent 加 ReAct 支持",),
        origin=ClassOrigin.PROMOTED,
    )


def research_class() -> BehaviorClass:
    return BehaviorClass(
        RESEARCH,
        "调研",
        "读代码、读资料，弄清楚一件事",
        "该调研一下了",
        excludes=(Exclusion("动手改代码", EDIT),),
        origin=ClassOrigin.PROMOTED,
    )


def request(key: str, name: str, *, lane: Lane = Lane.SESSION, summary: str | None = None) -> ClassifyRequest:
    return ClassifyRequest(key, lane, OccurrenceContent(name=name, summaries=(summary,) if summary else ()))


def answer(*rows: tuple[str, str, str | None]) -> dict[str, Any]:
    """``(记录编号, 选择, 提议名)`` → 一份模型输出；理由统一写"理由"。"""

    return {"items": [{"record": r, "reason": "理由", "choice": c, "proposed": p} for r, c, p in rows]}


class NameMatchingProvider(ScriptedProvider):
    """归类调用的确定性假模型：记录的原话与某个类名逐字相同就归那一类，在 ``not_events`` 里就答「不是一件事」，
    否则答「都不是」并把原话当提议名。只认白天归类的提示词形状（``## 词表`` / ``## 待归类的记录``）。"""

    def __init__(self, *, not_events: frozenset[str] = frozenset()) -> None:
        super().__init__([])
        self.not_events = not_events

    async def complete_async(self, request: PreparedChatRequest) -> ModelResponse:
        text = request.request.messages[-1].content or ""
        self.prompts.append(text)
        self.calls += 1
        head, _, tail = text.partition("## 待归类的记录")
        classes = {
            line.split("  ", 1)[1].strip(): line.split("  ", 1)[0]
            for line in head.splitlines()
            if line.startswith("C") and "  " in line
        }
        items = []
        for line in tail.splitlines():
            if not line.startswith("R") or "  " not in line:
                continue
            record, rest = line.split("  ", 1)
            name = rest.split("　｜", 1)[0]
            if name in self.not_events:
                items.append({"record": record, "reason": "小动作", "choice": "不是一件事", "proposed": None})
            elif name in classes:
                items.append({"record": record, "reason": "同名", "choice": classes[name], "proposed": None})
            else:
                items.append({"record": record, "reason": "没有合适的类", "choice": "都不是", "proposed": name})
        return ModelResponse(
            content=json.dumps({"items": items}, ensure_ascii=False),
            model=self.model,
            provider=self.provider_name,
            finish_reason="stop",
        )


class RoutingProvider(ScriptedProvider):
    """白天归类的调用按路由表确定性作答，其余调用（拆改提议、每晚新增）按脚本回放。

    ``routes``：原话 → 依次偏好的类名，取清单里第一个出现的；没有路由的原话与类名逐字相同就归那一类，否则答「都不是」。
    同一份清单答同样的话——锚点"站得稳"、新清单是否把锚点拉走，都由清单里有哪些类名决定。
    """

    def __init__(self, bodies: list[Any], routes: Mapping[str, tuple[str, ...]] | None = None) -> None:
        super().__init__(bodies)
        self.routes = dict(routes or {})
        self.classify_calls = 0

    async def complete_async(self, request: PreparedChatRequest) -> ModelResponse:
        text = request.request.messages[-1].content or ""
        if "## 待归类的记录" not in text:
            return await super().complete_async(request)
        self.prompts.append(text)
        self.classify_calls += 1
        head, _, tail = text.partition("## 待归类的记录")
        classes = {
            line.split("  ", 1)[1].strip(): line.split("  ", 1)[0]
            for line in head.splitlines()
            if line.startswith("C") and "  " in line
        }
        items = []
        for line in tail.splitlines():
            if not line.startswith("R") or "  " not in line:
                continue
            record, rest = line.split("  ", 1)
            name = rest.split("　｜", 1)[0]
            chosen = next((classes[c] for c in self.routes.get(name, (name,)) if c in classes), None)
            choice = chosen or "都不是"
            items.append({"record": record, "reason": "路由", "choice": choice, "proposed": None if chosen else name})
        return ModelResponse(
            content=json.dumps({"items": items}, ensure_ascii=False),
            model=self.model,
            provider=self.provider_name,
            finish_reason="stop",
        )


def named_classes(names: tuple[str, ...], *, lane: Lane = Lane.SESSION) -> tuple[BehaviorClass, ...]:
    return tuple(
        BehaviorClass(ClassId(lane, number), name, f"{name}的判据", f"该{name}了", origin=ClassOrigin.PROMOTED)
        for number, name in enumerate(names, start=1)
    )


def kind_components(
    tree: Any,
    lock_store: Any,
    *,
    names: tuple[str, ...],
    provider: ScriptedProvider,
    config: BehaviorKindConfig | None = None,
) -> tuple[Any, Any, Any]:
    """归约侧的词表两块（白天归类 + 整理活）与它们共用的存储；``names`` 非空时先把它们写成第 1 版（会话 lane）。"""

    from habitus.behavior.kinds.changes import AddClass, ChangeReason, VersionRecord
    from habitus.behavior.kinds.classify import DaytimeClassifier
    from habitus.behavior.kinds.nightly import NightlyGrower, PendingRecheck
    from habitus.behavior.kinds.revision import AnchorGate, Reviser
    from habitus.behavior.kinds.store import BehaviorKindStore
    from habitus.behavior.reduction.kinds_jobs import VocabularyJobs
    from habitus.behavior.reduction.kinds_migration import KindMigrator
    from habitus.behavior.reduction.kinds_step import KindStamping

    resolved = config or BehaviorKindConfig()
    store = BehaviorKindStore(tree.root, config=resolved)
    if names:
        operations = tuple(AddClass(item) for item in named_classes(names))
        store.append(VersionRecord(1, NOW, ChangeReason.NIGHTLY, operations), expected_version=0)
    caller = KindModelCaller(client_for(provider), config=resolved)
    classifier = DaytimeClassifier(caller)
    stamping = KindStamping(store, classifier, lane=resolved.default_lane)
    jobs = VocabularyJobs(
        tree=tree,
        stamping=stamping,
        migrator=KindMigrator(tree, lock_store, store),
        grower=NightlyGrower(caller, PendingRecheck(classifier)),
        reviser=Reviser(caller, AnchorGate(classifier)),
        zone=UTC,
    )
    return stamping, jobs, store
