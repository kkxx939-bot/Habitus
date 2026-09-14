"""关联测试的独立现场：一个候选在某天的两次发生，带前因、情境与未兑现前提，外加脚本化的模型后端。

夹具自成一体（不 import 归组的现场，也不 import 别的测试文件）：关联与归组问的是两个问题，
共用一份现场会让其中一边的改动悄悄改掉另一边的判据；而归组那半在第 7 步会整块删掉，现在把
关联挂上去等于给它绑一个将死的依赖。
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterator, Mapping
from datetime import date, datetime, timedelta, timezone
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
from habitus.scene.association.model import (
    AssociationInput,
    CauseRow,
    DayFacts,
    OccurrenceRow,
    PendingRow,
    SituationRow,
)

OFFSET = timezone(timedelta(hours=8))
DAY = date(2026, 9, 11)


def instant(hour: int, minute: int = 0, *, day: date = DAY) -> datetime:
    return datetime(day.year, day.month, day.day, hour, minute, tzinfo=OFFSET)


def occurrence(no: int, hour: int, name: str, *, kind: str = "杂事", goal: str | None = None) -> OccurrenceRow:
    return OccurrenceRow(
        no=no,
        uri=f"behavior://tree/occurrence/{no}",
        name=name,
        kind_token=kind,
        started_at=instant(hour),
        summary=f"{name}的一句话",
        goal=goal,
    )


def association_input(**overrides: object) -> AssociationInput:
    fields: dict[str, object] = {
        "kind_token": "打球",
        "day": DAY,
        "targets": (3,),
        "occurrences": (
            occurrence(1, 8, "吃早饭"),
            occurrence(2, 18, "和朋友通电话"),
            occurrence(3, 19, "打球", kind="打球", goal="活动一下"),
        ),
        "facts": DayFacts(weekday=4, month=9, day_note="工作日", observed_gaps=((instant(13), instant(14)),)),
        "situations": (SituationRow(no=1, text="周五下班后自己去", days=(date(2026, 9, 4),)),),
        "causes": (
            CauseRow(
                no=1,
                uri="behavior://tree/occurrence/earlier",
                name="和朋友通电话",
                started_at=instant(18, day=date(2026, 9, 6)),
                summary="约这周打一次球",
            ),
        ),
        "pending": (
            PendingRow(
                no=1,
                text="和朋友约好这周打一次球",
                producer_uri="behavior://tree/occurrence/earlier",
                created_on=date(2026, 9, 6),
                consumed_by="打球",
            ),
        ),
        "origin": None,
    }
    fields.update(overrides)
    return AssociationInput(**fields)  # type: ignore[arg-type]


def two_target_input(**overrides: object) -> AssociationInput:
    """同一天打了两次球：一次调用同时问两次发生，互为上下文。"""

    rows = (
        occurrence(1, 8, "吃早饭"),
        occurrence(2, 10, "打球", kind="打球"),
        occurrence(3, 18, "和朋友通电话"),
        occurrence(4, 19, "打球", kind="打球"),
    )
    fields: dict[str, object] = {"targets": (2, 4), "occurrences": rows}
    fields.update(overrides)
    return association_input(**fields)


def entry(**overrides: object) -> dict[str, object]:
    """一条合格的输出；测试各自改其中一个字段来验证降级。"""

    row: dict[str, object] = {
        "occurrence_no": 3,
        "context": "周五晚上，下午刚和朋友通过电话说好一起打",
        "cites": [2, 4],
        "causes": [2],
        "situation_no": None,
        "new_situation": "和朋友约好之后一起去打",
        "consumed": [1],
        "left": [],
    }
    row.update(overrides)
    return row


def output(*entries: dict[str, object]) -> dict[str, object]:
    return {"entries": list(entries or (entry(),))}


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


__all__ = [
    "DAY",
    "OFFSET",
    "association_input",
    "entry",
    "instant",
    "occurrence",
    "output",
    "RecordingClient",
    "ScriptedProvider",
    "model_config",
    "recording_client",
    "two_target_input",
]
