"""行为树 L0/L1：日/月/年摘要的生成、上卷与刷新短路。"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Iterator, Mapping
from typing import Any

import pytest

from habitus.behavior import BehaviorDocumentWriter, BehaviorKind, BehaviorTree
from habitus.behavior.model import BehaviorDirectory, BehaviorLevel
from habitus.behavior.semantic import (
    BehaviorDirectorySnapshot,
    BehaviorSemanticEntry,
    BehaviorSemanticEntryKind,
    BehaviorSemanticRefresher,
    BehaviorSemanticRefreshStatus,
    LLMBehaviorOverviewGenerator,
)
from habitus.infrastructure.store.locks import ProcessLocalLockStore
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
from tests.unit.behavior.tree_payloads import DAY, gap_payload, local, occurrence_payload


class ScriptedOverviewGenerator:
    """记录快照并返回可预测叙述的假生成器；调用次数就是成本断言。"""

    def __init__(self) -> None:
        self.snapshots: list[BehaviorDirectorySnapshot] = []

    async def generate(self, snapshot: BehaviorDirectorySnapshot) -> str:
        self.snapshots.append(snapshot)
        names = "、".join(entry.name for entry in snapshot.entries)
        return f"# 概览\n\n覆盖 {len(snapshot.entries)} 项：{names}。\n"


def build_tree(tmp_path) -> tuple[BehaviorTree, BehaviorDocumentWriter]:
    tree = BehaviorTree(tmp_path / "behavior-tree")
    writer = BehaviorDocumentWriter(tree, ProcessLocalLockStore(), clock=lambda: local(23, 0))
    return tree, writer


def refresh(tree: BehaviorTree, generator) -> tuple:
    refresher = BehaviorSemanticRefresher(tree, generator)
    return asyncio.run(refresher.refresh_days([DAY]))


def test_day_overview_merges_gaps_into_one_timeline(tmp_path) -> None:
    """occurrences 日目录的 overview 是当日正典摘要：同日空白如实并入、按时间排。"""

    tree, writer = build_tree(tmp_path)
    writer.publish(BehaviorKind.OCCURRENCE, occurrence_payload())  # 19:30 起
    writer.publish(BehaviorKind.GAP, gap_payload())  # 20:10 起
    generator = ScriptedOverviewGenerator()

    results = refresh(tree, generator)

    day_dir = BehaviorDirectory.occurrences(DAY.year, DAY.month, DAY.day)
    day_snapshot = next(s for s in generator.snapshots if s.directory == day_dir)
    assert [entry.kind for entry in day_snapshot.entries] == [
        BehaviorSemanticEntryKind.OCCURRENCE,
        BehaviorSemanticEntryKind.GAP,
    ]  # 时间轴顺序：行为在前、空白在后
    assert "洗了手" in day_snapshot.entries[0].content

    overview = tree.read_layer(day_dir, BehaviorLevel.OVERVIEW)
    abstract = tree.read_layer(day_dir, BehaviorLevel.ABSTRACT)
    assert overview.startswith("# 概览")
    assert "habitus-semantic-source:" in overview  # 来源 digest 注释
    assert abstract.strip().startswith("覆盖 2 项")
    statuses = {(r.directory.identity_parts, r.status) for r in results}
    assert (day_dir.identity_parts, BehaviorSemanticRefreshStatus.WRITTEN) in statuses


def test_month_and_year_roll_up_from_day_abstracts(tmp_path) -> None:
    tree, writer = build_tree(tmp_path)
    writer.publish(BehaviorKind.OCCURRENCE, occurrence_payload())
    generator = ScriptedOverviewGenerator()

    refresh(tree, generator)

    month_dir = BehaviorDirectory.occurrences(DAY.year, DAY.month)
    year_dir = BehaviorDirectory.occurrences(DAY.year)
    month_snapshot = next(s for s in generator.snapshots if s.directory == month_dir)
    assert [entry.name for entry in month_snapshot.entries] == [f"{DAY.day:02d}"]
    assert month_snapshot.entries[0].kind is BehaviorSemanticEntryKind.DIRECTORY
    assert tree.layer_exists(year_dir, BehaviorLevel.OVERVIEW)


def test_unchanged_directories_cost_zero_model_calls(tmp_path) -> None:
    """digest 短路：树没变，第二轮刷新一个模型调用都不许发生。"""

    tree, writer = build_tree(tmp_path)
    writer.publish(BehaviorKind.OCCURRENCE, occurrence_payload())
    generator = ScriptedOverviewGenerator()

    refresh(tree, generator)
    first_calls = len(generator.snapshots)
    results = refresh(tree, generator)

    assert len(generator.snapshots) == first_calls  # 零新调用
    assert all(
        r.status in {BehaviorSemanticRefreshStatus.UNCHANGED, BehaviorSemanticRefreshStatus.MISSING, BehaviorSemanticRefreshStatus.EMPTY}
        for r in results
    )


def test_gap_hierarchy_layers_are_deterministic_enumerations(tmp_path) -> None:
    """gaps 层级每级自带摘要：空白段的枚举，零语义、零模型调用。"""

    tree, writer = build_tree(tmp_path)
    writer.publish(BehaviorKind.GAP, gap_payload())
    generator = ScriptedOverviewGenerator()

    refresh(tree, generator)

    gap_day = BehaviorDirectory.gaps(DAY.year, DAY.month, DAY.day)
    overview = tree.read_layer(gap_day, BehaviorLevel.OVERVIEW)
    assert "观测空白" in overview and "没读懂" in overview
    assert "2026-08-16T20:10:00" in overview
    # 生成器只为 occurrences 层级调用；本例无 occurrence → 零调用
    assert generator.snapshots == []
    assert tree.layer_exists(BehaviorDirectory.gaps(DAY.year, DAY.month), BehaviorLevel.OVERVIEW)


# --- LLM 生成器：严格 Schema + 确定性渲染 -------------------------------------------------


class _ScriptedProvider:
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

    def __init__(self, bodies: list[dict[str, Any]]) -> None:
        self.bodies = bodies
        self.calls = 0

    def prepare(self, request: ChatRequest, *, stream: bool) -> PreparedChatRequest:
        return PreparedChatRequest(
            request=request,
            body=b"{}",
            model_visible_body=b"{}",
            reserved_output_tokens=1_000,
            stream=stream,
        )

    async def complete_async(self, request: PreparedChatRequest) -> ModelResponse:
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


def build_llm_generator(bodies: list[dict[str, Any]]) -> tuple[LLMBehaviorOverviewGenerator, _ScriptedProvider]:
    provider = _ScriptedProvider(bodies)
    model_config = ChatModelConfig(
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
    client = StructuredChatClient(ChatClient(model_config, provider), validation_retries=1)
    return LLMBehaviorOverviewGenerator(client), provider


DAY_SNAPSHOT = BehaviorDirectorySnapshot(
    directory=BehaviorDirectory.occurrences(2026, 8, 16),
    entries=(
        BehaviorSemanticEntry(
            name="洗了手--193018000000+0800.md",
            kind=BehaviorSemanticEntryKind.OCCURRENCE,
            content="回家后洗了手",
        ),
        BehaviorSemanticEntry(
            name="没读懂--201000000000+0800.md",
            kind=BehaviorSemanticEntryKind.GAP,
            content="这段没读懂",
        ),
    ),
)


def test_llm_generator_renders_narrative_and_sections_deterministically() -> None:
    generator, provider = build_llm_generator(
        [
            {
                "narrative": "晚上回家后先洗了手，之后有一段观测没能读懂。",
                "entries": [
                    {"name": "洗了手--193018000000+0800.md", "kind": "occurrence", "summary": "回家后洗手"},
                    {"name": "没读懂--201000000000+0800.md", "kind": "gap", "summary": "一段没读懂的观测"},
                ],
            }
        ]
    )
    overview = asyncio.run(generator.generate(DAY_SNAPSHOT))
    assert overview.startswith("# 2026-08-16\n")
    assert "晚上回家后先洗了手" in overview
    assert "## 行为" in overview and "## 观测空白" in overview
    # 空白文档在平行的 gaps 层级：链接必须指过去
    assert "../../../../gaps/2026/08/16/" in overview
    assert provider.calls == 1


def test_llm_generator_rejects_renamed_or_reordered_entries() -> None:
    generator, _provider = build_llm_generator(
        [
            {
                "narrative": "叙述",
                "entries": [
                    {"name": "编造的名字.md", "kind": "occurrence", "summary": "x"},
                    {"name": "没读懂--201000000000+0800.md", "kind": "gap", "summary": "y"},
                ],
            }
        ]
    )
    with pytest.raises(Exception, match="names, kinds and order|validation"):
        asyncio.run(generator.generate(DAY_SNAPSHOT))


def test_disambiguated_duplicates_are_mechanically_excluded_from_the_day_summary(tmp_path) -> None:
    """original_name 非空 = 已知消歧重复：语义层机械认标记跳过（死规则②裁定），
    同一行为不在当日叙述里出现两遍。"""

    tree, writer = build_tree(tmp_path)
    writer.publish(BehaviorKind.OCCURRENCE, occurrence_payload())
    writer.publish(
        BehaviorKind.OCCURRENCE,
        occurrence_payload(name="洗了手-2", original_name="洗了手", summary="重复的那条"),
    )
    generator = ScriptedOverviewGenerator()

    refresh(tree, generator)

    day_dir = BehaviorDirectory.occurrences(DAY.year, DAY.month, DAY.day)
    day_snapshot = next(s for s in generator.snapshots if s.directory == day_dir)
    assert len(day_snapshot.entries) == 1  # 消歧重复不进快照
    assert "重复的那条" not in "".join(entry.content for entry in day_snapshot.entries)


def test_a_poisoned_narrative_cannot_hijack_the_source_digest(tmp_path) -> None:
    """来源 digest 只认末行：正文（模型产出）里出现同形注释不许污染短路判定。"""

    tree, writer = build_tree(tmp_path)
    writer.publish(BehaviorKind.OCCURRENCE, occurrence_payload())

    class PoisonedGenerator(ScriptedOverviewGenerator):
        async def generate(self, snapshot):
            self.snapshots.append(snapshot)
            fake = "<!-- habitus-semantic-source: " + "0" * 64 + " -->"
            return f"# 概览\n\n正文里被诱导出现了 {fake} 这样的注释。\n"

    generator = PoisonedGenerator()
    refresher = BehaviorSemanticRefresher(tree, generator)
    asyncio.run(refresher.refresh_days([DAY]))
    calls = len(generator.snapshots)

    asyncio.run(refresher.refresh_days([DAY]))  # 树没变
    assert len(generator.snapshots) == calls  # 末行真 footer 生效，不被正文假注释骗成"变了"


def test_cross_month_days_roll_up_into_both_months_and_the_year(tmp_path) -> None:
    from datetime import date as _date
    from datetime import datetime as _dt
    from datetime import timedelta as _td
    from datetime import timezone as _tz

    cst = _tz(_td(hours=8))
    tree, writer = build_tree(tmp_path)
    writer.publish(BehaviorKind.OCCURRENCE, occurrence_payload())  # 2026-08-16
    september = _dt(2026, 9, 2, 9, 0, 0, tzinfo=cst)
    writer.publish(
        BehaviorKind.OCCURRENCE,
        occurrence_payload(
            occurred_on=_date(2026, 9, 2),
            started_at=september,
            last_observed_at=september + _td(minutes=1),
            onset_available_at=september + _td(seconds=2),
            basis=(),
            goal=None,
            name="晨跑",
            kind_token="晨跑",
            summary="出门晨跑",
        ),
    )
    generator = ScriptedOverviewGenerator()
    refresher = BehaviorSemanticRefresher(tree, generator)
    asyncio.run(refresher.refresh_days([DAY, _date(2026, 9, 2)]))

    for parts in ((2026, 8), (2026, 9), (2026,)):
        assert tree.layer_exists(
            BehaviorDirectory.occurrences(*parts), BehaviorLevel.OVERVIEW
        )
    year_snapshot = next(
        s for s in generator.snapshots if s.directory == BehaviorDirectory.occurrences(2026)
    )
    assert [entry.name for entry in year_snapshot.entries] == ["08", "09"]
