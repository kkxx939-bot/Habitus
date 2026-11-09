"""夜批：第 N 晚把语义树的几步按固定顺序跑一遍（语义树新方案 ``13`` 第五节）。

顺序是**线性的、不级联的**（派生树按固定先后处理已封口的历史，不做"输入又变了就重建"）：

```
B1  同步词表（裁定 20）：读上次同步之后的变更
      · 每个类一个基础概念：新类生成、改名 / 改判据跟着改、停用的标停用；汇总概念的成员改写成现编号
      · 迁移改了编号的那些已映射日子：重映射
B2  概念   类清单变了的 lane 交触点①写细分 / 汇总 / 情境概念（``scene_authoring``）  ← 写概念的 LLM 触点
B3  映射   今晚封口的那天 + 还没映射的、概念集变了要回填的日子，最新的先、每晚有预算
           （续跑按每条记录"挂在它类上的细分概念"认口径：只有真受影响的记录才问模型；情境变了就地重算）  ← 映射的 LLM 触点
B4  关系   （有模型建议时）候选调节条件补问、先验补问（都按输入摘要缓存）→
           每条 lane 全部两两检验（截止日 = 序列的截止日）→ 接上这条 lane 盘上之前最近的一晚折叠状态
           （候选 / 成立 / 失效 / 前向未复现）→ 这一晚的关系表与迁移日志落盘                 ← 纯算法、零模型
```

B4 读的是盘上**全部**已映射的命中（不只今天）：关系每晚全量重算，不做增量统计。

**读的是第 N 晚的事件序列**（``series.reader`` 读一次，组合根交进来，机会表用同一份）：映射哪几条、属于哪条 lane、
当天的空白都按它，行为树只用来取那几条的全文。第 N 晚只映射 N 之前（已封口）的日子。

**桥不读命中**：概念记着自己的类，节律按类编号从序列算（``temporal``；基础 / 细分 = 自己那个类，汇总 = 成员类合起来），
所以桥在映射之前就定了。常态是映射的**材料**，按裁定**不含当天**。

**跑法是可注入的对象**：组合根的 ``runtime/nightly.py`` 把结算、它与机会表串起来，``NightlyWorker`` 定拍。

**失败在哪停**：映射里模型失败不塌整天（映射器自己记未决）。

**不卡事件循环**：夜批跑在本地服务的事件循环上，读整支概念、同步词表、节律、时间线、两两检验与折叠、机会表的材料这些
纯计算都下沉到线程（``asyncio.to_thread``）；循环上只留调模型与落盘。
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, tzinfo

from habitus.behavior.tree import BehaviorTree
from habitus.behavior.uri import BehaviorURI
from habitus.conditions import DayTypeCalendar
from habitus.foundation.tracing import set_attributes, span
from habitus.infrastructure.lifecycle import RetryLedger
from habitus.runtime.scene_advice import RelationAdvice
from habitus.runtime.scene_authoring import ConceptAuthoring
from habitus.runtime.scene_night_mapping import DayMappingReport, MappingStage, select_mapping_days
from habitus.runtime.scene_night_relations import RelationStage
from habitus.scene.concepts.calendar_situations import calendar_key, most_specific
from habitus.scene.concepts.catalog import ClassCatalog
from habitus.scene.concepts.model import ConceptOrigin, ConceptSet, ContextScope
from habitus.scene.concepts.rhythm import Rhythm, RhythmRulers, rhythms_of
from habitus.scene.concepts.store import ConceptStore
from habitus.scene.concepts.sync import SyncPlan, plan_sync
from habitus.scene.occurrences.baselines import RECENT_WINDOW_DAYS
from habitus.scene.occurrences.mapper import ConceptMapper
from habitus.scene.occurrences.store import ConceptHitStore
from habitus.scene.recurrence import Recurrence, coverage_by_lane, recurrences_of
from habitus.scene.relations import (
    LaneTimeline,
    RelationConfig,
    Segment,
)
from habitus.scene.relations.state import Status
from habitus.scene.relations.store import RelationStore, RelationStoreError
from habitus.series import EventSeries
from habitus.temporal import Coverage


@dataclass(frozen=True)
class SceneNightConfig:
    """一夜要用到的数值。都是**待定值**，跑真实数据定完再写进 ``config.scene``。"""

    recent_days: int = RECENT_WINDOW_DAYS
    #: 常态按峰分份时两边各展几槽的容差，组合根从 ``config.scene.baseline_slack_slots`` 抄过来（裁定 59）。
    slack_slots: int = 0
    #: B3 一晚最多映射几天（今晚那天之外，补映射与回填共用）。保护闸：第一次装上、或概念集一变，历史可能有几百天，
    #: 一夜全问模型既慢又贵；有预算就一晚一晚往前补，最新的先。
    mapping_days_per_night: int = 14

    def __post_init__(self) -> None:
        if self.mapping_days_per_night <= 0:
            raise ValueError("mapping_days_per_night must be a positive integer")
        if self.slack_slots < 0:
            raise ValueError("slack_slots must be a non-negative integer")
        if self.recent_days <= 0:
            raise ValueError("recent_days must be a positive integer")


@dataclass(frozen=True)
class NightMaterials:
    """B4 这一晚用过的材料，交给排在语义树之后的那几拍复用（预测层的机会表要在**同一条**时间线上算）：
    同步词表之后的概念集、各 lane 的时间线。只在内存里传一晚，不落盘。"""

    concepts: ConceptSet
    timelines: Mapping[str, LaneTimeline]
    #: 每个类的复现规律（裁定 59）：预测层「离上次够久」从这里读，由机会表落盘。
    recurrences: Mapping[str, Recurrence] = field(default_factory=dict)
    #: 各 lane 的观测覆盖：机会表抄成每天看着的比例（账的久别用，26 号清单 3.3）。
    coverage: Mapping[str, Coverage] = field(default_factory=dict)


@dataclass(frozen=True)
class NightReport:
    """一夜的账：每一步各报自己的数，出了岔子的都在 ``signals`` 里，不静默。"""

    night: date
    concepts: int
    #: B3：这一晚映射了哪几天、问了模型的记录数、续跑的、只重算情境的、未决的；还剩几天没轮到（下一晚接着）。
    mapped_days: tuple[date, ...]
    mapped: int
    resumed: int
    refreshed: int
    unresolved: int
    backlog: int
    remapped_days: int
    #: B3：模型这一次没答成的（那几天不盖章、下一晚重问）；两遍判得不一样的（映射器一致性的生产读数）。
    model_failed: int = 0
    inconsistent: int = 0
    #: B4：每条 lane 各状态的关系数（样本不够 / 已检 / 候选 / 成立 / 失效 / 前向未复现）与这一晚的迁移条数。
    relations: Mapping[str, Mapping[str, int]] = field(default_factory=dict)
    transitions: int = 0
    signals: tuple[str, ...] = ()
    #: 这一晚的材料（见 ``NightMaterials``）；不进比较与打印。
    materials: NightMaterials | None = field(default=None, compare=False, repr=False)

    def summary(self) -> str:
        counts = "；".join(
            f"{lane} 候选 {numbers.get('candidate', 0)}、成立 {numbers.get('established', 0)}、失效 {numbers.get('expired', 0)}、前向未复现 {numbers.get('rejected', 0)}"
            for lane, numbers in sorted(self.relations.items())
        )
        return (
            f"第 {self.night} 晚：概念 {self.concepts} · 映射 {len(self.mapped_days)} 天 {self.mapped} 条"
            f"（续跑 {self.resumed}、只重算情境 {self.refreshed}、未决 {self.unresolved}"
            f"（其中两遍不一致 {self.inconsistent}、模型没答成 {self.model_failed}）、还欠 {self.backlog} 天）· "
            f"迁移重映射 {self.remapped_days} 天 · 关系 {counts or '无'}（迁移 {self.transitions} 条）"
        )


def _sync_attributes(plan: SyncPlan, concepts: ConceptSet, remapped: int) -> dict[str, object]:
    return {
        "version": plan.version,
        "written": len(plan.write),
        "rejected": len(plan.rejected),
        "moved": len(plan.moved),
        "remapped_days": remapped,
        "concepts": len(concepts),
    }


def _mapping_attributes(mapping: list[DayMappingReport], days: tuple[date, ...]) -> dict[str, object]:
    return {
        "mapped_days": len(days),
        "mapped": sum(item.mapped for item in mapping),
        "resumed": sum(item.resumed for item in mapping),
        "refreshed": sum(item.refreshed for item in mapping),
        "unresolved": sum(item.unresolved for item in mapping),
        "model_failed": sum(item.model_failed for item in mapping),
        "inconsistent": sum(item.inconsistent for item in mapping),
    }


def line_days(lines: Mapping[str, LaneTimeline]) -> tuple[date, ...]:
    """各条 lane 人在场的日子（日历备注要给的那些天）。"""

    return tuple(sorted({day for line in lines.values() for day in line.days()}))


class SceneNightlyRun:
    """一夜的跑法。构造时拿到的都是**已经建好的东西**，这个类不读配置、不认识 YAML（配置的换算在模块底部的
    ``relation_config``，由组合根调）。

    ``catalog`` 是词表口（组合根用 ``VocabularyReader`` 装的 ``ClassCatalog``）；``mapper_for`` 是"拿这一版
    概念集建一个映射器"（概念集在同步词表之后才定）。
    """

    def __init__(
        self,
        *,
        behavior_tree: BehaviorTree,
        catalog: ClassCatalog,
        concepts: ConceptStore,
        hits: ConceptHitStore,
        calendar: DayTypeCalendar,
        timezone: tzinfo,
        mapper_for: Callable[[ConceptSet], ConceptMapper],
        relations: RelationStore,
        relation_config: RelationConfig,
        authoring: ConceptAuthoring | None = None,
        advice: RelationAdvice | None = None,
        subject: str | None = None,
        rulers: RhythmRulers,
        mapping_retries: RetryLedger,
        config: SceneNightConfig | None = None,
    ) -> None:
        resolved = config or SceneNightConfig()
        self.behavior_tree = behavior_tree
        self.catalog = catalog
        self.concepts = concepts
        self.hits = hits
        self.calendar = calendar
        self.timezone = timezone
        self.mapper_for = mapper_for
        self.relations = relations
        self.relation_config = relation_config
        self.authoring = authoring
        self.advice = advice
        self.subject = subject
        self.config = resolved
        self.rulers = rulers
        self.mapping_retries = mapping_retries
        #: B3 映射与 B4 关系各是一个 stage；一夜的数值与模型建议每次从这里现读（构造后可换）。
        self.mapping = MappingStage(
            behavior_tree=behavior_tree,
            hits=hits,
            calendar=calendar,
            timezone=timezone,
            subject=subject,
            mapping_retries=mapping_retries,
            config=lambda: self.config,
        )
        self.relation_stage = RelationStage(
            catalog=catalog,
            concepts=concepts,
            hits=hits,
            calendar=calendar,
            relations=relations,
            relation_config=relation_config,
            advice=lambda: self.advice,
        )

    async def run(self, *, series: EventSeries, now: datetime) -> NightReport:
        """第 ``series.cutoff`` 晚，``now`` 是夜批时刻。节律从同一份 ``series`` 经 temporal 算（裁定 59）。B1–B4 各一个 span
        （``scene.b1``…``scene.b4``），属性是各步的输入、输出条数与关键读数。"""

        if now.utcoffset() is None:
            raise TypeError("now must be a timezone-aware datetime")
        signals: list[str] = []
        slack_minutes = self.config.slack_slots * self.relation_config.slot_minutes
        with span("scene.b1") as attributes:
            plan, concepts, mapper, coverage, rhythms, remapped = await self._sync(series, now, slack_minutes, signals)
            set_attributes(attributes, _sync_attributes(plan, concepts, remapped))
        if self.authoring is not None:
            with span("scene.b2", {"concepts_before": len(concepts)}) as attributes:
                before = set(concepts)
                concepts = await self.authoring.run(self.concepts, self.catalog.classes(), series, rhythms, signals)
                mapper = self.mapper_for(concepts)
                rhythms = await asyncio.to_thread(self._rhythms, series, concepts, coverage)
                # 只记条数：概念名是用户派生文字，不进 span（codex R3）
                set_attributes(attributes, {"concepts": len(concepts), "added": len(set(concepts) - before)})
        with span("scene.b3") as attributes:
            mapping, days, backlog = await self.mapping.map_backlog(
                series, concepts, mapper, rhythms, slack_minutes, now, signals
            )
            set_attributes(attributes, _mapping_attributes(mapping, days))
        with span("scene.b4") as attributes:
            relations, transitions, lines = await self.relation_stage.run(
                series, concepts, signals, vocabulary=plan.version, now=now
            )
            set_attributes(attributes, {"lanes": len(relations), "transitions": transitions})
        return NightReport(
            night=series.cutoff,
            concepts=len(concepts),
            mapped_days=days,
            mapped=sum(item.mapped for item in mapping),
            resumed=sum(item.resumed for item in mapping),
            refreshed=sum(item.refreshed for item in mapping),
            unresolved=sum(item.unresolved for item in mapping),
            model_failed=sum(item.model_failed for item in mapping),
            inconsistent=sum(item.inconsistent for item in mapping),
            backlog=backlog,
            remapped_days=remapped,
            relations=relations,
            transitions=transitions,
            signals=tuple(signals),
            materials=await asyncio.to_thread(self._materials, series, concepts, lines, coverage),
        )

    async def _sync(
        self, series: EventSeries, now: datetime, slack_minutes: int, signals: list[str]
    ) -> tuple[SyncPlan, ConceptSet, ConceptMapper, Mapping[str, Coverage], Mapping[str, Rhythm], int]:
        """B1 同步词表：概念落盘、迁移牵到的已映射日子重映射、记下同步到的版本；顺带算一晚只算一次的覆盖与节律。"""

        changes = self.catalog.changes_since(self.concepts.synced_version())
        # 日历情境的取值从数据里攒出来：序列里出现过的日子，日历给了哪些（键, 取值）就长哪些（裁定 59 第 7 条）
        observed = {
            item for day in sorted({record.day for record in series.records}) for item in self.calendar.of_day(day)
        }
        existing = await asyncio.to_thread(self.concepts.read_all)
        plan = await asyncio.to_thread(plan_sync, existing, self.catalog.classes(), changes, now=now, observed=observed)
        concepts = await asyncio.to_thread(self._write_concepts, plan, signals)
        mapper = self.mapper_for(concepts)
        # 各 lane 的观测覆盖只取决于序列：一晚算一次，节律（两次）与机会表的材料共用
        coverage = await asyncio.to_thread(coverage_by_lane, series, self.rulers)
        rhythms = await asyncio.to_thread(self._rhythms, series, concepts, coverage)
        remapped = await self._redo_moved(
            plan, concepts, mapper, rhythms, slack_minutes, series=series, now=now, signals=signals
        )
        self.concepts.write_synced_version(plan.version)
        return plan, concepts, mapper, coverage, rhythms, remapped

    def _materials(
        self,
        series: EventSeries,
        concepts: ConceptSet,
        lines: Mapping[str, LaneTimeline],
        coverage: Mapping[str, Coverage],
    ) -> NightMaterials:
        return NightMaterials(
            concepts=concepts,
            timelines=lines,
            recurrences=recurrences_of(series, self.rulers, coverage),
            coverage=coverage,
        )

    # ── B1 同步词表 ─────────────────────────────────────────────────────────

    def _write_concepts(self, plan: SyncPlan, signals: list[str]) -> ConceptSet:
        """概念落盘（新生成 / 改名改判据 / 停用 / 汇总成员改写），读回整个概念集。"""

        for definition in plan.write:
            self.concepts.write(definition)
        concepts = self.concepts.read_all()
        signals.extend(
            f"vocabulary: 「{definition.label}」（{definition.name}）没同步：{reason}（盘上没写，名字改开后下一晚补上）"
            for definition, reason in plan.rejected
        )
        calendar = 0
        for definition in plan.write:
            if definition.source.origin is ConceptOrigin.CALENDAR:
                calendar += 1
            elif definition.retired:
                signals.append(f"vocabulary: 「{definition.label}」停用了")
            else:
                signals.append(f"vocabulary: 「{definition.label}」（{definition.name}）已同步")
        if calendar:
            signals.append(
                f"vocabulary: 补上日历情境 {calendar} 个（数据里新出现的周几、几号、月末、日型取值，裁定 59）"
            )
        return concepts

    async def _redo_moved(
        self,
        plan: SyncPlan,
        concepts: ConceptSet,
        mapper: ConceptMapper,
        rhythms: Mapping[str, Rhythm],
        slack_minutes: int,
        *,
        series: EventSeries,
        now: datetime,
        signals: list[str],
    ) -> int:
        """迁移改了编号的那些已映射日子重映射（没映射过的由 B3 照常处理），返回重映射了几天。

        也包括它们之后 ``recent_days`` 天里已映射的日子：要"近几天同类记录"的细分概念，材料随别的日子的编号变了
        （续跑按材料摘要认出来、只重判那几条）。不限于今天之前——夜批补跑旧日子时，更晚的日子也可能已经映射过。
        """

        done = set(self.hits.days_done())
        moved = {BehaviorURI.parse(item.uri).to_address().occurred_on for item in plan.moved}
        reach = (
            mapper.config.recent_days
            if any(concepts[identity].context is ContextScope.RECENT for identity in concepts.behaviors())
            else 0
        )
        affected = {moved_day + timedelta(days=offset) for moved_day in moved for offset in range(reach + 1)}
        remapped = 0
        for past in sorted(affected & done):
            if past >= series.cutoff:
                continue  # 还没封口的日子不在这一份序列里，等它封口那一晚照常映射
            context = self.mapping.context(past, concepts, rhythms, slack_minutes)
            mapping = await self.mapping.map(past, mapper, context, series, now=now, force=True)
            if not (past in moved or mapping.changed):
                continue
            remapped += 1
            signals.append(f"vocabulary: {past} 受迁移影响 → 重判 {mapping.rewritten} 条")
        return remapped

    # ── B2–B3 ───────────────────────────────────────────────────────────────

    def _rhythms(
        self, series: EventSeries, concepts: ConceptSet, coverage: Mapping[str, Coverage]
    ) -> Mapping[str, Rhythm]:
        return rhythms_of(
            series, concepts, self.rulers, anchors=self._anchors(concepts, series.cutoff), coverage=coverage
        )

    def _anchors(self, concepts: ConceptSet, night: date) -> dict[str, tuple[str, ...]]:
        """后果 → 上一晚已经成立的日历关系说的那几种日子（只留最具体的那一族；裁定 59）。读不到上一晚就没有。"""

        held: dict[str, list[str]] = {}
        for lane in self.relations.lanes():
            try:
                state = self.relations.previous(lane, night)
            except RelationStoreError:
                # 读不到上一晚就按没有锚日算（见文档串）
                continue
            if state is None:
                continue
            for relation in state.of(Status.ESTABLISHED):
                key = relation.key
                if key.segment is Segment.SITUATION and not key.condition and relation.upward:
                    if calendar_key(concepts, key.antecedent) is not None:
                        held.setdefault(key.consequent, []).append(key.antecedent)
        return {
            consequent: tuple(concepts[item].name for item in most_specific(concepts, antecedents))
            for consequent, antecedents in held.items()
        }


__all__ = [
    "NightMaterials",
    "NightReport",
    "SceneNightConfig",
    "SceneNightlyRun",
    "select_mapping_days",
]
