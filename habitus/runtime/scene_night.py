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
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, date, datetime, timedelta, tzinfo

from habitus.behavior.tree import BehaviorTree
from habitus.behavior.uri import BehaviorURI
from habitus.conditions import DayTypeCalendar
from habitus.infrastructure.lifecycle import RetryLedger
from habitus.runtime.scene_advice import RelationAdvice
from habitus.runtime.scene_authoring import ConceptAuthoring
from habitus.runtime.scene_situations import SceneDayContext
from habitus.scene.advice.store import AdviceStoreError
from habitus.scene.concepts.calendar_situations import calendar_key, most_specific, value_sets_to_add
from habitus.scene.concepts.catalog import CatalogChanges, ClassCatalog
from habitus.scene.concepts.model import ConceptOrigin, ConceptSet, ContextScope
from habitus.scene.concepts.rhythm import Rhythm, RhythmRulers, rhythms_of
from habitus.scene.concepts.store import ConceptStore
from habitus.scene.concepts.sync import SyncPlan, plan_sync
from habitus.scene.occurrences.baselines import RECENT_WINDOW_DAYS
from habitus.scene.occurrences.mapper import ConceptMapper, ConceptMapperError, DayMappingReport, map_closed_day
from habitus.scene.occurrences.model import ConceptHits
from habitus.scene.occurrences.situations import history_days
from habitus.scene.occurrences.store import ConceptHitStore
from habitus.scene.recurrence import Recurrence, coverage_by_lane, recurrences_of
from habitus.scene.relations import (
    LaneTests,
    LaneTimeline,
    RelationConfig,
    RelationKey,
    Segment,
    build_timelines,
)
from habitus.scene.relations.state import LaneState, Status, Transition, fold, revert_touched
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

    async def run(self, *, series: EventSeries, now: datetime) -> NightReport:
        """第 ``series.cutoff`` 晚，``now`` 是夜批时刻。节律从同一份 ``series`` 经 temporal 算（裁定 59）。"""

        if now.utcoffset() is None:
            raise TypeError("now must be a timezone-aware datetime")
        signals: list[str] = []
        changes = self.catalog.changes_since(self.concepts.synced_version())
        # 日历情境的取值从数据里攒出来：序列里出现过的日子，日历给了哪些（键, 取值）就长哪些（裁定 59 第 7 条）
        observed = {
            item for day in sorted({record.day for record in series.records}) for item in self.calendar.of_day(day)
        }
        existing = await asyncio.to_thread(self.concepts.read_all)
        plan = await asyncio.to_thread(plan_sync, existing, self.catalog.classes(), changes, now=now, observed=observed)
        concepts = await asyncio.to_thread(self._write_concepts, plan, signals)
        slack_minutes = self.config.slack_slots * self.relation_config.slot_minutes
        mapper = self.mapper_for(concepts)
        # 各 lane 的观测覆盖只取决于序列：一晚算一次，节律（两次）与机会表的材料共用
        coverage = await asyncio.to_thread(coverage_by_lane, series, self.rulers)
        rhythms = await asyncio.to_thread(self._rhythms, series, concepts, coverage)
        remapped = await self._redo_moved(
            plan, concepts, mapper, rhythms, slack_minutes, series=series, now=now, signals=signals
        )
        self.concepts.write_synced_version(plan.version)

        if self.authoring is not None:
            concepts = await self.authoring.run(self.concepts, self.catalog.classes(), series, rhythms, signals)
            mapper = self.mapper_for(concepts)
            rhythms = await asyncio.to_thread(self._rhythms, series, concepts, coverage)

        mapping, days, backlog = await self._map_backlog(series, concepts, mapper, rhythms, slack_minutes, now, signals)
        relations, transitions, lines = await self._relations(
            series, concepts, signals, vocabulary=plan.version, now=now
        )
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

    # ── B3 映射 ─────────────────────────────────────────────────────────────

    async def _map_backlog(
        self,
        series: EventSeries,
        concepts: ConceptSet,
        mapper: ConceptMapper,
        rhythms: Mapping[str, Rhythm],
        slack_minutes: int,
        now: datetime,
        signals: list[str],
    ) -> tuple[list[DayMappingReport], tuple[date, ...], int]:
        """要映射的日子：序列里有记录、还没以现在的概念集口径完成的（没映射过、或概念集变了要回填），最新的先；
        今晚封口的那天总在里面，其余按预算。返回（各天的报告, 映射了哪几天, 还剩几天）。"""

        recorded = sorted({record.day for record in series.records}, reverse=True)
        # 完成标记按这一天自己的口径认（那天出现的类各自的口径 + 情境；E9），条数也要与序列对得上（盖章之后又补发了记录就重做；E1）
        current = {
            day
            for day in self.hits.days_done()
            if (marker := self.hits.read_marker(day)) is not None
            and marker.records == len(series.on(day))
            and marker.mapper == mapper.day_version(record.kind_token for record in series.on(day))
        }
        pending = [day for day in recorded if day not in current]
        today = series.last_day
        # 模型没答成的日子按重试账退避、次数用完就放弃，不再每晚重排、不占 14 天预算（26 号清单 2.4.6）；
        # 账的键带着这一天的映射口径，概念集一变就是新键，重新给机会。今晚封口的那天照常排进来。
        versions = {day: mapper.day_version(record.kind_token for record in series.on(day)) for day in pending}
        chosen, waiting = select_mapping_days(
            pending,
            today=today,
            budget=self.config.mapping_days_per_night,
            blocked=lambda day: self.mapping_retries.blocked(_mapping_key(day, versions[day])),
        )
        reports = []
        # 按日子先后映射（挑哪几天仍是最新的先）：读历史的派生情境（「昨天吃过早餐」「连续研发中」）要前几天先映射好
        for day in sorted(chosen):
            context = self._context(day, concepts, rhythms, slack_minutes)
            if day == today:
                signals.extend(f"baseline: {key} 样本不够，引用它的候选会记未决" for key in context.baselines.missing)
                signals.extend(f"drift: {item.render()}" for item in context.baselines.drifting)
            key = _mapping_key(day, versions[day])
            try:
                report = await self._map(day, mapper, context, series, now=now)
            except ConceptMapperError as exc:
                # 一条记录写不出（材料超了记录的上限等）：这一天不盖章、记进重试账，其余日子与 B4 照常（26 号清单 2.4.6）
                decision = self.mapping_retries.record_failure(key, exc)
                signals.append(
                    f"mapping: {day} 写不出（{exc}），"
                    f"{'先不再重排（概念集变了会重新给机会）' if decision.give_up else '按重试账退避后重做'}"
                )
                continue
            if report.model_failed:
                decision = self.mapping_retries.record_failure(
                    key, MappingModelFailed(f"{report.model_failed} unresolved")
                )
                if decision.give_up:
                    signals.append(f"mapping: {day} 模型几次都没答成，先不再重排（概念集变了会重新给机会）")
            else:
                self.mapping_retries.clear(key)
            reports.append(report)
        await self._revisit_readers(
            [report.day for report in reports], concepts, mapper, rhythms, slack_minutes, series, now, signals
        )
        backlog = len(pending) - len(chosen) - len(waiting)
        if backlog:
            signals.append(f"mapping: 还有 {backlog} 天没轮到（没映射过或概念集变了要回填），之后几晚接着补")
        if waiting:
            signals.append(f"mapping: {len(waiting)} 天模型没答成，在退避或已放弃")
        return reports, tuple(chosen), backlog

    async def _revisit_readers(
        self,
        mapped: Sequence[date],
        concepts: ConceptSet,
        mapper: ConceptMapper,
        rhythms: Mapping[str, Rhythm],
        slack_minutes: int,
        series: EventSeries,
        now: datetime,
        signals: list[str],
    ) -> None:
        """这一晚映射过的日子之后、读得到它们的那几天（派生情境往回看的天数之内），早已映射完的再过一遍：
        候选的输入没变就只就地重算情境、不问模型（续跑），这样前一天后补的命中能进后一天的「昨天…」「连续…」。"""

        depth = len(history_days(concepts, series.last_day)) - 1
        if depth <= 0 or not mapped:
            return
        done = set(self.hits.days_done())
        readers = {day + timedelta(days=step) for day in mapped for step in range(1, depth + 1)}
        for day in sorted(readers - set(mapped)):
            if day not in done or day >= series.cutoff:
                continue
            context = self._context(day, concepts, rhythms, slack_minutes)
            try:
                await self._map(day, mapper, context, series, now=now)
            except ConceptMapperError as exc:
                signals.append(f"mapping: {day} 回访重算情境时写不出（{exc}），之后照常补")

    # ── B4 关系 ─────────────────────────────────────────────────────────────

    async def _relations(
        self,
        series: EventSeries,
        concepts: ConceptSet,
        signals: list[str],
        *,
        vocabulary: int = 0,
        now: datetime | None = None,
    ) -> tuple[dict[str, dict[str, int]], int, dict[str, LaneTimeline]]:
        """第 ``series.cutoff`` 晚：每条 lane 检验、接上盘上之前最近的一晚折叠、落盘。返回各 lane 的计数、迁移条数与时间线。

        ``vocabulary`` 是这一晚同步到的词表版本，记在关系状态上。接的那一晚的状态记着它当时的词表版本，这之间拆分 / 合并改了
        成员的类，涉及它们的关系先退回候选（裁定 59）；按状态上记的版本算，同一晚重跑也退回得一样。折叠之后，同一件事有两个以上
        单个取值（周几、几号、日型）的日历关系处在候选或成立的，补一个取值集合的日历情境，下一晚起一起检验。"""

        hits = await asyncio.to_thread(self._all_hits)
        lines = await asyncio.to_thread(
            build_timelines, series, concepts, hits, self.relation_config, calendar=self.calendar
        )
        counts: dict[str, dict[str, int]] = {}
        moved = 0
        for lane, line in sorted(lines.items()):
            previous = self._previous(lane, series.cutoff, signals)
            reverted: tuple[Transition, ...] = ()
            if previous is not None:
                touched = _touched(concepts, _changed_classes(self.catalog.changes_since(previous.vocabulary)))
                if touched:
                    previous, reverted = revert_touched(previous, touched, series.cutoff)
            # 第二个条件起只长在接的那一晚已成立的带条件关系上（裁定 62 第 1 条）：退回之后再数哪些成立
            established = tuple(item.key for item in previous.of(Status.ESTABLISHED)) if previous is not None else ()
            tests = LaneTests(line, concepts, self.relation_config, series.cutoff, established=established)
            weights: dict[RelationKey, float] = {}
            if self.advice is not None:
                try:
                    proposals = await self.advice.proposals(tests, series.cutoff, signals)
                except AdviceStoreError as exc:
                    # 提议簿读不了：不带它们折叠会把接着的带条件关系当成"行没了"判失效，这条 lane 这一晚不写、沿用上一晚的
                    signals.append(f"relations: {lane} 提议簿读不了（{exc}），这一晚的关系表沿用上一晚的")
                    continue
                tests = LaneTests(
                    line, concepts, self.relation_config, series.cutoff, proposals=proposals, established=established
                )
                weights = await self.advice.weights(tests, signals)
            state, transitions = await asyncio.to_thread(fold, previous, tests, weights or None)
            state = replace(state, vocabulary=vocabulary)
            transitions = (*reverted, *transitions)
            try:
                self.relations.write(state, transitions)
            except RelationStoreError as exc:
                # 这条 lane 这一晚写不出（超限、盘上已有更晚一晚）：只让它这一晚失败、明天的预测层读上一代的表，
                # 不让整晚停、别的 lane 照常（26 号清单 2.5.2）
                signals.append(f"relations: {lane} 这一晚的关系表写不出（{exc}），沿用上一晚的")
                continue
            self._grow_value_sets(concepts, state, now or datetime.now(UTC), signals)
            counts[lane] = {status.value: state.count(status) for status in Status}
            moved += len(transitions)
            signals.extend(
                f"relation: {lane} {concepts.label_of(item.key.antecedent)} → {concepts.label_of(item.key.consequent)}"
                f"（{item.key.segment.value}{'｜' + item.key.condition if item.key.condition else ''}）"
                f"{'—' if item.before is None else item.before.value} → {item.after.value}：{item.reason}"
                for item in transitions
            )
        return counts, moved, lines

    def _all_hits(self) -> dict[str, ConceptHits]:
        return {record.occurrence_uri: record for day in self.hits.days_done() for record in self.hits.read_day(day)}

    def _grow_value_sets(self, concepts: ConceptSet, state: LaneState, now: datetime, signals: list[str]) -> None:
        held: dict[str, list[str]] = {}
        for relation in state.relations:
            key = relation.key
            if (
                key.segment is Segment.SITUATION
                and not key.condition
                and relation.upward
                and relation.status in (Status.CANDIDATE, Status.ESTABLISHED)
            ):
                held.setdefault(key.consequent, []).append(key.antecedent)
        for definition in value_sets_to_add(concepts, held, now=now):
            self.concepts.write(definition)
            signals.append(f"vocabulary: 补上日历取值集合「{definition.name}」，下一晚起一起检验（裁定 59）")

    # ── B1 同步词表 ─────────────────────────────────────────────────────────

    def _previous(self, lane: str, night: date, signals: list[str]) -> LaneState | None:
        """这一晚要接的那一晚。最近那一晚的文件坏了：接在更早一个读得了的晚上之后，并报出来（第四轮评审 E10）——
        不然夜批永远写不出新的一晚，预测层也一直读着坏文件硬拒，两边互相卡死，只能人工恢复。新的一晚写出来，预测层下一拍就好了。"""

        try:
            return self.relations.previous(lane, night)
        except RelationStoreError as exc:
            for earlier in reversed(self.relations.nights(lane)):
                if earlier >= night:
                    continue
                try:
                    state = self.relations.read(lane, earlier)
                except RelationStoreError:
                    # 这一晚也读不了：接着往更早找；都读不了时下面整体报一次
                    continue
                signals.append(f"relations: {lane} 更晚的关系表读不了（{exc}），这一晚接在第 {earlier} 晚之后折叠")
                return state
            signals.append(f"relations: {lane} 之前的关系表都读不了（{exc}），这一晚从头折叠")
            return None

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
            context = self._context(past, concepts, rhythms, slack_minutes)
            mapping = await self._map(past, mapper, context, series, now=now, force=True)
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

    def _context(
        self, day: date, concepts: ConceptSet, rhythms: Mapping[str, Rhythm], slack_minutes: int
    ) -> SceneDayContext:
        return SceneDayContext(
            day,
            concepts=concepts,
            hits=self.hits,
            calendar=self.calendar,
            timezone=self.timezone,
            subject=self.subject,
            recent_days=self.config.recent_days,
            rhythms=rhythms,
            slack_minutes=slack_minutes,
        )

    async def _map(
        self,
        day: date,
        mapper: ConceptMapper,
        context: SceneDayContext,
        series: EventSeries,
        *,
        now: datetime,
        force: bool = False,
    ) -> DayMappingReport:
        return await map_closed_day(
            self.behavior_tree,
            self.hits,
            mapper,
            day,
            now=now,
            series=series,
            situation_for=context.situation_for,
            baseline_for=context.baseline_for,
            force=force,
        )


def select_mapping_days(
    pending: Sequence[date], *, today: date | None, budget: int, blocked: Callable[[date], bool]
) -> tuple[list[date], list[date]]:
    """这一晚映射哪几天（今晚封口的那天总在里面，其余最新的先、按预算）、哪几天在重试账里等着（退避中或已放弃）。"""

    waiting = [day for day in pending if day != today and blocked(day)]
    chosen = [day for day in pending if day == today]
    chosen += [day for day in pending if day != today and day not in waiting][:budget]
    return chosen, waiting


class MappingModelFailed(RuntimeError):
    """这一天有记录模型没答成（可重试：传输、限流、输出不合格）。"""


def _mapping_key(day: date, version: str) -> str:
    return f"map:{day.isoformat()}:{version}"


__all__ = [
    "NightMaterials",
    "NightReport",
    "SceneNightConfig",
    "SceneNightlyRun",
    "select_mapping_days",
]


def _changed_classes(changes: CatalogChanges) -> frozenset[str]:
    """这一段词表变更里成员变了的类：拆出与交出的、并掉与并入的、条目被搬过的两头。"""

    found = {*changes.split_from, *changes.split_from.values(), *changes.merged_into, *changes.merged_into.values()}
    for moved in changes.moved:
        found.update((moved.source, moved.target))
    return frozenset(found)


def _touched(concepts: ConceptSet, changed: frozenset[str]) -> frozenset[str]:
    """成员变了的类牵到的行为概念：基础、细分、包含它的汇总。"""

    if not changed:
        return frozenset()
    return frozenset(identity for identity in concepts.behaviors() if set(concepts.classes_of(identity)) & changed)
