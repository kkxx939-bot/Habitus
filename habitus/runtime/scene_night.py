"""夜批：第 N 晚把语义树的几步按固定顺序跑一遍（语义树新方案 ``13`` 第五节）。

顺序是**线性的、不级联的**（派生树按固定先后处理已封口的历史，不做"输入又变了就重建"）：

```
B1  同步词表（裁定 20）：读上次同步之后的变更
      · 每个类一个基础概念：新类生成、改名 / 改判据跟着改、停用的标停用；汇总概念的成员改写成现编号
      · 迁移改了编号的那些已映射日子：重映射
    （预测树由组合根用同一份序列先建好，本模块收到的是那一代树，见 ``runtime/nightly.py``）
B2  概念   类清单变了的 lane 交触点①写细分 / 汇总 / 情境概念（``scene_authoring``）  ← 写概念的 LLM 触点
B3  映射   今晚封口的那天 + 还没映射的、概念集变了要回填的日子，最新的先、每晚有预算
           （续跑按每条记录"挂在它类上的细分概念"认口径：只有真受影响的记录才问模型；情境变了就地重算）  ← 映射的 LLM 触点
B4  关系   （有模型建议时）候选调节条件补问、先验补问（都按输入摘要缓存）→
           每条 lane 全部两两检验（截止日 = 序列的截止日）→ 接上这条 lane 盘上之前最近的一晚折叠状态
           （候选 / 成立 / 失效 / 前向未复现）→ 这一晚的关系表与迁移日志落盘                 ← 纯算法、零模型
```

B4 读的是盘上**全部**已映射的命中（不只今天）：关系每晚全量重算，不做增量统计。

**读的是第 N 晚的事件序列**（与预测树同一份，``series.reader`` 读一次，组合根交进来）：映射哪几条、属于哪条 lane、
当天的空白都按它，行为树只用来取那几条的全文。第 N 晚只映射 N 之前（已封口）的日子。

**桥不读命中**：概念记着自己的类，曲线按类编号从预测树取（基础 / 细分 = 自己那个类，汇总 = 成员类现编号相加），
所以桥在映射之前就定了。常态是映射的**材料**，按裁定**不含当天**。

**还没接进常驻 worker**（第 11 步）：跑法是**可注入的对象**，组合根的 ``runtime/nightly.py`` 把预测树与它串起来。

**失败在哪停**：映射里模型失败不塌整天（映射器自己记未决）。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, tzinfo

from habitus.behavior.tree import BehaviorTree
from habitus.behavior.uri import BehaviorURI
from habitus.config import HabitusConfig
from habitus.prediction.model import PredictionTree
from habitus.runtime.scene_advice import RelationAdvice
from habitus.runtime.scene_authoring import ConceptAuthoring
from habitus.runtime.scene_rhythms import TreeRhythms, class_curves
from habitus.runtime.scene_situations import SceneDayContext
from habitus.scene.calendar import DayTypeCalendar
from habitus.scene.concepts.catalog import ClassCatalog
from habitus.scene.concepts.model import ConceptSet, ContextScope
from habitus.scene.concepts.rhythm import Rhythm
from habitus.scene.concepts.store import ConceptStore
from habitus.scene.concepts.sync import SyncPlan, plan_sync
from habitus.scene.occurrences.baselines import RECENT_WINDOW_DAYS
from habitus.scene.occurrences.mapper import ConceptMapper, DayMappingReport, map_closed_day
from habitus.scene.occurrences.store import ConceptHitStore
from habitus.scene.relations import (
    LaneTests,
    LaneTimeline,
    RelationConfig,
    RelationKey,
    RelationThresholds,
    build_timelines,
)
from habitus.scene.relations.state import LaneState, Status, fold
from habitus.scene.relations.store import RelationStore, RelationStoreError
from habitus.series import EventSeries


@dataclass(frozen=True)
class SceneNightConfig:
    """一夜要用到的数值。都是**待定值**，跑真实数据定完再写进 ``config.scene``。"""

    recent_days: int = RECENT_WINDOW_DAYS
    #: 常态按峰分份时两边各展几槽的容差 = 预测树的 ``prediction.pool_half_width``（2026-10-01 用户定复用它）。
    #: 组合根从预测配置抄过来；这里不另起一个数。
    slack_slots: int = 0
    #: B3 一晚最多映射几天（今晚那天之外，补映射与回填共用）。保护闸：第一次装上、或概念集一变，历史可能有几百天，
    #: 一夜全问模型既慢又贵；有预算就一晚一晚往前补，最新的先。
    mapping_days_per_night: int = 14

    def __post_init__(self) -> None:
        if (
            isinstance(self.mapping_days_per_night, bool)
            or not isinstance(self.mapping_days_per_night, int)
            or self.mapping_days_per_night <= 0
        ):
            raise ValueError("mapping_days_per_night must be a positive integer")
        if isinstance(self.slack_slots, bool) or not isinstance(self.slack_slots, int) or self.slack_slots < 0:
            raise ValueError("slack_slots must be a non-negative integer")
        if isinstance(self.recent_days, bool) or not isinstance(self.recent_days, int) or self.recent_days <= 0:
            raise ValueError("recent_days must be a positive integer")


def relation_config(config: HabitusConfig) -> RelationConfig:
    """关系检验的配置：和时间有关的三个数只认预测树（槽宽、转移窗口、久别重来），统计门槛取 ``config.scene.relations``
    （没写的用 ``RelationThresholds`` 的默认值）。"""

    prediction = config.prediction
    if not prediction.enabled:
        raise ValueError("relation tests take their clock from the prediction tree; config.prediction must be enabled")
    return RelationConfig(
        slot_minutes=prediction.slot_minutes,  # type: ignore[arg-type]
        transition_window_slots=prediction.transition_window_slots,  # type: ignore[arg-type]
        recurrence_window_days=prediction.recurrence_window_days,  # type: ignore[arg-type]
        thresholds=RelationThresholds(**config.scene.relations.overrides()),
    )


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
        config: SceneNightConfig | None = None,
    ) -> None:
        if not isinstance(behavior_tree, BehaviorTree):
            raise TypeError("behavior_tree must be a BehaviorTree")
        if not isinstance(catalog, ClassCatalog):
            raise TypeError("catalog must implement ClassCatalog")
        resolved = config or SceneNightConfig()
        if not isinstance(resolved, SceneNightConfig):
            raise TypeError("config must be SceneNightConfig")
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

    async def run(self, *, series: EventSeries, tree: PredictionTree, now: datetime) -> NightReport:
        """第 ``series.cutoff`` 晚。``tree`` 是用同一份 ``series`` 建出来的那一代，``now`` 是夜批时刻。"""

        if not isinstance(now, datetime) or now.utcoffset() is None:
            raise TypeError("now must be a timezone-aware datetime")
        if not isinstance(series, EventSeries):
            raise TypeError("series must be an EventSeries")
        signals: list[str] = []
        plan = plan_sync(
            self.concepts.read_all(),
            self.catalog.classes(),
            self.catalog.changes_since(self.concepts.synced_version()),
            now=now,
        )
        concepts = self._write_concepts(plan, signals)
        slack_minutes = self.config.slack_slots * tree.slot_minutes
        mapper = self.mapper_for(concepts)
        rhythms = self._rhythms(tree, concepts)
        remapped = await self._redo_moved(
            plan, concepts, mapper, rhythms, slack_minutes, series=series, now=now, signals=signals
        )
        self.concepts.write_synced_version(plan.version)

        if self.authoring is not None:
            concepts = await self.authoring.run(self.concepts, self.catalog.classes(), series, rhythms, signals)
            mapper = self.mapper_for(concepts)
            rhythms = self._rhythms(tree, concepts)

        mapping, days, backlog = await self._map_backlog(series, concepts, mapper, rhythms, slack_minutes, now, signals)
        relations, transitions = await self._relations(series, concepts, signals)
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
        chosen = [day for day in pending if day == today]
        chosen += [day for day in pending if day != today][: self.config.mapping_days_per_night]
        reports = []
        for day in chosen:
            context = self._context(day, concepts, rhythms, slack_minutes)
            if day == today:
                signals.extend(f"baseline: {key} 样本不够，引用它的候选会记未决" for key in context.baselines.missing)
                signals.extend(f"drift: {item.render()}" for item in context.baselines.drifting)
            reports.append(await self._map(day, mapper, context, series, now=now))
        backlog = len(pending) - len(chosen)
        if backlog:
            signals.append(f"mapping: 还有 {backlog} 天没轮到（没映射过或概念集变了要回填），之后几晚接着补")
        return reports, tuple(chosen), backlog

    # ── B4 关系 ─────────────────────────────────────────────────────────────

    async def _relations(
        self, series: EventSeries, concepts: ConceptSet, signals: list[str]
    ) -> tuple[dict[str, dict[str, int]], int]:
        """第 ``series.cutoff`` 晚：每条 lane 检验、接上盘上之前最近的一晚折叠、落盘。"""

        hits = {record.occurrence_uri: record for day in self.hits.days_done() for record in self.hits.read_day(day)}
        lines = build_timelines(series, concepts, hits, self.relation_config)
        counts: dict[str, dict[str, int]] = {}
        moved = 0
        notes = {day: note for day in line_days(lines) if (note := self.calendar.describe(day))}
        for lane, line in sorted(lines.items()):
            tests = LaneTests(line, concepts, self.relation_config, series.cutoff, notes=notes)
            weights: dict[RelationKey, float] = {}
            if self.advice is not None:
                proposals = await self.advice.proposals(tests, series.cutoff, signals)
                tests = LaneTests(line, concepts, self.relation_config, series.cutoff, notes=notes, proposals=proposals)
                weights = await self.advice.weights(tests, signals)
            state, transitions = fold(self._previous(lane, series.cutoff, signals), tests, weights or None)
            self.relations.write(state, transitions)
            counts[lane] = {status.value: state.count(status) for status in Status}
            moved += len(transitions)
            signals.extend(
                f"relation: {lane} {concepts.label_of(item.key.antecedent)} → {concepts.label_of(item.key.consequent)}"
                f"（{item.key.segment.value}{'｜' + item.key.condition if item.key.condition else ''}）"
                f"{'—' if item.before is None else item.before.value} → {item.after.value}：{item.reason}"
                for item in transitions
            )
        return counts, moved

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
        for definition in plan.write:
            if definition.retired:
                signals.append(f"vocabulary: 「{definition.label}」停用了")
            else:
                signals.append(f"vocabulary: 「{definition.label}」（{definition.name}）已同步")
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

    @staticmethod
    def _rhythms(tree: PredictionTree, concepts: ConceptSet) -> Mapping[str, Rhythm]:
        labels = {identity: concepts[identity].label for identity in concepts.behaviors()}
        return TreeRhythms(tree, class_curves(concepts), labels).rhythms(concepts.behaviors())

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


__all__ = ["NightReport", "SceneNightConfig", "SceneNightlyRun", "relation_config"]
