"""B1 同步词表（裁定 20）：让基础概念跟上词表；迁移改了编号的那些已映射日子重映射（映射的 LLM 触点）。

``plan`` 是纯读盘、纯计算；``run`` 把概念落盘（按全部类核对，幂等），``remap`` 时接着重映射、记下同步到的版本。模型队列走
``remap``；换代不调模型，只落概念——有迁移牵到的日子要重映射时不记版本（记了队列就读不到那份迁移清单了），所以谁先做结果
都一样。
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta

from habitus.behavior.uri import BehaviorURI
from habitus.conditions import DayTypeCalendar
from habitus.runtime.scene_night_mapping import MappingStage
from habitus.scene.concepts.catalog import ClassCatalog
from habitus.scene.concepts.model import ConceptOrigin, ConceptSet, ContextScope
from habitus.scene.concepts.rhythm import RhythmRulers
from habitus.scene.concepts.store import ConceptStore
from habitus.scene.concepts.sync import SyncPlan, plan_sync
from habitus.scene.occurrences.mapper import ConceptMapper
from habitus.scene.occurrences.store import ConceptHitStore
from habitus.scene.recurrence import coverage_by_lane
from habitus.series import EventSeries
from habitus.temporal import Coverage


@dataclass(frozen=True)
class Synced:
    """同步词表之后的那一份：同步到的版本、落盘后的概念集、各 lane 的观测覆盖、迁移重映射了几天。"""

    version: int
    concepts: ConceptSet
    coverage: Mapping[str, Coverage]
    remapped: int = 0


class SyncStage:
    def __init__(
        self,
        *,
        catalog: ClassCatalog,
        concepts: ConceptStore,
        hits: ConceptHitStore,
        calendar: DayTypeCalendar,
        mapping: MappingStage,
        mapper_for: Callable[[ConceptSet], ConceptMapper],
        rulers: RhythmRulers,
    ) -> None:
        self.catalog = catalog
        self.concepts = concepts
        self.hits = hits
        self.calendar = calendar
        self.mapping = mapping
        self.mapper_for = mapper_for
        self.rulers = rulers

    async def plan(self, series: EventSeries, now: datetime) -> SyncPlan:
        """这一次同步词表要做的事（纯读盘、纯计算）。"""

        changes = self.catalog.changes_since(self.concepts.synced_version())
        # 日历情境的取值从数据里攒出来：序列里出现过的日子，日历给了哪些（键, 取值）就长哪些（裁定 59 第 7 条）
        observed = {
            item for day in sorted({record.day for record in series.records}) for item in self.calendar.of_day(day)
        }
        existing = await asyncio.to_thread(self.concepts.read_all)
        return await asyncio.to_thread(plan_sync, existing, self.catalog.classes(), changes, now=now, observed=observed)

    def owed(self, plan: SyncPlan) -> bool:
        """同步词表还欠着：有概念要写、有迁移牵到的日子要重映射、或同步到的版本还没记。"""

        return bool(plan.write or plan.moved) or plan.version != self.concepts.synced_version()

    async def run(
        self, series: EventSeries, now: datetime, signals: list[str], *, remap: bool, plan: SyncPlan | None = None
    ) -> Synced:
        """B1 同步词表：概念落盘；``remap`` 时迁移牵到的已映射日子重映射（调模型）、记下同步到的版本。不 ``remap``（换代）时
        只在没有迁移牵到的日子要重映射时才记版本——记了队列就读不到那份迁移清单了。顺带算一晚只算一次的覆盖。``plan`` 是
        刚算好的那一份（不给就现算）。"""

        if plan is None:
            plan = await self.plan(series, now)
        concepts = await asyncio.to_thread(self._write_concepts, plan, signals)
        # 各 lane 的观测覆盖只取决于序列：节律与机会表的材料共用
        coverage = await asyncio.to_thread(coverage_by_lane, series, self.rulers)
        remapped = 0
        if remap:
            remapped = await self._redo_moved(plan, concepts, self.mapper_for(concepts), series=series, signals=signals)
        if (remap or not plan.moved) and plan.version != self.concepts.synced_version():
            self.concepts.write_synced_version(plan.version)
        return Synced(plan.version, concepts, coverage, remapped)

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
        *,
        series: EventSeries,
        signals: list[str],
    ) -> int:
        """迁移改了编号的那些已映射日子重判受影响的那几条（没映射过的由 B3 照常处理），返回重映射了几天。重判过的日子
        完成章撤掉，由封口重盖（``MappingStage.seal``）。

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
            rewritten, removed = await self.mapping.rejudge_day(past, concepts, mapper, series)
            if not (past in moved or rewritten or removed):
                continue
            remapped += 1
            signals.append(f"vocabulary: {past} 受迁移影响 → 重判 {rewritten} 条")
        return remapped


__all__ = ["SyncStage", "Synced"]
