"""B1 同步词表（裁定 20）：让行为概念跟上在用的词（不在用了的停用）、补上数据里新出现的日历情境，记下同步到的版本。不调模型。

``plan`` 是纯读盘、纯计算；``run`` 把概念落盘（按全部在用的词核对，幂等）、记下同步到的版本（在用状态的修订号）。
在用的词由调用方显式交进来（换代交这一代数出来的那一份，词表方案 2026-11-14）；没给就读盘上此刻的在用状态。
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime

from habitus.conditions import DayTypeCalendar
from habitus.scene.concepts.catalog import CatalogWords, ClassCatalog
from habitus.scene.concepts.model import ConceptOrigin, ConceptSet
from habitus.scene.concepts.store import ConceptStore
from habitus.scene.concepts.sync import SyncPlan, plan_sync
from habitus.scene.recurrence import coverage_by_lane
from habitus.series import EventSeries
from habitus.temporal import Coverage, Rulers


@dataclass(frozen=True)
class Synced:
    """同步词表之后的那一份：同步到的版本、落盘后的概念集、各 lane 的观测覆盖。"""

    version: int
    concepts: ConceptSet
    coverage: Mapping[str, Coverage]


class SyncStage:
    def __init__(
        self,
        *,
        catalog: ClassCatalog,
        concepts: ConceptStore,
        calendar: DayTypeCalendar,
        rulers: Rulers,
    ) -> None:
        self.catalog = catalog
        self.concepts = concepts
        self.calendar = calendar
        self.rulers = rulers

    async def plan(self, series: EventSeries, now: datetime, words: CatalogWords | None = None) -> SyncPlan:
        """这一次同步词表要做的事（纯读盘、纯计算）。``words`` 是这一代在用的词（没给就读盘上此刻的）。"""

        if words is None:
            words = await asyncio.to_thread(self.catalog.current)
        # 日历情境的取值从数据里攒出来：序列里出现过的日子，日历给了哪些（键, 取值）就长哪些（裁定 59 第 7 条）
        observed = {
            item for day in sorted({record.day for record in series.records}) for item in self.calendar.of_day(day)
        }
        existing = await asyncio.to_thread(self.concepts.read_all)
        return await asyncio.to_thread(plan_sync, existing, words, now=now, observed=observed)

    def owed(self, plan: SyncPlan) -> bool:
        """同步词表还欠着：有概念要写、或同步到的版本还没记。"""

        return bool(plan.write) or plan.version != self.concepts.synced_version()

    async def run(
        self, series: EventSeries, now: datetime, signals: list[str], *, plan: SyncPlan | None = None
    ) -> Synced:
        """B1 同步词表：概念落盘、记下同步到的版本。顺带算一晚只算一次的覆盖。``plan`` 是刚算好的那一份（不给就现算）。"""

        if plan is None:
            plan = await self.plan(series, now)
        concepts = await asyncio.to_thread(self._write_concepts, plan, signals)
        # 各 lane 的观测覆盖只取决于序列：机会表的材料共用
        coverage = await asyncio.to_thread(coverage_by_lane, series, self.rulers)
        if plan.version != self.concepts.synced_version():
            self.concepts.write_synced_version(plan.version)
        return Synced(plan.version, concepts, coverage)

    def _write_concepts(self, plan: SyncPlan, signals: list[str]) -> ConceptSet:
        """概念落盘（新生成 / 改名改定义 / 新的日历情境），读回整个概念集。"""

        for definition in plan.write:
            self.concepts.write(definition)
        concepts = self.concepts.read_all()
        signals.extend(
            f"vocabulary: 「{definition.label}」（{definition.name}）没同步：{reason}（盘上没写）"
            for definition, reason in plan.rejected
        )
        calendar = 0
        for definition in plan.write:
            if definition.source.origin is ConceptOrigin.CALENDAR:
                calendar += 1
            elif definition.retired:
                signals.append(f"vocabulary: 「{definition.label}」（{definition.name}）不再在用，概念停用")
            else:
                signals.append(f"vocabulary: 「{definition.label}」（{definition.name}）已同步")
        if calendar:
            signals.append(
                f"vocabulary: 补上日历情境 {calendar} 个（数据里新出现的周几、几号、月末、日型取值，裁定 59）"
            )
        return concepts


__all__ = ["SyncStage", "Synced"]
