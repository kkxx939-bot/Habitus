"""``SceneNightlyRun.generate`` 的那一段串法（从 ``runtime.scene_night`` 拆出来的）：同步词表 → 封口（按并入之前的概念集核）
→ 并入待生效的概念 → 这一代的节律落盘 → 按缓存折叠 → 材料。没有模型调用（调模型的那几段归模型队列）。

``failures`` 给了的时候一步抛了就记下、走退路接着做。
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Mapping
from datetime import datetime

from habitus.foundation.tracing import set_attributes, span
from habitus.runtime.scene_night import NightReport, SceneNightlyRun
from habitus.runtime.scene_night_seal import SealReport
from habitus.runtime.scene_night_sync import Synced
from habitus.scene.concepts.model import ConceptSet
from habitus.scene.concepts.sync import SyncPlan
from habitus.scene.recurrence import coverage_by_lane
from habitus.scene.relations import LaneTimeline
from habitus.series import EventSeries
from habitus.temporal import Coverage


def _sync_attributes(plan: SyncPlan, concepts: ConceptSet) -> dict[str, object]:
    return {
        "version": plan.version,
        "written": len(plan.write),
        "rejected": len(plan.rejected),
        "moved": len(plan.moved),
        "concepts": len(concepts),
    }


class NightRun:
    """``SceneNightlyRun.generate`` 的那一段串法：B1 → 封口 → 并入待生效的概念 → 这一代的节律落盘 → 折叠 → 材料。``failures``
    给了的时候一步抛了就记下、走退路接着做。"""

    def __init__(
        self,
        scene: SceneNightlyRun,
        series: EventSeries,
        now: datetime,
        failures: list[tuple[str, BaseException]] | None,
        relations_held: str | None = None,
    ) -> None:
        self.scene = scene
        self.series = series
        self.now = now
        self.failures = failures
        self.relations_held = relations_held
        self.signals: list[str] = []

    async def run(self) -> NightReport:
        scene, series = self.scene, self.series
        synced = await self._b1()
        concepts = synced.concepts
        # 封口按并入之前的概念集核（命中记录是对着它映射的）；再把待生效的概念并进来——它们的映射与关系从下一代起有
        sealed = await self._seal(concepts)
        concepts, merged = await self._merge(concepts)
        await self._freeze(concepts, synced.coverage)
        relations, transitions, lines = await self._b4(concepts, synced.version)
        return NightReport(
            night=series.cutoff,
            concepts=len(concepts),
            relations=relations,
            transitions=transitions,
            signals=tuple(self.signals),
            relations_held=self.relations_held,
            sealed_days=len(sealed.sealed),
            seal_rewritten=sealed.rewritten,
            unsealed_days=sealed.unready_days,
            unsealed_records=sealed.unready_records,
            reader_days=sealed.readers,
            seal_seconds=sealed.seconds,
            merged_concepts=merged,
            materials=await asyncio.to_thread(scene._materials, series, concepts, lines, synced.coverage),
        )

    async def _seal(self, concepts: ConceptSet) -> SealReport:
        """封口（换代）：已封口的日子里每条记录都作数的，重算情境、写回、核条数、盖章。没做成就这一代不盖新章——那几天
        对细分概念算看不到，下一代再封。"""

        scene = self.scene
        started = time.monotonic()
        try:
            with span("scene.seal") as attributes:
                report = await asyncio.to_thread(
                    scene.sealing.seal, self.series, concepts, scene.mapper_for(concepts), self.now, self.signals
                )
                seconds = max(0.0, time.monotonic() - started)
                set_attributes(attributes, {"seal_seconds": round(seconds, 3)})
            return SealReport(
                report.sealed, report.rewritten, report.unready_days, report.unready_records, report.readers, seconds
            )
        except Exception as exc:
            self._tolerate("seal", exc)
        return SealReport(seconds=max(0.0, time.monotonic() - started))

    async def _merge(self, concepts: ConceptSet) -> tuple[ConceptSet, int]:
        """把待生效的概念并进概念集（换代）。没做成就这一代不并——它们留着，下一代再并。"""

        scene = self.scene
        try:
            merged = await asyncio.to_thread(scene.merge_pending, self.now, self.signals)
            if merged:
                concepts = await asyncio.to_thread(scene.concepts.read_all)
            return concepts, merged
        except Exception as exc:
            self._tolerate("merge", exc)
        return concepts, 0

    async def _freeze(self, concepts: ConceptSet, coverage: Mapping[str, Coverage]) -> None:
        """这一代的节律落盘一份（映射这一天的记录读它）。写不出只报出：读的那边会按截止到那一天的序列现算。"""

        try:
            await asyncio.to_thread(self.scene.freeze_rhythms, self.series, concepts, coverage)
        except Exception as exc:
            self._tolerate("rhythms", exc)

    def _tolerate(self, step: str, exc: Exception) -> None:
        """一步抛了：严格跑法（没给 ``failures``）原样抛，尽力而为的记下这一步、走退路。"""

        if self.failures is None:
            raise exc
        self.failures.append((step, exc))

    async def _b1(self) -> Synced:
        scene = self.scene
        try:
            with span("scene.b1") as attributes:
                plan = await scene.syncing.plan(self.series, self.now)
                synced = await scene.syncing.run(self.series, self.now, self.signals, remap=False, plan=plan)
                set_attributes(attributes, _sync_attributes(plan, synced.concepts))
            return synced
        except Exception as exc:
            self._tolerate("b1", exc)
        # 同步没做成：按盘上已同步到的那一版与现有的概念集接着做，下一次再同步
        concepts = await asyncio.to_thread(scene.concepts.read_all)
        coverage = await asyncio.to_thread(coverage_by_lane, self.series, scene.rulers)
        return Synced(scene.concepts.synced_version(), concepts, coverage)

    async def _b4(
        self, concepts: ConceptSet, version: int
    ) -> tuple[Mapping[str, Mapping[str, int]], int, Mapping[str, LaneTimeline]]:
        stage = self.scene.relation_stage
        if self.relations_held is not None:
            # 模型整个调不通：不拿这一代没长成的词表、概念与映射判出一张新表（同一代不重写，判出来就一直留着）
            self.signals.append(f"relations: 这一代不判关系（{self.relations_held}），机会表沿用上一代的关系表")
            return {}, 0, await stage.timelines(self.series, concepts)
        try:
            with span("scene.b4") as attributes:
                timelines = await stage.timelines(self.series, concepts)
                relations, transitions, lines = await stage.run(
                    self.series, concepts, self.signals, vocabulary=version, now=self.now, lines=timelines
                )
                set_attributes(attributes, {"lanes": len(relations), "transitions": transitions})
            return relations, transitions, lines
        except Exception as exc:
            self._tolerate("b4", exc)
        # 关系判定没做成：这一代不写关系表（已写出的 lane 留着），时间线照算，机会表退到上一代关系表
        return {}, 0, await stage.timelines(self.series, concepts)


__all__ = ["NightRun"]
