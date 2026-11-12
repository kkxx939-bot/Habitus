"""封口（实时化阶段 2「映射按槽」）：已封口的日子里每条记录都作数的，统一重算情境、写回、核条数、盖日完成章。

规则在领域包（``scene.occurrences.daily`` 的 ``plan_seal`` / ``apply_seal``）；这里取材料、调用、落盘：

- ``seal_plans``（纯读盘、纯计算）：还没以现在的口径盖章的已封口日子各自的封口计划；读得到它们的那几天（派生情境往回看的天数
  之内）、早已盖章的，情境要跟着重算的也给出来（读者日）。
- ``seal``：换代的语义树一步调——写回、盖章。有记录还不作数的日子不盖章（那一天对细分概念算看不到，由模型队列补）。
  封口按并入待生效概念之前的概念集核（命中记录是对着它映射的）。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta

from habitus.foundation.tracing import annotate
from habitus.runtime.scene_night_mapping import MappingStage
from habitus.scene.concepts.model import ConceptSet
from habitus.scene.occurrences.daily import DaySeal, apply_seal, day_documents, plan_seal
from habitus.scene.occurrences.mapper import ConceptMapper, ConceptMapperError
from habitus.scene.occurrences.situations import history_days
from habitus.series import EventSeries


@dataclass(frozen=True)
class SealReport:
    """一次封口：盖了章的日子、写回了情境的记录数、还不能盖章的日子与其中还不作数的记录数、跟着重算情境的读者日。"""

    sealed: tuple[date, ...] = ()
    rewritten: int = 0
    unready_days: int = 0
    unready_records: int = 0
    readers: int = 0
    seconds: float = 0.0


class SealStage:
    def __init__(self, mapping: MappingStage) -> None:
        #: 映射的编排：封口用它的取材料口（``records_of``）、还没盖章的日子（``pending_days``）与同一个命中仓。
        self.mapping = mapping

    def seal_plans(
        self,
        series: EventSeries,
        concepts: ConceptSet,
        mapper: ConceptMapper,
        signals: list[str] | None = None,
    ) -> tuple[list[DaySeal], list[DaySeal]]:
        """已封口的日子里：还没盖章的各自的封口计划、读得到它们而早已盖章的日子里情境要跟着重算的（读者日）。纯读盘、纯计算。"""

        pending = sorted(self.mapping.pending_days(series, mapper))
        plans = [plan for day in pending if (plan := self._plan_day(day, series, concepts, mapper, signals))]
        depth = len(history_days(concepts, series.last_day)) - 1
        sealing = {plan.day for plan in plans if plan.ready}
        readers: list[DaySeal] = []
        if depth > 0 and sealing:
            done = self.mapping.hits.days_done()
            wanted = {day + timedelta(days=step) for day in sealing for step in range(1, depth + 1)}
            for day in sorted((wanted & done) - set(pending)):
                if day >= series.cutoff:
                    continue
                plan = self._plan_day(day, series, concepts, mapper, signals)
                if plan is not None and plan.ready and plan.rewrite:
                    readers.append(plan)
        return plans, readers

    def _plan_day(
        self,
        day: date,
        series: EventSeries,
        concepts: ConceptSet,
        mapper: ConceptMapper,
        signals: list[str] | None,
    ) -> DaySeal | None:
        try:
            documents = day_documents(self.mapping.behavior_tree, series, day)
            records, context = self.mapping.records_of(day, concepts, mapper, series)
            context.hold_today()  # 封口时这一天的命中不动了：只读一遍
            return plan_seal(documents, self.mapping.hits, mapper, records, situation_for=context.situation_for)
        except ConceptMapperError as exc:
            if signals is not None:
                signals.append(f"mapping: {day} 封不了口（{exc}），之后照常补")
            return None

    def seal(
        self,
        series: EventSeries,
        concepts: ConceptSet,
        mapper: ConceptMapper,
        now: datetime,
        signals: list[str],
    ) -> SealReport:
        """封口：每条记录都作数的已封口日子重算情境、写回、核条数、盖章；读者日的情境跟着重算、重盖。有记录还不作数的
        日子不盖章（那一天对细分概念算看不到，由模型队列补）。"""

        plans, readers = self.seal_plans(series, concepts, mapper, signals)
        count = len(readers)
        sealed: list[date] = []
        rewritten = 0
        for plan, pending in (*((plan, True) for plan in plans), *((plan, False) for plan in readers)):
            if not plan.ready:
                continue
            try:
                apply_seal(self.mapping.hits, mapper, plan, now=now)
            except ValueError as exc:
                signals.append(f"mapping: {plan.day} 盖不了章（{exc}），之后照常补")
                continue
            rewritten += len(plan.rewrite)
            if pending:
                sealed.append(plan.day)
        unready = [plan for plan in plans if not plan.ready]
        if unready:
            signals.append(
                f"mapping: {len(unready)} 天还有 {sum(plan.not_current for plan in unready)} 条记录没映射好，"
                "这几天先不盖章（对细分概念算看不到），模型队列补上后下一代盖"
            )
        annotate(
            {
                "sealed_days": len(sealed),
                "seal_rewritten": rewritten,
                "unsealed_days": len(unready),
                "unsealed_records": sum(plan.not_current for plan in unready),
                "reader_days": count,
            }
        )
        return SealReport(
            sealed=tuple(sealed),
            rewritten=rewritten,
            unready_days=len(unready),
            unready_records=sum(plan.not_current for plan in unready),
            readers=count,
        )


__all__ = ["SealReport", "SealStage"]
