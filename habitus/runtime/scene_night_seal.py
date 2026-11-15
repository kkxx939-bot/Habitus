"""封口（换代的语义树一步）：已封口的日子里还没封好、或记录变了的，命中记录由代码直接生成、核条数、盖日完成章。不调模型。

规则在领域包（``scene.occurrences.daily`` 的 ``plan_seal`` / ``apply_seal``）；这里取材料、调用、落盘：

- ``pending_days``：序列里有记录、还没以现在的摘要盖章的日子——完成标记没有、条数对不上、或这一天的摘要变了（补进或延长了
  记录、日历情境多了）。
- ``seal``：这些日子各封一次口。读盘一侧出错（盘上的命中记录读不了）往上抛，封口这一步失败；写盘一侧出错（写不出、
  核条数对不上）只报出这一天，下一代再封。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

from habitus.conditions import DayTypeCalendar
from habitus.foundation.tracing import annotate
from habitus.scene.concepts.model import ConceptSet
from habitus.scene.occurrences.daily import apply_seal, day_digest, plan_seal
from habitus.scene.occurrences.situations import SituationInputs, situation_hits
from habitus.scene.occurrences.store import ConceptHitStore
from habitus.series import EventSeries


@dataclass(frozen=True)
class SealReport:
    """一次封口：盖了章的日子、写了的命中记录数、封不了的日子数、这一段用了多久。"""

    sealed: tuple[date, ...] = ()
    rewritten: int = 0
    failed_days: int = 0
    seconds: float = 0.0


class SealStage:
    def __init__(self, *, hits: ConceptHitStore, calendar: DayTypeCalendar) -> None:
        self.hits = hits
        self.calendar = calendar

    def pending_days(self, series: EventSeries, concepts: ConceptSet) -> tuple[date, ...]:
        """序列里有记录、还没以现在的摘要盖完成章的日子（按日子先后）。"""

        done = self.hits.days_done()
        pending = []
        for day in sorted({record.day for record in series.records}):
            marker = self.hits.read_marker(day) if day in done else None
            records = series.on(day)
            if marker is None or marker.records != len(records) or marker.digest != day_digest(records, concepts):
                pending.append(day)
        return tuple(pending)

    def seal(self, series: EventSeries, concepts: ConceptSet, now: datetime, signals: list[str]) -> SealReport:
        """封口：还没封好的已封口日子各自生成命中记录、核条数、盖章。"""

        sealed: list[date] = []
        rewritten = failed = 0
        for day in self.pending_days(series, concepts):
            situations = situation_hits(concepts, SituationInputs(day=day, calendar=dict(self.calendar.of_day(day))))
            # 读盘一侧（盘上的命中记录读不了、序列上的记录生成不了命中记录）不接：往上抛，封口这一步失败、换代记下降级——
            # 读不了的文件不会自己好，收成信号只会让那一天每一代都封不了口却没人察觉
            plan = plan_seal(self.hits, day, series.on(day), concepts, situations, now=now)
            try:
                apply_seal(self.hits, plan, now=now)
            except ValueError as exc:
                # 写盘一侧（清旧、写、核条数、盖章）没做成：这一天不盖章，下一代按摘要认出来再封
                failed += 1
                signals.append(f"seal: {day} 封不了口（{exc}），下一代再封")
                continue
            sealed.append(day)
            rewritten += len(plan.write)
        annotate({"sealed_days": len(sealed), "seal_rewritten": rewritten, "unsealed_days": failed})
        return SealReport(sealed=tuple(sealed), rewritten=rewritten, failed_days=failed)


__all__ = ["SealReport", "SealStage"]
