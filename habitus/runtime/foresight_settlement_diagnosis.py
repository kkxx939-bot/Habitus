"""结算的只读诊断（裁定 54）：最近一段里结过的预期，每一次落空归到哪一关，外加按切片的校准表。

只读账本、单元历史、机会表与行为树，不结算、不落账；标志怎么读、怎么归因是纯规则（``foresight.diagnosis``）。
每本账的线（``tolerance``）每次由结算那一拍交进来（结算那边可以在构造后换线）。
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from habitus.foresight import Evidence, Standing, Tolerance, climb, current_facts
from habitus.foresight.diagnosis import (
    DIAGNOSIS_DAYS,
    BasisSeen,
    Diagnosis,
    MissFacts,
    Settled,
    calibration_sample,
    counts_as_miss,
    diagnose,
    miss_facts,
)
from habitus.foresight.history import UnitHistory
from habitus.foresight.ledger import AccountFact, Expectation
from habitus.foresight.units import UNIT
from habitus.runtime.foresight_history_store import HistoryStore
from habitus.runtime.foresight_ledger import ForesightLedgerStore
from habitus.runtime.foresight_opportunities import OpportunityStore
from habitus.runtime.foresight_tree_rows import basis_row, rows_between
from habitus.scene.views import DayIndexCache
from habitus.series.reader import admitted

if TYPE_CHECKING:  # 只给类型检查看：结算模块在运行时导入本模块
    from habitus.runtime.foresight_settlement import BehaviorReadSide


class SettlementDiagnosis:
    def __init__(
        self,
        ledger: ForesightLedgerStore,
        read_side: BehaviorReadSide,
        *,
        evidence: Evidence,
        current: Callable[[str], Sequence[str]],
        opportunities: OpportunityStore | None,
        lane: str,
        history: HistoryStore | None,
    ) -> None:
        self.ledger = ledger
        self.read_side = read_side
        self.evidence = evidence
        self.current = current
        self.opportunities = opportunities
        self.lane = lane
        self.history = history

    def diagnose(self, *, now: datetime, days: int = DIAGNOSIS_DAYS, tolerance: Tolerance | None) -> Diagnosis:
        """最近 ``days`` 天里结过的预期：每一次落空归到哪一关（``foresight.diagnosis``），外加按切片的校准表。只读。"""

        if tolerance is None:
            raise ValueError("diagnosis needs the tolerance of each account's line")
        if days < 1:
            raise ValueError("days must be a positive integer")
        cache = DayIndexCache(
            self.read_side.behavior_tree,
            subject=self.read_side.subject,
            admits=admitted,
            calendar=self.read_side.calendar,
        )
        table = self.opportunities.latest(self.lane) if self.opportunities is not None else None
        listed = {relation.key for relation in table.relations} if table is not None else None
        start = now.date() - timedelta(days=days)
        accounts: dict[str, tuple[AccountFact, ...]] = {}
        lane = table.lane if table is not None else self.lane
        history = self.history.latest(lane) if self.history is not None else None
        misses: list[MissFacts] = []
        settled_rows: list[Settled] = []
        for day in self.ledger.days_with_expectations():
            if day < start:
                continue
            outcomes = {item.expectation_id: item.outcome for item in self.ledger.settlements_on(day)}
            deliveries = self.ledger.deliveries_on(day)
            spoken = {item.expectation_id for item in deliveries if item.spoken}
            subsumed = {item.expectation_id for item in deliveries if item.reason == UNIT}
            for item in self.ledger.expectations_on(day):
                outcome = outcomes.get(item.expectation_id)
                if outcome is None:
                    continue
                facts = accounts.get(item.account)
                if facts is None:
                    facts = accounts.setdefault(item.account, current_facts(self.ledger.facts_of(item.account)))
                # 这一次现在算什么，按账上现在算数的那笔事实（重结、后补都改的是它），不按当初的结算记录
                now_as = next((fact for fact in facts if fact.expectation_id == item.expectation_id), None)
                if now_as is None:
                    continue
                sample = calibration_sample(item, now_as)
                if sample is not None:
                    settled_rows.append(sample)
                if not counts_as_miss(now_as, outcome):
                    continue
                if item.expectation_id in subsumed:
                    continue  # 单元替它一并决定过：这一刻的落空记在依据那条上，一刻只算一次
                misses.append(
                    self._miss(
                        item,
                        now_as,
                        spoken=item.expectation_id in spoken,
                        standing=self._standing_at(item, history, tolerance),
                        cache=cache,
                        listed=listed,
                    )
                )
        return diagnose(misses, settled_rows, alpha=self.evidence.alpha)

    def _standing_at(self, item: Expectation, history: UnitHistory | None, tolerance: Tolerance) -> Standing:
        """这次落空那一刻（最晚时限）这条边的位置：单元历史（``history``，夜里按当晚语义树重标的最晚一晚，一次诊断只读一回）里
        这条边在那一刻之前的那几刻，按那时生效的设置与上一次的级。拿此刻的位置去判一个月前的落空，过了断点又恢复的会被说成
        预算内，正在断的会把之前的也拖成说不清；按刻的时刻截到那一刻（``edge_reading(visible_at=)``），之后的刻不算。"""

        at = item.deadline
        moments = history.of(item.kind_token) if history is not None else ()
        return climb(
            moments,
            item.account,
            now=at,
            tolerance=tolerance,
            evidence=self.evidence,
            setting=_latest_by(self.ledger.settings_of(item.kind_token), at, lambda found: found.at),
            previous=_latest_by(self.ledger.levels_of(item.account), at, lambda found: found.at),
        )

    def _miss(
        self,
        item: Expectation,
        fact: AccountFact,
        *,
        spoken: bool,
        standing: Standing,
        cache: DayIndexCache,
        listed: set[str] | None,
    ) -> MissFacts:
        """一次落空的材料从树上取（窗里的行、那一天与依据那几次的日历说法、依据是谁先发起的），标志怎么读在 ``foresight.diagnosis.miss_facts``。"""

        basis = [row for row in (basis_row(cache, uri) for uri in item.basis) if row is not None]
        return miss_facts(
            item,
            fact,
            spoken=spoken,
            standing=standing,
            current=self.current,
            window=rows_between(cache, item.observed_from.date(), item.deadline.date()),
            today_note=cache.day(item.day).day_note,
            basis=[BasisSeen(cache.day(row.at.date()).day_note, row.prompted_by) for row in basis],
            listed=listed,
        )


def _latest_by[T](items: Iterable[T], at: datetime, moment: Callable[[T], datetime]) -> T | None:
    earlier = [item for item in items if moment(item) <= at]
    return max(earlier, key=moment) if earlier else None


__all__ = ["SettlementDiagnosis"]
