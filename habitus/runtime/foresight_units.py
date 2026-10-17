"""行为单元在组合根的那一半（裁定 56 第 12 条，方案第九节 6g）：把这一拍到了时窗的预期按行为拢成单元、从账本读出历史上的
每一刻、算单元读数与档；给可见面读单元视图。算法本身在 ``foresight.units``（纯函数），这里只管读账本、拢组、缓存。

**拢组**：这一拍到了时窗、还没决定过的预期按类分组；同一类里说中的范围（起算 → 最晚时限）和它们连成一片、还没到时窗的那几条
一起拉进来——它们是同一刻的另外几个输入，单元替它们一并决定（记「单元已选」），不再等各自时窗开了再各说各的。
**历史**：同一类过去的预期，回看到遗忘让一刻的分量低于 α 为止（半衰期 × log₂(1/α)，不另配数）；按天缓存，结完的日子不再读。
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta

from habitus.foresight import Evidence, Standing, current_facts
from habitus.foresight.ladder import PREDICTION
from habitus.foresight.ledger import AccountFact, Expectation
from habitus.foresight.opportunities import OpportunityTable, TreeReading
from habitus.foresight.units import (
    Check,
    Input,
    Moment,
    UnitReading,
    basis,
    moments,
    self_check,
    tree_reading,
    unit_level,
    unit_reading,
)
from habitus.runtime.foresight_ledger import ForesightLedgerStore


@dataclass(frozen=True)
class Unit:
    """这一拍的一个单元：说话的那条（依据）、一并决定的另外几条、全部输入、单元读数与档、依据那本账换上单元档之后的位置。"""

    basis: Expectation
    members: tuple[Expectation, ...]
    inputs: tuple[Input, ...]
    reading: UnitReading
    level: int
    standing: Standing

    @property
    def kind_token(self) -> str:
        return self.basis.kind_token


@dataclass(frozen=True)
class UnitView:
    """可见面读的一个单元：行为、每个输入（账、语义树读数、两个数对不对得上）。"""

    kind_token: str
    inputs: tuple[tuple[str, Standing, Check], ...]


class UnitPlanner:
    def __init__(self, ledger: ForesightLedgerStore, *, evidence: Evidence, prior_strength: float) -> None:
        if not isinstance(ledger, ForesightLedgerStore):
            raise TypeError("ledger must be a ForesightLedgerStore")
        if not isinstance(evidence, Evidence):
            raise TypeError("evidence must be an Evidence")
        if isinstance(prior_strength, bool) or not isinstance(prior_strength, int | float) or prior_strength < 0:
            raise ValueError("prior_strength must be a non-negative number")
        self.ledger = ledger
        self.evidence = evidence
        self.prior_strength = float(prior_strength)
        #: 结完的日子：（日子 → 那天的预期）——之后不会再变，不再读。
        self._final_days: dict[date, tuple[Expectation, ...]] = {}

    def plan(
        self,
        opened: Sequence[Expectation],
        waiting: Sequence[Expectation],
        *,
        standing_of: Callable[[Expectation], Standing],
        previous_level: Callable[[str], int | None],
        table: OpportunityTable | None,
        now: datetime,
    ) -> tuple[Unit, ...]:
        """``opened`` 是这一拍到了时窗、没决定过的；``waiting`` 是还没到时窗、没决定过的（同一刻的另外几个输入从这里拉）。"""

        facts: dict[str, tuple[AccountFact, ...]] = {}
        histories: dict[str, tuple[Moment, ...]] = {}
        units = []
        for group in _groups(opened, waiting):
            inputs = tuple(Input(item, standing_of(item), *tree_reading(table, item)) for item in group)
            kind = group[0].kind_token
            if kind not in histories:
                histories[kind] = self._history(kind, now, facts)
            reading = unit_reading(
                inputs, histories[kind], now=now, evidence=self.evidence, prior_strength=self.prior_strength
            )
            chosen = basis(inputs)
            level = unit_level(chosen.standing, reading, previous=previous_level(chosen.account))
            units.append(
                Unit(
                    basis=chosen.expectation,
                    members=tuple(item for item in group if item.expectation_id != chosen.expectation.expectation_id),
                    inputs=inputs,
                    reading=reading,
                    level=level,
                    standing=replace(chosen.standing, level=level),
                )
            )
        return tuple(units)

    @staticmethod
    def views(inputs_by_kind: Mapping[str, Sequence[tuple[str, Standing, TreeReading | None]]]) -> tuple[UnitView, ...]:
        """可见面：每个行为的输入与自检（调用方给出每个输入的账身份、此刻的位置与语义树读数）。"""

        return tuple(
            UnitView(kind, tuple((account, standing, self_check(standing, reading)) for account, standing, reading in inputs))
            for kind, inputs in sorted(inputs_by_kind.items())
        )

    # ── 历史 ────────────────────────────────────────────────────────────────

    def _history(self, kind: str, now: datetime, facts: dict[str, tuple[AccountFact, ...]]) -> tuple[Moment, ...]:
        start = now.date() - timedelta(days=math.ceil(self.evidence.half_life_days * math.log2(1.0 / self.evidence.alpha)))
        expectations = [
            item for day in self.ledger.days_with_expectations() if start <= day <= now.date() for item in self._on(day)
        ]
        mine = [item for item in expectations if item.kind_token == kind and item.formed_at < now]

        def prediction_of(item: Expectation) -> AccountFact | None:
            if item.account not in facts:
                facts[item.account] = current_facts(self.ledger.facts_of(item.account))
            return next(
                (
                    fact
                    for fact in facts[item.account]
                    if fact.expectation_id == item.expectation_id and fact.axis == PREDICTION
                ),
                None,
            )

        return moments(mine, prediction_of)

    def _on(self, day: date) -> tuple[Expectation, ...]:
        cached = self._final_days.get(day)
        if cached is not None:
            return cached
        found = self.ledger.expectations_on(day)
        if self.ledger.is_final(day):
            self._final_days[day] = found
        return found


def _groups(opened: Sequence[Expectation], waiting: Iterable[Expectation]) -> list[list[Expectation]]:
    """按类、按说中的范围连成一片拢组；同一刻还没到时窗的那几条也拉进来。"""

    by_kind: dict[str, list[Expectation]] = {}
    for item in opened:
        by_kind.setdefault(item.kind_token, []).append(item)
    pending: dict[str, list[Expectation]] = {}
    for item in waiting:
        pending.setdefault(item.kind_token, []).append(item)
    found: list[list[Expectation]] = []
    for kind in sorted(by_kind):
        for group in _connected(sorted(by_kind[kind], key=lambda item: (item.observed_from, item.expectation_id))):
            start = min(item.observed_from for item in group)
            end = max(item.deadline for item in group)
            group += [
                item
                for item in pending.get(kind, ())
                if item.observed_from <= end and item.deadline >= start and item not in group
            ]
            found.append(sorted(group, key=lambda item: (item.window[0], item.expectation_id)))
    return found


def _connected(items: Sequence[Expectation]) -> list[list[Expectation]]:
    groups: list[list[Expectation]] = []
    end: datetime | None = None
    for item in items:
        if groups and end is not None and item.observed_from <= end:
            groups[-1].append(item)
            end = max(end, item.deadline)
        else:
            groups.append([item])
            end = item.deadline
    return groups


__all__ = ["Unit", "UnitPlanner", "UnitView"]
