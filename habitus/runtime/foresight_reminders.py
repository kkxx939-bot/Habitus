"""某一天各条提醒走到了哪一步（方案 36 第 2 步 ``habitus reminders``；R8 实测时看"问的比例"、放回次数、宿主没取的比例）。

只读：从账本拼——决定说的那条递送（「待取」与写好的话）、出口认出提问后追加的「说了」（经哪个宿主）、终局的「宿主没取」、
回话（正 / 中 / 负）、结算的结果；还没过最晚时限的，再从出口的取走记录拼上谁取走了、放回了几次（过了时限那份记录就清掉了）。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

from habitus.foresight.ledger.model import AWAITING_HOST, HOST_NEVER_CLAIMED, SPOKEN, Delivery
from habitus.runtime.foresight_ledger import ForesightLedgerStore
from habitus.runtime.foresight_outlet import Handoff, ReminderOutlet

#: 进了出口的三种递送：等取、说出口了、到时限没人取 / 没问出口
_OUTLET_REASONS = frozenset({AWAITING_HOST, SPOKEN, HOST_NEVER_CLAIMED})


@dataclass(frozen=True)
class ReminderProgress:
    """一条提醒：``stage`` 是此刻走到的那一步——待取 / 取走了 / 说了 / 宿主没取。"""

    expectation_id: str
    kind: str
    text: str
    deadline: datetime
    stage: str
    awaiting_at: datetime
    spoken_at: datetime | None
    spoken_via: str | None
    response: str | None
    responded_at: datetime | None
    outcome: str | None
    attempts: int | None
    handoffs: tuple[Handoff, ...]


@dataclass(frozen=True)
class ReminderDay:
    day: date
    reminders: tuple[ReminderProgress, ...]

    def count(self, stage: str) -> int:
        return sum(1 for item in self.reminders if item.stage == stage)

    @property
    def responded(self) -> int:
        return sum(1 for item in self.reminders if item.response is not None)


def reminder_day(ledger: ForesightLedgerStore, outlet: ReminderOutlet | None, day: date) -> ReminderDay:
    """拼出 ``day`` 那天进了出口的每条提醒，按开始等取的时刻排。"""

    deliveries: dict[str, list[Delivery]] = {}
    for item in ledger.deliveries_on(day):
        if item.reason in _OUTLET_REASONS:
            deliveries.setdefault(item.expectation_id, []).append(item)
    responses = {item.expectation_id: item for item in ledger.responses_on(day)}
    outcomes = {item.expectation_id: item.outcome for item in ledger.settlements_on(day)}
    trails = outlet.trails() if outlet is not None else {}
    rows: list[ReminderProgress] = []
    for expectation_id, items in deliveries.items():
        expectation = ledger.expectation(day, expectation_id)
        awaiting = [item for item in items if item.reason == AWAITING_HOST]
        if expectation is None or not awaiting:
            continue
        first = min(awaiting, key=lambda item: item.decided_at)
        spoken = next((item for item in sorted(items, key=lambda d: d.decided_at) if item.reason == SPOKEN), None)
        never = any(item.reason == HOST_NEVER_CLAIMED for item in items)
        trail = trails.get(expectation_id)
        holding = trail is not None and any(handoff.ended is None for handoff in trail.handoffs)
        stage = SPOKEN if spoken else HOST_NEVER_CLAIMED if never else "取走了" if holding else AWAITING_HOST
        response = responses.get(expectation_id)
        rows.append(
            ReminderProgress(
                expectation_id=expectation_id,
                kind=outlet.kind_name(expectation.kind_token) if outlet is not None else expectation.kind_token,
                text=first.text or "",
                deadline=expectation.deadline,
                stage=stage,
                awaiting_at=first.decided_at,
                spoken_at=None if spoken is None else spoken.decided_at,
                spoken_via=None if spoken is None else spoken.via,
                response=None if response is None else response.polarity,
                responded_at=None if response is None else response.responded_at,
                outcome=outcomes.get(expectation_id),
                attempts=None if trail is None else trail.attempts,
                handoffs=() if trail is None else trail.handoffs,
            )
        )
    rows.sort(key=lambda row: (row.awaiting_at, row.expectation_id))
    return ReminderDay(day, tuple(rows))


__all__ = ["ReminderDay", "ReminderProgress", "reminder_day"]
