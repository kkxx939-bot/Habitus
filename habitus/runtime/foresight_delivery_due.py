"""递送那一拍的读账分拨：从账本读出这一拍要看的那几拨预期，只读、不决定、不落账（规则在 ``foresight.delivery`` 与
``foresight.units``）。"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime

from habitus.foresight.delivery import PENDING, closes, hand_over
from habitus.foresight.ledger import AWAITING_HOST, Delivery, Expectation
from habitus.foresight.units import UNIT, UNIT_CITE, joinable, meets
from habitus.runtime.foresight_ledger import ForesightLedgerStore


@dataclass(frozen=True)
class DueScan:
    opened: tuple[Expectation, ...]
    closing: tuple[tuple[Expectation, Delivery], ...]
    waiting: tuple[Expectation, ...]
    #: 依据的预期身份 → 它在「单元已选」里替哪几条一并决定过。
    members: Mapping[str, tuple[Expectation, ...]]
    #: 这一拍刚到时窗、并进一个已经决定过的单元的：（这条, 依据的身份, 依据那次决定的档）。
    joins: tuple[tuple[Expectation, str, int], ...] = ()
    #: 最近一条决定是「待取」、还没说出口的：（这条, 那条「待取」）。
    awaiting: tuple[tuple[Expectation, Delivery], ...] = ()
    #: 依据没说成、该有成员接手的那几刻：（这一刻的全部输入——依据与它吸收的成员, 能接手且这一拍到了自己时窗末的那几条的身份）。
    handovers: tuple[tuple[tuple[Expectation, ...], tuple[str, ...]], ...] = ()
    #: 这一拍接手的候选成员 → 它那条「单元已选」（接手后当时窗末的「等时窗末」用：档与决定时刻）。
    subsumed: Mapping[str, Delivery] = field(default_factory=dict)


def scan_due(
    ledger: ForesightLedgerStore,
    now: datetime,
    since: datetime,
    *,
    voiced: Callable[[Expectation], bool],
    claimed: Callable[[str], bool],
) -> DueScan:
    """四拨：时窗与这一拍的区间（``since`` 到此刻：``JudgeMaterials.since``）有交集、还没结算、账上还没有决定的（开了）；只决定过「等时窗末」、这一拍
    到了时窗末的（该来了还没来，``delivery.closes``）；还没到时窗、没决定过的（单元从这里拉同一刻的另外几个输入）；最近一条决定是
    「待取」、还没说出口的（等宿主）。过了时窗终点还没决定过的不再递：那是结算去记的事。另记每条依据替哪几条一并决定过
    （「单元已选」引它）：时窗末开口时小包与话照样带上同一刻的全部输入。``voiced`` 按此刻的机会表判一条预期能不能当开口的
    依据（``units.voiced``）：不能的依据记了影子，被它吸收的成员照常接手（``delivery.hand_over``）。"""

    fresh: list[Expectation] = []
    closing: list[tuple[Expectation, Delivery]] = []
    #: 每条预期（还没结的）与它的全部决定：依据没说成时找接手的成员要看依据与成员两边。
    known: dict[str, Expectation] = {}
    decided: dict[str, list[Delivery]] = {}
    awaiting: list[tuple[Expectation, Delivery]] = []
    waiting: list[Expectation] = []
    members: dict[str, list[Expectation]] = {}
    #: 已经作为依据决定过、还没结的（类 → (预期, 最近那次决定的档)）：后来才形成的同一刻的输入并进它，不另起一个单元。
    bases: dict[str, list[tuple[Expectation, int]]] = {}
    instant = now.astimezone(UTC)
    for day in ledger.open_days():
        if day > now.date():
            continue
        decisions = _by_expectation(ledger.deliveries_on(day))
        for item in ledger.unsettled_expectations(day):
            made = decisions.get(item.expectation_id, [])
            known[item.expectation_id] = item
            decided[item.expectation_id] = made
            for cite in (cited for made_one in made if made_one.reason == UNIT for cited in made_one.evidence):
                members.setdefault(cite.removeprefix(UNIT_CITE), []).append(item)
            start, end = (item.window[0].astimezone(UTC), item.window[1].astimezone(UTC))
            if not made:
                if start <= instant and end > since:
                    fresh.append(item)
                elif start > instant:
                    waiting.append(item)
                continue
            if all(made_one.reason != UNIT for made_one in made):
                bases.setdefault(item.kind_token, []).append((item, made[-1].level))
            pending = [d for d in made if d.reason == PENDING]
            if pending and len(pending) == len(made) and closes(item, pending[-1], since, instant):
                closing.append((item, pending[-1]))
            latest = max(made, key=lambda d: d.decided_at)
            if latest.reason == AWAITING_HOST and not any(d.spoken for d in made):
                awaiting.append((item, latest))  # 等宿主来取：做了记「已发生」、过时限记「宿主没取」
    opened, joins = _join_fresh(fresh, bases, members)
    handovers, subsumed = hand_over(known, decided, members, instant, since, voiced=voiced, taken=claimed)
    grouped = {key: tuple(value) for key, value in members.items()}
    return DueScan(
        tuple(opened), tuple(closing), tuple(waiting), grouped, tuple(joins), tuple(awaiting), handovers, subsumed
    )


def _by_expectation(deliveries: Iterable[Delivery]) -> dict[str, list[Delivery]]:
    decisions: dict[str, list[Delivery]] = {}
    for delivery in deliveries:
        decisions.setdefault(delivery.expectation_id, []).append(delivery)
    return decisions


def _join_fresh(
    fresh: list[Expectation],
    bases: Mapping[str, list[tuple[Expectation, int]]],
    members: dict[str, list[Expectation]],
) -> tuple[list[Expectation], list[tuple[Expectation, str, int]]]:
    """同一刻不拆成几个单元：先到时窗的那条（日子、情境）已在时窗开时决定过，同一刻后来才形成的事件那条并进它；
    两条都以当天为单位、起算不在同一天的不是同一刻（``units.joinable``，裁定 81、90），各自决定。"""

    opened: list[Expectation] = []
    joins: list[tuple[Expectation, str, int]] = []
    for item in fresh:
        base = next(
            (
                (other, level)
                for other, level in bases.get(item.kind_token, ())
                if meets(item, other.observed_from, other.deadline) and joinable(other, item)
            ),
            None,
        )
        if base is None:
            opened.append(item)
        else:
            joins.append((item, base[0].expectation_id, base[1]))
            members.setdefault(base[0].expectation_id, []).append(item)
    return opened, joins


__all__ = ["DueScan", "scan_due"]
