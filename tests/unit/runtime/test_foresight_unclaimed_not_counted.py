"""宿主没取走的不算说过（A3；裁定 96 第 9、10 条）：

「待取」记在最晚时限那一刻或之后的（出口到最晚时限起不给取），和记了「宿主没取」的，都不算今天说过，不占当天的名额和同类。
还在等宿主、或已被宿主取走还没到最晚时限的，照旧算说过。「待取」之后记了「已发生」「依据失效」的，这次不动，仍占名额。

入口是 ``DeliveryStage.run_once``，每 15 分钟一拍；重启用新建的 ``DeliveryStage`` 读同一份账本。断言落在账上每条预期的递送原因和
时刻上。共同摆法：E 时窗 09:50–10:13，最晚时限 10:15，10:15 那一拍记「待取」（一写下就取不走）。
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from habitus.foresight import Seen
from habitus.foresight.hierarchy import KindHierarchy
from habitus.foresight.ledger import Delivery, Expectation
from habitus.foresight.opportunities import OpportunityTable, RelationOpportunities, TreeReading
from habitus.runtime.foresight_delivery import DeliveryStage, JudgeMaterials
from habitus.runtime.foresight_ledger import ForesightLedgerStore
from tests.unit.runtime.delivery_ticks import (
    DAY,
    MIDNIGHT,
    book,
    clock,
    listed,
    made,
    said,
    site,
    walk,
)
from tests.unit.runtime.test_foresight_delivery import (
    DAY0,
    EVIDENCE,
    LOSS,
    Outlet,
    earn,
    expectation,
    seeded_of,
    stage,
)
from tests.unit.runtime.test_foresight_delivery_handover import moment, reasons

E_KIND, X_KIND = "s-k0301", "s-k0302"
RELATION = "rel-x"
FLAT = KindHierarchy()


def basis_item(**overrides: Any) -> Expectation:
    """E：时窗 09:50–10:13；默认最晚时限 10:15（越过时窗终点的第一拍正好等于它）。"""

    options: dict[str, Any] = {"deadline": clock(10, 15), **overrides}
    return made("e", E_KIND, (clock(9, 50), clock(10, 13)), **options)


def other_item(window: tuple[datetime, datetime] | None = None, **overrides: Any) -> Expectation:
    """另一件事 X：默认时窗 10:20–10:40（10:30 开窗、10:45 到时窗末）。"""

    return made("x", X_KIND, window or (clock(10, 20), clock(10, 40)), **overrides)


UNCLAIMED = [("等时窗末", "10:00"), ("待取", "10:15"), ("宿主没取", "10:30")]


def restarted(ledger: Any, outlet: Outlet, *, daily_cap: int) -> DeliveryStage:
    """进程重启：新建的递送那一拍，读同一份账本。"""

    return DeliveryStage(
        ledger,
        tolerance=LOSS,
        evidence=EVIDENCE,
        daily_cap=daily_cap,
        tick_seconds=900.0,
        outlet=outlet,
        history=seeded_of(ledger).store,
    )


def relation(*, status: str = "established", reading: TreeReading | None = None) -> RelationOpportunities:
    return RelationOpportunities(
        key=RELATION,
        antecedent="c-a",
        consequent="c-b",
        antecedent_label="前因",
        consequent_label="后果",
        antecedent_classes=("s-k0303",),
        consequent_classes=(X_KIND,),
        segment="chain",
        status=status,
        upward=True,
        items=(),
        reading=reading,
    )


def table_of(*relations: RelationOpportunities) -> OpportunityTable:
    return OpportunityTable("session", DAY, DAY, 15, 3, relations, ())


# ── 复现：取不走的「待取」、「宿主没取」不占名额、同类 ────────────────


def test_a3_1_a_request_nobody_could_take_leaves_the_cap_free_for_a_later_one(tmp_path: Path) -> None:
    """A3-1：名额 1，另一件事 10:45 到时窗末：E 没说成，不占名额，另一件事记「待取」。"""

    run, ledger, outlet = site(tmp_path, basis_item(), other_item(), daily_cap=1)
    walk(run, clock(9, 45), clock(10, 45), kinds=FLAT)
    assert book(ledger, "e", "x") == {"e": UNCLAIMED, "x": [("等时窗末", "10:30"), ("待取", "10:45")]}
    assert listed(outlet) == ["x"]


def test_a3_2_the_cap_is_free_within_the_very_tick_the_request_was_written(tmp_path: Path) -> None:
    """A3-2（专门用例）：名额 1，另一件事同一拍 10:15 到时窗末：名额在这一拍就放开，不是下一拍。E 排在前面先写下「待取」，
    这条「待取」写在最晚时限那一刻、取不走，不占名额，另一件事同一拍记「待取」。"""

    other = other_item((clock(9, 50), clock(10, 10)))
    run, ledger, _outlet = site(tmp_path, basis_item(), other, daily_cap=1)
    walk(run, clock(9, 45), clock(10, 15), kinds=FLAT)
    assert book(ledger, "e", "x") == {
        "e": [("等时窗末", "10:00"), ("待取", "10:15")],
        "x": [("等时窗末", "10:00"), ("待取", "10:15")],
    }


def test_a3_3_the_cap_is_free_within_the_tick_that_records_the_host_never_claimed(tmp_path: Path) -> None:
    """A3-3（专门用例）：名额 1，另一件事 10:30 到时窗末，正是 E 记「宿主没取」的那一拍：这一拍就放开，另一件事记「待取」。"""

    other = other_item((clock(10), clock(10, 20)))
    run, ledger, _outlet = site(tmp_path, basis_item(), other, daily_cap=1)
    walk(run, clock(9, 45), clock(10, 30), kinds=FLAT)
    assert book(ledger, "e", "x") == {"e": UNCLAIMED, "x": [("等时窗末", "10:00"), ("待取", "10:30")]}


def test_a3_4_the_same_kind_at_another_moment_is_not_silenced(tmp_path: Path) -> None:
    """A3-4：同一类、说中的范围不相连的另一刻（11:30 开窗）：E 没说成，不算「同类已说」，另一刻记「待取」。"""

    later = made("y", E_KIND, (clock(11, 30), clock(11, 40)), deadline=clock(14), observed=clock(11), trigger="other")
    run, ledger, _outlet = site(tmp_path, basis_item(), later)
    walk(run, clock(9, 45), clock(11, 45), kinds=FLAT)
    assert book(ledger, "e", "y") == {"e": UNCLAIMED, "y": [("等时窗末", "11:30"), ("待取", "11:45")]}


def test_a3_6_a_request_that_could_be_taken_but_never_was_stops_counting_once_it_is_closed(tmp_path: Path) -> None:
    """A3-6：E 最晚时限 10:30（10:15 待取时还取得走），一直没人取，10:45 记「宿主没取」；另一件事 10:45 到时窗末：记「待取」。"""

    run, ledger, _outlet = site(tmp_path, basis_item(deadline=clock(10, 30)), other_item(), daily_cap=1)
    walk(run, clock(9, 45), clock(10, 45), kinds=FLAT)
    assert book(ledger, "e", "x") == {
        "e": [("等时窗末", "10:00"), ("待取", "10:15"), ("宿主没取", "10:45")],
        "x": [("等时窗末", "10:30"), ("待取", "10:45")],
    }


def test_a3_7_after_a_restart_a_request_never_claimed_is_not_counted(tmp_path: Path) -> None:
    """A3-7：进程重启后读账，E 是「待取、宿主没取」：不算今天说过。"""

    run, ledger, outlet = site(tmp_path, basis_item(), other_item(), daily_cap=1)
    walk(run, clock(9, 45), clock(10, 30), kinds=FLAT)
    walk(restarted(ledger, outlet, daily_cap=1), clock(10, 45), clock(10, 45), kinds=FLAT)
    assert book(ledger, "e", "x") == {"e": UNCLAIMED, "x": [("等时窗末", "10:30"), ("待取", "10:45")]}


def test_a3_8_after_a_restart_a_request_written_at_its_deadline_is_not_counted(tmp_path: Path) -> None:
    """A3-8：进程重启后读账，E 只有最晚时限那一拍的「待取」：不算今天说过（重启后的第一拍，另一件事就记「待取」）。"""

    other = other_item((clock(10), clock(10, 20)))
    run, ledger, outlet = site(tmp_path, basis_item(), other, daily_cap=1)
    walk(run, clock(9, 45), clock(10, 15), kinds=FLAT)
    assert book(ledger, "e") == {"e": [("等时窗末", "10:00"), ("待取", "10:15")]}
    walk(restarted(ledger, outlet, daily_cap=1), clock(10, 30), clock(10, 30), kinds=FLAT)
    assert book(ledger, "x") == {"x": [("等时窗末", "10:00"), ("待取", "10:30")]}


# ── 旧摆法在 A3 之后的结果（这三条原来写在已有测试里，已有测试改成了真的「说了」）──


def test_a3_old_shape_a_request_closed_as_never_claimed_no_longer_fills_the_cap(tmp_path: Path) -> None:
    """A3 旧摆法一（原 ``test_an_awaiting_request_is_handed_to_the_outlet_…``）：三条里一条说了、一条已发生、一条宿主没取，名额 3：
    宿主没取的那条不占名额，后来的那条记「待取」，不再是「名额已满」。"""

    outlet = Outlet()
    run, ledger = stage(tmp_path, outlet=outlet)
    for kind in ("s-k0002", "s-k0003", "s-k0004"):
        earn(ledger, kind=kind)
        ledger.record_expectation(expectation(f"t-{kind}", DAY0, kind=kind))
    run.daily_cap = 3
    asyncio.run(run.run_once(DAY0 + timedelta(minutes=15)))
    asyncio.run(run.run_once(DAY0 + timedelta(minutes=41)))
    ledger.record_delivery(
        DAY0.date(),
        Delivery("t-s-k0002", DAY0 + timedelta(minutes=50), 1, True, "说了", 1.0, "题目", via="codex", turn="c1"),
    )
    done = JudgeMaterials(None, lambda uris: {}, happened=lambda item: item.kind_token == "s-k0003")
    asyncio.run(run.run_once(DAY0 + timedelta(minutes=55), done))
    assert asyncio.run(run.run_once(DAY0 + timedelta(hours=2, minutes=1), done)).by_reason == {"宿主没取": 1}
    earn(ledger, kind="s-k0005")
    ledger.record_expectation(expectation("t-late", DAY0 + timedelta(minutes=50), kind="s-k0005"))
    asyncio.run(run.run_once(DAY0 + timedelta(minutes=65)))
    assert asyncio.run(run.run_once(DAY0 + timedelta(minutes=91))).by_reason == {"待取": 1}


def test_a3_old_shape_a_request_nobody_took_no_longer_silences_the_same_kind(tmp_path: Path) -> None:
    """A3 旧摆法二（原 ``test_one_class_is_spoken_once_a_day_even_from_several_accounts``）：那条「待取」到最晚时限都没人取：
    同一类另一刻记「待取」，不再是「同类已说」。"""

    run, ledger = stage(tmp_path)
    for trigger in ("rel", "interval"):
        earn(ledger, trigger=trigger)
    ledger.record_expectation(expectation("a", DAY0, trigger="rel"))
    ledger.record_expectation(expectation("b", DAY0, trigger="interval"))
    asyncio.run(run.run_once(DAY0 + timedelta(minutes=15)))
    asyncio.run(run.run_once(DAY0 + timedelta(minutes=41)))
    ledger.record_expectation(expectation("d", DAY0 + timedelta(hours=3), trigger="rel"))
    asyncio.run(run.run_once(DAY0 + timedelta(hours=3, minutes=15)))
    assert asyncio.run(run.run_once(DAY0 + timedelta(hours=3, minutes=41))).by_reason == {"待取": 1}


def test_a3_old_shape_a_request_past_its_deadline_no_longer_fills_the_cap_for_a_member(tmp_path: Path) -> None:
    """A3 旧摆法三（原 ``test_the_daily_cap_does_not_count_the_moment_that_was_never_said_but_counts_the_rest``）：别的类那条「待取」
    已过最晚时限：不占名额，接手的成员记「待取」，不再是「名额已满」。"""

    run, ledger = moment(tmp_path)
    for kind in ("s-k0003", "s-k0004"):
        earn(ledger, kind=kind)
        ledger.record_expectation(expectation(f"x-{kind}", DAY0, kind=kind))
    asyncio.run(run.run_once(DAY0 + timedelta(minutes=17)))
    asyncio.run(run.run_once(DAY0 + timedelta(minutes=41)))
    late = expectation("x-late", DAY0, kind="s-k0005")
    ledger.record_expectation(late)
    ledger.record_delivery(
        late.day, Delivery("x-late", DAY0 + timedelta(minutes=60), 1, True, "说了", 1.0, "话", via="c", turn="t")
    )
    asyncio.run(run.run_once(DAY0 + timedelta(minutes=121)))
    assert reasons(ledger, "later") == ["单元已选", "待取"]


# ── 不变：还算说过的 ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("taken", [True, False], ids=["A3-9 已被取走", "A3-10 挂着没人取"])
def test_a3_9_and_10_a_request_still_open_keeps_its_place(tmp_path: Path, taken: bool) -> None:
    """A3-9（已被取走）、A3-10：E 最晚时限还远，「待取」已被取走还没说出口、或挂着没人取：照旧占名额，另一件事「名额已满」。"""

    claimed: set[str] = set()
    run, ledger, _outlet = site(
        tmp_path, basis_item(deadline=clock(14)), other_item(), daily_cap=1, claimed=claimed.__contains__
    )
    walk(run, clock(9, 45), clock(10, 15), kinds=FLAT)
    if taken:
        claimed.add("e")
    walk(run, clock(10, 30), clock(10, 45), kinds=FLAT)
    assert book(ledger, "e", "x") == {
        "e": [("等时窗末", "10:00"), ("待取", "10:15")],
        "x": [("等时窗末", "10:30"), ("名额已满", "10:45")],
    }


def test_a3_11_a_reminder_that_was_said_keeps_its_place(tmp_path: Path) -> None:
    """A3-11：E 已「说了」：占名额，另一件事「名额已满」。"""

    run, ledger, _outlet = site(tmp_path, basis_item(deadline=clock(14)), other_item(), daily_cap=1)
    walk(run, clock(9, 45), clock(10, 15), kinds=FLAT)
    ledger.record_delivery(DAY, said("e", clock(10, 20)))
    walk(run, clock(10, 30), clock(10, 45), kinds=FLAT)
    assert book(ledger, "x") == {"x": [("等时窗末", "10:30"), ("名额已满", "10:45")]}


def test_a3_12_a_question_recognised_late_counts_even_after_never_claimed(tmp_path: Path) -> None:
    """A3-12（转写晚到）：账上先记「宿主没取」，后补一条时限前的「说了」：说出口的永远算，另一件事「名额已满」。"""

    other = other_item((clock(11), clock(11, 10)))
    run, ledger, _outlet = site(tmp_path, basis_item(deadline=clock(10, 30)), other, daily_cap=1)
    walk(run, clock(9, 45), clock(10, 45), kinds=FLAT)
    assert book(ledger, "e") == {"e": [("等时窗末", "10:00"), ("待取", "10:15"), ("宿主没取", "10:45")]}
    ledger.record_delivery(DAY, said("e", clock(10, 20)))
    walk(run, clock(11), clock(11, 15), kinds=FLAT)
    assert book(ledger, "x") == {"x": [("等时窗末", "11:00"), ("名额已满", "11:15")]}


def test_a3_13_a_request_withdrawn_because_the_node_retired_is_not_counted(tmp_path: Path) -> None:
    """A3-13：「待取」之后「已停用」：不算今天说过，另一件事记「待取」。"""

    run, ledger, _outlet = site(tmp_path, basis_item(deadline=clock(14)), other_item(), daily_cap=1)
    walk(run, clock(9, 45), clock(10, 15), kinds=FLAT)
    walk(run, clock(10, 30), clock(10, 45), kinds=KindHierarchy(in_use=frozenset({X_KIND})))
    assert book(ledger, "e", "x") == {
        "e": [("等时窗末", "10:00"), ("待取", "10:15"), ("已停用", "10:30")],
        "x": [("等时窗末", "10:30"), ("待取", "10:45")],
    }


def test_a3_14_a_request_followed_by_happened_keeps_its_place(tmp_path: Path) -> None:
    """A3-14：「待取」之后「已发生」：这次不动，仍占名额，另一件事「名额已满」。"""

    run, ledger, _outlet = site(tmp_path, basis_item(deadline=clock(14)), other_item(), daily_cap=1)
    rows = (Seen(clock(10, 20), clock(10, 25), E_KIND, "occ:done.md"),)
    walk(run, clock(9, 45), clock(10, 45), kinds=FLAT, rows=rows)
    assert book(ledger, "e", "x") == {
        "e": [("等时窗末", "10:00"), ("待取", "10:15"), ("已发生", "10:30")],
        "x": [("等时窗末", "10:30"), ("名额已满", "10:45")],
    }


def test_a3_15_a_request_followed_by_basis_withdrawn_keeps_its_place(tmp_path: Path) -> None:
    """A3-15：「待取」之后「依据失效」：这次不动，仍占名额，另一件事「名额已满」。"""

    event = basis_item(deadline=clock(14), trigger=RELATION, source="事件")
    run, ledger, _outlet = site(tmp_path, event, other_item(), daily_cap=1)
    walk(run, clock(9, 45), clock(10, 15), kinds=FLAT, table=table_of(relation()))
    walk(run, clock(10, 30), clock(10, 45), kinds=FLAT, table=table_of())
    assert book(ledger, "e", "x") == {
        "e": [("等时窗末", "10:00"), ("待取", "10:15"), ("依据失效", "10:30")],
        "x": [("等时窗末", "10:30"), ("名额已满", "10:45")],
    }


# ── 不变：名额空出来也不开口的 ────────────────────────────────────────────────


@pytest.mark.parametrize(
    "voiceless",
    [relation(status="candidate"), relation(reading=TreeReading(0.5, 0.45, large_enough=False))],
    ids=["候选关系", "巧合"],
)
def test_a3_16_a_kind_that_cannot_speak_stays_silent_when_the_cap_frees_up(
    tmp_path: Path, voiceless: RelationOpportunities
) -> None:
    """A3-16（不能开口）：另一件事是候选关系或巧合：名额空出来也不开口，记影子。"""

    other = other_item(trigger=RELATION, source="事件")
    run, ledger, _outlet = site(tmp_path, basis_item(), other, daily_cap=1)
    walk(run, clock(9, 45), clock(10, 45), kinds=FLAT, table=table_of(voiceless))
    assert book(ledger, "e", "x") == {"e": UNCLAIMED, "x": [("影子", "10:30")]}


def test_a3_17_an_antecedent_that_failed_stays_silent_when_the_cap_frees_up(tmp_path: Path) -> None:
    """A3-17（前因不成立）：另一件事开口前核出前因不成立：记「前因不成立」，名额空出来也不说。"""

    started = clock(10)
    other = other_item(trigger=RELATION, source="事件", instance=started.astimezone(DAY0.tzinfo).isoformat())
    run, ledger, _outlet = site(tmp_path, basis_item(), other, daily_cap=1)
    rows = (Seen(started, started, "s-k0304", "occ:antecedent.md"),)
    walk(run, clock(9, 45), clock(10, 45), kinds=FLAT, table=table_of(relation()), rows=rows)
    assert book(ledger, "e", "x") == {"e": UNCLAIMED, "x": [("等时窗末", "10:30"), ("前因不成立", "10:45")]}


def test_a3_18_a_member_takes_over_when_its_basis_could_not_be_taken(tmp_path: Path) -> None:
    """A3-18：E 是依据，成员 M 到自己时窗末接手：同一刻不算，M 记「待取」。"""

    member = made("m", E_KIND, (clock(10, 20), clock(10, 40)), deadline=MIDNIGHT, trigger="m")
    run, ledger, _outlet = site(tmp_path, basis_item(), member, daily_cap=1)
    walk(run, clock(9, 45), clock(10, 45), kinds=FLAT)
    assert book(ledger, "e", "m") == {"e": UNCLAIMED, "m": [("单元已选", "10:00"), ("待取", "10:45")]}


def test_a3_19_the_tick_that_writes_the_request_still_reports_it(tmp_path: Path) -> None:
    """A3-19：同一拍 E 记「待取」，这一拍的报告 ``by_reason`` 含待取 1；E 的两条记录照写。"""

    run, ledger, outlet = site(tmp_path, basis_item())
    walk(run, clock(9, 45), clock(10), kinds=FLAT)
    report = asyncio.run(run.run_once(clock(10, 15), JudgeMaterials(None, lambda uris: {}, kinds=FLAT)))
    assert report.by_reason == {"待取": 1} and report.spoken == 0 and report.open_requests == 1
    assert book(ledger, "e") == {"e": [("等时窗末", "10:00"), ("待取", "10:15")]}
    assert listed(outlet) == ["e"]


class _Crash(RuntimeError):
    """进程在这一拍中途崩溃。"""


def test_a3_20_a_rerun_of_the_deadline_tick_after_a_crash_does_not_count_the_untakeable_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A3-20：E 在 10:15（正好等于它的最晚时限）记了取不走的「待取」，进程写完 E 之后崩溃；新建的 ``DeliveryStage`` 读同一份账本
    重跑 10:15 这一拍，另一件事同一拍到时窗末，名额 1：这时 E 只能从账上读到，读账也不算它，另一件事记「待取」。"""

    other = other_item((clock(9, 50), clock(10, 10)))
    run, ledger, outlet = site(tmp_path, basis_item(), other, daily_cap=1)
    walk(run, clock(9, 45), clock(10), kinds=FLAT)
    written = ledger.record_delivery

    def crash_after_e(day: Any, delivery: Delivery) -> bool:
        if delivery.expectation_id == "x":
            raise _Crash
        return written(day, delivery)

    monkeypatch.setattr(ledger, "record_delivery", crash_after_e)
    with pytest.raises(_Crash):
        walk(run, clock(10, 15), clock(10, 15), kinds=FLAT)
    assert book(ledger, "e", "x") == {
        "e": [("等时窗末", "10:00"), ("待取", "10:15")],
        "x": [("等时窗末", "10:00")],
    }
    reopened = ForesightLedgerStore(ledger.root)
    walk(restarted(reopened, outlet, daily_cap=1), clock(10, 15), clock(10, 15), kinds=FLAT)
    assert book(reopened, "e", "x") == {
        "e": [("等时窗末", "10:00"), ("待取", "10:15")],
        "x": [("等时窗末", "10:00"), ("待取", "10:15")],
    }
