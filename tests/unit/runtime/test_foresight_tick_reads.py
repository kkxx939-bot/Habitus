"""预测拍不重复读账（丙批处三 B1、B2、B4–B8、B11）：两拍之间账没变就不再打开记录文件，账变了下一拍一定读到。

读文件的次数在 ``JsonFiles.load`` 这一层按路径数；入口是真实的 ``DeliveryStage._due``、``spoken_today`` 与 ``ForesightTick.run_once``。
B1、B2、B4 在改动前的代码上失败（那正是要修的重复读），其余改动前后都通过。
"""

from __future__ import annotations

import asyncio
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pytest

from habitus.conditions import LocalCalendar
from habitus.foresight import Evidence, NoUnsealed, Tolerance
from habitus.foresight.ledger import Delivery, Expectation, Settlement
from habitus.runtime.foresight import ForesightReadSide
from habitus.runtime.foresight_delivery import DeliveryStage
from habitus.runtime.foresight_delivery_tally import spoken_today
from habitus.runtime.foresight_history_store import HistoryStore
from habitus.runtime.foresight_ledger import ForesightLedgerStore
from habitus.runtime.foresight_opportunities import OpportunityStore
from habitus.runtime.foresight_outlet import ReminderOutlet
from habitus.runtime.foresight_outlet_ledgers import BranchOutlet
from habitus.runtime.foresight_outlet_store import OutletStore
from habitus.runtime.foresight_settlement import SettlementStage
from habitus.runtime.foresight_tick import ForesightTick, TickReport
from tests.unit.kind_ids import kind_id
from tests.unit.runtime.test_foresight_delivery import Outlet, Seeded
from tests.unit.runtime.test_foresight_outlet import claude_question, turn
from tests.unit.runtime.test_foresight_settlement_reads import WATERMARKS, Opened
from tests.unit.runtime.test_foresight_tick import TODAY as THREE_MORNINGS_TODAY
from tests.unit.runtime.test_foresight_tick import site_with_three_mornings, stores, tick
from tests.unit.scene.fixtures import SUBJECT, Site, at, publish

CST = timezone(timedelta(hours=8))
NOW = datetime(2026, 12, 10, 12, 0, tzinfo=CST)  # 第 90 天 12:00
TODAY = NOW.date()
SINCE = NOW - timedelta(minutes=15)
LOSS = Tolerance(line=2 / 3)
EVIDENCE = Evidence(alpha=0.05, half_life=8.0)
#: 账本的记录目录（预测根下还住着机会表、单元历史、出口状态，它们不是账本）。
RECORD_TOPS = frozenset({"expectations", "deliveries", "responses", "settlements", "rulings", "omissions"})
PER_ACCOUNT_TOPS = frozenset({"accounts", "units"})


def records_in(opened: Opened) -> Counter[str]:
    """这一段打开过的账本记录文件（路径 → 次数）；水位、身份索引与预测根下别的东西不算。"""

    return Counter(
        {
            path: count
            for path, count in opened.everything().items()
            if path.split("/")[0] in RECORD_TOPS | PER_ACCOUNT_TOPS
        }
    )


def made(identity: str, day: date, *, kind: str = "s-k0100", trigger: str = "rel") -> Expectation:
    """默认这一天 14:00–15:00 的时窗、23:00 最晚时限：今天的这一条在 12:00 还没到时窗。"""

    return Expectation(
        expectation_id=identity,
        kind_token=kind,
        day=day,
        formed_at=datetime.combine(day, datetime.min.time(), CST) + timedelta(minutes=5),
        source="日子",
        trigger=trigger,
        trigger_instance=f"behavior://{identity}.md",
        window=(
            datetime.combine(day, datetime.min.time(), CST) + timedelta(hours=14),
            datetime.combine(day, datetime.min.time(), CST) + timedelta(hours=15),
        ),
        deadline=datetime.combine(day, datetime.min.time(), CST) + timedelta(hours=23),
        basis=(),
        opportunities=13,
        hits=10,
        observed_from=datetime.combine(day, datetime.min.time(), CST),
    )


def seed(ledger: ForesightLedgerStore, days: int, *, per_day: int, deliveries: int, settled: int) -> list[date]:
    """铺 ``days`` 个开着的日子（今天和往前几天），再铺一个已经结完的更早的日子（盘上就有结完水位）。"""

    closed = TODAY - timedelta(days=days)
    ledger.record_expectation(made(f"{closed.isoformat()}-e0", closed))
    ledger.mark_final(closed, at=NOW - timedelta(days=days - 1))
    opened = [TODAY - timedelta(days=back) for back in range(days - 1, -1, -1)]
    for day in opened:
        items = [
            made(f"{day.isoformat()}-e{index:02d}", day, kind=f"s-k{100 + index:04d}", trigger=f"rel-{index}")
            for index in range(per_day)
        ]
        for item in items:
            ledger.record_expectation(item)
        noon = datetime.combine(day, datetime.min.time(), CST) + timedelta(hours=9)
        ledger.record_delivery(day, Delivery(items[0].expectation_id, noon, 1, True, "说了", 1.0, "该做了"))
        for index in range(1, deliveries):
            item = items[index % per_day]
            ledger.record_delivery(
                day, Delivery(item.expectation_id, noon + timedelta(seconds=index), 0, False, "影子", 0.0)
            )
        for item in items[-settled:] if settled else ():
            ledger.record_settlement(Settlement(item.expectation_id, item.kind_token, day, NOW, "过期待决"))
    return opened


def delivery_stage(ledger: ForesightLedgerStore, *, outlet: Any = None) -> DeliveryStage:
    return DeliveryStage(
        ledger,
        tolerance=LOSS,
        evidence=EVIDENCE,
        daily_cap=3,
        tick_seconds=900.0,
        outlet=outlet if outlet is not None else Outlet(),
        history=HistoryStore(ledger.root),
    )


def due_and_spoken(stage: DeliveryStage, now: datetime = NOW, since: datetime = SINCE) -> tuple[Any, dict[str, Any]]:
    """一拍里读账的那两处：四拨到期的、今天说过的。"""

    return stage._due(now, since, voiced=lambda item: True), spoken_today(stage.ledger, now)


def whole_tick(tmp_path: Path, stage: DeliveryStage, clock: Callable[[], datetime]) -> ForesightTick:
    """真实的整拍：没有机会表（不建预期），递送照走。"""

    site = Site(tmp_path, now=NOW)
    read_side = ForesightReadSide(site.behavior_tree, SUBJECT, LocalCalendar(), ZoneInfo("Asia/Shanghai"))
    return ForesightTick(
        stage.ledger,
        OpportunityStore(stage.ledger.root),
        read_side,
        NoUnsealed(),
        delivery=stage,
        tick_seconds=900.0,
        clock=clock,
        recent=35,
        tolerance=LOSS,
    )


def run(tick: ForesightTick) -> TickReport:
    return asyncio.run(tick.run_once())


def ground(tmp_path: Path, *, days: int = 3) -> tuple[ForesightLedgerStore, DeliveryStage, Opened]:
    ledger = ForesightLedgerStore(tmp_path / "foresight")
    seed(ledger, days, per_day=4, deliveries=10, settled=1)
    # 今天还有一条没决定过、还没到时窗的：``_due`` 的「等时窗」那一拨不是空的
    ledger.record_expectation(made(f"{TODAY.isoformat()}-waiting", TODAY, kind="s-k0200", trigger="rel-waiting"))
    return ledger, delivery_stage(ledger), Opened(ledger.root)


def new_delivery_file(ledger: ForesightLedgerStore, item: Delivery, day: date = TODAY) -> str:
    """另一个实例写入一条递送，返回它相对账本根的路径。"""

    before = set(ledger.root.rglob("*.json"))
    assert ForesightLedgerStore(ledger.root).record_delivery(day, item)
    (path,) = set(ledger.root.rglob("*.json")) - before
    return path.relative_to(ledger.root.resolve()).as_posix()


# ── 处三 B1：两拍之间账没变 ───────────────────────────────────────────────────


def test_b1_the_second_pass_opens_no_ledger_record_when_nothing_changed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """处三 B1：``_due`` 与 ``spoken_today`` 连跑两遍、中间没有新文件：第二遍不打开任何账本记录文件。"""

    _ledger, stage, opened = ground(tmp_path)
    opened.install(monkeypatch)
    first = due_and_spoken(stage)
    assert records_in(opened), "第一遍要真的读到账"
    opened.reset()
    assert due_and_spoken(stage) == first
    assert not records_in(opened), f"第二遍打开了 {sum(records_in(opened).values())} 次记录文件"


def test_b1_the_second_whole_tick_opens_no_ledger_record_when_nothing_changed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """处三 B1：真实的 ``ForesightTick.run_once`` 连跑两拍、中间没有新文件：第二拍不打开任何账本记录文件。"""

    _ledger, stage, opened = ground(tmp_path)
    clock = [NOW]
    tick = whole_tick(tmp_path, stage, lambda: clock[0])
    opened.install(monkeypatch)
    assert run(tick).decided == 0
    assert records_in(opened)
    opened.reset()
    clock[0] = NOW + timedelta(minutes=15)
    assert run(tick).decided == 0
    assert not records_in(opened), f"第二拍打开了 {sum(records_in(opened).values())} 次记录文件"


def test_b1_a_tick_that_forms_expectations_opens_no_ledger_record_once_the_ledger_is_still(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """处三 B1：有机会表、这一拍真的建预期的整拍（夹具照 ``test_foresight_tick``）：第一拍建下预期，账不再变之后，再一拍不打开
    任何账本记录文件。身份索引是查重的点读，不算记录文件。"""

    site, _codes = site_with_three_mornings(tmp_path)
    ledger, opportunities = stores(tmp_path, site)
    publish(site.behavior_tree, THREE_MORNINGS_TODAY, "讨论方案", 10, 0)
    forming = tick(tmp_path, site, ledger, opportunities, at(THREE_MORNINGS_TODAY, 10, 15))
    assert run(forming).recorded == 1
    run(forming)  # 第一拍写下的那一条读进来
    opened = Opened(ledger.root)
    opened.install(monkeypatch)
    again = run(forming)
    assert (again.formed, again.recorded) == (1, 0)
    assert not records_in(opened), f"账没变的这一拍打开了 {sum(records_in(opened).values())} 次记录文件"


# ── 处三 B2：两拍之间有一个新文件 ─────────────────────────────────────────────


def test_b2_the_second_pass_opens_only_the_one_new_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """处三 B2：两遍之间另一个实例写入一条递送：第二遍只打开这一个新文件，而且只打开一次。"""

    ledger, stage, opened = ground(tmp_path)
    opened.install(monkeypatch)
    due_and_spoken(stage)
    added = Delivery(f"{TODAY.isoformat()}-e01", NOW - timedelta(minutes=1), 0, False, "影子", 0.0)
    path = new_delivery_file(ledger, added)
    opened.reset()
    due_and_spoken(stage)
    assert records_in(opened) == Counter({path: 1})


def test_b2_the_second_whole_tick_opens_only_the_one_new_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """处三 B2：真实的整拍：两拍之间写入一条递送，第二拍只打开这一个新文件。"""

    ledger, stage, opened = ground(tmp_path)
    clock = [NOW]
    tick = whole_tick(tmp_path, stage, lambda: clock[0])
    opened.install(monkeypatch)
    run(tick)
    added = Delivery(f"{TODAY.isoformat()}-e01", NOW + timedelta(minutes=1), 0, False, "影子", 0.0)
    path = new_delivery_file(ledger, added)
    opened.reset()
    clock[0] = NOW + timedelta(minutes=15)
    run(tick)
    assert records_in(opened) == Counter({path: 1})


# ── 处三 B4：实盘量级 ─────────────────────────────────────────────────────────


def test_b4_at_production_scale_the_second_pass_opens_only_the_watermark(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """处三 B4：第 90 天 12:00、账本 7 个开着的日子（每天 30 条预期、100 条递送、8 条结算）：``_due`` 加 ``spoken_today``
    打开文件的次数从约 1,893 降到只剩水位文件（不超过 17 次）。"""

    ledger = ForesightLedgerStore(tmp_path / "foresight")
    assert len(seed(ledger, 7, per_day=30, deliveries=100, settled=8)) == 7
    stage = delivery_stage(ledger)
    opened = Opened(ledger.root)
    opened.install(monkeypatch)
    due_and_spoken(stage)
    opened.reset()
    due_and_spoken(stage)
    everything = opened.everything()
    left = {path for path in everything if path not in WATERMARKS}
    assert not left, f"第二遍一共打开 {sum(everything.values())} 次，水位以外还有 {len(left)} 个文件"
    assert sum(everything.values()) <= 17


# ── 处三 B5：结果不变 ─────────────────────────────────────────────────────────


def test_b5_a_warmed_up_instance_and_a_fresh_one_see_the_same_due_and_spoken(tmp_path: Path) -> None:
    """处三 B5：同一份账本，读过的实例与全新实例：``_due`` 的各拨结果与 ``spoken_today`` 逐项相等，账变了之后也相等。"""

    ledger, warm, _opened = ground(tmp_path)

    def fresh() -> DeliveryStage:
        return delivery_stage(ForesightLedgerStore(ledger.root))

    first_due, first_spoken = due_and_spoken(fresh())
    assert due_and_spoken(warm) == (first_due, first_spoken)
    assert first_spoken and first_due.waiting
    assert due_and_spoken(warm) == (first_due, first_spoken)
    identity = f"{TODAY.isoformat()}-e01"
    new_delivery_file(ledger, Delivery(identity, NOW - timedelta(minutes=2), 1, True, "说了", 1.0, "再说一次"))
    changed_due, changed_spoken = due_and_spoken(warm)
    assert identity in changed_spoken and changed_spoken != first_spoken
    assert (changed_due, changed_spoken) == due_and_spoken(fresh())


# ── 处三 B6：宿主刚取走 ───────────────────────────────────────────────────────


HOTEL = kind_id("订酒店")
SESSION, HOST = "claude-s1", "claude-code"


def hotel() -> Expectation:
    formed = datetime(2026, 12, 10, 14, 7, tzinfo=CST)
    return Expectation(
        expectation_id="hotel",
        kind_token=HOTEL,
        day=formed.date(),
        formed_at=formed,
        source="日子",
        trigger="rel",
        trigger_instance="behavior://hotel.md",
        window=(formed + timedelta(minutes=4), formed + timedelta(minutes=51)),
        deadline=formed + timedelta(hours=3),
        basis=(),
        opportunities=8,
        hits=8,
        observed_from=formed,
    )


def clocked(hour: int, minute: int) -> datetime:
    return datetime(2026, 12, 10, hour, minute, tzinfo=CST)


def test_b6_what_the_outlet_writes_between_two_ticks_is_read_and_not_said_again(tmp_path: Path) -> None:
    """处三 B6：两拍之间宿主取走、出口写入「说了」（出口用自己的账本实例）：下一拍读到，这条今天算说过、不再决定、不再交给出口。"""

    root = tmp_path / "foresight"
    ledger = ForesightLedgerStore(root)
    seeded = Seeded(root, ledger)
    seeded.earn(kind=HOTEL)
    outlet = ReminderOutlet(
        {"session": OutletStore(root / "outlet")},
        {"session": ForesightLedgerStore(root)},
        liveness=timedelta(hours=48),
        lease=timedelta(minutes=30),
        max_attempts=3,
        claim_limit=3,
        similarity=0.6,
        kind_labels={"session": lambda token: "订酒店" if token == HOTEL else None},
    )
    stage = DeliveryStage(
        ledger,
        tolerance=LOSS,
        evidence=EVIDENCE,
        daily_cap=3,
        tick_seconds=900.0,
        outlet=BranchOutlet(outlet, "session"),
        history=seeded.store,
    )
    ledger.record_expectation(hotel())
    assert outlet.claim(HOST, SESSION, clocked(9, 0), 3) == ()
    clock = [clocked(14, 15)]
    tick = whole_tick(tmp_path, stage, lambda: clock[0])
    opening = run(tick).delivery
    assert opening is not None and opening.by_reason == {"等时窗末": 1}
    clock[0] = clocked(15, 0)
    closing = run(tick).delivery
    assert closing is not None and closing.by_reason == {"待取": 1} and closing.open_requests == 1
    assert "hotel" in spoken_today(ledger, clock[0])
    (claimed,) = outlet.claim(HOST, SESSION, clocked(15, 10), 3)
    outlet.observe(SESSION, turn(claude_question(claimed.text, at=clocked(15, 11)), at=clocked(15, 10)), host=HOST)
    clock[0] = clocked(15, 15)
    after = run(tick).delivery
    assert after is not None and after.decided == 0 and after.spoken == 0 and after.open_requests == 0
    assert [item.reason for item in ledger.deliveries_on(TODAY)].count("说了") == 1
    assert [item.reason for item in ledger.deliveries_on(TODAY)].count("待取") == 1
    assert list(spoken_today(ledger, clock[0])) == ["hotel"]


# ── 处三 B7：两拍之间结算 ─────────────────────────────────────────────────────


@pytest.mark.parametrize("settled_by", ["same", "other"])
def test_b7_what_the_generation_settles_between_two_ticks_is_no_longer_unsettled(
    tmp_path: Path, settled_by: str
) -> None:
    """处三 B7：两拍之间换代写了结算和结完标记（同一个账本实例，或另一个实例）：下一拍这条不再算没结，这一天也不再开着。"""

    ledger = ForesightLedgerStore(tmp_path / "foresight")
    item = made("waiting", TODAY)
    ledger.record_expectation(item)
    stage = delivery_stage(ledger)
    clock = [NOW]
    tick = whole_tick(tmp_path, stage, lambda: clock[0])
    assert run(tick).decided == 0
    due, _spoken = due_and_spoken(stage)
    assert [found.expectation_id for found in due.waiting] == ["waiting"]
    assert ledger.unsettled_expectations(TODAY) == (item,)

    site = Site(tmp_path, now=NOW)
    SettlementStage(
        ledger if settled_by == "same" else ForesightLedgerStore(ledger.root),
        ForesightReadSide(site.behavior_tree, SUBJECT, LocalCalendar(), ZoneInfo("Asia/Shanghai")),
        closed_days=lambda: (TODAY,),
        evidence=EVIDENCE,
        clock=lambda: NOW.astimezone(UTC),
    ).run_once()

    assert ledger.unsettled_expectations(TODAY) == () and ledger.open_days() == ()
    due, _spoken = due_and_spoken(stage)
    assert due.waiting == () and due.opened == () and due.closing == ()
    clock[0] = NOW + timedelta(minutes=15)
    assert run(tick).decided == 0


# ── 处三 B8：重结打开了旧日子 ─────────────────────────────────────────────────


def test_b8_a_reopened_day_is_open_again_with_all_of_its_records(tmp_path: Path) -> None:
    """处三 B8：结完的日子读过之后被 ``reopen``：``open_days`` 里又有这一天，这一天的账读到的是完整的，之后新写的也读得到。"""

    ledger = ForesightLedgerStore(tmp_path / "foresight")
    earlier = TODAY - timedelta(days=1)
    first = made("first", earlier)
    ledger.record_expectation(first)
    spoken = Delivery("first", NOW - timedelta(days=1), 1, True, "说了", 1.0, "该做了")
    ledger.record_delivery(earlier, spoken)
    settled = Settlement("first", first.kind_token, earlier, NOW, "过期待决")
    ledger.record_settlement(settled)
    ledger.mark_final(earlier, at=NOW)
    assert ledger.open_days() == () and ledger.final_through() == earlier
    stage = delivery_stage(ledger)
    due_and_spoken(stage)
    assert ledger.expectations_on(earlier) == (first,) and ledger.deliveries_on(earlier) == (spoken,)

    ledger.reopen(earlier)

    assert ledger.open_days() == (earlier,) and not ledger.is_final(earlier)
    assert ledger.expectations_on(earlier) == (first,)
    assert ledger.deliveries_on(earlier) == (spoken,)
    assert ledger.settlements_on(earlier) == (settled,)
    assert ledger.unsettled_expectations(earlier) == ()
    second = made("second", earlier, kind="s-k0101", trigger="rel-2")
    ForesightLedgerStore(ledger.root).record_expectation(second)
    assert ledger.unsettled_expectations(earlier) == (second,)
    due, _spoken = due_and_spoken(stage, now=clocked(12, 0) - timedelta(days=1), since=SINCE - timedelta(days=1))
    assert [found.expectation_id for found in due.waiting] == ["second"]


# ── 处三 B11：坏文件 ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class BadFile:
    name: str
    top: str
    count: int


@pytest.mark.parametrize(
    "bad",
    [
        BadFile("delivery", "deliveries", 1),
        BadFile("expectation", "expectations", 1),
        BadFile("settlement", "settlements", 1),
    ],
    ids=lambda bad: bad.name,
)
def test_b11_a_bad_file_is_counted_on_every_tick(tmp_path: Path, bad: BadFile) -> None:
    """处三 B11：账上有一份坏文件：每拍的 ``unreadable`` 条数都一样（不能只有第一拍报）；修好之后归零。"""

    ledger, stage, _opened = ground(tmp_path)
    directory = ledger.root / bad.top / f"{TODAY.year:04d}" / f"{TODAY.month:02d}" / f"{TODAY.day:02d}"
    path = sorted(directory.glob("*.json"))[0]
    good = path.read_bytes()
    path.write_bytes(b'{"half": ')
    clock = [NOW]
    tick = whole_tick(tmp_path, stage, lambda: clock[0])
    for offset in range(3):
        clock[0] = NOW + timedelta(minutes=15 * offset)
        assert run(tick).unreadable == bad.count
    path.write_bytes(good)
    clock[0] = NOW + timedelta(minutes=45)
    assert run(tick).unreadable == 0
