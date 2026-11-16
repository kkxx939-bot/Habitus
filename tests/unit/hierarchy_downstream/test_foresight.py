"""丙一预测层那一段（预期表 D4 与场景 C16–C27；共享说明 3.3、3.4、3.6），外加"不在用的类现在也建日子预期"的现状（3.2）。

预测层由组合根的 builder 装（``site.World.foresight``：``build_foresight_components`` 带词表状态 ``kind_store``、
``build_foresight_tick``、``build_reminded_lookup``），走它的口子：结算那一拍（``settlement.run_once``）、预测拍（递送）、
归约盖 ``reminded`` 的查法、提醒出口的可取清单与宿主来取。预期直接写进账本（形成预期那一半的端到端在
``test_bookkeeping.py::test_c8``）；账的级用账上的流水 + 单元历史快照垫到提醒档（``World.earn``）；宿主活着由出口的 ``claim`` 定。

改动前（``780e298``）"他做没做这件事"按编号完全相等认、没有"上下级已说""已停用"：C16、C18、C19、C20、C22、C24、C26 与
"不在用的类建日子预期"在旧代码上应当失败；C17、C21、C23、C25、C27 是"反过来不认 / 范围不重叠 / 兄弟 / 账不互相当先验 /
一拍用开头那一份"的守门，旧代码上本来就成立。
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from habitus.foresight.model import NoUnsealed
from habitus.runtime.foresight import build_reminded_lookup
from tests.unit.hierarchy_downstream.site import (
    PLAN,
    SUMMARY,
    WEEKLY,
    WRITE_DOC,
    World,
    at,
    expectation,
    spoken,
)

DAY = date(2026, 4, 14)  # 周二


def _ticks(world: World, start: datetime, end: datetime, ticker: Any | None = None) -> None:
    """从 ``start`` 到 ``end``（含）每一拍。"""

    ticker = ticker or world.ticker()
    moment = start
    while moment <= end:
        world.tick(moment, ticker)
        moment += timedelta(minutes=15)


def _day_expectation(identity: str, kind: str, window: tuple[int, int]) -> Any:
    """当天的日子预期：从零点起看、最晚时限到午夜；``window`` 是时窗的起止（当天第几分钟）。"""

    return expectation(
        identity,
        kind,
        at(DAY, 0),
        window=window,
        deadline_minutes=24 * 60 - 1,
    )


def _ready(world: World, *kinds: str) -> None:
    """这几个类在用、各自那本账到提醒档；有宿主活着。"""

    world.use(*kinds)
    for kind in kinds:
        world.earn(kind, before=at(DAY, 0))
    world.host_alive(at(DAY, 8))


# ── 结算：上级的预期认下级的记录；下级的不认上级的（C16、C17）──────────────────────


def test_c16_a_parent_expectation_is_done_by_a_record_on_its_child(tmp_path: Path) -> None:
    world = World(tmp_path)
    world.use(WRITE_DOC, WEEKLY)
    world.ledger.record_expectation(expectation("doc", WRITE_DOC, at(DAY, 9)))
    uri = world.record(WEEKLY, at(DAY, 9, 20))
    world.set_clock(at(DAY + timedelta(days=1), 4))
    world.settle(DAY)
    (settled,) = world.ledger.settlements_on(DAY)
    assert (settled.occurrence_uri, settled.outcome) == (uri, "准时")


def test_c17_a_child_expectation_is_not_done_by_its_sibling_or_its_parent(tmp_path: Path) -> None:
    """守门：一条下级的预期只认下级自己和它的下级（旧代码上本来就成立）。"""

    world = World(tmp_path)
    world.use(WRITE_DOC, WEEKLY, PLAN)
    world.ledger.record_expectation(expectation("weekly", WEEKLY, at(DAY, 9)))
    world.record(PLAN, at(DAY, 9, 20))
    world.record(WRITE_DOC, at(DAY, 9, 30))
    world.set_clock(at(DAY + timedelta(days=1), 4))
    world.settle(DAY)
    (settled,) = world.ledger.settlements_on(DAY)
    assert settled.occurrence_uri is None


# ── 递送：今天已做（C18）、被提醒带出来（C19）─────────────────────────────────────


def test_c18_a_parent_day_expectation_is_already_done_when_the_child_was_done_today(tmp_path: Path) -> None:
    world = World(tmp_path)
    world.use(WRITE_DOC, WEEKLY)
    world.ledger.record_expectation(_day_expectation("doc-day", WRITE_DOC, (14 * 60, 15 * 60)))
    world.record(WEEKLY, at(DAY, 10))
    world.tick(at(DAY, 14))
    assert world.deliveries(DAY) == {"doc-day": ["已发生"]}


def test_c19_a_record_on_the_child_after_the_parent_reminder_counts_as_reminded(tmp_path: Path) -> None:
    world = World(tmp_path)
    world.use(WRITE_DOC, WEEKLY)
    spoken(world, expectation("doc", WRITE_DOC, at(DAY, 8, 50), deadline_minutes=180), at(DAY, 9))
    lookup = build_reminded_lookup(world.foresight)
    assert (lookup(WEEKLY, at(DAY, 9, 30)), lookup(WRITE_DOC, at(DAY, 9, 30))) == (True, True)


# ── 上级和下级只说一条（C20–C24、D4「上下级已说」）────────────────────────────────


def test_c20_d4_parent_and_child_due_in_the_same_tick_with_overlapping_ranges_only_the_child_is_said(
    tmp_path: Path,
) -> None:
    world = World(tmp_path)
    _ready(world, WRITE_DOC, WEEKLY)
    world.ledger.record_expectation(expectation("doc", WRITE_DOC, at(DAY, 9)))
    world.ledger.record_expectation(expectation("weekly", WEEKLY, at(DAY, 9)))
    _ticks(world, at(DAY, 9, 15), at(DAY, 9, 45))
    made = world.deliveries(DAY)
    assert (made["weekly"][-1], made["doc"][-1]) == ("待取", "上下级已说")


def test_c21_parent_said_in_the_morning_does_not_block_the_child_in_the_evening_without_overlap(
    tmp_path: Path,
) -> None:
    """守门：两条预期说中的范围不重叠（上午 09:00–10:00、晚上 18:00–19:00），各说各的（旧代码上本来就成立）。"""

    world = World(tmp_path)
    _ready(world, WRITE_DOC, WEEKLY)
    spoken(world, expectation("doc", WRITE_DOC, at(DAY, 9), deadline_minutes=60), at(DAY, 9, 40))
    world.ledger.record_expectation(expectation("weekly", WEEKLY, at(DAY, 18), deadline_minutes=60))
    _ticks(world, at(DAY, 18, 15), at(DAY, 18, 45))
    assert world.deliveries(DAY)["weekly"][-1] == "待取"


def test_c22_d4_child_said_in_the_morning_blocks_the_parent_in_the_afternoon_with_overlap(tmp_path: Path) -> None:
    world = World(tmp_path)
    _ready(world, WRITE_DOC, WEEKLY)
    spoken(world, expectation("weekly", WEEKLY, at(DAY, 9), deadline_minutes=180), at(DAY, 9, 40))
    world.ledger.record_expectation(_day_expectation("doc-day", WRITE_DOC, (14 * 60, 15 * 60)))
    _ticks(world, at(DAY, 14), at(DAY, 15, 15))
    assert world.deliveries(DAY)["doc-day"][-1] == "上下级已说"


def test_c23_a_sibling_said_in_the_morning_does_not_block(tmp_path: Path) -> None:
    """守门：兄弟不是上下级（旧代码上本来就成立）。"""

    world = World(tmp_path)
    _ready(world, WEEKLY, PLAN)
    spoken(world, expectation("weekly", WEEKLY, at(DAY, 9), deadline_minutes=180), at(DAY, 9, 40))
    world.ledger.record_expectation(_day_expectation("plan-day", PLAN, (14 * 60, 15 * 60)))
    _ticks(world, at(DAY, 14), at(DAY, 15, 15))
    assert world.deliveries(DAY)["plan-day"][-1] == "待取"


def test_c24_d4_first_and_third_levels_due_together_with_the_second_idle_only_the_third_is_said(
    tmp_path: Path,
) -> None:
    world = World(tmp_path)
    grown = world.grow(WEEKLY, "写需求书", written=at(DAY - timedelta(days=30), 9))
    _ready(world, WRITE_DOC, grown)
    world.ledger.record_expectation(expectation("doc", WRITE_DOC, at(DAY, 9)))
    world.ledger.record_expectation(expectation("third", grown, at(DAY, 9)))
    _ticks(world, at(DAY, 9, 15), at(DAY, 9, 45))
    made = world.deliveries(DAY)
    assert (made["third"][-1], made["doc"][-1]) == ("待取", "上下级已说")


# ── 账：上级和下级两本账，不互相当先验（C25）────────────────────────────────────


def test_c25_the_child_account_neither_inherits_nor_lends_to_the_parent_account(tmp_path: Path) -> None:
    """守门：写文档的账到了提醒档，写周报（同一种触发）从影子起；写周报做到了只记在它自己的账上（旧代码上本来就成立）。"""

    world = World(tmp_path)
    world.use(WRITE_DOC, WEEKLY)
    parent = world.earn(WRITE_DOC, before=at(DAY, 0))
    before = len(world.ledger.facts_of(parent))
    weekly = expectation("weekly", WEEKLY, at(DAY, 9))
    world.ledger.record_expectation(weekly)
    world.tick(at(DAY, 9, 15))
    assert world.deliveries(DAY) == {"weekly": ["影子"]}
    world.record(WEEKLY, at(DAY, 9, 20))
    world.set_clock(at(DAY + timedelta(days=1), 4))
    world.settle(DAY)
    assert len(world.ledger.facts_of(parent)) == before
    assert [fact.good for fact in world.ledger.facts_of(weekly.account)] == [True]


# ── 节点不在用以后不说（C26、D4「已停用」）─────────────────────────────────────


def test_c26_d4_after_the_node_leaves_use_nothing_more_is_said_and_the_waiting_request_is_withdrawn(
    tmp_path: Path,
) -> None:
    world = World(tmp_path)
    _ready(world, WRITE_DOC)
    # 两条说中的范围不相连（09:00–10:30、10:45–12:45）：后一条不并进前一条的单元
    world.ledger.record_expectation(expectation("early", WRITE_DOC, at(DAY, 9), deadline_minutes=90))
    world.ledger.record_expectation(expectation("later", WRITE_DOC, at(DAY, 10, 45), deadline_minutes=120))
    ticker = world.ticker()
    _ticks(world, at(DAY, 9, 15), at(DAY, 9, 45), ticker)
    assert world.deliveries(DAY)["early"][-1] == "待取"
    world.set_clock(at(DAY, 9, 50))
    world.stop_using(WRITE_DOC)
    world.tick(at(DAY, 10), ticker)
    outlet = world.foresight.outlet
    claimed = outlet.claim(outlet.tools[0].host, "after", at(DAY, 10, 5), 3)
    _ticks(world, at(DAY, 10, 15), at(DAY, 11, 30), ticker)
    made = world.deliveries(DAY)
    assert (made["early"][-1], made["later"][-1], [item.request_id for item in claimed]) == ("已停用", "已停用", [])


# ── 一拍开头取一份下级列表（C27）─────────────────────────────────────────────────


class _GrowsMidTick(NoUnsealed):
    """这一拍读还没封口的判断时（拍已经开始），词表在写文档下面长出一个候选。"""

    def __init__(self, world: World) -> None:
        self.world = world
        self.grown: str | None = None

    async def prepare(self, *, until: datetime) -> None:
        if self.grown is None:
            self.grown = self.world.grow(WRITE_DOC, "写需求书")


def test_c27_the_descendants_are_read_once_at_the_start_of_the_tick(tmp_path: Path) -> None:
    """守门：拍中间词表长了新候选，这一拍仍用开头那一份（旧代码上本来就成立：旧代码根本不认下级）。"""

    world = World(tmp_path)
    world.use(WRITE_DOC)
    coming = str(world.vocabulary().next_id())
    world.record(coming, at(DAY, 8, 30))
    world.ledger.record_expectation(_day_expectation("doc-day", WRITE_DOC, (9 * 60, 10 * 60)))
    hook = _GrowsMidTick(world)
    world.tick(at(DAY, 9), world.ticker(unsealed=hook))
    assert hook.grown == coming
    assert world.deliveries(DAY) == {"doc-day": ["影子"]}


# ── 现状：不在用的类现在也有日子预期（3.2 说按在用节点算）─────────────────────────────


def test_an_idle_class_with_records_on_several_days_gets_no_day_expectation(tmp_path: Path) -> None:
    world = World(tmp_path)
    world.use(WEEKLY)
    for day in (date(2026, 4, 1), date(2026, 4, 3), date(2026, 4, 6), date(2026, 4, 8)):
        world.record(WEEKLY, at(day, 9))
        world.record(SUMMARY, at(day, 15))
    night = date(2026, 4, 10)
    report = world.generate(night)
    world.opportunities(report)
    world.day_expectations()
    days = {item.kind_token for item in world.ledger.expectations_on(night) if item.source == "日子"}
    assert (WEEKLY in days, SUMMARY in days) == (True, False)
