"""裁定 89 第 1、2 条的回归（由 ``37-材料/readme/repro-audit123`` 的复现脚本改成，断言已翻转）：走真实链路——真实行为树、
每晚照夜批同一套调用建机会表（read_series → build_timelines → build_table）、白天每 15 分钟一拍 ``ForesightTick`` +
``DeliveryStage``、每晚 ``SettlementStage``。替身只有两处：「离上次够久」这本账的档由 ``Seeded.earn`` 种成提醒档，出口是
"宿主一直活着"。

1. 天天做的事漏过一次以后，单独漏做的那天照常到时窗末进「待取」：历史上隔过一次 2 天不再把时窗撑到次日；真正 1、2 天混着的、
   纯隔天的、天天做没漏过的，时窗与原来一样。
2. 会话 lane 一整天没有会话：对预测账等同于上游报了全天「未观测」——中性、不算机会、单元那一刻 ``came=None``、覆盖 0、跨这天的
   间隔 ``seen`` 降下来；那天有别的会话、只是没做，照旧是一笔坏。递送不变（裁定 68 第 2 条）。
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from habitus.behavior.kinds.ids import Lane, not_event_token
from habitus.conditions import LocalCalendar
from habitus.foresight import Moment, NoUnsealed
from habitus.foresight.history import record_of
from habitus.foresight.opportunities import build_table
from habitus.foresight.triggers import day_drafts
from habitus.runtime.foresight import ForesightReadSide
from habitus.runtime.foresight_days import DayExpectationStage
from habitus.runtime.foresight_delivery import DeliveryStage
from habitus.runtime.foresight_ledger import ForesightLedgerStore
from habitus.runtime.foresight_opportunities import OpportunityStore
from habitus.runtime.foresight_settlement import SettlementStage
from habitus.runtime.foresight_tick import ForesightTick
from habitus.scene.concepts import ConceptSet
from habitus.scene.concepts.rhythm import RhythmRulers
from habitus.scene.recurrence import recurrences_of
from habitus.scene.relations import RelationConfig, build_timelines
from habitus.series.reader import read_series
from habitus.temporal.coverage import coverage_of
from tests.unit.kind_ids import kind_id
from tests.unit.runtime.test_foresight_delivery import EVIDENCE, LOSS, Outlet, Seeded
from tests.unit.scene.concept_fixtures import base
from tests.unit.scene.fixtures import SUBJECT, Site, at, publish, publish_gap

ZONE = ZoneInfo("Asia/Shanghai")
CONFIG = RelationConfig(
    slot_minutes=15, transition_window_slots=3, absence_typical_multiple=8.0, absence_min_gaps=2
)
RULERS = RhythmRulers(15, 2700, 3, 5.0, 5.0, 0.10, 35)
TICK = timedelta(minutes=15)
START = date(2026, 10, 2)
HABIT = "吃维生素"
OTHER = "看新闻"  # 他那天别的会话


@dataclass
class World:
    site: Site
    ledger: ForesightLedgerStore
    opportunities: OpportunityStore
    delivery: DeliveryStage
    concepts: ConceptSet

    @property
    def kind(self) -> str:
        return kind_id(HABIT)


def world(tmp: Path) -> World:
    site = Site(tmp, now=at(START, 3, 0))
    root = tmp / "foresight" / "session"
    ledger, opportunities = ForesightLedgerStore(root), OpportunityStore(root)
    seeded = Seeded(root, ledger)
    seeded.earn(kind=kind_id(HABIT), trigger="interval", hits=8)  # 替身：这本「离上次够久」的账已在提醒档
    delivery = DeliveryStage(
        ledger,
        tolerance=LOSS,
        evidence=EVIDENCE,
        daily_cap=3,
        tick_seconds=TICK.total_seconds(),
        outlet=Outlet(alive=True),
        history=seeded.store,
    )
    return World(site, ledger, opportunities, delivery, ConceptSet([base(HABIT)]))


def nightly(w: World, night: date) -> None:
    series = read_series(w.site.behavior_tree, cutoff=night)
    lines = build_timelines(series, w.concepts, {}, CONFIG)
    if "session" not in lines:
        return
    w.opportunities.write(
        build_table(
            lines["session"],
            (),
            w.concepts,
            night=night,
            relation_night=night,
            slot_minutes=15,
            window_slots=3,
            recurrences=recurrences_of(series, RULERS),
        )
    )


def read_side(w: World) -> ForesightReadSide:
    return ForesightReadSide(w.site.behavior_tree, SUBJECT, LocalCalendar(), ZONE)


def settle(w: World, through: date, *, lane: str = "session") -> None:
    """接法同组合根（runtime/foresight.py 的 SettlementStage）：带机会表与线——补做强度、裁不裁后补都要机会表。"""

    closed = tuple(START + timedelta(days=offset) for offset in range((through - START).days + 1))
    now = datetime.combine(through + timedelta(days=1), time(4), tzinfo=ZONE)
    SettlementStage(
        w.ledger,
        read_side(w),
        closed_days=lambda: closed,
        current=lambda token: (token,),
        migrating=lambda: False,
        evidence=EVIDENCE,
        opportunities=w.opportunities,
        lane=lane,
        clock=lambda: now.astimezone(UTC),
        tolerance=LOSS,
    ).run_once()


def day_expectations(w: World, day: date, *, at: time = time(1, 15)) -> None:
    """夜里那条线的最后一步：用这一晚的表建当天的日子预期（裁定 94 第 1 条；前一天定稿后，这里取 01:15）。"""

    asyncio.run(
        DayExpectationStage(
            w.ledger,
            w.opportunities,
            read_side(w),
            NoUnsealed(),
            recent=35,
            tolerance=LOSS,
            clock=lambda: datetime.combine(day, at, tzinfo=ZONE),
        ).run_once()
    )


def morning(w: World, day: date, *, done: bool) -> None:
    """夜里那条线建好当天的日子预期，再 08:00–09:15 每 15 分钟一拍（时窗 08:30–08:45、09:00 那拍记「待取」）；做了就在
    08:30 发生、08:45 那拍时已进树。"""

    day_expectations(w, day)
    now = datetime.combine(day, time(8, 0), tzinfo=ZONE)
    while now <= datetime.combine(day, time(9, 15), tzinfo=ZONE):
        if done and now.time() == time(8, 45):
            publish(w.site.behavior_tree, day, HABIT, 8, 30)
        clock = now
        asyncio.run(
            ForesightTick(
                w.ledger,
                w.opportunities,
                read_side(w),
                NoUnsealed(),
                clock=lambda c=clock: c,
                delivery=w.delivery,
                tick_seconds=TICK.total_seconds(),
                recent=35,
                tolerance=LOSS,
            ).run_once()
        )
        now += TICK


def interval_items(w: World, day: date):  # type: ignore[no-untyped-def]
    return [item for item in w.ledger.expectations_on(day) if item.kind_token == w.kind and item.trigger == "interval"]


def reasons(w: World, day: date, expectation_id: str) -> list[str]:
    return [
        item.reason
        for item in sorted(w.ledger.deliveries_on(day), key=lambda item: item.decided_at)
        if item.expectation_id == expectation_id
    ]


def fact_of(w: World, item):  # type: ignore[no-untyped-def]
    found = [fact for fact in w.ledger.facts_of(item.account) if fact.expectation_id == item.expectation_id]
    return found[-1] if found else None


def window_of(item) -> tuple[str, str]:  # type: ignore[no-untyped-def]
    return tuple(moment.astimezone(ZONE).strftime("%m-%d %H:%M") for moment in item.window)  # type: ignore[return-value]


# ── 第 1 条 ────────────────────────────────────────────────────────────────


def test_a_daily_habits_second_single_miss_is_still_reminded_that_day(tmp_path: Path) -> None:
    """每天 08:30，10-14 漏、10-22 再漏（两天都开过别的会话）：两次漏做那天时窗都只在当天、09:00 都进「待取」、结成落空；
    10-21 那条也没有被撑到次日，10-22 那条不再被并进 10-21 的单元记「单元已选」（裁定 81 那条并单元路径自然不再触发）。"""

    w = world(tmp_path)
    misses = {date(2026, 10, 14), date(2026, 10, 22)}
    day = START
    while day <= date(2026, 10, 23):
        nightly(w, day)
        done = day not in misses
        if day >= date(2026, 10, 13):
            morning(w, day, done=done)
            settle(w, day - timedelta(days=1))
        elif done:
            publish(w.site.behavior_tree, day, HABIT, 8, 30)
        publish(w.site.behavior_tree, day, OTHER, 15, 0)
        day += timedelta(days=1)
    settle(w, date(2026, 10, 23))
    for miss in sorted(misses):
        (item,) = interval_items(w, miss)
        label = miss.strftime("%m-%d")
        assert window_of(item) == (f"{label} 08:30", f"{label} 08:45")
        assert reasons(w, miss, item.expectation_id)[:2] == ["等时窗末", "待取"]
        fact = fact_of(w, item)
        assert fact is not None and (fact.kind, fact.good, fact.strength) == ("过期待决", False, 1.0)
    (eve,) = interval_items(w, date(2026, 10, 21))
    assert window_of(eve) == ("10-21 08:30", "10-21 08:45")
    assert "单元已选" not in reasons(w, date(2026, 10, 22), interval_items(w, date(2026, 10, 22))[0].expectation_id)


def _windows_after(
    tmp: Path, gaps: list[int], *, sessions_on_skipped_days: bool = True
) -> list[tuple[str, str] | None]:
    """按给定间隔发布 08:30 的一件事（没做的日子照样开过别的会话，除非 ``sessions_on_skipped_days`` 为假），最后一次之后离上次
    1、2、3 天各在 08:00 读一次「离上次够久」：建不建、时窗从哪天几点到哪天几点（日子写成离上次第几天）。"""

    w = world(tmp)
    days = [START]
    for gap in gaps:
        days.append(days[-1] + timedelta(days=gap))
    day = START
    while day <= days[-1]:
        if day in days:
            publish(w.site.behavior_tree, day, HABIT, 8, 30)
        if day in days or sessions_on_skipped_days:
            publish(w.site.behavior_tree, day, OTHER, 15, 0)
        day += timedelta(days=1)
    found: list[tuple[str, str] | None] = []
    for elapsed in (1, 2, 3):
        today = days[-1] + timedelta(days=elapsed)
        nightly(w, today)
        table = w.opportunities.latest("session")
        assert table is not None
        moment = Moment(at=datetime.combine(today, time(8, 0), tzinfo=ZONE), slot=32)
        drafts = [
            draft
            for draft in day_drafts(table, moment, done_today=frozenset(), recent=35, line=LOSS.line)
            if draft.kind_token == w.kind and draft.trigger == "interval"
        ]
        if not drafts:
            found.append(None)
            continue
        (draft,) = drafts
        found.append(
            tuple(  # type: ignore[arg-type]
                f"+{(moment_.astimezone(ZONE).date() - days[-1]).days} {moment_.astimezone(ZONE):%H:%M}"
                for moment_ in draft.window
            )
        )
    return found


def test_only_a_real_one_or_two_day_rhythm_widens_the_window_to_the_next_day(tmp_path: Path) -> None:
    """复现报告那张四行表：天天做只隔过一次 2 天的，与天天做没漏过的一样只在当天；1、2 天混着的、纯隔天的与原来一样。
    超过了历史还没做的日子（裁定 90）按常见间隔每条只管一天：天天做的离上次 2、3 天各一条，1、2 天混着的（常见间隔 1 天）
    离上次 3 天一条，纯隔天的离上次 3 天不是它的日子（下一条在 4 天）。"""

    every_day = [("+1 08:30", "+1 08:45"), ("+2 08:30", "+2 08:45"), ("+3 08:30", "+3 08:45")]
    assert _windows_after(tmp_path / "daily", [1] * 15) == every_day
    assert _windows_after(tmp_path / "one-miss", [1] * 9 + [2] + [1] * 5) == every_day
    assert _windows_after(tmp_path / "mixed", [1, 2, 1, 2, 1, 1, 2, 2, 1, 2]) == [
        ("+1 08:30", "+2 08:30"),
        ("+1 08:30", "+2 08:30"),
        ("+3 08:30", "+3 08:45"),
    ]
    assert _windows_after(tmp_path / "every-other", [2] * 10) == [None, ("+2 08:30", "+2 08:45"), None]


def test_a_two_day_gap_across_a_day_without_any_session_does_not_widen_the_window(tmp_path: Path) -> None:
    """跨过一整天没会话的"隔 2 天"看不见那天他做没做（``seen`` 为 0）：不当成他隔了两天，时窗照样只在当天。"""

    found = _windows_after(tmp_path / "unseen", [1, 2, 1, 2, 1, 1, 2, 2, 1, 2], sessions_on_skipped_days=False)
    assert found[0] == ("+1 08:30", "+1 08:45")


# ── 第 2 条 ────────────────────────────────────────────────────────────────


def _one_empty_day(
    tmp: Path, *, gap: bool, other_session: bool, lane: str = "session", other: str = OTHER
) -> dict[str, object]:
    """每天 08:30 吃维生素；10-16 没做，那天：什么都没有 / 上游报了全天「未观测」/ 开过别的会话。10-18 04:00 结算。"""

    w = world(tmp)
    empty = date(2026, 10, 16)
    day = START
    while day <= empty + timedelta(days=1):
        nightly(w, day)
        if day < empty - timedelta(days=1):
            publish(w.site.behavior_tree, day, HABIT, 8, 30)
        else:
            if day == empty and gap:
                publish_gap(w.site.behavior_tree, day, (0, 0), (23, 59))
            morning(w, day, done=day != empty)
            if day == empty and other_session:
                publish(w.site.behavior_tree, day, other, 15, 0)
        day += timedelta(days=1)
    settle(w, empty + timedelta(days=1), lane=lane)
    (item,) = interval_items(w, empty)
    fact = fact_of(w, item)
    series = read_series(w.site.behavior_tree, cutoff=empty + timedelta(days=2))
    nightly(w, empty + timedelta(days=2))
    table = w.opportunities.latest("session")
    assert table is not None and fact is not None
    kind = table.kind(w.kind)
    assert kind is not None
    return {
        "deliveries": reasons(w, empty, item.expectation_id),
        "fact": (fact.kind, fact.good, fact.strength),
        "came": record_of(item, fact).came,
        "coverage": coverage_of(series, "session", slot_minutes=15).day_share(empty),
        "across": [(entry.days, round(entry.seen, 3)) for entry in kind.intervals if entry.days > 1],
    }


def test_a_whole_day_without_any_session_is_an_observation_gap_for_the_ledger(tmp_path: Path) -> None:
    silent = _one_empty_day(tmp_path / "silent", gap=False, other_session=False)
    reported = _one_empty_day(tmp_path / "reported", gap=True, other_session=False)
    # 与上游报了全天「未观测」一样：中性、单元那一刻不知道、覆盖 ≈0、跨这天的间隔看不见
    assert silent["fact"] == reported["fact"] == ("过期待决", False, 0.0)
    assert silent["came"] is None and reported["came"] is None
    assert silent["coverage"] == 0.0 and reported["coverage"] < 0.01
    assert silent["across"] == reported["across"] == [(2, 0.0)]
    # 递送不变（裁定 68 第 2 条）：白天还不知道这天会空着，时窗末照样进「待取」
    assert silent["deliveries"][:2] == ["等时窗末", "待取"]  # type: ignore[index]


def test_a_day_with_other_sessions_but_no_habit_is_still_a_miss(tmp_path: Path) -> None:
    watched = _one_empty_day(tmp_path, gap=False, other_session=True)
    assert watched["fact"] == ("过期待决", False, 1.0) and watched["came"] is False
    assert watched["coverage"] == 1.0 and watched["across"] == [(2, 1.0)]


def test_a_day_whose_only_session_was_not_an_event_is_still_watched(tmp_path: Path) -> None:
    """那天只有一轮被判成「非事件」（不进序列）：那天还是开过会话，在看、没做，照旧是一笔坏。"""

    chatted = _one_empty_day(tmp_path, gap=False, other_session=True, other=not_event_token(Lane.SESSION))
    assert chatted["fact"] == ("过期待决", False, 1.0) and chatted["coverage"] == 1.0


def test_the_physical_lanes_settlement_keeps_an_empty_day_watched(tmp_path: Path) -> None:
    """物理 lane 的上游没在看时自己报空白：结算按物理 lane 的口径，空着的一天照旧是在看、没做。"""

    physical = _one_empty_day(tmp_path, gap=False, other_session=False, lane="physical")
    assert physical["fact"] == ("过期待决", False, 1.0)
