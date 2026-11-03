"""裁定 90 的回归（由 ``37-材料/readme/repro-90/repro90_chain.py`` 改成，两条断言已翻转）：连漏期间照常建预期，什么时候不再
开口交给账的"规律断了"与提醒反馈。走真实链路——真实行为树、每晚照夜批同一套调用建机会表、白天每 15 分钟一拍 ``ForesightTick``
+ ``DeliveryStage``、每晚 ``SettlementStage``；他每天 15:00 另有一条别的会话（整天没会话 = 没在看，裁定 89 第 2 条）。

1. 建没建（档由 ``Seeded.earn`` 种成提醒档，只看建预期、递送、结算）：天天做没漏过连漏 5 天每天一条、每条只管当天、都进「待取」；
   每周一次连漏 2 周第 2 周那天也有一条；1、2 天真混着的空到第 3、4 天各一条，第 1 天那条两天时窗不再被并进前一天已做完的单元。
2. 什么时候停（不种档：每晚结算之后 ``HistoryStage`` 按当晚的表重标单元历史，档从影子自己升上来，"规律断了"随结算生效）：
   天天做连漏 8 天，第 1–6 天提醒、第 7 天起不开口；宿主每次问出口、他不回应也没做，第 4 天起不开口。一笔落空要等最晚时限
   那天（次日）封口才结算，结成过期待决的那天还要等"下一次机会之前补没补做"裁完才定稿（``runtime.foresight_settlement``
   ``_rule``），定稿了才进单元历史——漏做后第 3 晚，所以"规律断了"比"漏一天当晚就算"晚两天生效。
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from pathlib import Path

from habitus.foresight.ledger import SPOKEN, Delivery
from habitus.runtime.foresight_delivery import DeliveryStage
from habitus.runtime.foresight_history import HistoryStage
from habitus.runtime.foresight_history_store import HistoryStore
from habitus.runtime.foresight_ledger import ForesightLedgerStore
from habitus.runtime.foresight_opportunities import OpportunityStore
from habitus.scene.concepts import ConceptSet
from habitus.scene.relations import build_timelines
from habitus.series.reader import read_series
from tests.unit.runtime.test_foresight_daily_miss_and_silent_day import (
    CONFIG,
    HABIT,
    OTHER,
    TICK,
    World,
    interval_items,
    morning,
    reasons,
    settle,
    window_of,
    world,
)
from tests.unit.runtime.test_foresight_daily_miss_and_silent_day import nightly as plain_nightly
from tests.unit.runtime.test_foresight_delivery import EVIDENCE, LOSS, Outlet
from tests.unit.scene.concept_fixtures import base
from tests.unit.scene.fixtures import Site, at, publish

START = date(2026, 10, 2)


def _run(
    w: World,
    done: list[bool],
    *,
    ticking_from: int,
    nightly=plain_nightly,  # type: ignore[no-untyped-def]
    speak: bool = False,
) -> None:
    """第 i 天做不做由 ``done[i]`` 定；``ticking_from`` 之前只发布，之后每天早上逐拍跑、前一天夜里结算。"""

    for index, did in enumerate(done):
        day = START + timedelta(days=index)
        if index >= ticking_from:
            settle(w, day - timedelta(days=1))
        nightly(w, day)
        if index >= ticking_from:
            morning(w, day, done=did)
            if speak:
                _speak(w, day)
        elif did:
            publish(w.site.behavior_tree, day, HABIT, 8, 30)
        publish(w.site.behavior_tree, day, OTHER, 15, 0)
    settle(w, START + timedelta(days=len(done)))


def _speak(w: World, day: date) -> None:
    """宿主把今天的「待取」问出口（他不回应）：照出口 ``ReminderOutlet._record_spoken`` 追加一条「说了」。"""

    for open_day in w.ledger.open_days():
        made = w.ledger.deliveries_on(open_day)
        for item in made:
            if item.reason == "待取" and not any(x.spoken for x in made if x.expectation_id == item.expectation_id):
                spoken_at = datetime.combine(day, time(9, 30), tzinfo=item.decided_at.tzinfo)
                w.ledger.record_delivery(
                    open_day, Delivery(item.expectation_id, spoken_at, item.level, True, SPOKEN, 1.0, "该吃了")
                )


def _reminded(w: World, day: date) -> bool:
    return any("待取" in reasons(w, day, item.expectation_id) for item in interval_items(w, day))


def _days(first: int, count: int) -> list[date]:
    return [START + timedelta(days=first + offset) for offset in range(count)]


# ── 1. 建没建 ────────────────────────────────────────────────────────────────


def test_a_daily_streak_builds_one_same_day_expectation_every_missed_day(tmp_path: Path) -> None:
    """天天做、从没漏过，第 20 天起连漏 5 天：每天一条、只管当天、各是新的身份、都到时窗末进「待取」；回来那天照建、记「已发生」。"""

    w = world(tmp_path)
    done = [index not in range(20, 25) for index in range(27)]
    _run(w, done, ticking_from=16)
    identities = set()
    for day in _days(20, 5):
        (item,) = interval_items(w, day)
        label = day.strftime("%m-%d")
        assert window_of(item) == (f"{label} 08:30", f"{label} 08:45")
        assert reasons(w, day, item.expectation_id)[:2] == ["等时窗末", "待取"]
        identities.add(item.trigger_instance)
    assert len(identities) == 5
    (back,) = interval_items(w, START + timedelta(days=25))
    assert "已发生" in reasons(w, back.day, back.expectation_id)
    assert not any(_reminded(w, day) for day, did in zip(_days(16, 27 - 16), done[16:], strict=True) if did)


def test_a_weekly_habit_skipped_two_weeks_is_reminded_in_the_second_week_too(tmp_path: Path) -> None:
    """每周一次，倒数第 3、2 周那两次没做：第 1 周（离上次 7 天）与第 2 周（14 天）各一条、都进「待取」，中间的日子不建
    （不变成天天催）；回来那天（21 天）照建、记「已发生」。"""

    w = world(tmp_path)
    days = 12 * 7 + 1
    done = [index % 7 == 0 for index in range(days)]
    first, second = days - 1 - 21, days - 1 - 14
    done[first] = done[second] = False
    _run(w, done, ticking_from=days - 30)
    for index in (first, second):
        assert _reminded(w, START + timedelta(days=index))
    built = [index for index in range(first, second + 8) if interval_items(w, START + timedelta(days=index))]
    assert built == [first, second, second + 7]
    back = START + timedelta(days=second + 7)
    (item,) = interval_items(w, back)
    assert "已发生" in reasons(w, back, item.expectation_id) and not _reminded(w, back)


def test_a_mixed_rhythm_left_undone_is_reminded(tmp_path: Path) -> None:
    """1、2 天真混着，最后一次之后空 4 天：第 1 天那条两天时窗自己决定（不并进前一天已做完的单元记「单元已选」），到时窗末进
    「待取」；第 2 天仍是它（同一个身份，不另建）；超过历史的第 3、4 天按常见间隔 1 天各一条、只管当天、都进「待取」。"""

    w = world(tmp_path)
    gaps = [1, 2, 1, 2, 1, 1, 2, 2, 1, 2]
    done, at_ = [False] * 30, 0
    for index in range(30):
        if at_ >= 30:
            break
        done[at_] = True
        at_ += gaps[index % len(gaps)]
    last = max(index for index, did in enumerate(done) if did)
    done = done[: last + 1] + [False] * 4 + [True]
    _run(w, done, ticking_from=16)
    first, second, third, fourth = _days(last + 1, 4)
    (item,) = interval_items(w, first)
    assert window_of(item) == (f"{first:%m-%d} 08:30", f"{second:%m-%d} 08:30")
    assert "单元已选" not in reasons(w, first, item.expectation_id) and _reminded(w, first)
    assert interval_items(w, second) == []
    for day in (third, fourth):
        (later,) = interval_items(w, day)
        assert window_of(later) == (f"{day:%m-%d} 08:30", f"{day:%m-%d} 08:45") and _reminded(w, day)


# ── 2. 什么时候停 ────────────────────────────────────────────────────────────


def _live(tmp: Path) -> tuple[World, HistoryStage]:
    site = Site(tmp, now=at(START, 3, 0))
    root = tmp / "foresight" / "session"
    ledger, opportunities = ForesightLedgerStore(root), OpportunityStore(root)
    store = HistoryStore(root)
    delivery = DeliveryStage(
        ledger,
        tolerance=LOSS,
        evidence=EVIDENCE,
        daily_cap=3,
        tick_seconds=TICK.total_seconds(),
        outlet=Outlet(alive=True),
        history=store,
    )
    stage = HistoryStage(ledger, store, evidence=EVIDENCE, statuses=("established",))
    return World(site, ledger, opportunities, delivery, ConceptSet([base(HABIT)])), stage


def _relabelling(stage: HistoryStage):  # type: ignore[no-untyped-def]
    """夜批同序：建机会表之后按当晚的表与时间线重标单元历史（``runtime.foresight_nightly``）。"""

    def nightly(w: World, night: date) -> None:
        plain_nightly(w, night)
        table = w.opportunities.latest("session")
        if table is None:
            return
        lines = build_timelines(read_series(w.site.behavior_tree, cutoff=night), w.concepts, {}, CONFIG)
        stage.run(night=night, table=table, line=lines["session"])

    return nightly


def test_a_broken_streak_stops_the_reminders_by_the_ledger_not_a_fixed_day(tmp_path: Path) -> None:
    """不种档：天天做 40 天（档自己升上来），第 40 天起连漏 8 天、宿主一直没来取：每天都建；第 1–6 天进「待取」，第 7 天"规律
    断了"已把边打回影子、不开口（停在第几天由账算出，不写死；第 7 天早上单元历史里已有 4 刻连错，上界跌破 θ）。"""

    w, stage = _live(tmp_path)
    done = [index not in range(40, 48) for index in range(49)]
    _run(w, done, ticking_from=2, nightly=_relabelling(stage))
    streak = _days(40, 8)
    assert all(interval_items(w, day) for day in streak)
    assert [_reminded(w, day) for day in streak] == [True] * 6 + [False] * 2
    assert "影子" in reasons(w, streak[-1], interval_items(w, streak[-1])[0].expectation_id)


def test_unanswered_spoken_reminders_stop_on_the_fourth_day(tmp_path: Path) -> None:
    """连漏 5 天，宿主每次都把提醒问出口、他不回应也没做：提醒反馈两笔坏（第 1、2 天的，第 3、4 晚结算写下）之后上界 0.583 < θ，
    第 4 天起不开口——比"规律断了"早两天。"""

    w, stage = _live(tmp_path)
    done = [index not in range(40, 45) for index in range(46)]
    _run(w, done, ticking_from=2, nightly=_relabelling(stage), speak=True)
    assert [_reminded(w, day) for day in _days(40, 5)] == [True, True, True, False, False]
