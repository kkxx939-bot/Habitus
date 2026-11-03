"""裁定 94 第 1 条的回归（D1、D4，``37-材料/readme/repro-d1/report.md``）：当天的日子预期是夜里那条线的最后一步，一天只建一次，
用这条线刚写的、含昨天的那张表；白天那一拍不按"日期变了"自己建。

走真实链路（接法同 ``test_foresight_daily_miss_and_silent_day``）：真实行为树、夜里那条线同一套调用建机会表（read_series →
build_timelines → build_table）、全天每 15 分钟一拍 ``ForesightTick``、前一天定稿后（这里 01:15）先结算到昨天、再建今天这一晚的
表、再由 ``DayExpectationStage`` 建当天的日子预期。

1. 零点到这条线跑完之间，白天那一拍一条日子预期也不建（盘上只有前一晚的表）；跑完之后才有，形成在 01:15、从零点起算。
2. 查天气：工作日 07:40、周六 09:32。周日那条用含周六的表建，时窗延到 09:32，周日 09:32 那次结「准时」（零点拿旧表建时
   时窗只到 07:55，结成「晚了但做了」）。
3. 零点到这条线之间做的那次算进当天：结当天那条的「早到」，不记成昨天那条的「后补」。
4. 后补边界（D4）：昨天漏了、今天照常 08:30 做的那次归今天那条（「准时」），昨天那条不裁「后补」。
5. 每天都建（裁定 94 第 2 条）：时窗整个落在零点到这条线之间的习惯，做了的照建结「准时」，没做的那天也照建、结「过期待决」，
   递送不为它开口（时窗已过）。
"""

from __future__ import annotations

import asyncio
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path

from habitus.foresight import NoUnsealed
from habitus.runtime.foresight_ledger import ForesightLedgerStore
from habitus.runtime.foresight_opportunities import OpportunityStore
from habitus.runtime.foresight_settlement import SettlementStage
from habitus.runtime.foresight_tick import ForesightTick
from habitus.runtime.nightly_worker import LastRun
from habitus.scene.concepts import ConceptSet
from tests.unit.kind_ids import kind_id
from tests.unit.runtime.test_foresight_daily_miss_and_silent_day import (
    LOSS,
    TICK,
    ZONE,
    day_expectations,
    nightly,
    read_side,
)
from tests.unit.runtime.test_foresight_delivery import EVIDENCE
from tests.unit.scene.concept_fixtures import base
from tests.unit.scene.fixtures import Site, at, publish

#: 前一天定稿、夜里那条线跑完的时刻（正常用法 D 日在 D+1 01:00 后定稿，裁定 92 第 1 条）。
LINE_AT = time(1, 15)
OTHER = "看新闻"  # 他每天别的会话
Done = dict[date, list[tuple[int, int]]]


class World:
    """出口与递送不在这几条里：只看建了什么、结成什么。"""

    def __init__(self, tmp: Path, habit: str, start: date) -> None:
        self.site = Site(tmp, now=at(start, 3, 0))
        root = tmp / "foresight" / "session"
        self.ledger, self.opportunities = ForesightLedgerStore(root), OpportunityStore(root)
        self.concepts = ConceptSet([base(habit)])
        self.habit = habit
        self.start = start
        #: 夜里那条线自己的记录：结算据它认停掉的那天（裁定 94 第 9 条）
        self.last_run = LastRun()

    @property
    def kind(self) -> str:
        return kind_id(self.habit)


def settle(w: World, through: date) -> None:
    """接法同组合根：定稿到 ``through``，次日 01:15 结算（带机会表与线，裁不裁后补要它们）。"""

    closed = tuple(w.start + timedelta(days=offset) for offset in range((through - w.start).days + 1))
    now = datetime.combine(through + timedelta(days=1), LINE_AT, tzinfo=ZONE)
    SettlementStage(
        w.ledger,
        read_side(w),  # type: ignore[arg-type]
        closed_days=lambda: closed,
        current=lambda token: (token,),
        migrating=lambda: False,
        evidence=EVIDENCE,
        opportunities=w.opportunities,
        lane="session",
        clock=lambda: now.astimezone(UTC),
        tolerance=LOSS,
        finished=w.last_run.finished,
    ).run_once()


def tick(w: World, now: datetime) -> None:
    asyncio.run(
        ForesightTick(
            w.ledger,
            w.opportunities,
            read_side(w),  # type: ignore[arg-type]
            NoUnsealed(),
            clock=lambda c=now: c,
            tick_seconds=TICK.total_seconds(),
            recent=35,
            tolerance=LOSS,
        ).run_once()
    )


def line(w: World, day: date, *, finish: bool = True) -> None:
    """前一天定稿之后的那条线：结算到昨天 → 今天这一晚的机会表 → 当天的日子预期（``finish`` 为假：表写出后、日子预期那一段
    之前停下了）。线自己记下这一晚跑完了没有。"""

    settle(w, day - timedelta(days=1))
    nightly(w, day)  # type: ignore[arg-type]
    if finish:
        day_expectations(w, day, at=LINE_AT)  # type: ignore[arg-type]
    w.last_run.write(day, finished=finish)


def run(
    tmp: Path,
    habit: str,
    done: Done,
    live: tuple[date, ...],
    *,
    start: date,
    stopped: tuple[date, ...] = (),
    stopped_after_table: tuple[date, ...] = (),
) -> World:
    """``live`` 之前的直接进树，前一晚的表先建好（``live[0]`` 零点那几拍盘上就是它）；之后逐日：每 15 分钟一拍，01:15 那一拍
    之前跑夜里那条线（``stopped`` 里的日子那一晚线停在结构之前：不结算、没有机会表、没有日子预期；``stopped_after_table`` 里的
    日子表写出了、日子预期那一段没跑完）；做了的在发生之后那一拍之前进树。最后再结算到次日（最后一天那几条的最晚时限那天定稿）。"""

    w = World(tmp, habit, start)
    day = start
    while day < live[0]:
        for hour, minute in done.get(day, []):
            publish(w.site.behavior_tree, day, habit, hour, minute)
        publish(w.site.behavior_tree, day, OTHER, 23, 50)
        day += timedelta(days=1)
    nightly(w, live[0] - timedelta(days=1))  # type: ignore[arg-type]
    w.last_run.write(live[0] - timedelta(days=1), finished=True)
    for day in live:
        pending = sorted(done.get(day, []))
        now = datetime.combine(day, time(0), tzinfo=ZONE)
        while now <= datetime.combine(day, time(23, 45), tzinfo=ZONE):
            while pending and datetime.combine(day, time(*pending[0]), tzinfo=ZONE) < now:
                hour, minute = pending.pop(0)
                publish(w.site.behavior_tree, day, habit, hour, minute)
            if now.time() == LINE_AT and day in stopped:
                w.last_run.write(day, finished=False)
            elif now.time() == LINE_AT:
                line(w, day, finish=day not in stopped_after_table)
            tick(w, now)
            now += TICK
        publish(w.site.behavior_tree, day, OTHER, 23, 50)
    settle(w, live[-1] + timedelta(days=1))  # 最后一天那几条的最晚时限在次日零点
    return w


def interval_items(w: World, day: date):  # type: ignore[no-untyped-def]
    return [item for item in w.ledger.expectations_on(day) if item.kind_token == w.kind and item.source == "日子"]


def outcome_of(w: World, item) -> str | None:  # type: ignore[no-untyped-def]
    found = [s.outcome for s in w.ledger.settlements_on(item.day) if s.expectation_id == item.expectation_id]
    return found[-1] if found else None


def rulings_of(w: World, item) -> list[str]:  # type: ignore[no-untyped-def]
    return [r.ruling for r in w.ledger.rulings_on(item.day) if r.expectation_id == item.expectation_id]


def hhmm(moment: datetime) -> str:
    return moment.astimezone(ZONE).strftime("%m-%d %H:%M")


def daily(start: date, end: date, hour: int, minute: int) -> Done:
    done: Done = {}
    day = start
    while day <= end:
        done[day] = [(hour, minute)]
        day += timedelta(days=1)
    return done


# ── 1 ────────────────────────────────────────────────────────────────────────


def test_no_day_expectation_is_built_between_midnight_and_the_nightly_line(tmp_path: Path) -> None:
    start, today = date(2026, 10, 2), date(2026, 10, 14)
    w = World(tmp_path, "吃维生素", start)
    for offset in range((today - start).days):
        day = start + timedelta(days=offset)
        publish(w.site.behavior_tree, day, w.habit, 8, 30)
    nightly(w, today - timedelta(days=1))  # type: ignore[arg-type]  # 前一晚的表：只到前天
    now = datetime.combine(today, time(0), tzinfo=ZONE)
    while now.time() < LINE_AT:
        tick(w, now)
        now += TICK
    assert interval_items(w, today) == []  # 零点那几拍盘上只有旧表：白天那一拍不建日子预期
    line(w, today)
    (item,) = interval_items(w, today)
    assert hhmm(item.formed_at) == "10-14 01:15" and hhmm(item.observed_from) == "10-14 00:00"
    assert item.night == today  # 含昨天的那一晚
    tick(w, now)
    assert interval_items(w, today) == [item]  # 之后的拍不再建第二条


# ── 2 ────────────────────────────────────────────────────────────────────────


def weather() -> Done:
    """查天气：工作日 07:40，04-18（周六）09:32；04-19（周日）09:32。"""

    done: Done = {date(2026, 4, 13) + timedelta(days=offset): [(7, 40)] for offset in range(5)}
    done[date(2026, 4, 18)] = [(9, 32)]
    done[date(2026, 4, 19)] = [(9, 32)]
    return done


def test_the_sunday_weather_window_reaches_saturdays_late_check_and_settles_on_time(tmp_path: Path) -> None:
    sunday = date(2026, 4, 19)
    w = run(tmp_path, "查天气", weather(), (date(2026, 4, 18), sunday), start=date(2026, 4, 13))
    (item,) = interval_items(w, sunday)
    # 含 04-18 的那张表：最近 35 次的最早到最晚，延到周六 09:32
    assert (hhmm(item.window[0]), hhmm(item.window[1])) == ("04-19 07:40", "04-19 09:32")
    assert item.night == sunday and hhmm(item.deadline) == "04-20 00:00"
    assert outcome_of(w, item) == "准时"


# ── 3、4 ────────────────────────────────────────────────────────────────────


def test_a_time_done_between_midnight_and_the_line_counts_for_today_not_as_yesterdays_catch_up(
    tmp_path: Path,
) -> None:
    start, missed, today = date(2026, 10, 2), date(2026, 10, 13), date(2026, 10, 14)
    done = daily(start, today, 8, 30)
    done[missed] = []
    done[today] = [(0, 40)]  # 昨天漏了，今天零点过后 00:40 就做了
    w = run(tmp_path, "吃维生素", done, (missed, today), start=start)
    (yesterday,) = interval_items(w, missed)
    (item,) = interval_items(w, today)
    assert outcome_of(w, yesterday) == "过期待决" and rulings_of(w, yesterday) == []
    assert outcome_of(w, item) == "早到"


def test_yesterdays_miss_does_not_take_todays_ordinary_time_as_a_catch_up(tmp_path: Path) -> None:
    """D4：后补只认过期待决之后、下一条预期开始看之前（今天这条从零点起看）补做的。"""

    start, missed, today = date(2026, 10, 2), date(2026, 10, 13), date(2026, 10, 14)
    done = daily(start, today, 8, 30)
    done[missed] = []
    w = run(tmp_path, "吃维生素", done, (missed, today), start=start)
    (yesterday,) = interval_items(w, missed)
    (item,) = interval_items(w, today)
    assert item.formed_at > yesterday.deadline  # 形成晚于前一条的最晚时限：只有一个来源
    assert item.observed_from == yesterday.deadline
    assert outcome_of(w, yesterday) == "过期待决" and rulings_of(w, yesterday) == []
    assert outcome_of(w, item) == "准时"


def test_a_day_whose_line_stopped_is_unseen_and_does_not_feed_a_catch_up(tmp_path: Path) -> None:
    """裁定 94 第 8 条：线停掉的那天（没有那一晚的机会表、没有日子预期）按看不到——后补只看到它开始为止。前一天漏了、停掉那天
    照常 08:30 做的那次不裁成前一天那条的「后补」（不改的话后补一直看到再下一天零点，正好把它算进去）。"""

    start, missed, dark, after = date(2026, 10, 2), date(2026, 10, 12), date(2026, 10, 13), date(2026, 10, 14)
    done = daily(start, after, 8, 30)
    done[missed] = []
    w = run(tmp_path, "吃维生素", done, (missed, dark, after), start=start, stopped=(dark,))
    (yesterday,) = interval_items(w, missed)
    assert interval_items(w, dark) == [] and dark not in w.opportunities.nights("session")
    assert outcome_of(w, yesterday) == "过期待决" and rulings_of(w, yesterday) == []
    (item,) = interval_items(w, after)
    assert outcome_of(w, item) == "准时"


def test_a_day_whose_line_stopped_after_writing_the_table_is_unseen_too(tmp_path: Path) -> None:
    """review-94b 中低 3：线在表写出之后才停（单元历史写失败、日子预期那一段抛错或一整天没准备好）——那天有机会表、没有日子
    预期，同样按看不到：前一天漏了，停掉那天照常 08:30 做的那次不裁成前一天那条的「后补」。原来只认"那一晚没有机会表"，这种认不出，
    D4 在那天重现；撤回改动这里就红。"""

    start, missed, dark, after = date(2026, 10, 2), date(2026, 10, 12), date(2026, 10, 13), date(2026, 10, 14)
    done = daily(start, after, 8, 30)
    done[missed] = []
    w = run(tmp_path, "吃维生素", done, (missed, dark, after), start=start, stopped_after_table=(dark,))
    (yesterday,) = interval_items(w, missed)
    assert interval_items(w, dark) == [] and dark in w.opportunities.nights("session")
    assert outcome_of(w, yesterday) == "过期待决" and rulings_of(w, yesterday) == []


# ── 5 ────────────────────────────────────────────────────────────────────────


def test_a_habit_whose_window_ends_before_the_line_is_built_every_day_and_a_miss_settles_overdue(
    tmp_path: Path,
) -> None:
    start, live = date(2026, 10, 2), (date(2026, 10, 14), date(2026, 10, 15))
    done = daily(start, live[-1], 0, 20)
    done[live[1]] = []  # 10-15 凌晨忘了交
    w = run(tmp_path, "交作业", done, live, start=start)
    (item,) = interval_items(w, live[0])
    assert outcome_of(w, item) == "准时"
    # 时窗 00:20–00:35 在这条线（01:15）跑完之前就关了：没做的那天也照建，不只留做了的一侧
    (missed,) = interval_items(w, live[1])
    assert hhmm(missed.formed_at) == "10-15 01:15" and hhmm(missed.observed_from) == "10-15 00:00"
    assert hhmm(missed.window[1]) < "10-15 01:15"
    assert outcome_of(w, missed) == "过期待决"
