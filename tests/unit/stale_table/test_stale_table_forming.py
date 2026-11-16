"""预测不等换代（2026-11-16，``预期.md`` 第 2、3 节）：今天这一代还没做成时，预测拍用盘上最近一张已完成的那一代的机会表建事件、
情境预期；今天这一代做成后下一拍起换用今天的表。日子预期按用户 2026-11-16 第 1 条（取代阶段 0 定的 C15）：从旧表那一晚到昨天
出现过的类旧表下不建、等今天的表；没出现过的类旧表下照建（那一半在 ``test_stale_table_verify_c``）。

都走真实的 ``ForesightTick.run_once``（接落盘水位、完成标记、递送），盘面见 ``harness``。改动后才有的读数按语义断言：
观测属性里有一个值是那张表的日子、预期身上有一个字段记着那张表的日子（``harness.mentions_day`` / ``records_table_day``）。
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from pathlib import Path

from tests.unit.scene.fixtures import publish
from tests.unit.stale_table.harness import (
    BASE,
    CODE,
    ZONE,
    World,
    days_after,
    mentions_day,
    records_table_day,
    world,
)

#: 现场：BASE 起 4 天（BASE … BASE+3）每天 10:00 讨论方案、10:20 修改代码；今天 = BASE+4。
TODAY = BASE + timedelta(days=4)
YESTERDAY = TODAY - timedelta(days=1)


def clock(day: date, hour: int, minute: int) -> datetime:
    return datetime.combine(day, time(hour, minute), tzinfo=ZONE)


def yesterday_done(tmp: Path, *, lanes: tuple[str, ...] = ("session",)) -> World:
    """昨天那一代完成了（表 + 完成标记），今天这一代还没成；今天 10:00 讨论方案刚做完。"""

    w = world(tmp, lanes=lanes)
    w.mornings(days_after(BASE, 4))
    for lane in lanes:
        w.table(YESTERDAY, lane=lane)
        w.mark(YESTERDAY, lane=lane)
    publish(w.site.behavior_tree, TODAY, "讨论方案", 10, 0)
    return w


def test_control_with_todays_table_the_tick_forms_the_event_expectation(tmp_path: Path) -> None:
    """对照（改动前后都该过）：今天这一代已做成，10:15 那一拍建出修改代码的事件预期。"""

    w = world(tmp_path)
    w.mornings(days_after(BASE, 4))
    w.generation(TODAY)
    publish(w.site.behavior_tree, TODAY, "讨论方案", 10, 0)
    report = w.tick(clock(TODAY, 10, 15))
    assert report.skipped is None and report.recorded == 1
    (event,) = w.expectations(TODAY, source="事件")
    assert event.kind_token == CODE and records_table_day(event, TODAY)


def test_d1_c1_before_todays_generation_the_tick_forms_from_yesterdays_table_and_says_it(tmp_path: Path) -> None:
    """D1 / C1：昨天那一代完成了、今天的还没成：这一拍用昨天的表建出事件预期，观测里记着用的是哪一天那一代的表；到时窗照常说
    （递送进「待取」）。"""

    w = yesterday_done(tmp_path)
    report = w.tick(clock(TODAY, 10, 15))
    events = w.expectations(TODAY, source="事件")
    assert [item.kind_token for item in events] == [CODE], f"旧表下没建事件预期（skipped={report.skipped!r}）"
    assert report.recorded == 1
    (tick_event,) = w.session.recorder.ticks()
    assert mentions_day(tick_event, YESTERDAY), f"观测里没有记那张表的日子：{dict(tick_event.attributes)}"
    w.earn(events[0])
    w.ticks(TODAY, time(10, 30), time(11, 0))
    assert "待取" in w.reasons(TODAY, events[0].expectation_id)


def test_d4_each_expectation_records_which_generations_table_it_was_formed_from(tmp_path: Path) -> None:
    """D4：旧表下建的那条记着它是按昨天那一代的表建的（预期自己的 ``day`` 就是今天，所以只断言"有一个字段是昨天"）。"""

    w = yesterday_done(tmp_path)
    w.tick(clock(TODAY, 10, 15))
    events = w.expectations(TODAY, source="事件")
    assert events, "旧表下没建事件预期"
    assert records_table_day(events[0], YESTERDAY)


def test_c2_c15_the_day_expectations_are_not_formed_from_an_old_table_but_by_todays_generation(
    tmp_path: Path,
) -> None:
    """C2（用户 2026-11-16 第 1 条，取代 C15）"出现过 → 不建"那一半：三类在旧表那一晚（昨天）都做过，旧表下预测拍只建事件、
    不建这几类的日子预期（旧表会把做过的当成久没做）；今天这一代做成时换代那一步用今天的表把日子预期建上；同一身份只建一次。
    "没出现过 → 用旧表建"那一半见 ``test_stale_table_verify_c``。"""

    w = yesterday_done(tmp_path)
    w.tick(clock(TODAY, 10, 15))
    assert w.expectations(TODAY, source="事件"), "旧表下事件预期照建（C15：别的种类照建）"
    assert w.expectations(TODAY, source="日子") == []
    w.generation(TODAY)  # 换代：今天的表、日子预期、完成标记
    built = w.expectations(TODAY, source="日子")
    assert built and all(records_table_day(item, TODAY) for item in built)
    w.tick(clock(TODAY, 11, 15))
    again = w.expectations(TODAY, source="日子")
    assert sorted(item.expectation_id for item in again) == sorted(item.expectation_id for item in built)
    assert len({item.expectation_id for item in again}) == len(again)


def test_c3_after_todays_generation_the_next_tick_switches_tables_catches_up_and_does_not_repeat(
    tmp_path: Path,
) -> None:
    """C3：接 C1、C2，一小时后今天这一代做成：下一拍起用今天的表（观测里是今天），从今天零点起回看补建（补建拍）；10:15 用旧表
    建的那条不重复建；同一条链今天只有一条进「待取」。"""

    w = yesterday_done(tmp_path)
    w.tick(clock(TODAY, 10, 15))
    early = w.expectations(TODAY, source="事件")
    assert len(early) == 1 and early[0].formed_at == clock(TODAY, 10, 15), "10:15 那一拍没有用昨天的表建出事件预期"
    w.earn(early[0])
    w.ticks(TODAY, time(10, 30), time(11, 0))
    w.generation(TODAY)
    switched = w.tick(clock(TODAY, 11, 15))
    assert mentions_day(w.session.recorder.ticks()[-1], TODAY)
    assert switched.catch_up is True  # 换表那一拍从今天零点起回看
    events = w.expectations(TODAY, source="事件")
    assert [item.expectation_id for item in events] == [early[0].expectation_id]
    w.ticks(TODAY, time(11, 30), time(13, 0))
    offered = [item for item in events if "待取" in w.reasons(TODAY, item.expectation_id)]
    assert len(offered) <= 1


def test_c4_a_half_written_todays_table_without_a_mark_is_not_used(tmp_path: Path) -> None:
    """C4：今天的表写出来了、日子预期那一步失败没有完成标记：预测拍不用今天这张半成品，用上一张完成了的（昨天）。"""

    w = yesterday_done(tmp_path)
    w.table(TODAY)  # 半成品：没有完成标记
    report = w.tick(clock(TODAY, 10, 15))
    events = w.expectations(TODAY, source="事件")
    assert events, f"今天的表没完成时没有退回上一张完成的表（skipped={report.skipped!r}）"
    assert records_table_day(events[0], YESTERDAY)
    assert mentions_day(w.session.recorder.ticks()[-1], YESTERDAY)


def test_c5_a_corrupt_todays_mark_falls_back_to_the_last_completed_table(tmp_path: Path) -> None:
    """C5：今天的完成标记文件坏了：预测拍用上一张完成了的表（健康面那一半在 ``test_stale_table_health``）。"""

    w = yesterday_done(tmp_path)
    w.table(TODAY)
    w.corrupt_mark(TODAY)
    report = w.tick(clock(TODAY, 10, 15))
    events = w.expectations(TODAY, source="事件")
    assert events, f"标记坏了时没有退回上一张完成的表（skipped={report.skipped!r}）"
    assert records_table_day(events[0], YESTERDAY)


def test_c6_a_table_from_two_weeks_ago_is_still_used(tmp_path: Path) -> None:
    """C6：最近一张完成的表是 14 天前的（两周没开机，刚开机，今天这一代还没成）：照样用它建预期，不设天数上限；到时窗说不说
    按现有"久别刚回来"的规矩（这里只断言递送做了决定）。"""

    w = world(tmp_path)
    w.mornings(days_after(BASE, 4))
    night = BASE + timedelta(days=4)
    w.table(night)
    w.mark(night)
    today = night + timedelta(days=14)
    publish(w.site.behavior_tree, today, "讨论方案", 10, 0)
    report = w.tick(clock(today, 10, 15))
    events = w.expectations(today, source="事件")
    assert events, f"14 天前的表没被用来建预期（skipped={report.skipped!r}）"
    assert records_table_day(events[0], night)
    assert mentions_day(w.session.recorder.ticks()[-1], night)
    w.ticks(today, time(10, 30), time(11, 0))
    assert w.reasons(today, events[0].expectation_id)  # 到时窗做了决定（说或按久别不说）


def test_c7_after_two_weeks_away_todays_generation_switches_the_table(tmp_path: Path) -> None:
    """C7：接 C6，半小时后今天这一代做成：换用今天的表（健康面那一项消失见 ``test_stale_table_health``）。"""

    w = world(tmp_path)
    w.mornings(days_after(BASE, 4))
    night = BASE + timedelta(days=4)
    w.table(night)
    w.mark(night)
    today = night + timedelta(days=14)
    publish(w.site.behavior_tree, today, "讨论方案", 10, 0)
    w.tick(clock(today, 10, 15))
    w.generation(today)
    w.tick(clock(today, 10, 45))
    assert mentions_day(w.session.recorder.ticks()[-1], today)
    assert mentions_day(w.session.recorder.ticks()[0], night), "换表之前那一拍没有用 14 天前的表"


def test_c8_ten_days_without_a_generation_keep_using_the_table_from_before(tmp_path: Path) -> None:
    """C8：有一天卡住没定稿，换代连着十天做不成：这十天预测拍都用卡住之前那张表建预期。"""

    w = world(tmp_path)
    w.mornings(days_after(BASE, 4))
    night = BASE + timedelta(days=4)
    w.table(night)
    w.mark(night)
    missing: list[date] = []
    for offset in range(1, 11):
        day = night + timedelta(days=offset)
        publish(w.site.behavior_tree, day, "讨论方案", 10, 0)
        w.tick(clock(day, 10, 15))
        events = w.expectations(day, source="事件")
        if not events or not records_table_day(events[0], night):
            missing.append(day)
        assert mentions_day(w.session.recorder.ticks()[-1], night), f"{day} 那一拍没有用卡住之前那张表"
    assert missing == []


def test_c12_each_branch_uses_its_own_table_and_mark(tmp_path: Path) -> None:
    """C12：物理一支今天这一代没做成、会话一支做成了：物理一支用它自己的旧表，会话一支用今天的表，互不影响。"""

    w = yesterday_done(tmp_path, lanes=("session", "physical"))
    w.generation(TODAY, lane="session")
    w.tick(clock(TODAY, 10, 15), lane="session")
    w.tick(clock(TODAY, 10, 15), lane="physical")
    session = w.expectations(TODAY, source="事件", lane="session")
    physical = w.expectations(TODAY, source="事件", lane="physical")
    assert session and records_table_day(session[0], TODAY)
    assert physical, "物理一支今天这一代没成时没有用它自己的旧表建预期"
    assert records_table_day(physical[0], YESTERDAY)
    assert mentions_day(w.lanes["physical"].recorder.ticks()[-1], YESTERDAY)
    assert mentions_day(w.lanes["session"].recorder.ticks()[-1], TODAY)


def test_c13_a_relation_gone_in_todays_table_keeps_its_old_expectation_but_gets_no_new_one(tmp_path: Path) -> None:
    """C13：旧表里有这条关系、今天的表里已经失效：换表之前按旧表建的那条照它自己的时窗走（照常递送）；换表之后不再为它建新的。"""

    w = yesterday_done(tmp_path)
    w.tick(clock(TODAY, 10, 15))
    early = w.expectations(TODAY, source="事件")
    assert len(early) == 1, "换表之前没有按旧表建出事件预期"
    w.ticks(TODAY, time(10, 30), time(11, 0))
    assert w.reasons(TODAY, early[0].expectation_id), "旧表下建的那条没有照它自己的时窗走"
    w.generation(TODAY, relations=())  # 今天这一代：这条关系失效
    publish(w.site.behavior_tree, TODAY, "讨论方案", 14, 0)
    w.ticks(TODAY, time(11, 15), time(14, 30))
    later = w.expectations(TODAY, source="事件")
    assert [item.expectation_id for item in later] == [early[0].expectation_id]


def test_c14_s5_an_expectation_from_an_old_table_is_settled_and_diagnosed_the_next_day(tmp_path: Path) -> None:
    """C14 / S5：旧表下建的预期第二天照常结算、诊断不报错；账上看得到它是按哪一代的表建的。"""

    w = yesterday_done(tmp_path)
    w.tick(clock(TODAY, 10, 15))
    events = w.expectations(TODAY, source="事件")
    assert events, "旧表下没建事件预期"
    w.ticks(TODAY, time(10, 30), time(11, 0))
    w.generation(TODAY + timedelta(days=1))  # 第二天那一代
    stage = w.settle(TODAY, first=BASE)
    settled = {item.expectation_id for item in w.session.ledger.settlements_on(TODAY)}
    assert events[0].expectation_id in settled
    stage.diagnose(now=clock(TODAY + timedelta(days=1), 5, 0))  # 只读诊断不抛
    (stored,) = [item for item in w.expectations(TODAY) if item.expectation_id == events[0].expectation_id]
    assert records_table_day(stored, YESTERDAY)
