"""定稿日规划：没有记录的日子也定稿，偏移沿用之前最近一条记录的。"""

from __future__ import annotations

import asyncio
from datetime import UTC, date, datetime, timedelta

from habitus.behavior.reduction.closure import DayClosureState, plan_closure
from tests.unit.behavior.test_reduction_runner import OBS_A, OBS_B, OBS_C, Harness, at, seed_wash_chain

EIGHT = timedelta(hours=8)
THU, FRI, SAT, SUN = date(2026, 10, 15), date(2026, 10, 16), date(2026, 10, 17), date(2026, 10, 18)


def _offsets(days_with_records: dict[date, tuple[timedelta, ...]]):
    return lambda day: days_with_records.get(day, ())


def test_a_silent_saturday_closes_with_fridays_offset() -> None:
    state = DayClosureState(closed={THU: "x"})
    horizon = datetime(2026, 10, 17, 16, 1, tzinfo=UTC)  # 周六本地 24:01（+08:00）
    plan = plan_closure(state, {FRI}, horizon=horizon, unsealed_days=frozenset(), offsets_of=_offsets({FRI: (EIGHT,)}))
    assert plan.closed_now == (FRI, SAT)
    assert plan.last_offset == EIGHT
    assert SUN not in plan.still_open  # 周日还没过完、又没有记录：不记账，下一轮再推


def test_persisted_offset_covers_days_after_the_last_record() -> None:
    state = DayClosureState(closed={FRI: "x"}, last_offset=EIGHT)
    horizon = datetime(2026, 10, 18, 16, 1, tzinfo=UTC)
    plan = plan_closure(state, set(), horizon=horizon, unsealed_days=frozenset(), offsets_of=_offsets({}))
    assert plan.closed_now == (SAT, SUN)


def test_a_lane_that_never_recorded_closes_nothing() -> None:
    plan = plan_closure(
        DayClosureState(),
        set(),
        horizon=datetime(2026, 10, 18, tzinfo=UTC),
        unsealed_days=frozenset(),
        offsets_of=_offsets({}),
    )
    assert plan.closed_now == ()
    assert plan.still_open == frozenset()


def test_unsealed_days_stay_open_while_later_silent_days_close() -> None:
    state = DayClosureState(closed={THU: "x"})
    horizon = datetime(2026, 10, 17, 16, 1, tzinfo=UTC)
    plan = plan_closure(
        state, {FRI}, horizon=horizon, unsealed_days=frozenset({FRI}), offsets_of=_offsets({FRI: (EIGHT,)})
    )
    assert FRI in plan.still_open
    assert SAT in plan.closed_now


def test_without_a_horizon_nothing_closes_and_touched_days_wait() -> None:
    plan = plan_closure(DayClosureState(), {FRI}, horizon=None, unsealed_days=frozenset(), offsets_of=_offsets({}))
    assert plan.closed_now == ()
    assert plan.still_open == frozenset({FRI})


def test_a_day_with_several_offsets_closes_at_its_latest_local_end() -> None:
    """同一天有 +08:00 与 +09:00 两种记录：+09:00 的日终早一小时，要等到 +08:00 的日终才定稿。"""

    state = DayClosureState(closed={THU: "x"})
    horizon = datetime(2026, 10, 16, 15, 30, tzinfo=UTC)  # +09:00 的周五已过完，+08:00 的还没有
    plan = plan_closure(
        state,
        {FRI},
        horizon=horizon,
        unsealed_days=frozenset(),
        offsets_of=_offsets({FRI: (EIGHT, timedelta(hours=9))}),
    )
    assert FRI in plan.still_open
    assert plan.last_offset == EIGHT


def test_the_runner_closes_a_silent_day_after_a_recorded_one(tmp_path) -> None:
    """真实 runner：16 日有一条链；18 日凌晨再跑一轮时，没有任何记录的 17 日也定稿。"""

    harness = Harness(tmp_path)
    seed_wash_chain(harness, harness.deliver(OBS_A, OBS_B, OBS_C))
    asyncio.run(harness.runner.run_once())
    harness.now = at(30 * 3600)  # 18 日 01:30 CST，封口视界已过 17 日本地结束
    asyncio.run(harness.runner.run_once())
    assert harness.runner.closed_days() == (date(2026, 8, 16), date(2026, 8, 17))
