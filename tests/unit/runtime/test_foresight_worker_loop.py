"""预测拍常驻循环的槽对齐（实时化方案第 5 节、E12）：测的是真实的 ``ForesightWorker._run_loop``，不只是 ``next_boundary``。

墙钟用可控的模拟时钟；``_wait`` 换成"把模拟时钟往前推要等的秒数"，``tick.run_once`` 换成只记下它被叫到的时刻与
``scheduled``、再按给定用时推进时钟的替身。循环里去掉对齐（起来立刻跑、或每拍固定等一拍的宽度）这里都会变红。
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from habitus.runtime.foresight_tick import TickReport
from habitus.runtime.foresight_worker import ForesightWorker

SHANGHAI = ZoneInfo("Asia/Shanghai")


def at(hour: int, minute: int, second: int = 0) -> datetime:
    return datetime(2026, 8, 18, hour, minute, second, tzinfo=SHANGHAI)


def loop(start: datetime, costs: list[float], ticks: int) -> list[tuple[datetime, datetime | None]]:
    """从 ``start`` 起跑真实循环，第 n 拍用时 ``costs[n]`` 秒；返回每拍（被叫到的模拟时刻，``scheduled``）。"""

    clock = [start]
    calls: list[tuple[datetime, datetime | None]] = []

    class Tick:
        async def run_once(self, *, scheduled: datetime | None = None) -> TickReport:
            calls.append((clock[0], scheduled))
            clock[0] += timedelta(seconds=costs[len(calls) - 1] if len(calls) <= len(costs) else 1.0)
            if len(calls) >= ticks:
                worker._stop_requested.set()
            return TickReport(clock[0], 0, 0)

    worker = ForesightWorker(
        Tick(),  # type: ignore[arg-type]
        interval_seconds=900.0,
        shutdown_timeout_seconds=1.0,
        zone=SHANGHAI,
        clock=lambda: clock[0],
    )

    async def wait(timeout: float) -> None:
        clock[0] += timedelta(seconds=timeout)
        await asyncio.sleep(0)

    worker._wait = wait  # type: ignore[method-assign]
    asyncio.run(worker._run_loop())
    return calls


def test_the_first_tick_waits_for_the_next_slot_boundary_and_every_tick_starts_on_one() -> None:
    """10:16:20 起来：第一拍等到 10:30（不是起来就跑），之后每拍都正好落在 15 分钟的边界上、``scheduled`` 就是那个边界。"""

    calls = loop(at(10, 16, 20), [5.0, 5.0, 5.0], ticks=3)
    assert [started for started, _scheduled in calls] == [at(10, 30), at(10, 45), at(11, 0)]
    assert [scheduled for _started, scheduled in calls] == [at(10, 30), at(10, 45), at(11, 0)]


def test_starting_on_a_boundary_runs_right_away() -> None:
    calls = loop(at(10, 15), [1.0, 1.0], ticks=2)
    assert [scheduled for _started, scheduled in calls] == [at(10, 15), at(10, 30)]


def test_a_tick_running_past_boundaries_does_not_replay_them_and_the_next_one_realigns() -> None:
    """第二拍拖了 40 分钟（10:45 → 11:25）：11:00、11:15 不补跑，下一拍在 11:30——仍在边界上，而不是"跑完再等一拍"的 11:40。"""

    calls = loop(at(10, 30), [5.0, 40 * 60.0, 5.0], ticks=3)
    assert [scheduled for _started, scheduled in calls] == [at(10, 30), at(10, 45), at(11, 30)]
    assert calls[2][0] == at(11, 30)


def test_a_wait_that_overshoots_the_boundary_starts_the_tick_late_but_keeps_the_schedule() -> None:
    """合盖醒来（一次等待把时钟推过了边界）：这一拍晚了也照样带着它该开始的那个边界，供 span 算"离槽边界晚了多少"。"""

    clock = [at(10, 16)]
    seen: list[tuple[datetime, datetime | None]] = []

    class Tick:
        async def run_once(self, *, scheduled: datetime | None = None) -> TickReport:
            seen.append((clock[0], scheduled))
            worker._stop_requested.set()
            return TickReport(clock[0], 0, 0)

    worker = ForesightWorker(
        Tick(),  # type: ignore[arg-type]
        interval_seconds=900.0,
        shutdown_timeout_seconds=1.0,
        zone=SHANGHAI,
        clock=lambda: clock[0],
    )

    async def wait(timeout: float) -> None:
        clock[0] += timedelta(seconds=timeout + 7.0)  # 每次醒来都比该醒的晚 7 秒
        await asyncio.sleep(0)

    worker._wait = wait  # type: ignore[method-assign]
    asyncio.run(worker._run_loop())
    ((started, scheduled),) = seen
    assert scheduled == at(10, 30) and started > scheduled
