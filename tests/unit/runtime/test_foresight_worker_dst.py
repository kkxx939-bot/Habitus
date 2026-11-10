"""预测拍的槽边界在夏令时那天（06-第二轮修复清单 G3）：边界按绝对时间推进，相邻两拍真实间隔始终是一个槽；不因墙钟回拨跳过重复的
那一小时、把系统一直在跑的那段写成「未观测」。真实 ``ForesightWorker._run_loop`` + 真实 ``TickTrack.begin``，模拟时钟按 UTC 走。"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

from habitus.runtime.foresight_tick import TickReport
from habitus.runtime.foresight_worker import ForesightWorker
from habitus.runtime.generation_watermark import TickTrack, TickWatermarkStore

LONDON = ZoneInfo("Europe/London")
SLOT = timedelta(minutes=15)


def day_of_ticks(tmp_path, day: datetime) -> tuple[list[datetime], list[tuple[datetime, datetime]]]:  # type: ignore[no-untyped-def]
    """从当地 ``day`` 零点前一刻起跑真实循环，直到当地第二天零点；每拍开头走一遍真实的 ``TickTrack.begin`` 并推进水位。"""

    clock = [datetime.combine(day.date(), datetime.min.time(), tzinfo=LONDON).astimezone(UTC) - timedelta(seconds=1)]
    end = datetime.combine(day.date() + timedelta(days=1), datetime.min.time(), tzinfo=LONDON).astimezone(UTC)
    unwatched: list[tuple[datetime, datetime]] = []
    track = TickTrack(
        TickWatermarkStore(tmp_path),
        slot=SLOT,
        tolerance_slots=2,
        unwatched=lambda start, stop: unwatched.append((start, stop)),
    )
    ticks: list[datetime] = []

    class Tick:
        async def run_once(self, *, scheduled: datetime | None = None) -> TickReport:
            now = clock[0].astimezone(LONDON)
            mark, _gap = await track.begin(now)
            await track.write(mark.ticked(now))
            ticks.append(clock[0])
            if clock[0] >= end:
                worker._stop_requested.set()
            return TickReport(now, 0, 0)

    worker = ForesightWorker(
        Tick(),  # type: ignore[arg-type]
        interval_seconds=SLOT.total_seconds(),
        shutdown_timeout_seconds=1.0,
        zone=LONDON,
        clock=lambda: clock[0],
    )

    async def wait(timeout: float) -> None:
        clock[0] += timedelta(seconds=timeout)
        await asyncio.sleep(0)

    worker._wait = wait  # type: ignore[method-assign]
    asyncio.run(worker._run_loop())
    return ticks, unwatched


def test_the_25_hour_day_ticks_every_slot_through_the_repeated_hour_and_writes_no_unwatched(tmp_path) -> None:
    ticks, unwatched = day_of_ticks(tmp_path, datetime(2026, 10, 25))
    assert {later - earlier for earlier, later in zip(ticks, ticks[1:], strict=False)} == {SLOT}
    assert len(ticks) == 25 * 4 + 1 and unwatched == []  # 零点到次日零点（含两端）
    repeated = [moment.astimezone(LONDON) for moment in ticks if moment.astimezone(LONDON).hour == 1]
    assert len(repeated) == 8  # 01:00–01:45 BST 与 01:00–01:45 GMT 各一遍（槽号按墙钟，两遍同一槽号）


def test_the_23_hour_day_ticks_every_slot(tmp_path) -> None:
    ticks, unwatched = day_of_ticks(tmp_path, datetime(2026, 3, 29))
    assert {later - earlier for earlier, later in zip(ticks, ticks[1:], strict=False)} == {SLOT}
    assert len(ticks) == 23 * 4 + 1 and unwatched == []
    assert all(moment.astimezone(LONDON).minute % 15 == 0 for moment in ticks)  # 仍落在墙钟的槽边界上
