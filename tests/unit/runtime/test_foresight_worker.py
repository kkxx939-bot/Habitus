"""预测层的常驻 worker：拍点对齐槽边界跑那一拍（实时化方案第 5 节）、失败记下来不死、停得下来。"""

from __future__ import annotations

import asyncio
from datetime import timedelta
from zoneinfo import ZoneInfo

import pytest

from habitus.conditions import LocalCalendar
from habitus.foresight import ForesightError, NoUnsealed
from habitus.foundation.observability import ObservationEvent, ObservationStatus
from habitus.runtime.foresight import ForesightReadSide
from habitus.runtime.foresight_tick import ForesightTick
from habitus.runtime.foresight_worker import ForesightWorker
from tests.unit.runtime.test_foresight_tick import TODAY, TOLERANCE, site_with_three_mornings, stores
from tests.unit.scene.fixtures import SUBJECT, at, publish


class Recorder:
    def __init__(self) -> None:
        self.events: list[ObservationEvent] = []

    def record(self, event: ObservationEvent) -> None:
        self.events.append(event)


def worker(tmp_path, *, broken: bool = False, recorder: Recorder | None = None) -> ForesightWorker:  # type: ignore[no-untyped-def]
    site, _codes = site_with_three_mornings(tmp_path)
    ledger, opportunities = stores(tmp_path, site)
    publish(site.behavior_tree, TODAY, "讨论方案", 10, 0)  # 刚讨论完：这一拍建一条「事件」预期（日子预期在夜里建）
    read_side = ForesightReadSide(site.behavior_tree, SUBJECT, LocalCalendar(), ZoneInfo("Asia/Shanghai"))
    tick = ForesightTick(
        ledger, opportunities, read_side, NoUnsealed(), clock=lambda: at(TODAY, 10, 15), recent=35, tolerance=TOLERANCE
    )
    if broken:  # 那一拍读不出来（派生树读不到；坏的机会表文件不再算：往前退一晚、计入观测）

        async def unreadable(_now: object) -> None:
            raise ForesightError("derived tree cannot be read")

        tick._read = unreadable  # type: ignore[method-assign,assignment]
    # 时钟停在 10:15，正好是一个槽边界：起来这一拍就跑，下一拍等 10:30
    return ForesightWorker(
        tick,
        interval_seconds=900.0,
        shutdown_timeout_seconds=5.0,
        zone=ZoneInfo("Asia/Shanghai"),
        observer=recorder,
        clock=lambda: at(TODAY, 10, 15),
    )


def test_the_loop_runs_a_tick_then_waits_and_stops(tmp_path) -> None:
    recorder = Recorder()
    run = worker(tmp_path, recorder=recorder)

    async def scenario() -> None:
        await run.start()
        await asyncio.sleep(0.05)
        assert run.running and run.last_error is None
        with pytest.raises(RuntimeError):
            await run.run_once()
        await run.stop()
        assert not run.running

    asyncio.run(scenario())
    loops = [event for event in recorder.events if event.operation == "tick_loop"]
    assert loops and loops[0].status is ObservationStatus.SUCCESS and loops[0].attributes["recorded"] == 1
    assert asyncio.run(run.run_once()).recorded == 0  # 手动再跑一拍：同一身份不再写


def test_a_failing_tick_is_recorded_and_the_loop_survives(tmp_path) -> None:
    recorder = Recorder()
    run = worker(tmp_path, broken=True, recorder=recorder)

    async def scenario() -> None:
        await run.start()
        await asyncio.sleep(0.05)
        assert run.running and run.last_error is not None
        await run.stop()

    asyncio.run(scenario())
    assert [event.status for event in recorder.events if event.operation == "tick_loop"] == [ObservationStatus.FAILURE]
    with pytest.raises(ValueError):
        ForesightWorker(run.tick, interval_seconds=0.0, shutdown_timeout_seconds=1.0, zone=ZoneInfo("Asia/Shanghai"))
    assert timedelta(seconds=run.interval_seconds) == timedelta(minutes=15)


def test_ticks_start_on_slot_boundaries_and_skip_the_ones_already_passed(tmp_path) -> None:
    """E12：拍点是当地零点起每一槽的边界；一拍拖过了几个边界，下一拍从此刻之后的边界接着走（不补跑）。"""

    run = worker(tmp_path)
    assert run.next_boundary(at(TODAY, 10, 16), inclusive=True) == at(TODAY, 10, 30)
    assert run.next_boundary(at(TODAY, 10, 30), inclusive=True) == at(TODAY, 10, 30)
    assert run.next_boundary(at(TODAY, 10, 30), inclusive=False) == at(TODAY, 10, 45)
    assert run.next_boundary(at(TODAY, 23, 50), inclusive=True) == at(TODAY + timedelta(days=1), 0, 0)
