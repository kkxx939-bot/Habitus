"""预测层的常驻 worker：按节拍跑那一拍、失败记下来不死、停得下来。"""

from __future__ import annotations

import asyncio
from datetime import timedelta
from zoneinfo import ZoneInfo

import pytest

from habitus.foresight import NoUnsealed
from habitus.foundation.observability import ObservationEvent, ObservationStatus
from habitus.prediction.store import PredictionTreeStore
from habitus.runtime.foresight import ForesightReadSide
from habitus.runtime.foresight_tick import ForesightTick
from habitus.runtime.foresight_worker import ForesightWorker
from habitus.scene import NominalCalendar
from tests.unit.runtime.test_foresight_tick import TODAY, site_with_three_mornings, stores
from tests.unit.scene.fixtures import SUBJECT, at


class Recorder:
    def __init__(self) -> None:
        self.events: list[ObservationEvent] = []

    def record(self, event: ObservationEvent) -> None:
        self.events.append(event)


def worker(tmp_path, *, broken: bool = False, recorder: Recorder | None = None) -> ForesightWorker:  # type: ignore[no-untyped-def]
    site, _codes = site_with_three_mornings(tmp_path)
    ledger, opportunities, prediction = stores(tmp_path, site)
    read_side = ForesightReadSide(site.behavior_tree, SUBJECT, NominalCalendar(), ZoneInfo("Asia/Shanghai"))
    tick = ForesightTick(
        ledger,
        opportunities,
        PredictionTreeStore(tmp_path / "missing", retained_generations=3) if broken else prediction,
        read_side,
        NoUnsealed(),
        clock=lambda: at(TODAY, 10, 15),
    )
    if broken:
        tick.prediction.root.mkdir(parents=True, exist_ok=True)
        (tick.prediction.root / "current.json").write_text("{", encoding="utf-8")
    return ForesightWorker(tick, interval_seconds=3600.0, shutdown_timeout_seconds=5.0, observer=recorder)


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
    assert loops and loops[0].status is ObservationStatus.SUCCESS and loops[0].attributes["recorded"] == 2
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
    with pytest.raises(TypeError):
        ForesightWorker(object(), interval_seconds=1.0, shutdown_timeout_seconds=1.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        ForesightWorker(run.tick, interval_seconds=0.0, shutdown_timeout_seconds=1.0)
    assert timedelta(seconds=run.interval_seconds) == timedelta(hours=1)
