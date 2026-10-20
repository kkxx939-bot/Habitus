"""第 N 晚在组合根的串法（第 8 步）：行为树读一次，语义树与机会表用同一份序列；整晚那一趟自己的常驻节拍（裁定 59）。"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from habitus.foundation.observability import ObservationEvent, ObservationStatus
from habitus.runtime.nightly import Nightly
from habitus.runtime.nightly_worker import NightlyWorker
from tests.unit.runtime.test_scene_night import NOW, TONIGHT, nightly, publish_tonight, site_with_three_bedtimes


def sealed_through(last):  # type: ignore[no-untyped-def]
    """归约说"到 ``last`` 为止都封口了"。"""

    return lambda: tuple(last - timedelta(days=offset) for offset in range(30))


def steady() -> tuple[int, bool]:
    return (1, False)


def assembled(tmp_path, *, clock=lambda: NOW, closed_days=None, vocabulary=steady):  # type: ignore[no-untyped-def]
    site, hits = site_with_three_bedtimes(tmp_path)
    scene, _provider = nightly(tmp_path, hits, site)
    closed = closed_days if closed_days is not None else sealed_through(TONIGHT)
    return site, Nightly(
        behavior_tree=site.behavior_tree,
        scene=scene,
        zone=ZoneInfo("Asia/Shanghai"),
        closed_days=closed,
        vocabulary=vocabulary,
        clock=clock,
    )


def test_one_read_feeds_the_semantic_layer_for_the_same_night(tmp_path, monkeypatch) -> None:
    site, run = assembled(tmp_path)
    publish_tonight(site)
    reads: list[object] = []
    original = run.series.read

    import threading

    threads: list[bool] = []

    def counting(*, cutoff):  # type: ignore[no-untyped-def]
        reads.append(cutoff)
        threads.append(threading.current_thread() is threading.main_thread())
        return original(cutoff=cutoff)

    monkeypatch.setattr(run.series, "read", counting)
    outcome = asyncio.run(run.run_once())
    assert reads == [NOW.astimezone(ZoneInfo("Asia/Shanghai")).date()]  # 行为树只读这一次
    assert threads == [False]  # 全量扫行为树下沉到线程，不卡事件循环（2026-10-18 四方评审）
    assert outcome.scene is not None and outcome.skipped is None
    assert outcome.scene.night == outcome.series.cutoff == outcome.night
    assert TONIGHT in outcome.scene.mapped_days


def test_nothing_runs_on_an_empty_behaviour_tree(tmp_path) -> None:
    """序列里一条记录都没有（上游刚接入）：语义树这一晚不跑，说出原因（清单 5.2：闸原来是"预测树发布了没有"）。"""

    _site, run = assembled(tmp_path)
    outcome = asyncio.run(run.run_once())
    assert outcome.scene is None and outcome.skipped == "序列里还没有记录"


def test_tonight_is_the_subjects_local_day() -> None:
    """零点刚过的上海：UTC 还是昨天，"今天"按主体所在地算。"""

    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as folder:
        _site, run = assembled(Path(folder), clock=lambda: datetime(2026, 8, 17, 16, 30, tzinfo=UTC))
        assert str(run.tonight()) == "2026-08-18"


def test_the_scene_run_must_read_the_same_tree(tmp_path) -> None:
    site, run = assembled(tmp_path)
    _other_site, other = assembled(tmp_path / "other")
    with pytest.raises(ValueError, match="same behaviour tree"):
        Nightly(
            behavior_tree=site.behavior_tree,
            scene=other.scene,
            zone=ZoneInfo("Asia/Shanghai"),
            closed_days=tuple,
            vocabulary=steady,
        )
    with pytest.raises(TypeError, match="zone"):
        Nightly(
            behavior_tree=site.behavior_tree,
            scene=run.scene,
            zone="Asia/Shanghai",
            closed_days=tuple,
            vocabulary=steady,
        )  # type: ignore[arg-type]


def test_the_night_stops_at_the_last_sealed_day(tmp_path) -> None:
    """昨天的链还没归约完（归约说只封口到前天）：截止日退到昨天，昨天这一晚不算（E1）——不把半天当成完整的一天映射、盖章。"""

    site, run = assembled(tmp_path, closed_days=sealed_through(TONIGHT - timedelta(days=1)))
    publish_tonight(site)
    outcome = asyncio.run(run.run_once())
    assert outcome.night == TONIGHT and outcome.series.cutoff == TONIGHT
    assert not outcome.series.on(TONIGHT)
    assert outcome.scene is None or TONIGHT not in outcome.scene.mapped_days


def test_nothing_runs_before_any_day_is_sealed(tmp_path) -> None:
    site, run = assembled(tmp_path, closed_days=tuple)
    publish_tonight(site)
    outcome = asyncio.run(run.run_once())
    assert outcome.scene is None and outcome.skipped == "一天都还没封口"


def test_nothing_runs_while_the_vocabulary_is_migrating(tmp_path) -> None:
    """词表迁移做到一半（树上的行改了一部分、新版本还没写）：这一晚不跑语义树，说出原因（E3）。"""

    site, run = assembled(tmp_path, vocabulary=lambda: (2, True))
    publish_tonight(site)
    outcome = asyncio.run(run.run_once())
    assert outcome.scene is None and outcome.skipped == "词表迁移还没做完"


def test_nothing_runs_when_the_vocabulary_changes_while_reading(tmp_path) -> None:
    states = iter([(1, False), (2, False)])
    site, run = assembled(tmp_path, vocabulary=lambda: next(states))
    publish_tonight(site)
    outcome = asyncio.run(run.run_once())
    assert outcome.scene is None and outcome.skipped == "读序列时词表变了"


class Recorder:
    def __init__(self) -> None:
        self.events: list[ObservationEvent] = []

    def record(self, event: ObservationEvent) -> None:
        self.events.append(event)


def test_the_nightly_worker_runs_a_night_then_waits_and_survives_a_failing_one(tmp_path) -> None:
    """整晚那一趟自己的常驻节拍（清单 5.1）：跑一趟、记下读数、等下一拍；一趟抛了只记 last_error，循环不死，下一趟成功就清掉。"""

    site, run = assembled(tmp_path)
    publish_tonight(site)
    recorder = Recorder()
    worker = NightlyWorker(run, interval_seconds=3600.0, shutdown_timeout_seconds=5.0, observer=recorder)

    async def scenario() -> None:
        await worker.start()
        await asyncio.sleep(0.5)
        assert worker.running and worker.last_error is None
        with pytest.raises(RuntimeError):
            await worker.run_once()
        await worker.stop()

    asyncio.run(scenario())
    (event,) = [item for item in recorder.events if item.operation == "night"]
    assert worker.last_outcome is not None and worker.last_outcome.scene is not None
    assert event.status is ObservationStatus.SUCCESS and event.attributes["night"] == str(worker.last_outcome.night)
    assert event.attributes["scene"] is True and event.attributes["records"] == len(worker.last_outcome.series.records)

    class Broken:
        async def run_once(self):  # type: ignore[no-untyped-def]
            raise OSError("disk")

    failing = NightlyWorker(Broken(), interval_seconds=3600.0, shutdown_timeout_seconds=5.0, observer=recorder)

    async def broken() -> None:
        await failing.start()
        await asyncio.sleep(0.05)
        assert failing.running and isinstance(failing.last_error, OSError)
        await failing.stop()

    asyncio.run(broken())
    assert recorder.events[-1].status is ObservationStatus.FAILURE
    with pytest.raises(TypeError):
        NightlyWorker(object(), interval_seconds=1.0, shutdown_timeout_seconds=1.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        NightlyWorker(run, interval_seconds=0.0, shutdown_timeout_seconds=1.0)


def test_a_night_whose_settlement_failed_leaves_an_error_on_the_worker(tmp_path) -> None:
    """整晚照跑完了、但结算抛了：常驻节拍在 ``last_error`` 上留痕（健康面读它），下一趟结算好了就清掉（2026-10-18 四方评审）。"""

    from habitus.runtime.nightly import NightOutcome
    from habitus.series import EventSeries

    outcomes = iter([NightOutcome(TONIGHT, EventSeries(cutoff=TONIGHT), None, None, settlement_failed="OSError")])

    class Stub:
        async def run_once(self):  # type: ignore[no-untyped-def]
            return next(outcomes, NightOutcome(TONIGHT, EventSeries(cutoff=TONIGHT), None, None))

    recorder = Recorder()
    worker = NightlyWorker(Stub(), interval_seconds=0.01, shutdown_timeout_seconds=5.0, observer=recorder)

    async def scenario() -> None:
        await worker.start()
        await asyncio.sleep(0.005)
        assert isinstance(worker.last_error, RuntimeError) and "OSError" in str(worker.last_error)
        await asyncio.sleep(0.05)
        assert worker.last_error is None  # 之后那一趟结算好了
        await worker.stop()

    asyncio.run(scenario())
    first = next(item for item in recorder.events if item.operation == "night")
    assert first.attributes["settlement_failed"] == "OSError"


def test_the_night_refreshes_the_calendar_before_anything_else(tmp_path) -> None:
    """每晚开头重读放假数据（用户 10-18：节假日在夜批时判断），先于结算与语义树。"""

    order: list[str] = []
    site, run = assembled(tmp_path)
    publish_tonight(site)
    run.refresh_calendar = lambda: order.append("日历")
    scene_run = run.scene.run

    async def scene_then(**kwargs):  # type: ignore[no-untyped-def]
        order.append("语义树")
        return await scene_run(**kwargs)

    run.scene.run = scene_then  # type: ignore[method-assign]
    asyncio.run(run.run_once())
    assert order == ["日历", "语义树"]
    with pytest.raises(TypeError, match="refresh_calendar"):
        Nightly(
            behavior_tree=site.behavior_tree,
            scene=run.scene,
            zone=ZoneInfo("Asia/Shanghai"),
            closed_days=tuple,
            vocabulary=steady,
            refresh_calendar=object(),
        )  # type: ignore[arg-type]
