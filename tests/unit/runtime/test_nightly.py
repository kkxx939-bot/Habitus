"""第 N 晚在组合根的串法（第 8 步）：行为树读一次，语义树与机会表用同一份序列；整晚那一趟自己的常驻节拍（裁定 59）。"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from datetime import time as clock_time
from zoneinfo import ZoneInfo

import pytest

from habitus.foundation.observability import ObservationEvent, ObservationStatus
from habitus.runtime.nightly import Nightly
from habitus.runtime.nightly_worker import LastRun, NightlyWorker
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


SHANGHAI = ZoneInfo("Asia/Shanghai")
FOUR = clock_time(4, 0)


def test_the_nightly_worker_catches_up_a_missed_night_then_waits_and_survives_a_failing_one(tmp_path) -> None:
    """整晚那一趟自己的常驻节拍：启动时错过了最近一个预定时刻（从没跑过）就补跑一趟、记下读数、等下一个预定时刻；
    一趟抛了只记 last_error，循环不死（裁定 67 第 7 条）。"""

    site, run = assembled(tmp_path)
    publish_tonight(site)
    recorder = Recorder()
    worker = NightlyWorker(
        run, at=FOUR, zone=SHANGHAI, shutdown_timeout_seconds=5.0, clock=lambda: NOW, observer=recorder
    )

    async def scenario() -> None:
        await worker.start()
        await asyncio.sleep(0.5)
        assert worker.running and worker.last_error is None
        with pytest.raises(RuntimeError):
            await worker.run_once()
        await worker.stop()

    asyncio.run(scenario())
    (event,) = [item for item in recorder.events if item.operation == "night"]  # 补跑一趟之后在等，不再跑
    assert worker.last_outcome is not None and worker.last_outcome.scene is not None
    assert event.status is ObservationStatus.SUCCESS and event.attributes["night"] == str(worker.last_outcome.night)
    assert event.attributes["scene"] is True and event.attributes["records"] == len(worker.last_outcome.series.records)

    class Broken:
        async def run_once(self):  # type: ignore[no-untyped-def]
            raise OSError("disk")

    failing = NightlyWorker(Broken(), at=FOUR, zone=SHANGHAI, shutdown_timeout_seconds=5.0, observer=recorder)

    async def broken() -> None:
        await failing.start()
        await asyncio.sleep(0.05)
        assert failing.running and isinstance(failing.last_error, OSError)
        await failing.stop()

    asyncio.run(broken())
    assert recorder.events[-1].status is ObservationStatus.FAILURE
    with pytest.raises(TypeError):
        NightlyWorker(object(), at=FOUR, zone=SHANGHAI, shutdown_timeout_seconds=1.0)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        NightlyWorker(run, at="04:00", zone=SHANGHAI, shutdown_timeout_seconds=1.0)  # type: ignore[arg-type]


def test_a_restart_only_catches_up_a_night_it_actually_missed(tmp_path) -> None:
    """上一趟开跑的时刻落盘：重启时上一趟在最近一个 04:00 之后就不跑（不在额度还没恢复的白天把唯一一次重试用掉），
    早于它才补跑；读不出就当没跑过。"""

    path = tmp_path / "nightly" / "last_run.json"
    now = datetime(2026, 10, 21, 15, 10, tzinfo=SHANGHAI)

    def worker(last: LastRun) -> NightlyWorker:
        return NightlyWorker(
            object.__new__(_Stub),
            at=FOUR,
            zone=SHANGHAI,
            shutdown_timeout_seconds=1.0,
            last_run=last,
            clock=lambda: now,
        )

    assert worker(LastRun(path, root=tmp_path)).missed()  # 从没跑过
    LastRun(path, root=tmp_path).write(datetime(2026, 10, 21, 4, 0, 5, tzinfo=SHANGHAI))
    assert not worker(LastRun(path, root=tmp_path)).missed()  # 今早 04:00 跑过了：15:10 重启不跑
    LastRun(path, root=tmp_path).write(datetime(2026, 10, 20, 4, 0, 5, tzinfo=SHANGHAI))
    assert worker(LastRun(path, root=tmp_path)).missed()  # 今早那一趟错过了：补跑
    path.write_text("{broken")
    assert worker(LastRun(path, root=tmp_path)).missed()


def test_the_due_times_are_the_subjects_local_clock() -> None:
    from habitus.runtime.nightly_worker import latest_due, next_due

    before = datetime(2026, 10, 21, 3, 59, tzinfo=SHANGHAI)
    after = datetime(2026, 10, 21, 4, 0, tzinfo=SHANGHAI)
    assert latest_due(before, at=FOUR, zone=SHANGHAI) == datetime(2026, 10, 20, 4, 0, tzinfo=SHANGHAI)
    assert latest_due(after, at=FOUR, zone=SHANGHAI) == after
    assert next_due(before, at=FOUR, zone=SHANGHAI) == after
    assert next_due(after, at=FOUR, zone=SHANGHAI) == datetime(2026, 10, 22, 4, 0, tzinfo=SHANGHAI)
    # 进程用 UTC 写法也一样：2026-10-20 20:30Z 是上海 10-21 04:30
    assert latest_due(datetime(2026, 10, 20, 20, 30, tzinfo=UTC), at=FOUR, zone=SHANGHAI) == after


class _Stub:
    async def run_once(self):  # type: ignore[no-untyped-def]
        raise AssertionError("not run")


def test_a_night_whose_settlement_or_retry_failed_leaves_an_error_on_the_worker(tmp_path) -> None:
    """整晚照跑完了、但结算或开头的重试抛了：常驻节拍在 ``last_error`` 上留痕（健康面读它），下一趟好了就清掉。"""

    from habitus.runtime.nightly import NightOutcome
    from habitus.series import EventSeries

    outcomes = iter(
        [
            NightOutcome(TONIGHT, EventSeries(cutoff=TONIGHT), None, None, settlement_failed="OSError"),
            NightOutcome(TONIGHT, EventSeries(cutoff=TONIGHT), None, None, retry_failed="ModelQuotaError"),
            NightOutcome(TONIGHT, EventSeries(cutoff=TONIGHT), None, None),
        ]
    )

    class Stub:
        async def run_once(self):  # type: ignore[no-untyped-def]
            return next(outcomes)

    recorder = Recorder()
    worker = NightlyWorker(Stub(), at=FOUR, zone=SHANGHAI, shutdown_timeout_seconds=5.0, observer=recorder)
    asyncio.run(worker._night())
    assert isinstance(worker.last_error, RuntimeError) and "OSError" in str(worker.last_error)
    asyncio.run(worker._night())
    assert "nightly retry failed: ModelQuotaError" in str(worker.last_error)
    asyncio.run(worker._night())
    assert worker.last_error is None
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


def test_the_retry_runs_before_the_settlement_and_its_failure_does_not_stop_the_night(tmp_path) -> None:
    """裁定 67 第 2、4 条：白天读不懂的先重试、落树，结算才看得到；重试抛错这一晚照常往下，健康面看得见。"""

    from habitus.behavior.retry import RetryReport

    order: list[str] = []
    site, run = assembled(tmp_path)
    publish_tonight(site)

    class Retry:
        async def run_once(self, *, through):  # type: ignore[no-untyped-def]
            order.append(f"重试到 {through}")
            return RetryReport(taken=1, recovered=1)

    class Settlement:
        def run_once(self):  # type: ignore[no-untyped-def]
            order.append("结算")

    run.retry = Retry()  # type: ignore[assignment]
    run.settlement = Settlement()  # type: ignore[assignment]
    outcome = asyncio.run(run.run_once())
    assert order == [f"重试到 {run.tonight() - timedelta(days=1)}", "结算"]  # 处理前一天及更早的（裁定 68 第 3 条）
    assert outcome.retried == RetryReport(taken=1, recovered=1) and outcome.retry_failed is None

    class Broken:
        async def run_once(self, *, through):  # type: ignore[no-untyped-def]
            raise OSError("disk")

    run.retry = Broken()  # type: ignore[assignment]
    outcome = asyncio.run(run.run_once())
    assert outcome.retry_failed == "OSError" and outcome.scene is not None


def test_the_worker_naps_in_short_slices_and_survives_a_marker_it_cannot_write(tmp_path) -> None:
    """裁定 68 第 4 条：等下一个 04:00 时一次最多睡 5 分钟、醒来重看墙钟（Mac 合盖时单调时钟不走）；``last_run`` 写不进
    照跑、健康面看得见；记下的时刻晚于此刻（时钟回拨过）当没跑过；一起来就把上一晚中途被杀剩下的重试项清掉。"""

    from habitus.runtime.nightly import NightOutcome
    from habitus.runtime.nightly_worker import MAX_NAP_SECONDS
    from habitus.series import EventSeries

    naps: list[float] = []
    cleaned: list[str] = []

    class Stub:
        def clean_leftovers(self) -> int:
            cleaned.append("清")
            return 0

        async def run_once(self):  # type: ignore[no-untyped-def]
            return NightOutcome(TONIGHT, EventSeries(cutoff=TONIGHT), None, None)

    class Unwritable(LastRun):
        def write(self, started_at):  # type: ignore[no-untyped-def]
            raise OSError("read-only")

    now = datetime(2026, 10, 21, 15, 10, tzinfo=SHANGHAI)
    worker = NightlyWorker(
        Stub(), at=FOUR, zone=SHANGHAI, shutdown_timeout_seconds=1.0, last_run=Unwritable(), clock=lambda: now
    )

    async def nap(seconds: float) -> None:
        naps.append(seconds)
        worker._stop_requested.set()

    worker._wait = nap  # type: ignore[method-assign]
    asyncio.run(worker._run_loop())
    assert cleaned == ["清"] and naps == [MAX_NAP_SECONDS]
    assert "last-run marker not written" in str(worker.last_error)

    future = LastRun()
    future.write(now + timedelta(days=2))
    assert NightlyWorker(
        Stub(), at=FOUR, zone=SHANGHAI, shutdown_timeout_seconds=1.0, last_run=future, clock=lambda: now
    ).missed()


def test_a_night_whose_whole_retry_batch_was_dropped_for_an_unavailable_model_is_degraded() -> None:
    from habitus.behavior.retry import RetryReport
    from habitus.runtime.nightly import NightOutcome
    from habitus.runtime.nightly_worker import _problems
    from habitus.series import EventSeries

    both = NightOutcome(
        TONIGHT,
        EventSeries(cutoff=TONIGHT),
        None,
        None,
        settlement_failed="OSError",
        retried=RetryReport(taken=3, unavailable=3),
    )
    assert _problems(both) == [
        "nightly settlement failed: OSError",
        "nightly retry dropped 3 items: model unavailable",
    ]
    fine = NightOutcome(
        TONIGHT, EventSeries(cutoff=TONIGHT), None, None, retried=RetryReport(taken=2, recovered=1, unavailable=1)
    )
    assert _problems(fine) == []
