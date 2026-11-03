"""第 N 晚在组合根的串法（第 8 步）：行为树读一次，语义树与机会表用同一份序列；整晚那一趟自己的常驻节拍（裁定 59）。"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from habitus.foundation.observability import ObservationEvent, ObservationStatus
from habitus.runtime.nightly import STEPS, Nightly
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


def test_the_line_is_ready_once_yesterday_is_closed(tmp_path) -> None:
    """裁定 94 第 1 条：夜里那条线前一天定稿完成就接着跑，不等钟点——昨天在归约的定稿日里，今天这一晚就能跑；还没定稿就等。"""

    closed: list[object] = []
    _site, run = assembled(tmp_path, closed_days=lambda: tuple(closed))
    today = run.tonight()
    assert run.ready() is None  # 一天都还没定稿
    closed.append(today - timedelta(days=2))
    assert run.ready() is None  # 前天定稿了、昨天还没有：零点到定稿之间不跑
    closed.append(today - timedelta(days=1))
    assert run.ready() == today


def test_the_nightly_worker_runs_once_yesterday_closes_then_waits_and_survives_a_failing_one(tmp_path) -> None:
    """整晚那一趟自己的常驻节拍：昨天定稿了、今天这一晚还没跑过就跑一趟、记下读数，之后不再跑；
    一趟抛了只记 last_error、不记跑过，循环不死。"""

    site, run = assembled(tmp_path)
    publish_tonight(site)
    recorder = Recorder()
    worker = NightlyWorker(run, shutdown_timeout_seconds=5.0, observer=recorder)

    async def scenario() -> None:
        await worker.start()
        await asyncio.sleep(0.5)
        assert worker.running and worker.last_error is None
        with pytest.raises(RuntimeError):
            await worker.run_once()
        await worker.stop()

    asyncio.run(scenario())
    (event,) = [item for item in recorder.events if item.operation == "night"]  # 跑过一趟之后在等，不再跑
    assert worker.last_outcome is not None and worker.last_outcome.scene is not None
    assert event.status is ObservationStatus.SUCCESS and event.attributes["night"] == str(worker.last_outcome.night)
    assert event.attributes["scene"] is True and event.attributes["records"] == len(worker.last_outcome.series.records)
    assert worker.last_run.read() == run.tonight()

    class Broken:
        def ready(self):  # type: ignore[no-untyped-def]
            return TONIGHT

        async def run_once(self, **_progress):  # type: ignore[no-untyped-def]
            raise OSError("disk")

    failing = NightlyWorker(Broken(), shutdown_timeout_seconds=5.0, observer=recorder)

    async def broken() -> None:
        await failing.start()
        await asyncio.sleep(0.05)
        assert failing.running and isinstance(failing.last_error, OSError)
        await failing.stop()

    asyncio.run(broken())
    assert recorder.events[-1].status is ObservationStatus.FAILURE
    assert failing.last_run.read() is None  # 抛了不算跑过：下一眼从断点接着跑（裁定 94 第 1 条）
    with pytest.raises(TypeError):
        NightlyWorker(object(), shutdown_timeout_seconds=1.0)  # type: ignore[arg-type]


def test_a_restart_runs_the_night_only_if_it_has_not_run_for_it(tmp_path) -> None:
    """跑过哪一晚落盘：重启时昨天已定稿、这一晚还没跑过就接着跑（服务夜里没跑的，起来从断点接着跑），跑过了就不跑；
    昨天还没定稿就等；读不出当没跑过；记下的那一晚晚于今天（时钟回拨过）也再跑一趟。"""

    path = tmp_path / "nightly" / "last_run.json"
    ready: list[object] = [TONIGHT]

    class Line:
        def ready(self):  # type: ignore[no-untyped-def]
            return ready[0]

        async def run_once(self):  # type: ignore[no-untyped-def]
            raise AssertionError("not run")

    def worker(last: LastRun) -> NightlyWorker:
        return NightlyWorker(Line(), shutdown_timeout_seconds=1.0, last_run=last)

    assert worker(LastRun(path, root=tmp_path)).due() == TONIGHT  # 从没跑过
    LastRun(path, root=tmp_path).write(TONIGHT, finished=True)
    assert worker(LastRun(path, root=tmp_path)).due() is None  # 这一晚跑过了：重启不跑
    LastRun(path, root=tmp_path).write(TONIGHT - timedelta(days=1), finished=True)
    assert worker(LastRun(path, root=tmp_path)).due() == TONIGHT  # 上一次是前一晚：接着跑
    LastRun(path, root=tmp_path).write(TONIGHT + timedelta(days=2), finished=True)
    assert worker(LastRun(path, root=tmp_path)).due() == TONIGHT
    path.write_text("{broken")
    assert worker(LastRun(path, root=tmp_path)).due() == TONIGHT
    ready[0] = None
    assert worker(LastRun(path, root=tmp_path)).due() is None  # 昨天还没定稿


def test_a_night_cut_off_midway_is_run_again_on_the_next_look(tmp_path) -> None:
    """被杀（取消）的那一趟不记这一晚：下一眼昨天照样定稿、这一晚还没记下，从头再走一遍——每一步重跑都幂等。"""

    from habitus.runtime.nightly import NightOutcome
    from habitus.series import EventSeries

    runs: list[str] = []

    class Line:
        def ready(self):  # type: ignore[no-untyped-def]
            return TONIGHT

        async def run_once(self, **_progress):  # type: ignore[no-untyped-def]
            runs.append("跑")
            if len(runs) == 1:
                raise asyncio.CancelledError
            return NightOutcome(TONIGHT, EventSeries(cutoff=TONIGHT), None, None, done=STEPS)

    worker = NightlyWorker(Line(), shutdown_timeout_seconds=1.0)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(worker._night(TONIGHT))
    assert worker.due() == TONIGHT
    asyncio.run(worker._night(TONIGHT))
    assert worker.due() is None and runs == ["跑", "跑"]


def test_a_stopped_night_or_kept_errors_leave_an_error_on_the_worker(tmp_path) -> None:
    """这一晚停下了（结算出错），或前置处理里有保持「错误·看不到」的：常驻节拍在 ``last_error`` 上留痕（健康面读它），
    下一趟好了就清掉。"""

    from habitus.behavior.retry import RetryReport
    from habitus.runtime.nightly import NightOutcome
    from habitus.series import EventSeries

    outcomes = iter(
        [
            NightOutcome(TONIGHT, EventSeries(cutoff=TONIGHT), None, None, stopped="结算出错: OSError"),
            NightOutcome(
                TONIGHT, EventSeries(cutoff=TONIGHT), None, None, retried=RetryReport(deterministic=1), done=STEPS
            ),
            NightOutcome(TONIGHT, EventSeries(cutoff=TONIGHT), None, None, done=STEPS),
        ]
    )

    class Stub:
        def ready(self):  # type: ignore[no-untyped-def]
            return TONIGHT

        async def run_once(self, **_progress):  # type: ignore[no-untyped-def]
            return next(outcomes)

    recorder = Recorder()
    worker = NightlyWorker(Stub(), shutdown_timeout_seconds=5.0, observer=recorder)
    asyncio.run(worker._night(TONIGHT))
    assert isinstance(worker.last_error, RuntimeError) and "OSError" in str(worker.last_error)
    asyncio.run(worker._night(TONIGHT))
    assert "nightly errors kept unseen: 处理不了 1" in str(worker.last_error)
    asyncio.run(worker._night(TONIGHT))
    assert worker.last_error is None
    first = next(item for item in recorder.events if item.operation == "night")
    assert first.attributes["stopped"] == "结算出错: OSError"


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


def test_a_calendar_refresh_that_raises_does_not_stop_the_night(tmp_path) -> None:
    """重读放假数据抛错：照用上一次读到的日历，重试、结算、语义树照跑，抛了什么记在这一晚的结果上、健康面看得见。"""

    from habitus.behavior.retry import RetryReport
    from habitus.runtime.nightly_worker import _problems

    order: list[str] = []
    site, run = assembled(tmp_path)
    publish_tonight(site)

    def broken() -> None:
        raise TypeError("unhashable type: 'list'")

    class Retry:
        async def run_once(self, *, through):  # type: ignore[no-untyped-def]
            order.append("重试")
            return RetryReport()

    class Settlement:
        def run_once(self):  # type: ignore[no-untyped-def]
            order.append("结算")

    run.refresh_calendar = broken
    run.retry = Retry()  # type: ignore[assignment]
    run.settlement = Settlement()  # type: ignore[assignment]
    for _ in range(2):
        order.clear()
        outcome = asyncio.run(run.run_once())
        assert order == ["重试", "结算"] and outcome.scene is not None
        assert outcome.calendar_failed == "TypeError"
        assert "nightly calendar refresh failed: TypeError" in _problems(outcome)


def test_the_retry_runs_before_the_settlement_and_a_retry_that_raises_stops_the_night(tmp_path) -> None:
    """裁定 67 第 2 条：白天读不懂的先重试、落树，结算才看得到。重试整个抛了是 bug（一项的 bug 由重试自己逐项拦）：这一晚
    停下、健康面报出，不接着跑结算与结构（裁定 94 第 8 条）。"""

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
    assert outcome.retried == RetryReport(taken=1, recovered=1) and outcome.stopped is None

    class Broken:
        async def run_once(self, *, through):  # type: ignore[no-untyped-def]
            raise OSError("disk")

    run.retry = Broken()  # type: ignore[assignment]
    order.clear()
    outcome = asyncio.run(run.run_once())
    assert outcome.stopped == "前置处理出错: OSError" and outcome.scene is None and order == []


def test_the_worker_looks_in_short_slices_and_survives_a_marker_it_cannot_write(tmp_path) -> None:
    """隔 5 分钟看一眼昨天定稿了没有、醒来重看墙钟（Mac 合盖时单调时钟不走，裁定 68 第 4 条）；``last_run`` 写不进照跑、
    健康面看得见；读不了定稿日照样活着；一起来就把上一晚中途被杀剩下的重试项清掉。"""

    from habitus.runtime.nightly import NightOutcome
    from habitus.runtime.nightly_worker import POLL_SECONDS
    from habitus.series import EventSeries

    naps: list[float] = []
    cleaned: list[str] = []

    class Stub:
        def clean_leftovers(self) -> int:
            cleaned.append("清")
            return 0

        def ready(self):  # type: ignore[no-untyped-def]
            return TONIGHT

        async def run_once(self, **_progress):  # type: ignore[no-untyped-def]
            return NightOutcome(TONIGHT, EventSeries(cutoff=TONIGHT), None, None, done=STEPS)

    class Unwritable(LastRun):
        def write(self, night):  # type: ignore[no-untyped-def]
            raise OSError("read-only")

    worker = NightlyWorker(Stub(), shutdown_timeout_seconds=1.0, last_run=Unwritable())

    async def nap(seconds: float) -> None:
        naps.append(seconds)
        worker._stop_requested.set()

    worker._wait = nap  # type: ignore[method-assign]
    asyncio.run(worker._run_loop())
    assert cleaned == ["清"] and naps == [POLL_SECONDS]
    assert "last-run marker not written" in str(worker.last_error)

    class Unreadable(Stub):
        def ready(self):  # type: ignore[no-untyped-def]
            raise OSError("day_closure.json")

    blind = NightlyWorker(Unreadable(), shutdown_timeout_seconds=1.0)

    async def stop(_seconds: float) -> None:
        blind._stop_requested.set()

    blind._wait = stop  # type: ignore[method-assign]
    asyncio.run(blind._run_loop())
    assert isinstance(blind.last_error, OSError) and blind.last_outcome is None


def test_a_stopped_night_still_lists_the_errors_kept_unseen() -> None:
    """停下的那一晚也列前置处理里保持「错误·看不到」的（同一晚两条都要看得见）；什么都没有就是空。"""

    from habitus.behavior.retry import RetryReport
    from habitus.runtime.nightly import NightOutcome
    from habitus.runtime.nightly_worker import _problems
    from habitus.series import EventSeries

    both = NightOutcome(
        TONIGHT,
        EventSeries(cutoff=TONIGHT),
        None,
        None,
        retried=RetryReport(taken=3, invalid=1, unavailable=1),
        stopped="模型调不通: quota",
    )
    assert _problems(both) == [
        "nightly stopped: 模型调不通: quota",
        "nightly errors kept unseen: 处理不了 0，内容安全 0，格式不合 1",
    ]
    fine = NightOutcome(
        TONIGHT, EventSeries(cutoff=TONIGHT), None, None, retried=RetryReport(taken=2, recovered=2), done=STEPS
    )
    assert _problems(fine) == []
