"""第 N 晚在组合根的串法（第 8 步）：行为树读一次，语义树与机会表用同一份序列；夜线的 Job 执行器（裁定 59、94）。

各段经 ``Nightly.run_step`` 跑（``line`` 按先后串起来，等于执行器在同一晚里的一趟）；执行器的节拍、租约与 Job 账见
``test_nightly_jobs.py``。
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from habitus.foundation.observability import ObservationEvent
from habitus.runtime.nightly import (
    DONE,
    HELD,
    NO_RECORDS,
    NOTHING_CLOSED,
    PREPARE,
    STEPS,
    STOPPED,
    STRUCTURE,
    Nightly,
    StepResult,
)
from habitus.runtime.nightly_jobs import NightJobs
from habitus.runtime.nightly_report import NightReports
from habitus.runtime.nightly_settings import NightlySettings
from habitus.runtime.nightly_trace import TraceFiles
from habitus.runtime.nightly_worker import NightlyWorker
from habitus.scene.relations.store import RelationStore
from tests.lock_store import ProcessLocalLockStore
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


async def line_async(run: Nightly, *, night=None, steps=STEPS) -> dict[str, StepResult]:  # type: ignore[no-untyped-def]
    """同一晚按先后跑各段，碰到没成的就停（执行器一趟的样子，不落 Job 账）。结构那一段的"没东西可算"交给日子预期那一段。"""

    tonight = night if night is not None else run.tonight()
    results: dict[str, StepResult] = {}
    skipped: str | None = None
    for step in steps:
        result = await run.run_step(step, tonight, structure_skipped=skipped)
        results[step] = result
        if step == STRUCTURE:
            raw = result.report.get("skipped")
            skipped = raw if isinstance(raw, str) else None
        if not result.done:
            break
    return results


def line(run: Nightly, **options) -> dict[str, StepResult]:  # type: ignore[no-untyped-def]
    return asyncio.run(line_async(run, **options))


def worker_for(  # type: ignore[no-untyped-def]
    run,
    tmp_path,
    *,
    clock=None,
    lock_store=None,
    settings=None,
    observer=None,
    worker_id=None,
    lease_key="scene:nightly:test",
):
    """一个挂在 ``tmp_path`` 上的执行器：Job 账、报告、trace 都在 ``tmp_path/nightly`` 下，语义树的关系表在 ``tmp_path/scene``。"""

    root = tmp_path / "nightly"
    return NightlyWorker(
        run,
        jobs=NightJobs(root / "jobs", artifact_root=tmp_path, kept_nights=366),
        lock_store=lock_store if lock_store is not None else ProcessLocalLockStore(),
        lease_key=lease_key,
        shutdown_timeout_seconds=1.0,
        settings=settings or NightlySettings(retry_base_seconds=60.0, retry_max_seconds=600.0),
        reports=NightReports(root / "reports", artifact_root=tmp_path, relations=RelationStore(tmp_path / "scene")),
        traces=TraceFiles(root / "traces", artifact_root=tmp_path),
        observer=observer,
        clock=clock or (lambda: NOW),
        worker_id=worker_id,
    )


class StubRetry:
    """重试那一段的替身：按顺序交出 ``reports``（最后一份之后重复它）；``order`` 记每次调到的 ``through``。"""

    branches = ()

    def __init__(self, *reports, order=None, label="重试") -> None:  # type: ignore[no-untyped-def]
        from habitus.behavior.retry import RetryReport

        self.reports = list(reports) or [RetryReport()]
        self.order = order if order is not None else []
        self.label = label
        self.restored: list[object] = []
        self.cleaned = 0

    def restore(self, checkpoint):  # type: ignore[no-untyped-def]
        self.restored.append(checkpoint)

    def checkpoint(self):  # type: ignore[no-untyped-def]
        return None

    def clean_leftovers(self) -> int:
        self.cleaned += 1
        return 0

    async def run_once(self, *, through):  # type: ignore[no-untyped-def]
        report = self.reports[0] if len(self.reports) == 1 else self.reports.pop(0)
        if isinstance(report, BaseException):
            raise report
        self.order.append(self.label.format(through=through))
        return report


class StubSettlement:
    ledger = None

    def __init__(self, order=None, fails=None) -> None:  # type: ignore[no-untyped-def]
        self.order = order if order is not None else []
        self.fails = fails

    def run_once(self):  # type: ignore[no-untyped-def]
        from habitus.runtime.foresight_settlement import SettlementReport

        self.order.append("结算")
        if self.fails is not None and self.fails():
            raise OSError("ledger")
        return SettlementReport(0, {}, 0, 0, ())


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
    results = line(run)
    tonight = NOW.astimezone(ZoneInfo("Asia/Shanghai")).date()
    assert reads == [tonight]  # 行为树只读这一次
    assert threads == [False]  # 全量扫行为树下沉到线程，不卡事件循环（2026-10-18 四方评审）
    structure = results[STRUCTURE]
    assert structure.outcome == DONE and "skipped" not in structure.report
    scene = structure.report["scene"]
    assert isinstance(scene, dict) and scene["night"] == structure.report["night"] == tonight.isoformat()
    assert TONIGHT.isoformat() in scene["mapped_days"]


def test_nothing_runs_on_an_empty_behaviour_tree(tmp_path) -> None:
    """序列里一条记录都没有（上游刚接入）：语义树这一晚不跑，说出原因（清单 5.2：闸原来是"预测树发布了没有"）；这一晚照样算跑完。"""

    _site, run = assembled(tmp_path)
    results = line(run)
    structure = results[STRUCTURE]
    assert structure.outcome == DONE and structure.report["skipped"] == NO_RECORDS and "scene" not in structure.report
    assert all(result.done for result in results.values()) and list(results) == list(STEPS)


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


def test_an_unknown_step_is_refused(tmp_path) -> None:
    _site, run = assembled(tmp_path)
    with pytest.raises(ValueError, match="unknown nightly step"):
        asyncio.run(run.run_step("weather", TONIGHT))


def test_the_night_stops_at_the_last_sealed_day(tmp_path) -> None:
    """昨天的链还没归约完（归约说只封口到前天）：截止日退到昨天，昨天这一晚不算（E1）——不把半天当成完整的一天映射、盖章。"""

    site, run = assembled(tmp_path, closed_days=sealed_through(TONIGHT - timedelta(days=1)))
    publish_tonight(site)
    structure = line(run)[STRUCTURE]
    assert structure.report["night"] == TONIGHT.isoformat()
    scene = structure.report.get("scene")
    assert scene is None or TONIGHT.isoformat() not in scene["mapped_days"]  # type: ignore[index]


def test_nothing_runs_before_any_day_is_sealed(tmp_path) -> None:
    site, run = assembled(tmp_path, closed_days=tuple)
    publish_tonight(site)
    structure = line(run)[STRUCTURE]
    assert (
        structure.outcome == DONE and structure.report["skipped"] == NOTHING_CLOSED and "scene" not in structure.report
    )


def test_nothing_runs_while_the_vocabulary_is_migrating(tmp_path) -> None:
    """词表迁移做到一半（树上的行改了一部分、新版本还没写）：这一晚不跑语义树，结构这一段还没准备好（E3）。"""

    site, run = assembled(tmp_path, vocabulary=lambda: (2, True))
    publish_tonight(site)
    structure = line(run)[STRUCTURE]
    assert structure.outcome == HELD and structure.reason == "词表迁移还没做完" and "scene" not in structure.report


def test_nothing_runs_when_the_vocabulary_changes_while_reading(tmp_path) -> None:
    states = iter([(1, False), (2, False)])
    site, run = assembled(tmp_path, vocabulary=lambda: next(states))
    publish_tonight(site)
    structure = line(run)[STRUCTURE]
    assert structure.outcome == HELD and structure.reason == "读序列时词表变了"


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
    line(run)
    assert order == ["日历", "语义树"]


def test_a_calendar_refresh_that_raises_does_not_stop_the_night(tmp_path) -> None:
    """重读放假数据抛错：照用上一次读到的日历，重试、结算、语义树照跑；抛了什么记在前置处理那一段的报告上，执行器据它在
    ``last_error`` 上留痕（健康面看得见）。"""

    order: list[str] = []
    site, run = assembled(tmp_path)
    publish_tonight(site)

    def broken() -> None:
        raise TypeError("unhashable type: 'list'")

    run.refresh_calendar = broken
    run.retry = StubRetry(order=order)  # type: ignore[assignment]
    run.settlement = StubSettlement(order)  # type: ignore[assignment]
    for _ in range(2):
        order.clear()
        results = line(run)
        assert order == ["重试", "结算"] and "scene" in results[STRUCTURE].report
        assert results[PREPARE].outcome == DONE and results[PREPARE].report["calendar_failed"] == "TypeError"
    worker = worker_for(run, tmp_path)
    asyncio.run(worker.look())
    assert "nightly calendar refresh failed: TypeError" in str(worker.last_error)


def test_the_retry_runs_before_the_settlement_and_a_retry_that_raises_stops_the_night(tmp_path) -> None:
    """裁定 67 第 2 条：白天读不懂的先重试、落树，结算才看得到。重试整个抛了是 bug（一项的 bug 由重试自己逐项拦）：这一段停下、
    健康面报出，不接着跑结算与结构（裁定 94 第 8 条）。"""

    from habitus.behavior.retry import RetryReport

    order: list[str] = []
    site, run = assembled(tmp_path)
    publish_tonight(site)

    run.retry = StubRetry(RetryReport(taken=1, recovered=1), order=order, label="重试到 {through}")  # type: ignore[assignment]
    run.settlement = StubSettlement(order)  # type: ignore[assignment]
    results = line(run)
    assert order == [f"重试到 {run.tonight() - timedelta(days=1)}", "结算"]  # 处理前一天及更早的（裁定 68 第 3 条）
    retry = results[PREPARE].report["retry"]
    assert isinstance(retry, dict) and retry["taken"] == 1 and retry["recovered"] == 1

    run.retry = StubRetry(OSError("disk"), order=order)  # type: ignore[assignment]
    order.clear()
    results = line(run)
    assert results[PREPARE].outcome == STOPPED and results[PREPARE].reason == "前置处理出错: OSError"
    assert list(results) == [PREPARE] and order == []
