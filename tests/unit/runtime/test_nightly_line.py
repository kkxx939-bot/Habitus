"""裁定 94 第 1、4、8 条：夜里一条线的先后、断点、停下。

- 线的先后钉死：探模型 → 重试 → 结算 → 词表（每晚新增、定期拆改）→ 语义树 → 机会表（单元同一拍）→ 当天的日子预期，挪动任何一步
  这里都红；词表那一段拿到的是这一晚（定期拆改的周期按它算，不挂钟点，裁定 94 第 9 条）；
- 只有整条线跑完才算跑过：某一段还没准备好（词表迁移中、锁被占），这一晚不记跑过，下一眼带着记下的进度从断点接着跑，
  之前成了的段不再跑；
- 一段抛了异常是 bug：这一晚停下、健康面报出、记成跑过，下一晚照常——不每隔几分钟整段重跑；
- 模型整个调不通（线上任何一处）：这一晚停下、不拿旧数据顶（不结算、不跑语义树、不建日子预期），记成跑过（下一晚照常），
  健康面写"模型调不通"。
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta

from habitus.behavior.reduction import BehaviorReductionBusyError
from habitus.behavior.retry import RetryReport
from habitus.model_client import ModelQuotaError, ModelUnreachable
from habitus.runtime.foresight_days import DayReport
from habitus.runtime.foresight_forming import Formed
from habitus.runtime.foresight_nightly import OpportunityReport
from habitus.runtime.nightly import LOCK_BUSY, MODEL_UNREACHABLE, STEPS
from habitus.runtime.nightly_worker import LastRun, NightlyWorker
from tests.unit.runtime.test_nightly import assembled
from tests.unit.runtime.test_scene_night import publish_tonight


def wired(tmp_path, order: list[str], *, migrating=None, settle_fails=None, retried=None):  # type: ignore[no-untyped-def]
    """真实语义树，其余各段换成只记先后的替身（挂在构造之后，绕开类型检查）。"""

    state = {"migrating": migrating or (lambda: False)}
    nights: list[object] = []
    site, run = assembled(tmp_path, vocabulary=lambda: (1, state["migrating"]()))
    publish_tonight(site)

    async def probe() -> None:
        order.append("探模型")

    class Retry:
        async def run_once(self, *, through):  # type: ignore[no-untyped-def]
            order.append("重试")
            return retried if retried is not None else RetryReport()

    class Settlement:
        def run_once(self):  # type: ignore[no-untyped-def]
            order.append("结算")
            if settle_fails is not None and settle_fails():
                raise OSError("ledger")

    class Foresight:
        def run(self, *, night, **_materials):  # type: ignore[no-untyped-def]
            order.append("机会表")
            return OpportunityReport(night)

    class Days:
        async def run_once(self):  # type: ignore[no-untyped-def]
            order.append("日子")
            return DayReport(datetime.now().astimezone(), Formed())

    async def vocabulary(night) -> None:  # type: ignore[no-untyped-def]
        order.append("词表")
        nights.append(night)

    scene_run = run.scene.run

    async def scene_then(**kwargs):  # type: ignore[no-untyped-def]
        order.append("语义树")
        return await scene_run(**kwargs)

    run.scene.run = scene_then  # type: ignore[method-assign]
    run.probe = probe
    run.retry = Retry()  # type: ignore[assignment]
    run.settlement = Settlement()  # type: ignore[assignment]
    run.foresight = Foresight()  # type: ignore[assignment]
    run.days = Days()  # type: ignore[assignment]
    run.vocabulary_jobs = (vocabulary,)
    run.vocabulary_nights = nights  # type: ignore[attr-defined]
    return run


def test_the_line_runs_its_steps_in_the_ruled_order(tmp_path) -> None:
    order: list[str] = []
    run = wired(tmp_path, order)
    steps: list[str] = []
    outcome = asyncio.run(run.run_once(on_step=steps.append))
    assert order == ["探模型", "重试", "结算", "词表", "语义树", "机会表", "日子"]
    assert outcome.complete and steps == list(STEPS)
    assert run.vocabulary_nights == [run.tonight()]  # type: ignore[attr-defined]


def test_a_held_step_is_not_a_finished_night_and_the_next_look_resumes_from_it(tmp_path) -> None:
    """词表迁移中：结构那一段停下，这一晚没跑完；下一眼带着进度来，前置处理、结算与每晚新增不再跑，从结构接着跑完。"""

    order: list[str] = []
    migrating = [True]
    run = wired(tmp_path, order, migrating=lambda: migrating[0])
    steps: list[str] = []
    first = asyncio.run(run.run_once(on_step=steps.append))
    assert order == ["探模型", "重试", "结算", "词表"] and first.skipped == "词表迁移还没做完"
    assert not first.complete and first.stopped is None
    assert first.done == ("prepare", "settle", "vocabulary") and steps == ["prepare", "settle", "vocabulary"]
    order.clear()
    migrating[0] = False
    second = asyncio.run(run.run_once(resume=steps, on_step=steps.append))
    assert order == ["语义树", "机会表", "日子"]
    assert second.complete and steps == list(STEPS)


def test_a_step_that_raises_stops_the_night_instead_of_looping(tmp_path) -> None:
    """裁定 94 第 8 条：结算抛了是 bug——这一晚停下（后面几段不跑、不拿旧数据顶）、记成跑过、健康面报出；同一晚不再整段重跑，
    下一晚照常从头跑。"""

    from habitus.runtime.health import _night_line_checks

    order: list[str] = []
    fails = [True]
    run = wired(tmp_path, order, settle_fails=lambda: fails[0])
    last = LastRun(tmp_path / "nightly" / "last_run.json", root=tmp_path)
    worker = NightlyWorker(run, shutdown_timeout_seconds=1.0, last_run=last)
    night = run.tonight()
    asyncio.run(worker._night(night))
    assert order == ["探模型", "重试", "结算"]  # 停在结算，不接着跑词表、语义树、机会表、日子预期
    outcome = worker.last_outcome
    assert outcome is not None and outcome.stopped == "结算出错: OSError" and not outcome.complete
    assert last.read() == night and worker.due() is None  # 记成跑过：下一眼（5 分钟后）不再整段重跑
    assert "nightly stopped: 结算出错: OSError" in str(worker.last_error)
    (check,) = _night_line_checks(outcome)
    assert check.name == "nightly_line" and check.detail == "结算出错: OSError"

    order.clear()
    fails[0] = False
    asyncio.run(worker._night(night + timedelta(days=1)))  # 下一晚照常
    assert order == ["探模型", "重试", "结算", "词表", "语义树", "机会表", "日子"] and worker.last_error is None


def test_an_unreachable_model_stops_the_night_without_stale_steps(tmp_path) -> None:
    order: list[str] = []
    run = wired(tmp_path, order)

    async def down() -> None:
        order.append("探模型")
        raise ModelUnreachable("MODEL_TRANSPORT")

    run.probe = down
    outcome = asyncio.run(run.run_once())
    assert order == ["探模型"]  # 不重试、不结算、不跑语义树、不建日子预期
    assert outcome.stopped == f"{MODEL_UNREACHABLE}: MODEL_TRANSPORT" and not outcome.complete


def test_the_retry_meeting_an_unusable_model_stops_the_night_and_keeps_its_counts(tmp_path) -> None:
    """重试里第一条会话轮就是模型用不了（``RetryReport.unavailable``）：停下，计数留在这一晚的结果上。"""

    order: list[str] = []
    run = wired(tmp_path, order, retried=RetryReport(taken=2, unavailable=1))
    outcome = asyncio.run(run.run_once())
    assert order == ["探模型", "重试"] and outcome.stopped is not None
    assert outcome.stopped.startswith(MODEL_UNREACHABLE) and outcome.retried == RetryReport(taken=2, unavailable=1)


def test_any_model_touchpoint_on_the_line_that_finds_the_model_unusable_stops_the_night(tmp_path) -> None:
    """裁定 94 第 8 条：线上任何一处调模型碰到"整个用不了"都停线、报"模型调不通"——语义树的触点（``ModelUnreachable``）、
    每晚新增词表直接调模型客户端原样抛出的（额度满）。"""

    order: list[str] = []
    run = wired(tmp_path, order)

    async def quota(_night) -> None:  # type: ignore[no-untyped-def]
        order.append("词表")
        raise ModelQuotaError("quota exhausted")

    run.vocabulary_jobs = (quota,)
    outcome = asyncio.run(run.run_once())
    assert order == ["探模型", "重试", "结算", "词表"]
    assert outcome.stopped is not None and outcome.stopped.startswith(MODEL_UNREACHABLE)

    order.clear()
    run = wired(tmp_path / "scene", order)

    async def touchpoint(**_kwargs):  # type: ignore[no-untyped-def]
        order.append("语义树")
        raise ModelUnreachable("model: ModelAuthenticationError: denied")

    run.scene.run = touchpoint  # type: ignore[method-assign]
    outcome = asyncio.run(run.run_once())
    assert order[-1] == "语义树" and "机会表" not in order
    assert outcome.stopped == f"{MODEL_UNREACHABLE}: model: ModelAuthenticationError: denied"


def test_a_busy_lock_on_the_vocabulary_is_not_ready_and_resumes_there(tmp_path) -> None:
    """每晚新增碰到 sweep 锁被另一进程拿着：还没准备好，不停线，下一眼从这一段接着跑。"""

    order: list[str] = []
    run = wired(tmp_path, order)
    busy = [True]

    async def vocabulary(_night) -> None:  # type: ignore[no-untyped-def]
        order.append("词表")
        if busy[0]:
            raise BehaviorReductionBusyError("sweep lock held")

    run.vocabulary_jobs = (vocabulary,)
    steps: list[str] = []
    first = asyncio.run(run.run_once(on_step=steps.append))
    assert first.skipped == LOCK_BUSY and first.stopped is None and first.done == ("prepare", "settle")
    order.clear()
    busy[0] = False
    second = asyncio.run(run.run_once(resume=steps, on_step=steps.append))
    assert order == ["词表", "语义树", "机会表", "日子"] and second.complete


def test_resuming_keeps_the_errors_the_preparation_left_unseen_on_the_health_surface(tmp_path) -> None:
    """裁定 94 第 8 条：从断点接着跑的那一趟不再做前置处理，健康面上那一晚"错误·看不到"的计数不丢。"""

    from habitus.runtime.health import _night_line_checks

    order: list[str] = []
    migrating = [True]
    kept = RetryReport(taken=3, deterministic=1, invalid=1, refused=1)
    run = wired(tmp_path, order, migrating=lambda: migrating[0], retried=kept)
    worker = NightlyWorker(run, shutdown_timeout_seconds=1.0, last_run=LastRun())
    night = run.tonight()
    asyncio.run(worker._night(night))
    migrating[0] = False
    asyncio.run(worker._night(night))
    outcome = worker.last_outcome
    assert outcome is not None and outcome.complete and outcome.retried == kept
    (check,) = _night_line_checks(outcome)
    assert check.name == "nightly_errors" and "格式不合 1" in check.detail
    assert "nightly errors kept unseen" in str(worker.last_error)


def test_the_worker_records_only_a_finished_or_stopped_night_and_keeps_the_progress(tmp_path) -> None:
    from habitus.runtime.health import _night_line_checks

    order: list[str] = []
    migrating = [True]
    run = wired(tmp_path, order, migrating=lambda: migrating[0])
    last = LastRun(tmp_path / "nightly" / "last_run.json", root=tmp_path)
    worker = NightlyWorker(run, shutdown_timeout_seconds=1.0, last_run=last)
    night = run.tonight()
    asyncio.run(worker._night(night))
    assert last.read() is None and worker.due() == night  # 没跑完：不记跑过，下一眼再来
    assert LastRun(last.path, root=tmp_path).steps(night) == {"prepare", "settle", "vocabulary"}  # 进度落了盘
    assert "nightly line not finished" in str(worker.last_error)
    order.clear()
    migrating[0] = False
    asyncio.run(worker._night(night))
    assert order == ["语义树", "机会表", "日子"]  # 从断点接着跑
    assert last.read() == night and worker.due() is None and worker.last_error is None

    async def down() -> None:
        raise ModelUnreachable("MODEL_AUTHENTICATION")

    run.probe = down
    tomorrow = night + timedelta(days=1)
    asyncio.run(worker._night(tomorrow))
    assert last.read() == tomorrow  # 停下的这一晚也算跑过：不在白天反复重探，下一晚照常
    assert "nightly stopped: 模型调不通" in str(worker.last_error)
    (check,) = _night_line_checks(worker.last_outcome)
    assert check.name == "nightly_line" and check.detail.startswith(MODEL_UNREACHABLE)


def test_the_last_run_file_reads_the_old_format_as_a_finished_night(tmp_path) -> None:
    path = tmp_path / "nightly" / "last_run.json"
    path.parent.mkdir(parents=True)
    path.write_text('{"night": "2026-08-18"}')
    last = LastRun(path, root=tmp_path)
    assert str(last.read()) == "2026-08-18" and last.steps(last.read()) == frozenset()  # type: ignore[arg-type]


def test_kept_errors_reach_the_health_surface() -> None:
    from habitus.runtime.health import _night_line_checks
    from habitus.runtime.nightly import NightOutcome
    from habitus.series import EventSeries
    from tests.unit.runtime.test_scene_night import TONIGHT

    outcome = NightOutcome(
        TONIGHT, EventSeries(cutoff=TONIGHT), None, retried=RetryReport(taken=3, deterministic=2, refused=1), done=STEPS
    )
    (check,) = _night_line_checks(outcome)
    assert check.name == "nightly_errors" and "处理不了 2" in check.detail and "内容安全 1" in check.detail

    # 停下的那一晚也列前置处理里保持「错误·看不到」的，格式不合夜里再败的也报（裁定 94 第 8 条）
    stopped = NightOutcome(
        TONIGHT,
        EventSeries(cutoff=TONIGHT),
        None,
        retried=RetryReport(taken=2, invalid=1, unavailable=1),
        stopped=f"{MODEL_UNREACHABLE}: quota",
    )
    line, kept = _night_line_checks(stopped)
    assert line.name == "nightly_line" and kept.name == "nightly_errors" and "格式不合 1" in kept.detail


def _branch(run_once):  # type: ignore[no-untyped-def]
    from types import SimpleNamespace

    from habitus.behavior.kinds.ids import Lane

    return SimpleNamespace(lane=Lane.SESSION, reduction_runner=SimpleNamespace(run_once=run_once))


def test_the_retry_reduction_reports_busy_unreachable_or_raises_a_bug() -> None:
    """裁定 94 第 9 条（review-94b 中 1）：重试段里那一次归约——锁被另一进程拿着是还没准备好（``BUSY``）；归类时模型整个用不了
    （归约报告 ``unavailable``，或触点抛的 ``ModelUnreachable`` / 模型客户端原样抛的"整个用不了"）是 ``UNREACHABLE``；别的异常
    是 bug，原样抛出（不再吞成一句写进没人读的 ``unreduced`` 的原因）。"""

    import pytest

    from habitus.behavior.reduction import BehaviorReductionReport
    from habitus.behavior.retry import Reduction
    from habitus.runtime.behavior import _reduce

    def reporting(unavailable: int):  # type: ignore[no-untyped-def]
        async def run_once() -> BehaviorReductionReport:
            return BehaviorReductionReport(0, 0, 0, 0, (), unavailable=unavailable)

        return run_once

    def raising(exc: BaseException):  # type: ignore[no-untyped-def]
        async def run_once() -> BehaviorReductionReport:
            raise exc

        return run_once

    assert asyncio.run(_reduce(_branch(reporting(0)))()) is Reduction.DONE
    assert asyncio.run(_reduce(_branch(reporting(2)))()) is Reduction.UNREACHABLE
    busy = raising(BehaviorReductionBusyError("sweep lock held"))
    assert asyncio.run(_reduce(_branch(busy))()) is Reduction.BUSY
    for down in (ModelUnreachable("down"), ModelQuotaError("quota")):
        assert asyncio.run(_reduce(_branch(raising(down)))()) is Reduction.UNREACHABLE
    with pytest.raises(KeyError):
        asyncio.run(_reduce(_branch(raising(KeyError("verdicts"))))())


def test_a_retry_still_waiting_on_a_lock_holds_the_line_before_settlement_and_resumes_there(tmp_path) -> None:
    """review-94b 中 1：前置处理里重试那一支还在等锁（``RetryReport.waiting``）——这一晚停在前置处理、不结算（写回的判断还没落树，
    结算了它们就会在白天落进已经结算的日子）；下一眼从前置处理接着跑，重试做完才结算。撤回 ``_prepare`` 里那道判断这里就红。"""

    order: list[str] = []
    reports = [RetryReport(taken=1, waiting=("session",)), RetryReport(taken=1, recovered=1)]

    class Retry:
        async def run_once(self, *, through):  # type: ignore[no-untyped-def]
            order.append("重试")
            return reports.pop(0)

    run = wired(tmp_path, order)
    run.retry = Retry()  # type: ignore[assignment]
    steps: list[str] = []
    first = asyncio.run(run.run_once(on_step=steps.append))
    assert order == ["探模型", "重试"] and first.skipped == LOCK_BUSY
    assert not first.complete and first.stopped is None and first.done == () and steps == []
    order.clear()
    second = asyncio.run(run.run_once(resume=steps, on_step=steps.append))
    assert order == ["探模型", "重试", "结算", "词表", "语义树", "机会表", "日子"] and second.complete
    assert second.retried == RetryReport(taken=1, recovered=1)


def test_the_worker_records_which_nights_the_whole_line_finished(tmp_path) -> None:
    """review-94b 中低 3 / 裁定 94 第 9 条："停掉的那天"读线自己记的这一晚跑到哪一步：日子预期那一段没跑完（停下了、没跑到、没开机）
    即算停。表写出后才停（日子预期那一段抛了）也认得出——原来只认"那一晚没有机会表"认不出这种。"""

    order: list[str] = []
    run = wired(tmp_path, order)
    last = LastRun(tmp_path / "nightly" / "last_run.json", root=tmp_path)
    worker = NightlyWorker(run, shutdown_timeout_seconds=1.0, last_run=last)
    night = run.tonight()
    asyncio.run(worker._night(night - timedelta(days=3)))  # 跑完
    days = run.days

    class Broken:
        async def run_once(self):  # type: ignore[no-untyped-def]
            order.append("日子")
            raise KeyError("ledger")

    run.days = Broken()  # type: ignore[assignment]
    asyncio.run(worker._night(night - timedelta(days=2)))  # 机会表写了、日子预期那一段抛了：停下
    assert worker.last_outcome is not None and worker.last_outcome.stopped == "日子预期出错: KeyError"
    run.days = days
    # night - 1：没开机，一趟都没跑
    asyncio.run(worker._night(night))  # 跑完

    reread = LastRun(last.path, root=tmp_path)
    assert [reread.finished(night - timedelta(days=offset)) for offset in range(4, -1, -1)] == [
        None,  # 第一次跑完之前：记录够不着
        True,
        False,  # 表写出后才停
        False,  # 没开机
        True,
    ]
    assert reread.read() == night


def test_the_last_run_file_keeps_its_record_of_finished_nights_bounded(tmp_path) -> None:
    from habitus.runtime.nightly_worker import KEPT_NIGHTS

    path = tmp_path / "nightly" / "last_run.json"
    last = LastRun(path, root=tmp_path)
    start = datetime(2025, 1, 1).date()
    last.write(start, finished=True)
    far = start + timedelta(days=2 * KEPT_NIGHTS)
    last.write(far, finished=True)  # 中间几百晚都没跑完
    reread = LastRun(path, root=tmp_path)
    assert reread.finished(far - timedelta(days=1)) is False and reread.finished(far) is True
    assert reread.finished(start) is None  # 留存之外：认不出
    assert len(path.read_bytes()) < 16_384


def test_the_opportunity_table_is_stale_once_its_night_is_behind_the_last_closed_day_plus_one(tmp_path) -> None:
    """裁定 68 第 4 条（健康面不写死周期）：机会表陈旧不再按"86400 秒的两倍"量，而是盘上最晚那一晚落后于"最后定稿日 + 1"（线在前一天
    定稿之后跑，裁定 94 第 1 条）。昨天已定稿、今天这一晚的表没出来就是陈旧——不必等过两天；撤回改动这里就红。"""

    from habitus.foresight.opportunities import OpportunityTable
    from habitus.runtime.foresight_opportunities import OpportunityStore
    from habitus.runtime.health import RuntimeHealthService, RuntimeHealthStatus

    check = RuntimeHealthService._opportunity_freshness_check
    store = OpportunityStore(tmp_path)
    today = datetime.now().astimezone().date()
    yesterday = today - timedelta(days=1)
    assert check(store, lambda: (yesterday,)).detail == "never_written"
    store.write(OpportunityTable("session", yesterday, yesterday, 15, 3, (), ()))
    stale = check(store, lambda: (yesterday - timedelta(days=1), yesterday))
    assert stale.status is RuntimeHealthStatus.DEGRADED and stale.detail == f"stale:night {yesterday} behind {today}"
    assert check(store, lambda: (yesterday - timedelta(days=1),)).status is RuntimeHealthStatus.HEALTHY
    assert check(store, lambda: ()).status is RuntimeHealthStatus.HEALTHY  # 还没有定稿日
    store.write(OpportunityTable("session", today, today, 15, 3, (), ()))
    fresh = check(store, lambda: (yesterday,))
    assert fresh.status is RuntimeHealthStatus.HEALTHY and fresh.detail == f"night:{today}"
