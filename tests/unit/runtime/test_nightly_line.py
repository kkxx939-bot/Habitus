"""裁定 94 第 1、4、8 条：夜里一条线的先后、断点、停下（每晚 5 段各一个 Job）。

- 线的先后钉死：探模型 → 重试 → 结算 → 词表（每晚新增、定期拆改）→ 语义树 → 机会表（单元同一拍）→ 当天的日子预期，挪动任何一步
  这里都红；词表那一段拿到的是这一晚（定期拆改的周期按它算，不挂钟点，裁定 94 第 9 条）；
- 只有 5 段都成了这一晚才算跑完：某一段还没准备好（词表迁移中、锁被占），Job 保持排队，下一眼从这一段接着跑，之前成了的段不再跑；
- 一段抛了异常是 bug：这一段停下、健康面报出，后面的段不跑；按退避到点再试，不每 5 分钟整段重跑；
- 模型整个调不通（线上任何一处）：这一段停下、不拿旧数据顶（不结算、不跑语义树、不建日子预期），健康面写"模型调不通"。
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
from habitus.runtime.nightly import (
    DAYS,
    HELD,
    LOCK_BUSY,
    MODEL_UNREACHABLE,
    PREPARE,
    SETTLE,
    STEPS,
    STOPPED,
    STRUCTURE,
    VOCABULARY,
)
from habitus.runtime.nightly_jobs import FAILED, QUEUED, SUCCEEDED, NightJob, NightJobs, NightLedger
from habitus.runtime.nightly_jobs import STOPPED as JOB_STOPPED
from tests.unit.runtime.test_nightly import StubRetry, StubSettlement, assembled, line, worker_for
from tests.unit.runtime.test_scene_night import NOW, publish_tonight


def wired(tmp_path, order: list[str], *, migrating=None, settle_fails=None, retried=None, clock=None):  # type: ignore[no-untyped-def]
    """真实语义树，其余各段换成只记先后的替身（挂在构造之后，绕开类型检查）。"""

    state = {"migrating": migrating or (lambda: False)}
    nights: list[object] = []
    site, run = assembled(tmp_path, vocabulary=lambda: (1, state["migrating"]()), clock=clock or (lambda: NOW))
    publish_tonight(site)

    async def probe() -> None:
        order.append("探模型")

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
    run.retry = StubRetry(*(() if retried is None else (retried,)), order=order)  # type: ignore[assignment]
    run.settlement = StubSettlement(order, fails=settle_fails)  # type: ignore[assignment]
    run.foresight = Foresight()  # type: ignore[assignment]
    run.days = Days()  # type: ignore[assignment]
    run.vocabulary_jobs = (vocabulary,)
    run.vocabulary_nights = nights  # type: ignore[attr-defined]
    return run


ALL = ["探模型", "重试", "结算", "词表", "语义树", "机会表", "日子"]


def test_the_line_runs_its_steps_in_the_ruled_order(tmp_path) -> None:
    order: list[str] = []
    run = wired(tmp_path, order)
    results = line(run)
    assert order == ALL
    assert list(results) == list(STEPS) and all(result.done for result in results.values())
    assert run.vocabulary_nights == [run.tonight()]  # type: ignore[attr-defined]


def test_a_held_step_is_not_a_finished_night_and_the_next_look_resumes_from_it(tmp_path) -> None:
    """词表迁移中：结构那一段还没准备好，Job 保持排队，这一晚没跑完；下一眼前置处理、结算与每晚新增不再跑，从结构接着跑完。"""

    order: list[str] = []
    migrating = [True]
    run = wired(tmp_path, order, migrating=lambda: migrating[0])
    worker = worker_for(run, tmp_path)
    first = asyncio.run(worker.look())
    assert order == ["探模型", "重试", "结算", "词表"]
    assert first is not None and not first.complete
    assert [first.job(step).state for step in STEPS] == [SUCCEEDED, SUCCEEDED, SUCCEEDED, QUEUED, QUEUED]
    held = first.job(STRUCTURE)
    assert held.reason == "词表迁移还没做完" and held.attempts == 0  # 没准备好不算一次尝试
    order.clear()
    migrating[0] = False
    second = asyncio.run(worker.look())
    assert order == ["语义树", "机会表", "日子"]
    assert second is not None and second.complete and worker.finished(run.tonight()) is True


def test_a_step_that_raises_stops_there_backs_off_and_retries_instead_of_looping(tmp_path) -> None:
    """裁定 94 第 8 条（Job 化之后）：结算抛了是 bug——这一段停下、带原因，后面几段不跑、不拿旧数据顶、健康面报出；退避到点之前
    下一眼不再整段重跑；到点再试、好了就从结算接着跑完。"""

    from habitus.runtime.health import _night_line_checks

    order: list[str] = []
    fails = [True]
    clock = [NOW]
    run = wired(tmp_path, order, settle_fails=lambda: fails[0])
    worker = worker_for(run, tmp_path, clock=lambda: clock[0])
    ledger = asyncio.run(worker.look())
    assert order == ["探模型", "重试", "结算"]  # 停在结算，不接着跑词表、语义树、机会表、日子预期
    assert ledger is not None
    stopped = ledger.job(SETTLE)
    assert stopped.state == JOB_STOPPED and stopped.reason == "结算出错: OSError" and stopped.attempts == 1
    assert stopped.next_attempt_at == NOW + timedelta(seconds=60)
    assert "settle stopped: 结算出错: OSError" in str(worker.last_error)
    (check,) = _night_line_checks(worker.jobs)
    assert check.name == "nightly_line" and "settle stopped: 结算出错: OSError" in check.detail

    order.clear()
    asyncio.run(worker.look())  # 5 分钟后：还没到点，不整段重跑
    assert order == []
    clock[0] = NOW + timedelta(seconds=61)
    fails[0] = False
    ledger = asyncio.run(worker.look())
    assert order == ["结算", "词表", "语义树", "机会表", "日子"]
    assert ledger is not None and ledger.complete and ledger.job(SETTLE).attempts == 2 and worker.last_error is None


def test_an_unreachable_model_stops_the_night_without_stale_steps(tmp_path) -> None:
    order: list[str] = []
    run = wired(tmp_path, order)

    async def down() -> None:
        order.append("探模型")
        raise ModelUnreachable("MODEL_TRANSPORT")

    run.probe = down
    results = line(run)
    assert order == ["探模型"]  # 不重试、不结算、不跑语义树、不建日子预期
    assert list(results) == [PREPARE]
    assert results[PREPARE].outcome == STOPPED and results[PREPARE].reason == f"{MODEL_UNREACHABLE}: MODEL_TRANSPORT"


def test_the_retry_meeting_an_unusable_model_stops_the_night_and_keeps_its_counts(tmp_path) -> None:
    """重试里第一条会话轮就是模型用不了（``RetryReport.unavailable``）：停下，计数留在前置处理那一段的报告上。"""

    order: list[str] = []
    run = wired(tmp_path, order, retried=RetryReport(taken=2, unavailable=1))
    results = line(run)
    prepared = results[PREPARE]
    assert order == ["探模型", "重试"] and prepared.outcome == STOPPED
    assert prepared.reason is not None and prepared.reason.startswith(MODEL_UNREACHABLE)
    retry = prepared.report["retry"]
    assert isinstance(retry, dict) and retry["taken"] == 2 and retry["unavailable"] == 1


def test_any_model_touchpoint_on_the_line_that_finds_the_model_unusable_stops_the_night(tmp_path) -> None:
    """裁定 94 第 8 条：线上任何一处调模型碰到"整个用不了"都停线、报"模型调不通"——语义树的触点（``ModelUnreachable``）、
    每晚新增词表直接调模型客户端原样抛出的（额度满）。"""

    order: list[str] = []
    run = wired(tmp_path, order)

    async def quota(_night) -> None:  # type: ignore[no-untyped-def]
        order.append("词表")
        raise ModelQuotaError("quota exhausted")

    run.vocabulary_jobs = (quota,)
    results = line(run)
    assert order == ["探模型", "重试", "结算", "词表"]
    reason = results[VOCABULARY].reason
    assert results[VOCABULARY].outcome == STOPPED and reason is not None and reason.startswith(MODEL_UNREACHABLE)

    order.clear()
    run = wired(tmp_path / "scene", order)

    async def touchpoint(**_kwargs):  # type: ignore[no-untyped-def]
        order.append("语义树")
        raise ModelUnreachable("model: ModelAuthenticationError: denied")

    run.scene.run = touchpoint  # type: ignore[method-assign]
    results = line(run)
    assert order[-1] == "语义树" and "机会表" not in order
    assert results[STRUCTURE].reason == f"{MODEL_UNREACHABLE}: model: ModelAuthenticationError: denied"


def test_a_busy_lock_on_the_vocabulary_is_not_ready_and_resumes_there(tmp_path) -> None:
    """每晚新增碰到 sweep 锁被另一进程拿着：还没准备好，Job 保持排队、不停线，下一眼从这一段接着跑。"""

    order: list[str] = []
    run = wired(tmp_path, order)
    busy = [True]

    async def vocabulary(_night) -> None:  # type: ignore[no-untyped-def]
        order.append("词表")
        if busy[0]:
            raise BehaviorReductionBusyError("sweep lock held")

    run.vocabulary_jobs = (vocabulary,)
    worker = worker_for(run, tmp_path)
    first = asyncio.run(worker.look())
    assert first is not None and first.job(VOCABULARY).state == QUEUED and first.job(VOCABULARY).reason == LOCK_BUSY
    assert [first.job(step).state for step in (PREPARE, SETTLE)] == [SUCCEEDED, SUCCEEDED]
    order.clear()
    busy[0] = False
    second = asyncio.run(worker.look())
    assert order == ["词表", "语义树", "机会表", "日子"] and second is not None and second.complete


def test_resuming_keeps_the_errors_the_preparation_left_unseen_on_the_health_surface(tmp_path) -> None:
    """裁定 94 第 8 条：从断点接着跑的那一趟不再做前置处理，健康面上那一晚"错误·看不到"的计数不丢（在 Job 账里，重启也不丢）。"""

    from habitus.runtime.health import _night_line_checks

    order: list[str] = []
    migrating = [True]
    kept = RetryReport(taken=3, deterministic=1, invalid=1, refused=1)
    run = wired(tmp_path, order, migrating=lambda: migrating[0], retried=kept)
    worker = worker_for(run, tmp_path)
    asyncio.run(worker.look())
    migrating[0] = False
    ledger = asyncio.run(worker.look())
    assert ledger is not None and ledger.complete
    restarted = worker_for(run, tmp_path)  # 重启：健康面照样从账上读
    (check,) = _night_line_checks(restarted.jobs)
    assert check.name == "nightly_errors" and "格式不合 1" in check.detail and "处理不了 1" in check.detail
    assert "nightly errors kept unseen" in str(worker.last_error)


def test_the_ledger_keeps_the_progress_on_disk_and_a_stopped_step_reaches_the_health_surface(tmp_path) -> None:
    """进度落在 Job 账上（重读盘上的账看得见）；没跑完这一晚就不算跑完；模型调不通停下的那一段健康面写"模型调不通"。"""

    from habitus.runtime.health import _night_line_checks

    order: list[str] = []
    migrating = [True]
    run = wired(tmp_path, order, migrating=lambda: migrating[0])
    worker = worker_for(run, tmp_path)
    night = run.tonight()
    asyncio.run(worker.look())
    reread = NightJobs(tmp_path / "nightly" / "jobs", artifact_root=tmp_path, kept_nights=366)
    ledger = reread.read(night)
    assert ledger is not None and [job.step for job in ledger.jobs if job.state == SUCCEEDED] == [
        PREPARE,
        SETTLE,
        VOCABULARY,
    ]
    assert reread.finished(night) is None  # 一晚都没跑完过：认不出
    order.clear()
    migrating[0] = False
    asyncio.run(worker.look())
    assert order == ["语义树", "机会表", "日子"] and reread.finished(night) is True and worker.last_error is None

    async def down() -> None:
        raise ModelUnreachable("MODEL_AUTHENTICATION")

    clock = [NOW + timedelta(days=1)]
    tomorrow_run = wired(tmp_path / "next", order, clock=lambda: clock[0])
    tomorrow_run.probe = down
    tomorrow_run.closed_days = lambda: (night,)
    later = worker_for(tomorrow_run, tmp_path, clock=lambda: clock[0])
    asyncio.run(later.look())
    assert reread.finished(night + timedelta(days=1)) is False
    (check,) = _night_line_checks(later.jobs)
    assert check.name == "nightly_line" and f"prepare stopped: {MODEL_UNREACHABLE}" in check.detail


def test_a_leftover_last_run_file_is_not_read(tmp_path) -> None:
    """Job 账取代了 ``last_run.json``：旧文件不读、不迁移（不兼容）——留在盘上也不影响"这一晚跑完没有"。"""

    path = tmp_path / "nightly" / "last_run.json"
    path.parent.mkdir(parents=True)
    path.write_text('{"night": "2026-08-18", "since": "2026-08-01", "finished": "2026-08-18", "missed": []}')
    jobs = NightJobs(tmp_path / "nightly" / "jobs", artifact_root=tmp_path, kept_nights=366)
    assert jobs.finished(datetime(2026, 8, 18).date()) is None and jobs.latest() is None


def _ledger(night, states, *, report=None, reasons=None) -> NightLedger:  # type: ignore[no-untyped-def]
    reasons = reasons or {}
    return NightLedger(
        night=night,
        created_at=NOW,
        jobs=tuple(
            NightJob(
                step,
                state=state,
                reason=reasons.get(step),
                report=(report or {}).get(step, {}),
            )
            for step, state in zip(STEPS, states, strict=True)
        ),
    )


def test_kept_errors_reach_the_health_surface(tmp_path) -> None:
    from habitus.runtime.health import _night_line_checks
    from tests.unit.runtime.test_scene_night import TONIGHT

    jobs = NightJobs(tmp_path / "jobs", artifact_root=tmp_path, kept_nights=366)
    retry = {"taken": 3, "deterministic": 2, "refused": 1, "invalid": 0}
    jobs.write(_ledger(TONIGHT, [SUCCEEDED] * 5, report={PREPARE: {"retry": retry}}))
    (check,) = _night_line_checks(jobs)
    assert check.name == "nightly_errors" and "处理不了 2" in check.detail and "内容安全 1" in check.detail

    # 停下的那一晚也列前置处理里保持「错误·看不到」的，格式不合夜里再败的也报（裁定 94 第 8 条）
    retry = {"taken": 2, "invalid": 1, "unavailable": 1}
    stopped = _ledger(
        TONIGHT + timedelta(days=1),
        [JOB_STOPPED, QUEUED, QUEUED, QUEUED, QUEUED],
        report={PREPARE: {"retry": retry}},
        reasons={PREPARE: f"{MODEL_UNREACHABLE}: quota"},
    )
    jobs.write(stopped)
    line_check, kept = _night_line_checks(jobs)
    assert line_check.name == "nightly_line" and kept.name == "nightly_errors" and "格式不合 1" in kept.detail


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
    """review-94b 中 1：前置处理里重试那一支还在等锁（``RetryReport.waiting``）——前置处理那个 Job 保持排队、不结算（写回的判断
    还没落树，结算了它们就会在白天落进已经结算的日子）；下一眼从前置处理接着跑，重试做完才结算。撤回 ``_prepare`` 里那道判断这里就红。"""

    order: list[str] = []
    run = wired(tmp_path, order)
    run.retry = StubRetry(  # type: ignore[assignment]
        RetryReport(taken=1, waiting=("session",)), RetryReport(taken=1, recovered=1), order=order
    )
    worker = worker_for(run, tmp_path)
    first = asyncio.run(worker.look())
    assert order == ["探模型", "重试"] and first is not None
    assert first.job(PREPARE).state == QUEUED and first.job(PREPARE).reason == LOCK_BUSY
    assert all(job.state == QUEUED for job in first.jobs)
    order.clear()
    second = asyncio.run(worker.look())
    assert order == ALL and second is not None and second.complete
    retry = second.job(PREPARE).report["retry"]
    assert isinstance(retry, dict) and retry["recovered"] == 1 and retry["waiting"] == []


def test_the_ledger_records_which_nights_the_whole_line_finished(tmp_path) -> None:
    """review-94b 中低 3 / 裁定 94 第 9 条："停掉的那天"读线自己记的这一晚跑到哪一步：日子预期那一段没跑完（停下了、失败了、
    没跑到、没开机）即算停——表写出后才停（日子预期那一段抛了）也认得出。第一次跑完之前认不出（None）。"""

    from tests.unit.runtime.test_scene_night import TONIGHT

    jobs = NightJobs(tmp_path / "jobs", artifact_root=tmp_path, kept_nights=366)
    night = TONIGHT
    jobs.write(_ledger(night - timedelta(days=4), [SUCCEEDED, JOB_STOPPED, QUEUED, QUEUED, QUEUED]))  # 第一次跑完之前
    jobs.write(_ledger(night - timedelta(days=3), [SUCCEEDED] * 5))  # 跑完
    jobs.write(_ledger(night - timedelta(days=2), [SUCCEEDED] * 4 + [FAILED]))  # 机会表写了、日子预期那一段失败
    # night - 1：没开机，没有账
    jobs.write(_ledger(night, [SUCCEEDED] * 5))
    assert [jobs.finished(night - timedelta(days=offset)) for offset in range(5, -1, -1)] == [
        None,  # 留存里第一次跑完之前
        None,
        True,
        False,  # 表写出后才停
        False,  # 没开机
        True,
    ]
    assert jobs.finished(night + timedelta(days=1)) is False  # 还没跑到


def test_the_ledger_keeps_its_record_of_finished_nights_bounded(tmp_path) -> None:
    jobs = NightJobs(tmp_path / "jobs", artifact_root=tmp_path, kept_nights=366)
    start = datetime(2025, 1, 1).date()
    jobs.write(_ledger(start, [SUCCEEDED] * 5))
    far = start + timedelta(days=2 * 366)
    jobs.write(_ledger(far, [SUCCEEDED] * 5))  # 中间几百晚都没跑完
    assert jobs.prune(far) == 1
    assert jobs.finished(far - timedelta(days=1)) is None and jobs.finished(far) is True  # 留存里只剩这一晚
    jobs.write(_ledger(far - timedelta(days=10), [SUCCEEDED] * 5))
    assert jobs.finished(far - timedelta(days=1)) is False and jobs.finished(start) is None  # 留存之外：认不出


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


def test_a_day_step_held_on_a_table_that_is_not_tonights_keeps_the_job_queued(tmp_path) -> None:
    """日子预期那一段：今晚的表还不是今天这一晚的，是还没准备好——Job 保持排队（原因照抄），不算停下。"""

    order: list[str] = []
    run = wired(tmp_path, order)

    class Stale:
        async def run_once(self):  # type: ignore[no-untyped-def]
            order.append("日子")
            return DayReport(datetime.now().astimezone(), Formed(), "今天这一晚的机会表还没有")

    run.days = Stale()  # type: ignore[assignment]
    results = line(run)
    assert results[DAYS].outcome == HELD and results[DAYS].reason == "今天这一晚的机会表还没有"
