"""夜线 Job 化 + 观测（2026-11-09 共享说明第 2 节）：每晚 5 段各一个耐久 Job、全局租约互斥、退避重试、断点续跑、过了那天没跑完记失败；
每段 span / 指标 / INFO 日志、每晚报告文件（含关系表双向变化）、本地 trace 文件、健康面读 Job 账、运行参数进统一配置。
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import date, timedelta
from types import SimpleNamespace

import pytest

from habitus.behavior.retry import RetryCheckpoint, RetryReport
from habitus.config.scene import SceneConfig
from habitus.foundation.observability import ObservationEvent, ObservationStatus
from habitus.foundation.tracing import open_trace, span
from habitus.infrastructure.observability.metrics import LONG_DURATION_BUCKETS, MetricRegistry
from habitus.infrastructure.store.sqlite import SQLiteLockStore
from habitus.runtime.nightly import DONE, PREPARE, SETTLE, STEPS, STOPPED, STRUCTURE, VOCABULARY, StepResult
from habitus.runtime.nightly_jobs import (
    FAILED,
    NIGHT_PASSED,
    QUEUED,
    RUNNING,
    SUCCEEDED,
    NightClaimLostError,
    NightJobs,
    NightJobsError,
    NightLedger,
    claim,
    claimed,
    decode_checkpoint,
    encode_checkpoint,
    renewed,
    settled,
)
from habitus.runtime.nightly_report import NightReports
from habitus.runtime.nightly_settings import NightlySettings
from habitus.runtime.nightly_trace import ModelCallTracer, TraceFiles
from habitus.scene.relations import RelationKey, Segment, stats
from habitus.scene.relations.engine import RelationTest, Verdict
from habitus.scene.relations.state import LaneState, Relation, Status, Transition
from habitus.scene.relations.store import RelationStore
from tests.lock_store import ProcessLocalLockStore
from tests.unit.runtime.test_nightly import Recorder, StubRetry, StubSettlement, line, worker_for
from tests.unit.runtime.test_nightly_line import ALL, wired
from tests.unit.runtime.test_scene_night import NOW


def test_yesterday_closing_queues_five_jobs_that_run_in_order(tmp_path) -> None:
    """共享说明 #1：昨天封口 → 这一晚入队 5 个 Job（prepare、settle、vocabulary、structure、days），按序执行，都成了这一晚才跑完；
    跑完之后下一眼不再跑。"""

    order: list[str] = []
    run = wired(tmp_path, order)
    worker = worker_for(run, tmp_path)
    night = run.tonight()
    ledger = asyncio.run(worker.look())
    assert order == ALL and ledger is not None and ledger.night == night
    assert [job.step for job in ledger.jobs] == list(STEPS) and ledger.complete
    assert all(job.attempts == 1 and job.started_at is not None and job.finished_at is not None for job in ledger.jobs)
    order.clear()
    asyncio.run(worker.look())
    assert order == []  # 这一晚跑完了：不再跑


def test_nothing_is_queued_before_yesterday_closes(tmp_path) -> None:
    order: list[str] = []
    run = wired(tmp_path, order)
    run.closed_days = lambda: ()
    worker = worker_for(run, tmp_path)
    assert asyncio.run(worker.look()) is None and worker.jobs.nights() == ()


def test_a_night_left_unfinished_when_the_day_turns_is_failed_and_the_next_night_runs(tmp_path) -> None:
    """过了那天还没跑完（迁移卡着结构那一段）：上一晚没成的段记失败（"这一晚过去了，没跑完"），不在新的日子上按新日子补算；
    下一晚照常入队、跑完。上一晚就算停下的那天（``finished`` 为假）。"""

    order: list[str] = []
    migrating = [True]
    clock = [NOW]
    run = wired(tmp_path, order, migrating=lambda: migrating[0], clock=lambda: clock[0])
    worker = worker_for(run, tmp_path, clock=lambda: clock[0])
    first_night = run.tonight()
    asyncio.run(worker.look())
    clock[0] = NOW + timedelta(days=1)
    run.closed_days = lambda: (first_night,)
    migrating[0] = False
    order.clear()
    ledger = asyncio.run(worker.look())
    old = worker.jobs.read(first_night)
    assert old is not None and old.job(STRUCTURE).state == FAILED and old.job(STRUCTURE).reason is not None
    assert old.job(STRUCTURE).reason.startswith(NIGHT_PASSED) and old.job(PREPARE).state == SUCCEEDED
    assert ledger is not None and ledger.night == first_night + timedelta(days=1) and ledger.complete
    assert order == ALL
    # 留存里第一次跑完的是下一晚：上一晚在它之前，认不出（None，结算不认它停了）；之后再有没跑完的就是 False
    assert worker.finished(first_night) is None and worker.finished(ledger.night) is True


def test_a_stopped_step_is_retried_with_backoff_and_fails_after_its_attempts_blocking_the_rest(tmp_path) -> None:
    """共享说明 #3：停下 → STOPPED 带原因，按 ``next_attempt_at`` 有界退避；次数用完 FAILED，后面的段不再跑；健康面读账报出。"""

    from habitus.runtime.health import _night_line_checks

    order: list[str] = []
    clock = [NOW]
    run = wired(tmp_path, order, settle_fails=lambda: True)
    settings = NightlySettings(max_attempts=3, retry_base_seconds=60.0, retry_max_seconds=90.0)
    worker = worker_for(run, tmp_path, clock=lambda: clock[0], settings=settings)
    waits: list[timedelta] = []
    for _ in range(3):
        ledger = asyncio.run(worker.look())
        assert ledger is not None
        job = ledger.job(SETTLE)
        if job.next_attempt_at is not None:
            waits.append(job.next_attempt_at - clock[0])
            clock[0] = job.next_attempt_at
    assert waits == [timedelta(seconds=60), timedelta(seconds=90)]  # 60 → 120 封顶 90
    assert ledger.job(SETTLE).state == FAILED and ledger.job(SETTLE).attempts == 3
    assert [ledger.job(step).state for step in (VOCABULARY, STRUCTURE)] == [QUEUED, QUEUED]
    order.clear()
    clock[0] += timedelta(hours=1)
    asyncio.run(worker.look())
    assert order == []  # 失败挡着后面
    (check,) = _night_line_checks(worker.jobs)
    assert "settle failed: 结算出错: OSError" in check.detail


def test_a_cancelled_step_goes_back_to_the_queue_and_the_next_look_resumes_it(tmp_path) -> None:
    """被取消（停机超时、租约丢了）的那一段放回排队、不算一次尝试；下一眼从这一段接着跑（每一段重跑都幂等）。"""

    order: list[str] = []
    run = wired(tmp_path, order)
    cut = [True]

    async def vocabulary(_night) -> None:  # type: ignore[no-untyped-def]
        order.append("词表")
        if cut[0]:
            cut[0] = False
            raise asyncio.CancelledError

    run.vocabulary_jobs = (vocabulary,)
    worker = worker_for(run, tmp_path)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(worker.look())
    ledger = worker.jobs.read(run.tonight())
    assert ledger is not None and ledger.job(VOCABULARY).state == QUEUED and ledger.job(VOCABULARY).attempts == 0
    order.clear()
    ledger = asyncio.run(worker.look())
    assert order == ["词表", "语义树", "机会表", "日子"] and ledger is not None and ledger.complete


def test_a_crashed_claim_is_taken_over_only_after_its_lease_expires_with_a_new_generation(tmp_path) -> None:
    """共享说明 #5、#6：进程在结构那一段中途被杀，账上留着"在跑"：租约没过期别人不接；过期了递增 generation 接管，从这一段接着跑；
    陈旧的认领续约、结算都被拒。"""

    order: list[str] = []
    clock = [NOW]
    run = wired(tmp_path, order)
    worker = worker_for(run, tmp_path, clock=lambda: clock[0])
    night = run.tonight()
    jobs = worker.jobs
    ledger = NightLedger.fresh(night, NOW)
    for step in (PREPARE, SETTLE, VOCABULARY):
        token = claim(ledger, step, worker_id="dead", now=NOW)
        ledger = claimed(ledger, token, now=NOW, ttl_seconds=120)
        ledger = settled(
            ledger,
            token,
            StepResult(step, DONE),
            now=NOW,
            duration_seconds=1.0,
            backoff=NightlySettings().backoff,
            max_attempts=3,
        )
    stale = claim(ledger, STRUCTURE, worker_id="dead", now=NOW)
    jobs.write(claimed(ledger, stale, now=NOW, ttl_seconds=120))
    asyncio.run(worker.look())
    assert order == []  # 租约还在：不接管
    clock[0] = NOW + timedelta(seconds=121)
    ledger = asyncio.run(worker.look())
    assert order == ["语义树", "机会表", "日子"] and ledger is not None and ledger.complete
    assert ledger.job(STRUCTURE).claim_generation == stale.claim_generation + 1
    current = jobs.read(night)
    assert current is not None
    with pytest.raises(NightClaimLostError):
        renewed(current, stale, now=clock[0], ttl_seconds=120)
    with pytest.raises(NightClaimLostError):
        settled(
            current,
            stale,
            StepResult(STRUCTURE, DONE),
            now=clock[0],
            duration_seconds=1.0,
            backoff=NightlySettings().backoff,
            max_attempts=3,
        )


def test_two_processes_on_one_root_run_each_step_once(tmp_path) -> None:
    """共享说明 #5：两个执行者挂同一个 storage root（同一份 SQLite 锁库）：同一晚同一段只跑一次——拿不到全局租约的那一个这一眼什么都不做。"""

    order: list[str] = []
    run = wired(tmp_path, order)
    gate = asyncio.Event

    async def scenario() -> None:
        started, release = gate(), gate()
        scene_run = run.scene.run

        async def slow(**kwargs):  # type: ignore[no-untyped-def]
            started.set()
            await release.wait()
            return await scene_run(**kwargs)

        run.scene.run = slow  # type: ignore[method-assign]
        locks = SQLiteLockStore(tmp_path / "locks.sqlite3")
        first = worker_for(run, tmp_path, lock_store=locks, worker_id="a")
        second = worker_for(run, tmp_path, lock_store=SQLiteLockStore(tmp_path / "locks.sqlite3"), worker_id="b")
        running = asyncio.create_task(first.look())
        await started.wait()
        assert await second.look() is None  # 租约在第一个手里
        release.set()
        ledger = await running
        assert ledger is not None and ledger.complete
        assert await second.look() is not None  # 跑完之后它拿得到租约，但没什么要跑的

    asyncio.run(scenario())
    assert order.count("结算") == 1 and order.count("日子") == 1


def test_a_retry_waiting_on_a_lock_survives_a_restart_through_the_ledger(tmp_path) -> None:
    """共享说明 #6：前置处理里重试那一支等锁时的状态落在前置处理那个 Job 上；重启（新执行器、新的重试段）先从账上恢复，不撤回
    等着的项（``clean_leftovers`` 不调）；下一眼接着跑完。"""

    order: list[str] = []
    run = wired(tmp_path, order)
    night = run.tonight()
    checkpoint = RetryCheckpoint(
        through=night - timedelta(days=1), waiting={"session": (("a" * 32,), ())}, counts={"taken": 1}
    )

    class Waiting(StubRetry):
        async def run_once(self, *, through):  # type: ignore[no-untyped-def]
            self.last = await super().run_once(through=through)
            return self.last

        def checkpoint(self):  # type: ignore[no-untyped-def]
            return checkpoint if self.last.waiting else None

    run.retry = Waiting(RetryReport(taken=1, waiting=("session",)), RetryReport(taken=1, recovered=1), order=order)  # type: ignore[assignment]
    asyncio.run(worker_for(run, tmp_path).look())
    ledger = NightJobs(tmp_path / "nightly" / "jobs", artifact_root=tmp_path, kept_nights=366).read(night)
    assert ledger is not None and ledger.job(PREPARE).retry_checkpoint is not None
    assert decode_checkpoint(ledger.job(PREPARE).retry_checkpoint) == checkpoint

    restarted_retry = Waiting(RetryReport(taken=1, recovered=1), order=order)
    run.retry = restarted_retry  # type: ignore[assignment]
    ledger = asyncio.run(worker_for(run, tmp_path).look())
    assert restarted_retry.cleaned == 0  # 等着的项不撤回
    assert restarted_retry.restored == [checkpoint]  # 先从账上恢复再跑
    assert ledger is not None and ledger.complete and ledger.job(PREPARE).retry_checkpoint is None


def test_a_fresh_process_cleans_the_leftover_retry_items_once_when_nothing_is_waiting(tmp_path) -> None:
    order: list[str] = []
    run = wired(tmp_path, order)
    worker = worker_for(run, tmp_path)
    asyncio.run(worker.look())
    asyncio.run(worker.look())
    assert run.retry.cleaned == 1  # type: ignore[union-attr]


def test_the_retry_checkpoint_round_trips_and_a_corrupt_one_is_refused() -> None:
    checkpoint = RetryCheckpoint(date(2026, 8, 18), {"session": (("a" * 32,), ("b" * 32,))}, {"taken": 2})
    assert decode_checkpoint(encode_checkpoint(checkpoint)) == checkpoint
    with pytest.raises(NightJobsError):
        decode_checkpoint({"through": "2026-08-18", "waiting": [], "counts": {}})


def test_a_corrupt_ledger_is_not_read_as_an_unrun_night(tmp_path) -> None:
    """账坏了不当没跑过、不静默重跑：执行器这一眼抛出（留在 ``last_error``），健康面报读不了；``finished`` 也抛出，不吞成
    "没跑完"（codex R2）。"""

    from habitus.runtime.health import _night_line_checks

    order: list[str] = []
    run = wired(tmp_path, order)
    worker = worker_for(run, tmp_path)
    path = worker.jobs.path(run.tonight())
    path.parent.mkdir(parents=True)
    path.write_text("{broken")
    with pytest.raises(NightJobsError):
        asyncio.run(worker.look())
    assert order == []
    (check,) = _night_line_checks(worker.jobs)
    assert check.detail == "unreadable:NightJobsError"
    with pytest.raises(NightJobsError):
        worker.finished(run.tonight())


def test_an_unreadable_ledger_stops_the_settlement_and_holds_the_daytime_forming(tmp_path) -> None:
    """codex R2：结算要问的那一晚的账读不了——``finished`` 抛出，结算那一段 STOPPED、带原因（不把它当成停掉的那天去截断后补）；
    白天那一拍不建预期、报出原因（不当跑完，也不当没跑完）。读得到的晚照常三值。"""

    from habitus.runtime.foresight_forming import LEDGER_UNREADABLE, tonight_table

    jobs = NightJobs(tmp_path / "jobs", artifact_root=tmp_path, kept_nights=366)
    first, broken = date(2026, 8, 16), date(2026, 8, 17)
    jobs.write(_complete(first))
    jobs.path(broken).write_text("{broken")
    assert jobs.finished(first) is True
    with pytest.raises(NightJobsError):
        jobs.finished(broken)

    order: list[str] = []
    run = wired(tmp_path / "line", order)

    class Settlement(StubSettlement):
        def run_once(self):  # type: ignore[no-untyped-def]
            jobs.finished(broken)  # 结算认停掉的那天时问到这一晚
            return super().run_once()

    run.settlement = Settlement(order)  # type: ignore[assignment]
    results = line(run)
    assert results[SETTLE].outcome == STOPPED and results[SETTLE].reason == "结算出错: NightJobsError"
    assert list(results) == [PREPARE, SETTLE]

    class Store:
        def latest(self, _lane):  # type: ignore[no-untyped-def]
            return SimpleNamespace(night=broken)

    table, reason = tonight_table(Store(), "session", broken, line_finished=jobs.finished)  # type: ignore[arg-type]
    assert table is not None and reason == LEDGER_UNREADABLE


def test_the_worker_looks_in_configured_slices_and_survives_an_unreadable_closure(tmp_path) -> None:
    """隔 ``scene.nightly_poll_seconds`` 看一眼（醒来重看墙钟，裁定 68 第 4 条）；读不了定稿日照样活着、留 ``last_error``。"""

    order: list[str] = []
    run = wired(tmp_path, order)
    worker = worker_for(run, tmp_path, settings=NightlySettings(poll_seconds=42.0))
    naps: list[float] = []

    async def nap(seconds: float) -> None:
        naps.append(seconds)
        worker._stop_requested.set()

    worker._wait = nap  # type: ignore[method-assign]
    asyncio.run(worker._run_loop())
    assert naps == [42.0] and order == ALL

    def unreadable():  # type: ignore[no-untyped-def]
        raise OSError("day_closure.json")

    blind_run = wired(tmp_path / "blind", order)
    blind_run.ready = unreadable  # type: ignore[method-assign]
    blind = worker_for(blind_run, tmp_path / "blind")

    async def stop(_seconds: float) -> None:
        blind._stop_requested.set()

    blind._wait = stop  # type: ignore[method-assign]
    asyncio.run(blind._run_loop())
    assert isinstance(blind.last_error, OSError)


def test_each_step_is_a_span_under_the_night_with_its_logical_data_in_the_local_trace(tmp_path) -> None:
    """共享说明 #7（trace 细度）：``nightly.night`` → 每段 ``nightly.<段>`` → 结构段里 ``scene.b1``…``scene.b4``（每条 lane 一个
    ``scene.b4.lane``）；每一次模型调用一个 ``model.call``，挂在发起它的段下；段的属性带结局、尝试次数与这一段报告摊开的读数。"""

    order: list[str] = []
    run = wired(tmp_path, order)
    tracer = ModelCallTracer()

    async def vocabulary(_night):  # type: ignore[no-untyped-def]
        order.append("词表")
        tracer.record(
            ObservationEvent(
                "model",
                "complete_async",
                ObservationStatus.SUCCESS,
                1.5,
                {"provider": "p", "model": "m", "touch": "NightlyGrowth", "retry_count": 1, "prompt_chars": 900},
            )
        )

    run.vocabulary_jobs = (vocabulary,)
    worker = worker_for(run, tmp_path)
    asyncio.run(worker.look())
    spans = TraceFiles(tmp_path / "nightly" / "traces", artifact_root=tmp_path).read(run.tonight())
    by_name = {item["name"]: item for item in spans}
    night = by_name["nightly.night"]
    assert {f"nightly.{step}" for step in STEPS} | {"scene.b1", "scene.b3", "scene.b4", "model.call"} <= set(by_name)
    for step in STEPS:
        assert by_name[f"nightly.{step}"]["parent_id"] == night["span_id"]
        assert by_name[f"nightly.{step}"]["attributes"]["outcome"] == "done"
    structure = by_name["nightly.structure"]
    assert by_name["scene.b1"]["parent_id"] == structure["span_id"]
    assert by_name["scene.b4"]["parent_id"] == structure["span_id"]
    call = by_name["model.call"]
    assert call["parent_id"] == by_name["nightly.vocabulary"]["span_id"]
    assert call["attributes"]["touch"] == "NightlyGrowth" and call["attributes"]["attempts"] == 2
    vocabulary_attributes = by_name["nightly.vocabulary"]["attributes"]
    assert vocabulary_attributes["model_calls"] == 1 and vocabulary_attributes["model_calls.touch.NightlyGrowth"] == 1
    assert night["attributes"]["model_calls"] == 1 and night["attributes"]["outcome.days"] == SUCCEEDED
    assert "records" in structure["attributes"] and "mapped" in by_name["scene.b3"]["attributes"]


def test_model_calls_outside_a_trace_are_not_spans() -> None:
    tracer = ModelCallTracer(metrics=Recorder())
    tracer.record(ObservationEvent("model", "complete", ObservationStatus.SUCCESS, 0.1, {"touch": "x"}))
    assert tracer.metrics.events == []  # type: ignore[union-attr]

    seen: list[str] = []

    class Sink:
        def started(self, record):  # type: ignore[no-untyped-def]
            return None

        def finished(self, record, handle):  # type: ignore[no-untyped-def]
            seen.append(record.name)

    with open_trace(Sink()), span("nightly.settle"):
        tracer.record(ObservationEvent("model", "complete", ObservationStatus.FAILURE, 0.1, {"error_code": "E"}))
    assert seen == ["model.call", "nightly.settle"]
    (event,) = tracer.metrics.events  # type: ignore[union-attr]
    assert event.attributes == {"step": "settle", "touch": "chat"} and event.status is ObservationStatus.FAILURE


def test_metrics_carry_step_durations_job_states_model_calls_and_stopped_is_not_success(tmp_path) -> None:
    """共享说明 #8：``/metrics`` 有每段耗时（桶到数小时）、各段各状态 Job 数、按段按触点的调模型次数；停下记降级，不记成功。"""

    order: list[str] = []
    registry = MetricRegistry()
    run = wired(tmp_path, order, settle_fails=lambda: True)
    worker = worker_for(run, tmp_path, observer=registry)
    asyncio.run(worker.look())
    text = registry.prometheus_text()
    assert (
        f'habitus_nightly_job_duration_seconds_bucket{{step="prepare",le="{LONG_DURATION_BUCKETS[-1]:.9g}"}} 1' in text
    )
    assert 'habitus_nightly_jobs{state="stopped",step="settle"} 1' in text or (
        'habitus_nightly_jobs{step="settle",state="stopped"} 1' in text
    )
    assert 'habitus_nightly_step_outcomes_total{step="settle",outcome="stopped"} 1' in text
    snapshot = registry.snapshot()
    assert snapshot.counters["operations_total[category=nightly,operation=step,status=degraded]"] == 1
    assert "operations_total[category=nightly,operation=step,status=success]" in snapshot.counters  # prepare 成了


def test_step_start_finish_and_stop_are_info_lines(tmp_path, caplog) -> None:
    """共享说明 #9：段开始 / 结束 / 停下各一行 INFO（晚、段、结局、原因）。"""

    order: list[str] = []
    run = wired(tmp_path, order, settle_fails=lambda: True)
    worker = worker_for(run, tmp_path)
    with caplog.at_level(logging.INFO, logger="habitus.nightly"):
        asyncio.run(worker.look())
    lines = [record.getMessage() for record in caplog.records if record.name == "habitus.nightly"]
    night = run.tonight().isoformat()
    assert f"nightly step started night={night} step=prepare attempt=1" in lines
    assert any(line.startswith(f"nightly step finished night={night} step=prepare outcome=done") for line in lines)
    assert any(
        line.startswith(f"nightly step stopped night={night} step=settle outcome=stopped reason=结算出错: OSError")
        for line in lines
    )
    assert all(record.levelno == logging.INFO for record in caplog.records if record.name == "habitus.nightly")


def test_the_nightly_report_keeps_every_step_and_the_vocabulary_report(tmp_path) -> None:
    """共享说明 #10：每晚报告文件 ``reports/<日期>.json``：每段结局、耗时、尝试、调模型次数、各段原有报告（词表段的不再丢）。"""

    from habitus.behavior.reduction import VocabularyJobReport

    order: list[str] = []
    run = wired(tmp_path, order)

    async def vocabulary(night):  # type: ignore[no-untyped-def]
        order.append("词表")
        return VocabularyJobReport((night,), 3, ("grew",), pool=7, created=("k1",), moves=2)

    run.vocabulary_jobs = (vocabulary,)
    worker = worker_for(run, tmp_path)
    asyncio.run(worker.look())
    report = json.loads((tmp_path / "nightly" / "reports" / f"{run.tonight().isoformat()}.json").read_text())
    assert report["complete"] is True and set(report["steps"]) == set(STEPS)
    branch = report["steps"]["vocabulary"]["report"]["branches"][0]
    assert branch["pool"] == 7 and branch["created"] == ["k1"] and branch["moves"] == 2 and branch["model_calls"] == 3
    assert report["steps"]["structure"]["state"] == SUCCEEDED and "scene" in report["steps"]["structure"]["report"]
    assert report["steps"]["prepare"]["attempts"] == 1 and report["steps"]["prepare"]["model_calls"] == {}
    assert "relations" in report


def _test(key: RelationKey, p: float, verdict: Verdict = Verdict.SIGNIFICANT) -> RelationTest:
    return RelationTest(
        key=key,
        verdict=verdict,
        antecedents=20,
        blocks=6,
        block_days=7.0,
        effect=stats.Effect(20, 0.9, 0.1),
        p_value=p,
        minimum_p=p / 10,
        upward=True,
    )


def test_the_report_lists_relations_added_removed_and_changed_against_the_previous_night(tmp_path) -> None:
    """共享说明 #10（用户 11-09：增加还是减少都要看）：关系表比上一晚多出 / 少了 / 状态变了的逐条，带前后状态、当晚 p 与最小可达 p、
    族、进族路径与迁移日志里的原因；按 lane、种类、状态、段、带不带条件计进指标。"""

    first, second = date(2026, 8, 17), date(2026, 8, 18)
    a = RelationKey("session", "起床", "吃早餐", Segment.CHAIN)
    b = RelationKey("session", "加班", "晚睡", Segment.REST_OF_DAY)
    child = RelationKey("session", "起床", "吃早餐", Segment.CHAIN, "concept:周一")
    store = RelationStore(tmp_path / "scene")
    store.write(
        LaneState(
            "session",
            first,
            3,
            (
                Relation(a, Status.CANDIDATE, True, tonight=_test(a, 0.001), discovered_on=first, forward_needed=3),
                Relation(b, Status.ESTABLISHED, True, tonight=_test(b, 0.002), established_on=first),
            ),
        ),
        (),
    )
    store.write(
        LaneState(
            "session",
            second,
            4,
            (
                Relation(a, Status.ESTABLISHED, True, tonight=_test(a, 0.0005), established_on=second),
                Relation(child, Status.CANDIDATE, True, tonight=_test(child, 0.004), discovered_on=second),
            ),
        ),
        (
            Transition(second, a, Status.CANDIDATE, Status.ESTABLISHED, "前向验证复现"),
            Transition(second, child, None, Status.CANDIDATE, "第 1–3 道都过了"),
        ),
    )
    section, changes = NightReports(tmp_path / "reports", artifact_root=tmp_path, relations=store).relation_changes(
        second
    )
    lane = section["session"]
    assert isinstance(lane, dict) and lane["previous_night"] == first.isoformat()
    assert (lane["added"], lane["removed"], lane["status"]) == (1, 1, 1)
    rows = {row["key"]: row for row in lane["changes"]}
    assert rows[a.text]["before"] == "candidate" and rows[a.text]["after"] == "established"
    assert rows[a.text]["reason"] == "前向验证复现" and rows[a.text]["p_value"] == 0.0005
    assert rows[a.text]["minimum_p"] == pytest.approx(0.00005) and rows[a.text]["family"] == "plain"
    assert rows[a.text]["entry_path"] == "outside"  # 跨夜带着的候选不进族
    assert rows[b.text]["kind"] == "removed" and rows[b.text]["after"] is None
    assert rows[child.text]["kind"] == "added" and rows[child.text]["family"] == "conditioned"
    assert rows[child.text]["entry_path"] == "parent"  # 父关系当晚过了第 1 道
    assert len(changes) == 3


def test_relation_changes_reach_the_metrics(tmp_path) -> None:
    registry = MetricRegistry()
    registry.record(
        ObservationEvent(
            "scene",
            "relation_change",
            ObservationStatus.SUCCESS,
            0.0,
            {
                "lane": "session",
                "kind": "added",
                "status_from": "none",
                "status_to": "candidate",
                "segment": "chain",
                "conditioned": True,
                "count": 4,
            },
        )
    )
    assert (
        'habitus_scene_relations_changed_total{lane="session",kind="added",status_from="none",status_to="candidate",'
        'segment="chain",conditioned="true"} 4'
    ) in registry.prometheus_text()


def test_the_health_surface_reads_the_ledger_after_a_restart(tmp_path) -> None:
    """共享说明 #3、#5：健康面读 Job 账（不读内存）：换一个执行器（重启）照样看得到上一晚停在哪、为什么。"""

    from habitus.runtime.health import _night_line_checks

    order: list[str] = []
    run = wired(tmp_path, order, settle_fails=lambda: True)
    asyncio.run(worker_for(run, tmp_path).look())
    restarted = worker_for(run, tmp_path)
    (check,) = _night_line_checks(restarted.jobs)
    assert check.name == "nightly_line" and "settle stopped: 结算出错: OSError (next attempt" in check.detail


def test_the_nightly_operating_numbers_live_in_the_unified_config() -> None:
    """共享说明 #12：原 ``POLL_SECONDS``、``KEPT_NIGHTS`` 与新加的租约、退避、次数都在 ``scene`` 配置里，示例 YAML 列全。"""

    from pathlib import Path

    import habitus.config as config_package

    defaults = SceneConfig()
    assert (defaults.nightly_poll_seconds, defaults.nightly_kept_nights) == (300.0, 366)
    assert (defaults.nightly_lease_ttl_seconds, defaults.nightly_heartbeat_seconds) == (120.0, 30.0)
    assert (defaults.nightly_max_attempts, defaults.nightly_retry_base_seconds, defaults.nightly_retry_max_seconds) == (
        3,
        900.0,
        3600.0,
    )
    for field, value in (
        ("nightly_poll_seconds", 0.5),
        ("nightly_kept_nights", 0),
        ("nightly_heartbeat_seconds", 200.0),
        ("nightly_max_attempts", 0),
        ("nightly_retry_base_seconds", 7200.0),
    ):
        with pytest.raises(ValueError, match=field):
            SceneConfig.from_mapping({field: value})
    example = (Path(config_package.__file__).parent / "example.yaml").read_text(encoding="utf-8")
    for field in (
        "nightly_poll_seconds",
        "nightly_kept_nights",
        "nightly_lease_ttl_seconds",
        "nightly_heartbeat_seconds",
        "nightly_max_attempts",
        "nightly_retry_base_seconds",
        "nightly_retry_max_seconds",
    ):
        assert f"#   {field}:" in example


def test_settling_with_a_held_result_requeues_without_spending_an_attempt() -> None:
    ledger = NightLedger.fresh(date(2026, 8, 18), NOW)
    token = claim(ledger, PREPARE, worker_id="w", now=NOW)
    ledger = claimed(ledger, token, now=NOW, ttl_seconds=60)
    assert ledger.job(PREPARE).state == RUNNING and ledger.job(PREPARE).attempts == 1
    held = settled(
        ledger,
        token,
        StepResult(PREPARE, "held", "锁被另一进程拿着"),
        now=NOW,
        duration_seconds=0.5,
        backoff=NightlySettings().backoff,
        max_attempts=3,
    )
    job = held.job(PREPARE)
    assert job.state == QUEUED and job.attempts == 0 and job.reason == "锁被另一进程拿着" and job.claim_id is None
    with pytest.raises(NightClaimLostError):
        claim(held, SETTLE, worker_id="w", now=NOW)  # 只能按先后认领


def _complete(night: date) -> NightLedger:
    ledger = NightLedger.fresh(night, NOW)
    return NightLedger(night, NOW, tuple(job.__class__(job.step, state=SUCCEEDED) for job in ledger.jobs))


def test_signals_with_user_text_stay_out_of_span_attributes_and_the_local_trace(tmp_path) -> None:
    """codex R3：信号是带用户派生文字的句子（概念名、关系两端名、迁移原因）——只进本地每晚报告，不进 OTLP 的 span 属性，
    本地 trace 文件里也没有；"模型调不通"后面的异常文字只留原因码。"""

    from habitus.infrastructure.observability.otel import _span_attributes
    from habitus.runtime.nightly_summary import reason_code
    from habitus.runtime.nightly_worker import flatten

    exported = _span_attributes(
        {"scene.signals": "relation: session 私人标签 → 私人后果", "scene.mapped": 3, **flatten({"signals": ["私人"]})}
    )
    assert exported == {"habitus.scene.mapped": 3}
    assert reason_code("模型调不通: provider said 私人内容") == "模型调不通"
    assert reason_code("这一晚过去了，没跑完（模型调不通: 私人）") == "这一晚过去了，没跑完（模型调不通）"
    assert reason_code("结算出错: OSError") == "结算出错: OSError"

    order: list[str] = []
    run = wired(tmp_path, order)
    worker = worker_for(run, tmp_path)
    asyncio.run(worker.look())
    report = json.loads((tmp_path / "nightly" / "reports" / f"{run.tonight().isoformat()}.json").read_text())
    signals = report["steps"]["structure"]["report"]["scene"]["signals"]
    assert any("「" in item for item in signals)  # 报告里照样有带概念名的信号
    text = (tmp_path / "nightly" / "traces" / f"{run.tonight().isoformat()}.jsonl").read_text()
    assert "signals" not in text and "「" not in text and "已同步" not in text


def test_the_unreachable_constant_matches_the_span_reason_code() -> None:
    from habitus.runtime.nightly import MODEL_UNREACHABLE
    from habitus.runtime.nightly_summary import UNREACHABLE

    assert UNREACHABLE == MODEL_UNREACHABLE


def test_a_busy_lease_renewal_is_retried_until_the_lease_would_expire(tmp_path) -> None:
    """验证轮"要人定"第 3 条：心跳一次续不上（SQLite busy）不取消正在跑的段，下一拍再续；只有续不上、租约就要到期才取消。"""

    from habitus.infrastructure.store.contracts.lock import LockLostError
    from habitus.runtime.nightly_worker import NightlyLeaseLostError

    class Flaky(ProcessLocalLockStore):
        def __init__(self, failures: int) -> None:
            super().__init__()
            self.failures = failures
            self.renewals = 0

        def renew(self, token, ttl_seconds=30):  # type: ignore[no-untyped-def]
            self.renewals += 1
            if self.renewals <= self.failures:
                raise LockLostError("Lock store busy while renewing")
            return super().renew(token, ttl_seconds)

    def slow_run(path, seconds):  # type: ignore[no-untyped-def]
        order: list[str] = []
        run = wired(path, order)

        async def vocabulary(_night):  # type: ignore[no-untyped-def]
            await asyncio.sleep(seconds)

        run.vocabulary_jobs = (vocabulary,)
        return run

    settings = NightlySettings(lease_ttl_seconds=2.0, heartbeat_seconds=0.2)
    busy_twice = Flaky(failures=2)
    worker = worker_for(slow_run(tmp_path / "a", 1.2), tmp_path / "a", lock_store=busy_twice, settings=settings)
    ledger = asyncio.run(worker.look())
    assert ledger is not None and ledger.complete and busy_twice.renewals >= 3

    never = Flaky(failures=10_000)
    lost = worker_for(slow_run(tmp_path / "b", 10.0), tmp_path / "b", lock_store=never, settings=settings)
    started = time.monotonic()
    with pytest.raises(NightlyLeaseLostError):
        asyncio.run(lost.look())
    assert 1.5 <= time.monotonic() - started < 2.0 + 1.0  # 撑到租约快到期才取消，不是第一次失败就取消
    stuck = lost.jobs.read(lost.nightly.tonight())
    assert stuck is not None and stuck.job(VOCABULARY).state == QUEUED  # 放回排队（放回的那次写在租约内）


def test_model_call_spans_name_the_backend_and_model_that_actually_answered(tmp_path) -> None:
    """``model.call`` 的后端与模型记答复上实际的那一对（适配器回退到别的后端时看得出来），路由配置的那一对另记。"""

    from habitus.model_client import ChatClient
    from habitus.model_client.contracts import ModelResponse
    from tests.unit.foresight.scripted_model import ScriptedProvider, model_config

    class Fallback(ScriptedProvider):
        async def complete_async(self, request):  # type: ignore[no-untyped-def]
            answered = await super().complete_async(request)
            return ModelResponse(content=answered.content, model="claude-sonnet-4-5", provider="claude")

    seen: list[dict[str, object]] = []

    class Sink:
        def started(self, record):  # type: ignore[no-untyped-def]
            return None

        def finished(self, record, handle):  # type: ignore[no-untyped-def]
            seen.append({"name": record.name, "calls": dict(record.model_calls), **record.attributes})

    client = ChatClient(model_config(), Fallback(["好"]), observer=ModelCallTracer())

    async def call() -> None:
        with open_trace(Sink()), span("nightly.vocabulary"):
            await client.complete_async("在吗")

    asyncio.run(call())
    call_span = next(item for item in seen if item["name"] == "model.call")
    assert call_span["provider"] == "claude" and call_span["model"] == "claude-sonnet-4-5"
    assert call_span["route_model"] == client.model and call_span["touch"] == "chat"
    outer = next(item for item in seen if item["name"] == "nightly.vocabulary")
    assert outer["calls"] == {("chat", "claude", "claude-sonnet-4-5"): 1}


def test_zero_counts_are_written_not_left_out() -> None:
    """计数为 0 的键照样写出：四族与按组、进族路径、各状态的新增 / 消失、按段跳过（分得清"是 0"还是"没记"）。"""

    from habitus.runtime.scene_night_relations import lane_attributes
    from habitus.scene.relations.engine import LaneNight
    from habitus.scene.relations.evaluation import _BREAKDOWN_KEYS, _families

    family, passed, limits, breakdown = _families({}, [], {}, q=0.1, outside=frozenset())
    assert set(breakdown) == set(_BREAKDOWN_KEYS) and set(breakdown.values()) == {0}
    state = LaneState("session", date(2026, 8, 18), 0, ())
    night = LaneNight("session", date(2026, 8, 18), (), 0, breakdown=dict.fromkeys(_BREAKDOWN_KEYS, 0))
    found = lane_attributes(night, None, state, (), {})
    assert found["groups_family"] == 0 and found["entry_carried"] == 0
    assert found["added.established"] == 0 and found["removed.candidate"] == 0
    assert found["skipped.chain"] == 0


def test_the_retry_cutoff_stays_the_night_before_the_job_even_across_midnight(tmp_path) -> None:
    """codex R4：前置处理的重试截止日固定为 Job 那一晚的前一天（裁定 68 第 3 条 2026-11-09 补记）——探模型、重读日历跨过零点
    之后才开始重试，也不按新的"今天"多处理一天。"""

    from datetime import UTC, datetime

    clock = [datetime(2026, 8, 18, 15, 59, 50, tzinfo=UTC)]  # 上海 23:59:50
    order: list[str] = []
    run = wired(tmp_path, order, clock=lambda: clock[0])
    night = run.tonight()
    assert night == date(2026, 8, 18)

    async def probe() -> None:
        clock[0] = datetime(2026, 8, 18, 16, 0, 30, tzinfo=UTC)  # 探针返回时已过零点

    run.probe = probe
    run.retry = StubRetry(order=order, label="重试到 {through}")  # type: ignore[assignment]
    result = asyncio.run(run.run_step(PREPARE, night))
    assert result.done and run.tonight() == date(2026, 8, 19)
    assert order == ["重试到 2026-08-17"]
