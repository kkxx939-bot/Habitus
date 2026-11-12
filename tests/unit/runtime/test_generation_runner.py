"""换代的那一眼（实时化方案第 3、9、10 节；阶段 2 第二小步起不是常驻 worker，由预测 worker 每槽在预测根租约之内调）：同一把
预测根租约下同一天只做一次、进程起来第一眼撤回剩下的重试项、定稿日不连续连着几眼就报卡住的那一天、每代一份报告、trace 里各步子 span 按先后挂在 ``generation.run`` 下、指标有每步耗时与各个退路的计数。"""

from __future__ import annotations

import asyncio
from datetime import timedelta

from habitus.foundation.observability import ObservationEvent
from habitus.infrastructure.observability import MetricRegistry
from habitus.runtime.generation_lease import Lease
from habitus.runtime.generation_report import GenerationReports
from habitus.runtime.generation_result import FALLBACKS, STEPS
from habitus.runtime.generation_runner import GenerationRunner, GenerationSettings
from habitus.runtime.generation_trace import GENERATION_SPAN, TraceFiles
from habitus.scene.relations.store import RelationStore
from tests.lock_store import ProcessLocalLockStore
from tests.unit.runtime.test_generation import TODAY, StubRetry, wired


class Recorder:
    def __init__(self) -> None:
        self.events: list[ObservationEvent] = []

    def record(self, event: ObservationEvent) -> None:
        self.events.append(event)


def worker_for(run, tmp_path, *, observer=None, stuck_ticks=3):  # type: ignore[no-untyped-def]
    root = tmp_path / "scene" / "generation"
    return GenerationRunner(
        run,
        settings=GenerationSettings(stuck_ticks=stuck_ticks),
        reports=GenerationReports(root / "reports", artifact_root=tmp_path, relations=RelationStore(tmp_path / "scene")),
        traces=TraceFiles(root / "traces", artifact_root=tmp_path),
        observer=observer,
    )


def test_two_workers_on_one_root_generate_the_day_once_and_clean_leftovers_once(tmp_path) -> None:
    """E10：两个进程挂同一个根（各一把预测根租约、共用租约库，像预测 worker 那样在租约之内看一眼）：同一天只做一次；
    进程起来第一眼撤回剩下的重试项，只撤一次。"""

    order: list[str] = []
    run = wired(tmp_path, order)
    locks = ProcessLocalLockStore()
    first, second = worker_for(run, tmp_path), worker_for(run, tmp_path)

    async def under_lease(runner):  # type: ignore[no-untyped-def]
        async with Lease(locks, "foresight:root:test", ttl_seconds=30.0, heartbeat_seconds=5.0, poll_seconds=0.05).held():
            return await runner.look()

    async def both():  # type: ignore[no-untyped-def]
        return await asyncio.gather(under_lease(first), under_lease(second))

    results = asyncio.run(both())
    assert sum(result is not None for result in results) == 1 and order.count("机会表") == 1
    assert asyncio.run(first.look()) is None  # 完成标记在：不再做
    retry = run.retry
    assert isinstance(retry, StubRetry) and retry.cleaned >= 1
    cleaned = retry.cleaned
    asyncio.run(first.look())
    assert retry.cleaned == cleaned


def test_a_stuck_day_is_reported_after_the_configured_looks(tmp_path) -> None:
    order: list[str] = []
    yesterday = TODAY - timedelta(days=1)
    run = wired(tmp_path, order, closed_days=lambda: (yesterday - timedelta(days=2), yesterday))
    worker = worker_for(run, tmp_path, stuck_ticks=2)
    asyncio.run(worker.look())
    assert worker.stuck is None  # 正常只持续一轮归约
    asyncio.run(worker.look())
    assert worker.stuck == yesterday - timedelta(days=1)


def test_each_step_is_a_child_span_in_order_and_the_report_and_metrics_carry_the_fallbacks(tmp_path) -> None:
    """E17 / 观测验收：trace 里每步一个子 span、按先后挂在 ``generation.run`` 下；每代一份报告；指标有每步耗时与退路计数。"""

    order: list[str] = []
    run = wired(tmp_path, order)
    run.retry = StubRetry(OSError("disk"), order=order)  # type: ignore[assignment]  # 结算留到下次
    registry = MetricRegistry()
    recorder = Recorder()

    class Both:
        def record(self, event: ObservationEvent) -> None:
            registry.record(event)
            recorder.record(event)

    worker = worker_for(run, tmp_path, observer=Both())
    result = asyncio.run(worker.look())
    assert result is not None and result.complete and worker.last_error is not None  # 退路照样报到健康面
    spans = worker.traces.read(TODAY)  # type: ignore[union-attr]
    (parent,) = [item for item in spans if item["name"] == GENERATION_SPAN]
    children = [item for item in spans if item["parent_id"] == parent["span_id"]]
    assert [item["name"] for item in children] == [f"generation.{step}" for step in STEPS]
    assert [item["started_at"] for item in children] == sorted(item["started_at"] for item in children)
    assert parent["attributes"]["fallback.settlement_deferred"] == 1
    report = worker.reports.read(TODAY)  # type: ignore[union-attr]
    assert report is not None and report["complete"] is True and report["fallbacks"]["settlement_deferred"] == 1
    snapshot = registry.snapshot()
    assert snapshot.counters["generation_fallbacks_total[kind=settlement_deferred]"] == 1
    assert snapshot.counters["generation_runs_total[outcome=complete]"] == 1
    assert {event.attributes["kind"] for event in recorder.events if event.operation == "fallback"} == set(FALLBACKS)
    assert any(key.startswith("generation_step_duration_seconds[step=opportunities]") for key in snapshot.duration_seconds)


def test_a_failed_must_step_is_retried_the_same_day_with_a_bounded_backoff_and_a_busy_lock_is_not(tmp_path) -> None:
    """F3（用户 11-10 失败重试第 2 条）：必须成的那一步没成，同一天按退避一直重试（5 分钟起、逐次加倍、封顶 1 小时、不放弃），
    不再每一眼从头重做整代；健康面报出下一次重试的时刻。锁被占（还没准备好）不算失败、不退避。写成了清零。"""

    from datetime import datetime

    from habitus.runtime.foresight_days import DayReport
    from habitus.runtime.foresight_forming import Formed
    from habitus.runtime.health import _generation_state_checks
    from tests.unit.runtime.test_generation import NOW

    order: list[str] = []
    clock = [NOW]
    run = wired(tmp_path, order, clock=lambda: clock[0])
    stale = [True]

    class Days:
        async def run_once(self):  # type: ignore[no-untyped-def]
            order.append("日子")
            return DayReport(datetime.now().astimezone(), Formed(), "今天这一代的机会表还没有" if stale[0] else None)

    run.days = Days()  # type: ignore[assignment]
    worker = worker_for(run, tmp_path)
    worker._clock = lambda: clock[0]  # type: ignore[method-assign]
    first = asyncio.run(worker.look())
    assert first is not None and first.failed == "days"
    assert worker.retry_at == NOW + timedelta(seconds=300)
    (check,) = [item for item in _generation_state_checks(worker) if item.name == "generation"]
    assert "next retry at" in check.detail
    order.clear()
    assert asyncio.run(worker.look()) is None and order == []  # 还没到点：这一眼不做
    clock[0] = NOW + timedelta(seconds=301)
    assert asyncio.run(worker.look()) is not None
    assert worker.retry_at == clock[0] + timedelta(seconds=600)  # 逐次加倍
    stale[0] = False
    clock[0] += timedelta(seconds=601)
    done = asyncio.run(worker.look())
    assert done is not None and done.complete and worker.retry_at is None


def test_a_held_generation_does_not_back_off(tmp_path) -> None:
    from habitus.behavior.reduction import BehaviorReductionBusyError

    order: list[str] = []
    migrating = [True]  # 修复轮 G1：有没做完的迁移时才去拿 sweep 锁
    run = wired(tmp_path, order, vocabulary=lambda: (1, migrating[0]))
    busy = [True]

    async def migrate() -> None:  # A1：换代里会碰 sweep 锁的只剩续做迁移（每晚新增归模型队列）
        if busy[0]:
            raise BehaviorReductionBusyError("sweep lock held")
        migrating[0] = False

    run.migration_jobs = (migrate,)
    worker = worker_for(run, tmp_path)
    held = asyncio.run(worker.look())
    assert held is not None and held.held == "migrate" and worker.retry_at is None and worker.last_error is None
    busy[0] = False
    assert asyncio.run(worker.look()) is not None and run.marks.finished(TODAY) is True  # 下一眼从头做、写成
