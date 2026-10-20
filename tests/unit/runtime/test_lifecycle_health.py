"""生命周期 Worker 的健康面：单个参与者失败、容量到告警线都报 degraded（非 critical），整体仍可就绪。"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

from habitus.infrastructure.lifecycle import CapacityReading, CapacitySnapshot, ParticipantReport
from habitus.infrastructure.store.contracts import PathLock
from habitus.infrastructure.store.locks import ProcessLocalLockStore
from habitus.runtime import (
    LifecycleMaintenanceCycleResult,
    LifecycleParticipantOutcome,
    RuntimeHealthService,
    build_runtime,
)
from tests.helpers import BASE_TIME
from tests.integration.test_runtime_assembly import runtime_config, runtime_dependencies


def _lifecycle_check(tmp_path: Path, cycle: LifecycleMaintenanceCycleResult | None):
    providers, vectors = runtime_dependencies()
    runtime = build_runtime(
        runtime_config(tmp_path),
        providers=providers,
        vector_stores=vectors,
        path_lock=PathLock(ProcessLocalLockStore()),
    )
    worker = runtime.components.workflow.lifecycle_worker
    worker._last_cycle = cycle  # 只读健康面的输入；一轮维护要真实 lease 与全部参与者，这里不必跑
    return RuntimeHealthService(runtime.components)._lifecycle_check()


def _cycle(*outcomes: LifecycleParticipantOutcome) -> LifecycleMaintenanceCycleResult:
    return LifecycleMaintenanceCycleResult(
        lease_acquired=True, started_at=BASE_TIME, finished_at=BASE_TIME + timedelta(seconds=1), outcomes=outcomes
    )


def test_failed_participant_degrades_without_blocking_readiness(tmp_path: Path) -> None:
    check = _lifecycle_check(
        tmp_path,
        _cycle(
            LifecycleParticipantOutcome("conversation-chain", report=ParticipantReport()),
            LifecycleParticipantOutcome("memory-l2", error_type="RuntimeError", message="boom"),
        ),
    )
    assert check.status.value == "degraded"
    assert not check.critical
    assert "memory-l2: RuntimeError" in check.detail


def test_all_participants_ok_is_healthy(tmp_path: Path) -> None:
    check = _lifecycle_check(
        tmp_path, _cycle(LifecycleParticipantOutcome("conversation-chain", report=ParticipantReport()))
    )
    assert check.status.value == "healthy"


def test_participant_signals_show_up_on_the_health_surface(tmp_path: Path) -> None:
    """参与者报的"要人看的情况"原先没人读（L10.10）：现在报 degraded（非 critical），内容带参与者名。"""

    signal = "1 conversation source deliveries are broken and wait for repair"
    check = _lifecycle_check(
        tmp_path,
        _cycle(
            LifecycleParticipantOutcome("conversation-source-recovery", report=ParticipantReport(signals=(signal,)))
        ),
    )
    assert check.status.value == "degraded"
    assert not check.critical
    assert f"conversation-source-recovery: {signal}" in check.detail


def _capacity_check(tmp_path: Path, *readings: CapacityReading):
    providers, vectors = runtime_dependencies()
    runtime = build_runtime(
        runtime_config(tmp_path),
        providers=providers,
        vector_stores=vectors,
        path_lock=PathLock(ProcessLocalLockStore()),
    )
    store = runtime.components.workflow.lifecycle_worker.capacity
    assert store is not None
    store.publish(CapacitySnapshot("sources", BASE_TIME, readings))
    return RuntimeHealthService(runtime.components, capacity_warning_share=0.8)._lifecycle_capacity_check()


def test_capacity_past_the_warning_share_degrades_without_blocking_readiness(tmp_path: Path) -> None:
    check = _capacity_check(tmp_path, CapacityReading("envelopes", 850, 1000), CapacityReading("locks", 5))
    assert check.status.value == "degraded"
    assert not check.critical
    assert "sources.envelopes: 850/1000" in check.detail


def test_capacity_below_the_warning_share_is_healthy(tmp_path: Path) -> None:
    check = _capacity_check(tmp_path, CapacityReading("envelopes", 100, 1000))
    assert check.status.value == "healthy"
