"""Runtime 健康、就绪和依赖状态的只读应用服务。"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from enum import Enum
from zoneinfo import ZoneInfo

from habitus.foresight import ForesightError
from habitus.memory.workflow import MemoryJobStatus
from habitus.runtime.components import RuntimeComponents
from habitus.runtime.foresight_opportunities import OpportunityStore
from habitus.runtime.generation_marks import GenerationMarkError
from habitus.runtime.generation_result import REWOUND
from habitus.runtime.generation_worker import GenerationWorker
from habitus.runtime.lifecycle import LifecycleWorkerState
from habitus.runtime.resident import ResidentWorker
from habitus.runtime.worker import MemoryWorkerState

_SIGNAL_DETAIL_LIMIT = 1_000
#: 只关记忆写入侧的检查：后台 Job 停住（失败待处置、或在等模型恢复）不影响已有记忆的召回。
_WRITE_SIDE_CHECKS = frozenset({"memory_worker", "memory_queue"})


class RuntimeHealthStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CLOSED = "closed"


@dataclass(frozen=True)
class RuntimeHealthCheck:
    name: str
    status: RuntimeHealthStatus
    detail: str
    critical: bool = True


@dataclass(frozen=True)
class RuntimeHealthReport:
    status: RuntimeHealthStatus
    ready: bool
    checked_at: datetime
    checks: tuple[RuntimeHealthCheck, ...]
    #: 召回所需的就绪：与 ``ready`` 同一判法，但不看只关写入侧的检查（记忆 Job 停住时召回照常可用）
    recall_ready: bool


class RuntimeHealthService:
    """聚合现有组件状态，不触发索引重建或业务写入。"""

    def __init__(
        self, components: RuntimeComponents, capacity_warning_share: float = 0.8, zone: ZoneInfo | None = None
    ) -> None:
        if not 0.0 < capacity_warning_share < 1.0:
            raise ValueError("capacity_warning_share must lie in (0, 1)")
        self.components = components
        self.capacity_warning_share = float(capacity_warning_share)
        self.zone = zone  # 配置的主体时区（``locale.timezone``）：与系统时区比（``timezone_checks``）；没给不比

    async def report(self, runtime_state: str, *, deep: bool = False) -> RuntimeHealthReport:
        if not runtime_state:
            raise ValueError("runtime_state must be non-empty text")
        checks = [
            self._runtime_check(runtime_state),
            self._lifecycle_storage_check(runtime_state),
            self._memory_worker_check(),
            self._lifecycle_check(),
            self._lifecycle_capacity_check(),
            self._observability_check(),
            *([] if self.zone is None else timezone_checks(self.zone, datetime.now().astimezone())),
        ]
        checks.extend(self._behavior_checks(runtime_state))
        checks.extend(self._generation_checks(runtime_state))
        checks.extend(self._foresight_checks(runtime_state))
        checks.append(await self._queue_check())
        checks.extend(
            await asyncio.gather(
                self._vector_check("memory_vector", self.components.memory.vector_index.store),
                self._vector_check("summary_vector", self.components.conversation.summary_vector_index.store),
            )
        )
        if deep:
            checks.extend(
                await asyncio.gather(
                    self._chat_check(),
                    _index_consistency_check(
                        "memory_index_consistency",
                        self.components.memory.vector_index,
                    ),
                    _index_consistency_check(
                        "summary_index_consistency",
                        self.components.conversation.summary_vector_index,
                    ),
                )
            )
        critical_unhealthy = any(
            check.critical and check.status in {RuntimeHealthStatus.UNHEALTHY, RuntimeHealthStatus.CLOSED}
            for check in checks
        )
        critical_degraded = any(check.critical and check.status is RuntimeHealthStatus.DEGRADED for check in checks)
        if runtime_state == "closed":
            status = RuntimeHealthStatus.CLOSED
        elif critical_unhealthy:
            status = RuntimeHealthStatus.UNHEALTHY
        elif critical_degraded:
            status = RuntimeHealthStatus.DEGRADED
        else:
            status = RuntimeHealthStatus.HEALTHY
        ready = runtime_state in {"ready", "running"} and not critical_unhealthy and not critical_degraded
        recall_ready = runtime_state in {"ready", "running"} and not any(
            check.critical and check.name not in _WRITE_SIDE_CHECKS and check.status is not RuntimeHealthStatus.HEALTHY
            for check in checks
        )
        return RuntimeHealthReport(status, ready, datetime.now(UTC), tuple(checks), recall_ready)

    def _behavior_checks(self, runtime_state: str) -> list[RuntimeHealthCheck]:
        """行为管线的健康面；**non-critical**——行为失败不阻断记忆主链的既定哲学在健康面同样成立。

        未启用时如实报 disabled（healthy）；启用后暴露两条循环的存活与最近错误——否则
        "启动失败不阻断"+"循环异常只进 last_error"的组合等于死活无人知（评审点名）。
        """

        behavior = self.components.behavior
        if behavior is None:
            return [RuntimeHealthCheck("behavior", RuntimeHealthStatus.HEALTHY, "disabled", critical=False)]
        checks = self._worker_checks(
            runtime_state,
            tuple((f"behavior_reduction_worker_{branch.lane.value}", branch.reduction_worker) for branch in behavior),
        )
        for branch in behavior:
            pinned = branch.reduction_runner.pinned
            if pinned:
                # 封口视界被哪几份观测钉住：比它晚的观测已被回执覆盖、它没有（裁定 94 第 8 条）——那天及以后定不了稿；逐份列出
                checks.append(
                    RuntimeHealthCheck(
                        f"behavior_seal_horizon_{branch.lane.value}",
                        RuntimeHealthStatus.DEGRADED,
                        "; ".join(
                            f"pinned by observation delivery {item.source_id} since {item.since.isoformat()}"
                            for item in pinned
                        )[:1_000],
                        critical=False,
                    )
                )
        return checks

    def _generation_checks(self, runtime_state: str) -> list[RuntimeHealthCheck]:
        """换代的健康面；同样 non-critical。换代死掉不会让任何请求出错——它只是让语义树和机会表停在昨天，而这**恰恰是最难自己发现的故障**：白天照常递送，只是依据
        越来越旧、新的预期一条都不建。所以循环存活必须上健康面；另报：今天没有这一代（必须成的那一步没成）、走了退路的步、
        定稿日不连续卡住的那一天、最新的完成标记读不了；接了预测层时再看机会表最晚一晚有多旧（清单 5.3）。
        """

        worker = self.components.generation
        if worker is None:
            return [RuntimeHealthCheck("generation", RuntimeHealthStatus.HEALTHY, "disabled", critical=False)]
        checks = self._worker_checks(runtime_state, (("generation_worker", worker),))
        checks.extend(_generation_state_checks(worker))
        calendar = getattr(getattr(worker.generation, "scene", None), "calendar", None)
        problems = tuple(getattr(calendar, "problems", ()))
        if problems:
            # 放假数据有年份读不了：那几年日型未知（建预期按名义周末），进程照常——但要让人看见、去改文件
            checks.append(
                RuntimeHealthCheck("calendar", RuntimeHealthStatus.DEGRADED, "; ".join(problems)[:500], critical=False)
            )
        foresight = self.components.foresight
        if foresight is not None:
            closed_days = getattr(worker.generation, "closed_days", None)
            checks.append(self._opportunity_freshness_check(foresight.opportunities, closed_days))
        return checks

    def _foresight_checks(self, runtime_state: str) -> list[RuntimeHealthCheck]:
        """预测层每槽一拍的循环；同样 non-critical。它死掉不会让任何请求出错——只是不再有判断。"""

        foresight = self.components.foresight
        if foresight is None:
            return [RuntimeHealthCheck("foresight", RuntimeHealthStatus.HEALTHY, "disabled", critical=False)]
        if foresight.worker is None:
            # 触发与递法还没接上：只有账本与换代里的结算，没有常驻循环可查
            return [RuntimeHealthCheck("foresight", RuntimeHealthStatus.HEALTHY, "ledger only", critical=False)]
        return self._worker_checks(runtime_state, (("foresight_worker", foresight.worker),))

    @staticmethod
    def _opportunity_freshness_check(
        store: OpportunityStore, closed_days: Callable[[], Iterable[date]] | None
    ) -> RuntimeHealthCheck:
        """机会表有多旧。循环活着不等于换代在产出——这正是上面那段 docstring 说最难自己发现的故障。

        不写死周期（裁定 68 第 4 条）：换代在前一天定稿之后做，所以该有的最晚一代是"最后定稿日 + 1"；盘上最晚那张表比它早就是
        陈旧（换代没做、没做成）。还没写过任何一代、还没有定稿日都不算故障：行为侧刚接入、第一代还没到。
        """

        lane = "session"
        try:
            night = store.latest_night(lane)
        except ForesightError as exc:
            return RuntimeHealthCheck(
                f"opportunities_{lane}",
                RuntimeHealthStatus.UNHEALTHY,
                f"unreadable:{type(exc).__name__}",
                critical=False,
            )
        if night is None:
            return RuntimeHealthCheck(
                f"opportunities_{lane}", RuntimeHealthStatus.HEALTHY, "never_written", critical=False
            )
        closed = sorted(closed_days()) if closed_days is not None else []
        if closed and night < closed[-1] + timedelta(days=1):
            expected = closed[-1] + timedelta(days=1)
            return RuntimeHealthCheck(
                f"opportunities_{lane}",
                RuntimeHealthStatus.DEGRADED,
                f"stale:night {night} behind {expected}",
                critical=False,
            )
        return RuntimeHealthCheck(
            f"opportunities_{lane}", RuntimeHealthStatus.HEALTHY, f"night:{night}", critical=False
        )

    @staticmethod
    def _worker_checks(runtime_state: str, workers: tuple[tuple[str, ResidentWorker], ...]) -> list[RuntimeHealthCheck]:
        """一组常驻循环的存活面：出过错报 degraded，该转不转报 unhealthy。"""

        checks: list[RuntimeHealthCheck] = []
        for name, worker in workers:
            if worker.last_error is not None:
                checks.append(
                    RuntimeHealthCheck(
                        name,
                        RuntimeHealthStatus.DEGRADED,
                        f"last_error:{type(worker.last_error).__name__}",
                        critical=False,
                    )
                )
            elif runtime_state == "running" and not worker.running:
                checks.append(
                    RuntimeHealthCheck(name, RuntimeHealthStatus.UNHEALTHY, "loop_not_running", critical=False)
                )
            else:
                checks.append(
                    RuntimeHealthCheck(
                        name,
                        RuntimeHealthStatus.HEALTHY,
                        "running" if worker.running else "stopped",
                        critical=False,
                    )
                )
        return checks

    @staticmethod
    def _runtime_check(state: str) -> RuntimeHealthCheck:
        if state == "closed":
            return RuntimeHealthCheck("runtime", RuntimeHealthStatus.CLOSED, state)
        if state == "created":
            return RuntimeHealthCheck("runtime", RuntimeHealthStatus.DEGRADED, "not_initialized")
        return RuntimeHealthCheck("runtime", RuntimeHealthStatus.HEALTHY, state)

    def _memory_worker_check(self) -> RuntimeHealthCheck:
        worker = self.components.workflow.worker
        if worker.state is MemoryWorkerState.FAILED:
            return RuntimeHealthCheck("memory_worker", RuntimeHealthStatus.UNHEALTHY, self._error(worker.last_error))
        if worker.state is MemoryWorkerState.BLOCKED:
            return RuntimeHealthCheck("memory_worker", RuntimeHealthStatus.DEGRADED, self._error(worker.last_error))
        return RuntimeHealthCheck("memory_worker", RuntimeHealthStatus.HEALTHY, worker.state.value)

    def _lifecycle_check(self) -> RuntimeHealthCheck:
        worker = self.components.workflow.lifecycle_worker
        if worker.state is LifecycleWorkerState.FAILED:
            return RuntimeHealthCheck("lifecycle_worker", RuntimeHealthStatus.UNHEALTHY, self._error(worker.last_error))
        if worker.last_error is not None:
            return RuntimeHealthCheck(
                "lifecycle_worker", RuntimeHealthStatus.DEGRADED, self._error(worker.last_error), critical=False
            )
        failed = () if worker.last_cycle is None else worker.last_cycle.failures
        if failed:
            detail = "; ".join(f"{item.name}: {item.error_type}" for item in failed)
            return RuntimeHealthCheck("lifecycle_worker", RuntimeHealthStatus.DEGRADED, detail, critical=False)
        signals = () if worker.last_cycle is None else worker.last_cycle.signals
        if signals:
            # 参与者报的"要人看的情况"（坏掉待修的源、隔离了的交付……）：不影响就绪，但要看得见
            detail = "; ".join(signals)[:_SIGNAL_DETAIL_LIMIT]
            return RuntimeHealthCheck("lifecycle_worker", RuntimeHealthStatus.DEGRADED, detail, critical=False)
        return RuntimeHealthCheck("lifecycle_worker", RuntimeHealthStatus.HEALTHY, worker.state.value, critical=False)

    def _lifecycle_capacity_check(self) -> RuntimeHealthCheck:
        """只读各参与者上一轮落盘的容量快照，不自己扫存储；到告警线报 degraded（不影响就绪）。"""

        store = self.components.workflow.lifecycle_worker.capacity
        if store is None:
            return RuntimeHealthCheck(
                "lifecycle_capacity", RuntimeHealthStatus.HEALTHY, "not configured", critical=False
            )
        try:
            snapshots = store.read_all()
        except Exception as exc:  # noqa: BLE001 - 快照坏了只是看不见水位，不该让健康检查本身失败
            return RuntimeHealthCheck(
                "lifecycle_capacity", RuntimeHealthStatus.DEGRADED, self._error(exc), critical=False
            )
        crowded = [
            f"{snapshot.participant}.{reading.name}: {reading.count}/{reading.limit}"
            for snapshot in snapshots
            for reading in snapshot.readings
            if reading.share is not None and reading.share >= self.capacity_warning_share
        ]
        if crowded:
            return RuntimeHealthCheck(
                "lifecycle_capacity", RuntimeHealthStatus.DEGRADED, "; ".join(crowded), critical=False
            )
        return RuntimeHealthCheck(
            "lifecycle_capacity", RuntimeHealthStatus.HEALTHY, f"{len(snapshots)} snapshots", critical=False
        )

    def _lifecycle_storage_check(self, runtime_state: str) -> RuntimeHealthCheck:
        recall_store = self.components.memory.search.recall_lifecycle.store
        summary_use = self.components.conversation.summary_use
        if runtime_state == "created":
            return RuntimeHealthCheck(
                "lifecycle_storage",
                RuntimeHealthStatus.DEGRADED,
                "not_initialized",
            )
        if not getattr(recall_store, "initialized", False) or not summary_use.initialized:
            return RuntimeHealthCheck(
                "lifecycle_storage",
                RuntimeHealthStatus.UNHEALTHY,
                "not_initialized",
            )
        try:
            pending_l2 = len(self.components.memory.lifecycle.operation_store.pending())
            pending_summaries = len(self.components.workflow.lifecycle.retirement_store.pending())
        except Exception as exc:
            return RuntimeHealthCheck(
                "lifecycle_storage",
                RuntimeHealthStatus.UNHEALTHY,
                type(exc).__name__,
            )
        return RuntimeHealthCheck(
            "lifecycle_storage",
            RuntimeHealthStatus.HEALTHY,
            f"l2_pending={pending_l2};summary_pending={pending_summaries}",
        )

    def _observability_check(self) -> RuntimeHealthCheck:
        manager = self.components.infrastructure.managed_observability
        if manager is None:
            return RuntimeHealthCheck(
                "observability",
                RuntimeHealthStatus.DEGRADED,
                "not_configured",
                critical=False,
            )
        status, detail = manager.health()
        resolved = RuntimeHealthStatus.HEALTHY if status == "healthy" else RuntimeHealthStatus.DEGRADED
        return RuntimeHealthCheck("observability", resolved, detail, critical=False)

    async def _queue_check(self) -> RuntimeHealthCheck:
        try:
            job = await asyncio.to_thread(self.components.workflow.jobs.oldest_uncommitted)
        except Exception as exc:
            return RuntimeHealthCheck("memory_queue", RuntimeHealthStatus.UNHEALTHY, type(exc).__name__)
        if job is None:
            self.components.infrastructure.observability.set_gauge("memory_queue_blocked", 0)
            return RuntimeHealthCheck("memory_queue", RuntimeHealthStatus.HEALTHY, "empty")
        if job.status is MemoryJobStatus.FAILED:
            self.components.infrastructure.observability.set_gauge("memory_queue_blocked", 1)
            return RuntimeHealthCheck("memory_queue", RuntimeHealthStatus.DEGRADED, f"failed:{job.memory_sequence}")
        if job.waiting_for_model:
            # 模型此刻用不了（额度、鉴权、断网……）：不是失败，长退避里等它恢复；要人看的多半是换 key 或充值
            assert job.next_attempt_at is not None
            self.components.infrastructure.observability.set_gauge("memory_queue_blocked", 1)
            return RuntimeHealthCheck(
                "memory_queue",
                RuntimeHealthStatus.DEGRADED,
                f"waiting_for_model:{job.memory_sequence};waits={job.model_waits};"
                f"next_attempt_at={job.next_attempt_at.isoformat()}",
            )
        self.components.infrastructure.observability.set_gauge("memory_queue_blocked", 0)
        return RuntimeHealthCheck(
            "memory_queue", RuntimeHealthStatus.HEALTHY, f"{job.status.value}:{job.memory_sequence}"
        )

    @staticmethod
    async def _vector_check(name: str, store: object) -> RuntimeHealthCheck:
        try:
            state = await store.state()  # type: ignore[attr-defined]
        except Exception as exc:
            return RuntimeHealthCheck(name, RuntimeHealthStatus.UNHEALTHY, type(exc).__name__)
        if state is None:
            return RuntimeHealthCheck(name, RuntimeHealthStatus.DEGRADED, "not_published")
        if not state.ready:
            return RuntimeHealthCheck(name, RuntimeHealthStatus.DEGRADED, "not_ready")
        return RuntimeHealthCheck(
            name, RuntimeHealthStatus.HEALTHY, f"generation={state.generation};records={state.record_count}"
        )

    async def _chat_check(self) -> RuntimeHealthCheck:
        result = await asyncio.to_thread(self.components.models.chat.health_check)
        if result.get("ok") is True:
            return RuntimeHealthCheck("chat_model", RuntimeHealthStatus.HEALTHY, "available", critical=False)
        return RuntimeHealthCheck(
            "chat_model",
            RuntimeHealthStatus.DEGRADED,
            str(result.get("error_code", "unavailable"))[:256],
            critical=False,
        )

    @staticmethod
    def _error(error: BaseException | None) -> str:
        return "none" if error is None else type(error).__name__


async def _index_consistency_check(name: str, index: object) -> RuntimeHealthCheck:
    """执行无副作用的真相源/派生索引审计，不在健康检查中重建。"""

    try:
        state = await index.store.state()  # type: ignore[attr-defined]
        if state is None:
            return RuntimeHealthCheck(name, RuntimeHealthStatus.DEGRADED, "not_published")
        if not state.ready:
            return RuntimeHealthCheck(name, RuntimeHealthStatus.DEGRADED, "not_ready")
        report = await index.audit_consistency()  # type: ignore[attr-defined]
    except Exception as exc:
        return RuntimeHealthCheck(name, RuntimeHealthStatus.UNHEALTHY, type(exc).__name__)
    if report.ok:
        return RuntimeHealthCheck(
            name,
            RuntimeHealthStatus.HEALTHY,
            f"expected={report.expected_count};indexed={report.indexed_count}",
        )
    return RuntimeHealthCheck(
        name,
        RuntimeHealthStatus.UNHEALTHY,
        (
            f"missing={len(report.missing_identities)};"
            f"stale={len(report.stale_identities)};"
            f"orphan={len(report.orphan_identities)}"
        ),
    )


def timezone_checks(zone: ZoneInfo, system: datetime) -> list[RuntimeHealthCheck]:
    """系统时区与配置时区（non-critical；用户 11-10 换时区，方案 19 第 6.1 节；不管换代接没接都比，用户 11-10）：运行时一切按
    配置的 ``locale.timezone`` 算（今天、槽、闸），``system`` 是此刻按系统本地时区的读法，系统时区只影响人在系统上看到的钟点——``system``（此刻、按系统本地时区）与它
    此刻的 UTC 偏移不一致时报出两边的名字与偏移，人可能改了一边忘了另一边。只比此刻的偏移：名字不同、偏移相同（Asia/Shanghai
    与 Asia/Singapore）不报。"""

    configured = system.astimezone(zone)
    if system.utcoffset() == configured.utcoffset():
        return []
    detail = f"system timezone {_system_zone_name(system)} {system:%z} differs from locale.timezone {zone.key} {configured:%z}"
    return [RuntimeHealthCheck("timezone", RuntimeHealthStatus.DEGRADED, detail, critical=False)]


def _system_zone_name(moment: datetime) -> str:
    """系统时区的名字：能拿到 IANA 名（tzlocal）就用它，拿不到用系统给的缩写。"""

    try:
        from tzlocal import get_localzone_name

        name = get_localzone_name()
    except Exception:  # noqa: BLE001 - tzlocal 在各平台抛的异常不一，名字只是给人看的
        name = None
    return name or moment.tzname() or "local"


def _generation_state_checks(worker: GenerationWorker) -> list[RuntimeHealthCheck]:
    """换代要人看的（都是 non-critical：白天照常递送）：

    - ``generation``：最近一次做的那一代没写成（必须成的那一步没成），今天没有这一代，带下一次重试的时刻；
    - ``generation_fallbacks``：最近一次写成的那一代走了退路（关系判定退到上一代、结算留到下次、单元历史跳过……），逐步列出；
    - ``generation_errors``：重试收尾之后仍保持「错误·看不到」的（我们自己处理不了的、内容安全拦下的、格式不合再试仍不合的）；
    - ``generation_stuck``：定稿日不连续、连着几眼都卡在同一天（有判断一直封不了口）；最新的完成标记读不了也报（日期回退见 ``_rewound_checks``）。"""

    checks: list[RuntimeHealthCheck] = []
    result = worker.last_result
    if result is not None and not result.complete and result.held is None:
        later = "" if worker.retry_at is None else f"; next retry at {worker.retry_at.isoformat()}"
        checks.append(
            RuntimeHealthCheck(
                "generation",
                RuntimeHealthStatus.DEGRADED,
                f"{result.day} not written: {result.failed} {result.reason or '-'}{later}"[:500],
                critical=False,
            )
        )
    try:
        mark = worker.generation.marks.latest()
    except GenerationMarkError as exc:
        checks.append(
            RuntimeHealthCheck(
                "generation_marks", RuntimeHealthStatus.DEGRADED, f"unreadable:{type(exc).__name__}", critical=False
            )
        )
        mark = None
    if mark is not None:
        degraded = [
            f"{step}: {value.get('reason') or '-'}"
            for step, value in mark.steps.items()
            if value.get("outcome") == "degraded"
        ]
        if degraded:
            checks.append(
                RuntimeHealthCheck(
                    "generation_fallbacks",
                    RuntimeHealthStatus.DEGRADED,
                    f"{mark.day} " + "; ".join(degraded)[:480],
                    critical=False,
                )
            )
        kept = [item for item in mark.problems if item.startswith("错误·看不到")]
        if kept:
            checks.append(
                RuntimeHealthCheck("generation_errors", RuntimeHealthStatus.DEGRADED, kept[0][:500], critical=False)
            )
    stuck = worker.stuck
    if stuck is not None:
        checks.append(
            RuntimeHealthCheck(
                "generation_stuck",
                RuntimeHealthStatus.DEGRADED,
                f"closed days are not contiguous: stuck at {stuck}",
                critical=False,
            )
        )
    return [*checks, *_rewound_checks(worker)]


def _rewound_checks(worker: GenerationWorker) -> list[RuntimeHealthCheck]:
    """``generation_rewound``：当地日期回到了已换代的日子之前（往西改了时区），日期追上之前不换代；追上后这一项自己消掉。"""

    if worker.rewound is None:
        return []
    today, latest = worker.rewound
    detail = f"{REWOUND}: today {today}, latest generation {latest}; no generation until the date catches up"
    return [RuntimeHealthCheck("generation_rewound", RuntimeHealthStatus.DEGRADED, detail, critical=False)]


__all__ = [
    "RuntimeHealthCheck",
    "RuntimeHealthReport",
    "RuntimeHealthService",
    "RuntimeHealthStatus",
]
