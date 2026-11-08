"""记忆主链唯一组合根的显式生命周期。"""

from __future__ import annotations

import asyncio
import time
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

from habitus.behavior.reduction import VocabularyJobReport
from habitus.config import HabitusConfig
from habitus.conversation import (
    ConversationSourceConsumer,
    ConversationSourceEnvelope,
    ConversationSourceOutputRepair,
    ConversationSourcePendingDelivery,
    ConversationSourceRecoveryResult,
    ConversationSourceRepairError,
    ConversationSourceRepairResult,
    conversation_source_request_digest,
)
from habitus.foundation.observability import (
    ObservationEvent,
    ObservationStatus,
    record_observation,
)
from habitus.infrastructure.observability import AuditRecord, ObservabilitySnapshot
from habitus.infrastructure.store.contracts import LockStoreSnapshot
from habitus.infrastructure.store.filesystem import is_path_alias, restrict_directory_to_owner
from habitus.memory.browse import MemoryLayerText, MemoryListing
from habitus.memory.conversation import (
    ConversationAddress,
    ConversationIngressRequest,
)
from habitus.memory.document import MemoryDocument
from habitus.memory.intention import MemoryIntentionRecallScope
from habitus.memory.model import MemoryKind
from habitus.memory.retrieval import MemorySearchResult
from habitus.memory.uri import MemoryURI
from habitus.memory.workflow import (
    ConversationLifecycleMaintenanceResult,
    ConversationMemoryFlushResult,
    ConversationMemoryIngestResult,
    MemoryChangeReceipt,
    MemoryChangeSource,
    MemoryJob,
    MemoryJobRunResult,
    MemoryJobStatus,
)
from habitus.pre.conversation import (
    ConversationAdaptation,
    ConversationAdapterContext,
    ConversationAdapterRegistry,
    ConversationBatch,
    ConversationSegment,
)
from habitus.runtime.behavior import BehaviorBranches, UnitOutcome
from habitus.runtime.behavior import deliver_physical as _deliver_physical
from habitus.runtime.behavior import run_vocabulary_job as _run_vocabulary_job
from habitus.runtime.components import RuntimeComponents
from habitus.runtime.consistency import MemoryConsistencyService, MemoryConsistencySnapshot
from habitus.runtime.foresight_outlet import ClaimedReminder
from habitus.runtime.foresight_reminders import ReminderDay, reminder_day
from habitus.runtime.health import RuntimeHealthReport, RuntimeHealthService
from habitus.runtime.lifecycle import LifecycleWorkerState
from habitus.runtime.lifecycle_sources import memory_jobs_restored
from habitus.runtime.runtime_jobs import FailedMemoryJobOperations, MemoryJobAbandonResult, MemoryJobRetryResult
from habitus.runtime.runtime_lifecycle import RuntimeLifecycle, publish_source_delivery_gauges
from habitus.runtime.runtime_state import RuntimeState, RuntimeStateError
from habitus.runtime.worker import MemoryWorkerState


class RuntimeInitializationError(RuntimeError):
    """Runtime 的共同存储根无法安全初始化。"""


class RuntimeShutdownTimeoutError(TimeoutError):
    """Runtime 有界关闭时仍有 Source Consumer 未完成。"""

    def __init__(
        self,
        timeout_seconds: float,
        pending_deliveries: tuple[ConversationSourcePendingDelivery, ...],
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self.pending_deliveries = pending_deliveries
        pending = ", ".join(f"{item.source_id}:{item.consumer.value}" for item in pending_deliveries)
        super().__init__(
            "runtime close timed out after "
            f"{timeout_seconds} seconds; pending Conversation Source consumers: "
            f"{pending or 'unknown'}"
        )


@dataclass(frozen=True)
class RuntimeInitialization:
    """一次成功初始化的可审计结果。"""

    memory_root: Path
    recovered_transaction_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.memory_root.is_absolute():
            raise ValueError("memory_root must be an absolute Path")
        if any(not identifier for identifier in self.recovered_transaction_ids):
            raise TypeError("recovered_transaction_ids must contain non-empty strings")


@dataclass(frozen=True)
class RuntimeConversationProtocolIngestResult:
    """外部协议转换事实与实际写入结果。"""

    adaptation: ConversationAdaptation
    ingest: ConversationMemoryIngestResult
    effective_after_turn: bool

    @property
    def next_sequence(self) -> int:
        """返回写入锁内确认的 Conversation 权威下一序号。"""

        return self.ingest.append.next_sequence


class RuntimeObservabilityRefresh:
    """只读刷新队列、Worker 和锁聚合值并发布 gauge；采集失败不阻断业务或指标端点。

    与门面同文件：锁快照的类型核对是这个文件已有的运行时类型检查（尺寸闸的类型检查基线按文件计），搬出去会变成新文件的新增条目。
    """

    def __init__(self, config: HabitusConfig, components: RuntimeComponents) -> None:
        self.config = config
        self.components = components

    async def refresh(self) -> ObservabilitySnapshot:
        attributes: dict[str, str | int | float | bool] = {}
        status = ObservationStatus.SUCCESS
        if not await self._collect_queue(attributes):
            status = ObservationStatus.DEGRADED
        if not await self._collect_locks(attributes):
            status = ObservationStatus.DEGRADED
        registry = self.components.infrastructure.observability
        self._publish_worker_gauges()
        manager = self.components.infrastructure.managed_observability
        if manager is not None:
            backend_status, _detail = manager.health()
            registry.set_gauge("observability_backend_healthy", 1 if backend_status == "healthy" else 0)
        observer = self.components.infrastructure.observer
        if observer is not None:
            observer.record(
                ObservationEvent(
                    category="observability",
                    operation="snapshot",
                    status=status,
                    duration_seconds=0.0,
                    attributes=attributes,
                )
            )
        return registry.snapshot()

    async def _collect_queue(self, attributes: dict[str, str | int | float | bool]) -> bool:
        """队列聚合值并进 ``attributes``；失败记错误类型并返回 False（降级）。"""

        try:
            queue = await asyncio.to_thread(self.components.workflow.jobs.observability_snapshot)
            attributes.update(
                queue_staged=queue.staged,
                queue_queued=queue.queued,
                queue_running=queue.running,
                queue_failed=queue.failed,
                queue_committed=queue.committed,
                queue_oldest_age_seconds=queue.oldest_age_seconds,
                queue_high_watermark=queue.high_watermark,
            )
        except Exception as exc:
            attributes["queue_error_type"] = type(exc).__name__
            return False
        return True

    async def _collect_locks(self, attributes: dict[str, str | int | float | bool]) -> bool:
        """锁库带快照能力时并进锁聚合值；失败记错误类型并返回 False（降级）。"""

        snapshotter = getattr(self.components.infrastructure.path_lock.lock_store, "observability_snapshot", None)
        if not callable(snapshotter):
            return True
        try:
            lock_snapshot = await asyncio.to_thread(
                snapshotter,
                warning_seconds=float(self.config.observability.lock_warning_seconds),
            )
            if not isinstance(lock_snapshot, LockStoreSnapshot):
                raise TypeError("lock store returned an invalid observability snapshot")
            attributes.update(
                active_locks=lock_snapshot.active_count,
                hanging_locks=lock_snapshot.hanging_count,
                max_active_lock_age_seconds=lock_snapshot.max_active_age_seconds,
            )
        except Exception as exc:
            attributes["lock_error_type"] = type(exc).__name__
            return False
        return True

    def _publish_worker_gauges(self) -> None:
        registry = self.components.infrastructure.observability
        for memory_worker_state in MemoryWorkerState:
            registry.set_gauge(
                "memory_worker_state",
                int(memory_worker_state is self.components.workflow.worker.state),
                labels={"worker_state": memory_worker_state.value},
            )
        for lifecycle_worker_state in LifecycleWorkerState:
            registry.set_gauge(
                "lifecycle_worker_state",
                int(lifecycle_worker_state is self.components.workflow.lifecycle_worker.state),
                labels={"worker_state": lifecycle_worker_state.value},
            )


class Runtime:
    """只管理组装、初始化顺序和 Job 执行入口，不承载领域规则。"""

    def __init__(
        self,
        config: HabitusConfig,
        components: RuntimeComponents,
        *,
        conversation_adapters: ConversationAdapterRegistry | None = None,
    ) -> None:
        if components.memory.tree.root != config.memory_root:
            raise ValueError("runtime components are bound to another memory root")
        if components.conversation.journal.layout.root != config.conversation_root:
            raise ValueError("runtime components are bound to another conversation root")
        if components.workflow.jobs.root != config.workflow_root:
            raise ValueError("runtime components are bound to another workflow root")
        self.config = config
        self.components = components
        self._state = RuntimeState.CREATED
        self._initialization: RuntimeInitialization | None = None
        self._consistency = MemoryConsistencyService(
            components.workflow.jobs,
            components.workflow.receipts,
            observer=components.infrastructure.observer,
        )
        self._conversation_adapters = (
            conversation_adapters if conversation_adapters is not None else ConversationAdapterRegistry.with_builtins()
        )
        self._health = RuntimeHealthService(components, capacity_warning_share=config.lifecycle.capacity_warning_share)
        self._lifecycle = RuntimeLifecycle(config, components, observe=self._observe)
        self._failed_jobs = FailedMemoryJobOperations(
            components, running=lambda: self._state is RuntimeState.RUNNING, observe=self._observe
        )
        self._observability = RuntimeObservabilityRefresh(config, components)

    @property
    def state(self) -> RuntimeState:
        return self._state

    @property
    def initialized(self) -> bool:
        return self._state in {RuntimeState.READY, RuntimeState.RUNNING}

    def initialize(self) -> RuntimeInitialization:
        """初始化树；无任务时恢复孤儿事务，有任务时留给租约持有者恢复。"""

        started = time.monotonic()
        if self._state in {RuntimeState.CLOSING, RuntimeState.CLOSED}:
            raise RuntimeStateError(f"{self._state.value} runtime cannot be initialized")
        if self._initialization is not None:
            return self._initialization
        try:
            self._ensure_storage_root()
            memory_root, recovered = self._lifecycle.initialize_stores()
            initialization = RuntimeInitialization(
                memory_root=memory_root,
                recovered_transaction_ids=recovered,
            )
            self._initialization = initialization
            self._state = RuntimeState.READY
        except Exception as exc:
            self._observe(
                "runtime",
                "initialization",
                ObservationStatus.FAILURE,
                started,
                {"error_type": type(exc).__name__},
            )
            raise
        self._observe(
            "runtime",
            "initialization",
            ObservationStatus.SUCCESS,
            started,
            {"recovered_transactions": len(recovered)},
        )
        return initialization

    async def start(self) -> None:
        """完成初始化后依次启动 Job Worker 和生命周期维护 Worker。"""

        started = time.monotonic()
        if self._state in {RuntimeState.CLOSING, RuntimeState.CLOSED}:
            raise RuntimeStateError(f"{self._state.value} runtime cannot be started")
        try:
            self.initialize()
            if self._state is RuntimeState.RUNNING:
                return
            await self._lifecycle.start(started)
            self._state = RuntimeState.RUNNING
        except BaseException as exc:
            self._observe(
                "runtime",
                "start",
                ObservationStatus.FAILURE,
                started,
                {"error_type": type(exc).__name__},
            )
            raise
        self._observe(
            "runtime",
            "start",
            ObservationStatus.SUCCESS,
            started,
            {"runtime_state": self._state.value},
        )

    async def stop(self) -> None:
        """停止认领并有界排空 Worker，保留 Runtime 供再次启动。"""

        if self._state is RuntimeState.CLOSED:
            return
        if self._state is RuntimeState.CLOSING:
            raise RuntimeStateError("closing runtime cannot be stopped")
        if self._state is RuntimeState.CREATED:
            return
        await self._lifecycle.stop()
        self._state = RuntimeState.READY

    async def deliver_physical(self, payload: object) -> tuple[UnitOutcome, ...]:
        """物理 lane 的正门：上游（手表等）的一份交付进来，一段一条写成判断，返回每一段的去向。

        交付的形状与对上游的约定见 ``behavior/fusion/lanes/physical``；不合约定的整份拒收。按上游的段编号幂等
        （同一段再送一遍不重复写，上游改判不收）；行为侧未启用时明确拒绝，而不是静默丢弃。
        同步的耐久落盘下沉线程，与 Runtime 其余同步存储调用同一形态。会话 lane 不走这里，走会话源的交付。
        """

        self._require_initialized("physical behavior delivery")
        return await asyncio.to_thread(
            _deliver_physical,
            self._require_behavior(),
            payload,
            observer=self.components.infrastructure.observer,
        )

    def reminders_enabled(self) -> bool:
        """提醒出口开着没有：预测层装上了、带着出口。没开时取口恒为空，服务不公布 ``reminders_v1``。"""

        foresight = self.components.foresight
        return foresight is not None and foresight.outlet is not None

    async def claim_reminders(
        self, *, host: str, session: str, at: datetime, limit: int
    ) -> tuple[ClaimedReminder, ...]:
        """提醒出口的取口（方案 32 第 4.1 条）：宿主在他敲一句话时来取此刻可取的提醒（带租约），顺手留下活性。预测层没开就是空。
        ``limit`` 是正整数；超过 ``config.foresight.outlet_claim_limit`` 的按这个上限取（宿主那边的上限各自配，不因为配得大就一条都取不到）。
        ``at`` 由宿主给、带时区。同步的文件读写下沉线程。"""

        self._require_initialized("reminder claim")
        if at.utcoffset() is None:
            raise ValueError("at must be a timezone-aware datetime")
        if limit < 1:
            raise ValueError("limit must be a positive integer")
        foresight = self.components.foresight
        if foresight is None or foresight.outlet is None:
            return ()
        return await asyncio.to_thread(
            foresight.outlet.claim, host, session, at, min(limit, self.config.foresight.outlet_claim_limit)
        )

    async def reminders(self, day: date | None = None) -> ReminderDay:
        """``day``（省略是本地今天）进了提醒出口的每条提醒走到了哪一步（方案 36 第 2 步）。预测层没开就是空。只读。"""

        self._require_initialized("reminder report")
        resolved = day if day is not None else datetime.now(self.config.locale.zone()).date()
        if isinstance(resolved, datetime):
            raise ValueError("day must be a date")
        foresight = self.components.foresight
        if foresight is None:
            return ReminderDay(resolved, ())
        return await asyncio.to_thread(reminder_day, foresight.ledger, foresight.outlet, resolved)

    async def run_vocabulary_job(self, job: str, *, lane: str) -> VocabularyJobReport:
        """词表整理活正门：``grow`` 立刻每晚新增、``revise`` 立刻定期拆改。两条 lane 的词表各是各的，``lane`` 说整理哪一支。"""

        self._require_initialized("behavior vocabulary job")
        branch = self._require_behavior().of(lane)
        return await _run_vocabulary_job(branch, job, observer=self.components.infrastructure.observer)

    def _require_behavior(self) -> BehaviorBranches:
        behavior = self.components.behavior
        if behavior is None:
            raise RuntimeStateError("behavior pipeline is not configured; set behavior.primary_subject")
        return behavior

    async def run_next(self) -> MemoryJobRunResult:
        """只在初始化完成后委托领域 Runner 处理最早的一项 Job。"""

        self._require_ready()
        return await self.components.workflow.worker.run_once()

    async def append_conversation(
        self,
        address: ConversationAddress,
        batch: ConversationBatch,
        *,
        after_turn: bool = False,
        omit_tool_call_ids: frozenset[str] = frozenset(),
        ingress: ConversationIngressRequest | None = None,
        protocol: str = "normalized",
    ) -> ConversationMemoryIngestResult:
        """先保存规范 Source，再并行分发 Memory 与 Behavior Projection。"""

        self._require_initialized("conversation append")
        request_digest = conversation_source_request_digest(
            conversation_id=address.conversation_id,
            started_on=address.started_on,
            protocol=protocol,
            batch=batch,
            after_turn=after_turn,
            omit_tool_call_ids=omit_tool_call_ids,
        )
        delivery_id = request_digest
        if ingress is not None:
            request_digest = ingress.request_digest
            delivery_id = ingress.delivery_id
        envelope = ConversationSourceEnvelope.create(
            conversation_id=address.conversation_id,
            started_on=address.started_on,
            protocol=protocol,
            batch=batch,
            after_turn=after_turn,
            omit_tool_call_ids=omit_tool_call_ids,
            delivery_id=delivery_id,
            request_digest=request_digest,
            recorded_at=datetime.now(UTC),
        )
        handle = self.components.conversation.source_coordinator.start(envelope)
        # 每个 Consumer 的成败由 ConversationConsumerDelivery 统一观察，覆盖前台、
        # 启动恢复和显式恢复三条路径；这里不再为单个 Consumer 另挂回调。
        result = await handle.wait(ConversationSourceConsumer.MEMORY)
        if not isinstance(result, ConversationMemoryIngestResult):
            raise RuntimeError("Memory Conversation Consumer completed without an ingest result")
        if result.jobs and self._state is RuntimeState.RUNNING:
            self.components.workflow.worker.wake()
        return result

    async def append_protocol_conversation(
        self,
        address: ConversationAddress,
        *,
        protocol: str,
        payload: object,
        start_sequence: int,
        occurred_at: datetime,
        after_turn: bool | None = None,
        omit_tool_call_ids: frozenset[str] = frozenset(),
        delivery_id: str | None = None,
    ) -> RuntimeConversationProtocolIngestResult:
        """适配 OpenAI/Anthropic 协议后进入唯一 Conversation 写入主链。"""

        self._require_initialized("protocol conversation append")
        context = ConversationAdapterContext(
            conversation_id=address.conversation_id,
            start_sequence=start_sequence,
            occurred_at=occurred_at,
        )
        adaptation = self._conversation_adapters.adapt(protocol, payload, context)
        resolved_after_turn = adaptation.after_turn if after_turn is None else after_turn
        ingress = None
        if delivery_id is not None:
            request_digest = conversation_source_request_digest(
                conversation_id=address.conversation_id,
                started_on=address.started_on,
                protocol=protocol,
                batch=adaptation.batch,
                after_turn=resolved_after_turn,
                omit_tool_call_ids=omit_tool_call_ids,
            )
            ingress = ConversationIngressRequest(delivery_id, request_digest)
        ingest = await self.append_conversation(
            address,
            adaptation.batch,
            after_turn=resolved_after_turn,
            omit_tool_call_ids=omit_tool_call_ids,
            ingress=ingress,
            protocol=protocol,
        )
        return RuntimeConversationProtocolIngestResult(adaptation, ingest, resolved_after_turn)

    def conversation_protocols(self) -> tuple[str, ...]:
        """返回当前显式注册的协议名，供 HTTP/Agent 接入层发现能力。"""

        return self._conversation_adapters.protocols()

    async def recover_conversation_sources(self) -> tuple[ConversationSourceRecoveryResult, ...]:
        """只处理非终态 Source Consumer，并保留两条交付链的隔离。"""

        self._require_initialized("conversation source recovery")
        results = await self.components.conversation.source_recovery.recover_pending()
        publish_source_delivery_gauges(self.components.infrastructure.observability, results)
        if self._state is RuntimeState.RUNNING and memory_jobs_restored(
            self.components.conversation.source_delivery, results
        ):
            self.components.workflow.worker.wake()
        return results

    def repair_conversation_source_outputs(
        self,
        source_id: str,
        consumer: ConversationSourceConsumer,
    ) -> ConversationSourceRepairResult:
        """人工修复同一 Source Consumer 上的多个孤儿 Output。

        与 FAILED Job 的重试入口同形态：正式运维动作只从 Runtime 发起，且必须
        指名具体来源。交付层保持 fail-closed，不会自动做这件事——出现多个孤儿
        意味着存储被外力动过，需要人确认过再修。修复后该 Consumer 会在下一次
        恢复时采用保留下来的那份，不会重新执行。
        """

        self._require_initialized("conversation source output repair")
        envelope = self.components.conversation.sources.read(source_id)
        if envelope is None:
            raise ConversationSourceRepairError(f"unknown Conversation Source: {source_id}")
        repair = ConversationSourceOutputRepair(self.components.conversation.source_delivery)
        repaired = repair.repair(envelope, ConversationSourceConsumer(consumer))
        # 修好了就清掉这一份的重试账：放弃过的也在下一轮周期恢复里照常再试
        self.components.conversation.source_recovery.forgive(source_id, ConversationSourceConsumer(consumer))
        return repaired

    async def flush_conversation(
        self,
        address: ConversationAddress,
    ) -> ConversationMemoryFlushResult:
        """在完整轮次边界封存剩余消息并返回提交任务。"""

        self._require_initialized("conversation flush")
        preview = await asyncio.to_thread(
            self.components.workflow.enqueuer.preview_retention,
            address,
            after_turn=False,
            flush=True,
        )
        boundary_hints = None
        if preview.should_seal:
            live = await asyncio.to_thread(
                self.components.conversation.journal.read_live,
                address,
            )
            boundary_hints = await self.components.conversation.boundary_scorer.score(live)
        jobs, retention = await asyncio.to_thread(
            self.components.workflow.enqueuer.enqueue_ready_segments,
            address,
            after_turn=False,
            flush=True,
            boundary_hints=boundary_hints,
        )
        result = ConversationMemoryFlushResult(jobs=jobs, retention=retention)
        if result.jobs and self._state is RuntimeState.RUNNING:
            self.components.workflow.worker.wake()
        return result

    async def read_live_conversation(self, address: ConversationAddress) -> ConversationBatch | None:
        """读取尚未封存的 Conversation 原始事实。"""

        self._require_initialized("conversation read")
        return await asyncio.to_thread(self.components.conversation.journal.read_live, address)

    async def conversation_cursor(self, address: ConversationAddress) -> int:
        """读取服务端耐久 Conversation 的下一消息序号。"""

        self._require_initialized("conversation cursor read")
        return await asyncio.to_thread(self.components.conversation.journal.next_sequence, address)

    async def list_conversation_history(
        self,
        address: ConversationAddress,
    ) -> tuple[ConversationSegment, ...]:
        """按消息范围返回当前仍保留的不可变 History Segment。"""

        self._require_initialized("conversation history read")
        return await asyncio.to_thread(self.components.conversation.journal.list_history, address)

    async def memory_consistency(self, job: MemoryJob) -> MemoryConsistencySnapshot:
        """读取异步写入当前是否已提交、失败或被人工放弃。"""

        self._require_initialized("memory consistency inspection")
        return await asyncio.to_thread(self._consistency.inspect, job)

    async def wait_memory_consistency(
        self,
        job: MemoryJob,
        *,
        timeout_seconds: float,
        poll_interval_seconds: float = 0.05,
    ) -> MemoryConsistencySnapshot:
        """等待指定 Job 达到可解释终态；超时不会取消后台任务。"""

        self._require_initialized("memory consistency wait")
        return await self._consistency.wait(
            job,
            timeout_seconds=timeout_seconds,
            poll_interval_seconds=poll_interval_seconds,
        )

    async def memory_change_receipt(self, job: MemoryJob) -> MemoryChangeReceipt | None:
        """按写入句柄读取可审计的 Memory Change Receipt。"""

        self._require_initialized("memory change receipt read")
        return await asyncio.to_thread(
            self.components.workflow.receipts.try_read,
            MemoryChangeSource.from_job(job),
        )

    async def list_memory_jobs(self, address: ConversationAddress) -> tuple[MemoryJob, ...]:
        """读取一个 Conversation 当前仍保留的后台任务。"""

        self._require_initialized("memory job read")
        return await asyncio.to_thread(self.components.workflow.jobs.list_for_conversation, address)

    async def failed_memory_job(self) -> MemoryJob | None:
        """读取当前阻塞全队列的最早 FAILED Job，供人工检查错误和版本。"""

        self._require_initialized("failed memory job inspection")
        job = await asyncio.to_thread(
            self.components.workflow.jobs.oldest_uncommitted,
        )
        return job if job is not None and job.status is MemoryJobStatus.FAILED else None

    async def retry_failed_memory_job(self, failed_job: MemoryJob) -> MemoryJobRetryResult:
        """以完整失败快照重新开放最早 Job，并恢复被它阻塞的常驻 Worker。"""

        if failed_job.status is not MemoryJobStatus.FAILED:
            raise ValueError("failed_job must be a FAILED MemoryJob snapshot")
        self._require_initialized("failed memory job retry")
        return await self._failed_jobs.retry(failed_job)

    async def abandon_failed_memory_job(
        self,
        failed_job: MemoryJob,
        *,
        reason: str,
    ) -> MemoryJobAbandonResult:
        """显式放弃无法修复的最早 Job；有任何未决或已提交副作用时拒绝执行。"""

        if failed_job.status is not MemoryJobStatus.FAILED:
            raise ValueError("failed_job must be a FAILED MemoryJob snapshot")
        if not reason.strip():
            raise ValueError("reason must be non-empty text")
        self._require_initialized("failed memory job abandonment")
        return await self._failed_jobs.abandon(failed_job, reason=reason)

    async def find_memory(
        self,
        query: str,
        *,
        target_uris: MemoryURI | str | tuple[MemoryURI | str, ...] | None = None,
        limit: int | None = None,
        score_threshold: float | None = None,
        kinds: tuple[MemoryKind, ...] = (),
        intention_scope: MemoryIntentionRecallScope = MemoryIntentionRecallScope.ACTIVE,
    ) -> MemorySearchResult:
        """不使用 Conversation 上下文，直接执行 Agent 记忆检索。"""

        self._require_initialized("memory search")
        return await self.components.memory.search.find(
            query,
            target_uris=target_uris,
            limit=limit,
            score_threshold=score_threshold,
            kinds=kinds,
            intention_scope=intention_scope,
        )

    async def list_memory(self, uri: MemoryURI | str | None = None) -> MemoryListing:
        """浏览记忆树：列一个目录（省略是根）的子目录与文档，只给 ``memory://`` 地址（方案 36 第 2 步）。"""

        self._require_initialized("memory browse")
        return await asyncio.to_thread(self.components.memory.browser.list, uri)

    async def read_memory(self, uri: MemoryURI | str) -> MemoryDocument | MemoryLayerText:
        """读一个记忆文档，或一个目录的摘要 / 概览层。"""

        self._require_initialized("memory browse")
        return await asyncio.to_thread(self.components.memory.browser.read, uri)

    async def search_memory(
        self,
        query: str,
        *,
        conversation: ConversationAddress | None = None,
        target_uris: MemoryURI | str | tuple[MemoryURI | str, ...] | None = None,
        limit: int | None = None,
        score_threshold: float | None = None,
        kinds: tuple[MemoryKind, ...] = (),
        intention_scope: MemoryIntentionRecallScope = MemoryIntentionRecallScope.ACTIVE,
    ) -> MemorySearchResult:
        """执行 Memory 主召回，并在信息不足时按需补充历史 Summary。"""

        self._require_initialized("memory search")
        return await self.components.memory.search.search(
            query,
            conversation=conversation,
            target_uris=target_uris,
            limit=limit,
            score_threshold=score_threshold,
            kinds=kinds,
            intention_scope=intention_scope,
        )

    async def maintain_conversation(
        self,
        address: ConversationAddress,
        *,
        now: datetime | None = None,
    ) -> ConversationLifecycleMaintenanceResult:
        """显式执行一次单 Conversation 维护；自动批量维护由 LifecycleWorker 负责。"""

        if not self.initialized:
            raise RuntimeStateError("conversation maintenance requires an initialized runtime")
        return await self.components.workflow.lifecycle.maintain_once(address, now=now)

    async def health(self, *, deep: bool = False) -> RuntimeHealthReport:
        """返回无副作用健康报告；deep 额外探测模型与两套派生索引。"""

        return await self._health.report(self._state.value, deep=deep)

    async def refresh_observability(self) -> ObservabilitySnapshot:
        """只读刷新队列、Worker 和锁聚合值；采集失败不阻断业务或指标端点。"""

        return await self._observability.refresh()

    async def recent_audit_events(self, *, limit: int = 100) -> tuple[AuditRecord, ...]:
        """读取最小安全与运维审计；没有启用审计时返回空结果。"""

        self._require_initialized("audit event read")
        manager = self.components.infrastructure.managed_observability
        if manager is None:
            return ()
        return await asyncio.to_thread(manager.recent_audit, limit=limit)

    def metrics_snapshot(self) -> ObservabilitySnapshot:
        """返回进程内指标不可变快照，供 HTTP/OTel Adapter 导出。"""

        return self.components.infrastructure.observability.snapshot()

    def prometheus_metrics(self) -> str:
        """返回 Prometheus exposition 文本，不启动 HTTP 服务。"""

        return self.components.infrastructure.observability.prometheus_text()

    async def close(self) -> None:
        """有界排空 Consumer；超时显式失败且不取消底层耐久交付。"""

        if self._state is RuntimeState.CLOSED:
            return
        previous_state = self._state
        self._state = RuntimeState.CLOSING
        if previous_state is not RuntimeState.CREATED:
            await self._lifecycle.stop_for_close()
        timeout_seconds = self.config.conversation.source.shutdown_timeout_seconds
        idle = await self.components.conversation.source_coordinator.wait_for_idle(timeout_seconds)
        if not idle:
            pending = self.components.conversation.source_coordinator.pending_deliveries
            if pending or not await self.components.conversation.source_coordinator.wait_for_idle(0):
                raise RuntimeShutdownTimeoutError(timeout_seconds, pending)
        await self._lifecycle.close_resources()
        self._state = RuntimeState.CLOSED

    async def __aenter__(self) -> Runtime:
        await self.start()
        return self

    async def __aexit__(
        self,
        _error_type: object,
        _error: object,
        _traceback: object,
    ) -> None:
        await self.close()

    def _require_ready(self) -> None:
        if self._state is not RuntimeState.READY:
            raise RuntimeStateError("manual run_next requires an initialized runtime with its worker stopped")

    def _require_initialized(self, operation: str) -> None:
        if not self.initialized:
            raise RuntimeStateError(f"{operation} requires an initialized runtime")

    def _ensure_storage_root(self) -> None:
        """逐级创建共同根目录，并拒绝沿途符号链接和非目录节点。"""

        root = Path(self.config.storage_root)
        missing: list[Path] = []
        current = root
        while not current.exists():
            if is_path_alias(current):
                raise RuntimeInitializationError("runtime storage path cannot traverse a symbolic link")
            missing.append(current)
            parent = current.parent
            if parent == current:
                raise RuntimeInitializationError("runtime storage root has no existing parent")
            current = parent
        if is_path_alias(current) or not current.is_dir():
            raise RuntimeInitializationError("runtime storage ancestor must be a real directory")
        for directory in reversed(missing):
            if is_path_alias(directory):
                raise RuntimeInitializationError("runtime storage path cannot traverse a symbolic link")
            try:
                directory.mkdir(mode=0o700)
            except FileExistsError:
                # 目录已在：下一行核它是不是真实目录
                pass
            if is_path_alias(directory) or not directory.is_dir():
                raise RuntimeInitializationError("runtime storage path is not a real directory")
            with suppress(OSError):
                restrict_directory_to_owner(directory)

    def _observe(
        self,
        category: str,
        operation: str,
        status: ObservationStatus,
        started: float,
        attributes: dict[str, str | int | float | bool],
    ) -> None:
        observer = self.components.infrastructure.observer
        if observer is None:
            return
        record_observation(observer, category, operation, status, started=started, attributes=attributes)


__all__ = [
    "MemoryJobAbandonResult",
    "MemoryJobRetryResult",
    "RuntimeConversationProtocolIngestResult",
    "Runtime",
    "RuntimeInitialization",
    "RuntimeInitializationError",
    "RuntimeShutdownTimeoutError",
    "RuntimeState",
    "RuntimeStateError",
]
