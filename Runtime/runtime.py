"""记忆主链唯一组合根的显式生命周期。"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from behavior.claim.service import ClaimNormalizationResult
from behavior.evidence.ingress import BehaviorEvidenceIngressResult
from Config import M2BOSConfig
from foundation.integrity import canonical_digest
from foundation.observability import (
    ObservabilitySnapshot,
    ObservationEvent,
    ObservationStatus,
    bind_observation_context,
)
from infrastructure.observability import AuditRecord
from infrastructure.store.contracts import LockStoreSnapshot
from memory.conversation import (
    ConversationAddress,
    ConversationIngressRequest,
    ConversationSummaryReference,
)
from memory.editor import MemoryTransactionJournalState
from memory.intention import MemoryIntentionRecallScope
from memory.model import MemoryKind
from memory.retrieval import MemorySearchResult
from memory.uri import MemoryURI
from memory.workflow import (
    ConversationLifecycleMaintenanceResult,
    ConversationMemoryFlushResult,
    ConversationMemoryIngestResult,
    MemoryChangeReceipt,
    MemoryChangeSource,
    MemoryJob,
    MemoryJobAbandonment,
    MemoryJobRunResult,
    MemoryJobStatus,
)
from pre.conversation import (
    ConversationAdaptation,
    ConversationAdapterContext,
    ConversationAdapterRegistry,
    ConversationBatch,
    ConversationSegment,
)
from Runtime.components import RuntimeComponents
from Runtime.consistency import MemoryConsistencyService, MemoryConsistencySnapshot
from Runtime.health import RuntimeHealthReport, RuntimeHealthService
from Runtime.lifecycle import LifecycleWorkerState
from Runtime.worker import MemoryWorkerState


class RuntimeState(str, Enum):
    """Runtime 的显式初始化、运行、停止和终止状态。"""

    CREATED = "created"
    READY = "ready"
    RUNNING = "running"
    CLOSED = "closed"


class RuntimeStateError(RuntimeError):
    """Runtime 操作与当前生命周期不相容。"""


class RuntimeInitializationError(RuntimeError):
    """Runtime 的共同存储根无法安全初始化。"""


@dataclass(frozen=True)
class RuntimeInitialization:
    """一次成功初始化的可审计结果。"""

    memory_root: Path
    behavior_root: Path
    recovered_transaction_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.memory_root, Path) or not self.memory_root.is_absolute():
            raise ValueError("memory_root must be an absolute Path")
        if not isinstance(self.behavior_root, Path) or not self.behavior_root.is_absolute():
            raise ValueError("behavior_root must be an absolute Path")
        if not isinstance(self.recovered_transaction_ids, tuple) or any(
            not isinstance(identifier, str) or not identifier for identifier in self.recovered_transaction_ids
        ):
            raise TypeError("recovered_transaction_ids must contain non-empty strings")


@dataclass(frozen=True)
class MemoryUseReceipt:
    """回答消费方显式确认实际使用的 L2 与 Summary 身份。"""

    memory_uris: tuple[MemoryURI, ...]
    summary_references: tuple[ConversationSummaryReference, ...]
    used_at: datetime

    def __post_init__(self) -> None:
        if not isinstance(self.memory_uris, tuple) or any(
            not isinstance(uri, MemoryURI) for uri in self.memory_uris
        ):
            raise TypeError("memory_uris must contain MemoryURI values")
        if not isinstance(self.summary_references, tuple) or any(
            not isinstance(reference, ConversationSummaryReference)
            for reference in self.summary_references
        ):
            raise TypeError("summary_references must contain ConversationSummaryReference values")
        if self.used_at.tzinfo is None or self.used_at.utcoffset() is None:
            raise ValueError("used_at must be timezone-aware")


@dataclass(frozen=True)
class MemoryJobRetryResult:
    """一次人工重试的失败快照、新状态和 Worker 恢复结果。"""

    failed_job: MemoryJob
    reopened_job: MemoryJob
    worker_restarted: bool

    def __post_init__(self) -> None:
        if not isinstance(self.failed_job, MemoryJob) or self.failed_job.status is not MemoryJobStatus.FAILED:
            raise ValueError("failed_job must be a FAILED MemoryJob snapshot")
        if not isinstance(self.reopened_job, MemoryJob) or self.reopened_job.status not in {
            MemoryJobStatus.STAGED,
            MemoryJobStatus.QUEUED,
        }:
            raise ValueError("reopened_job must be a STAGED or QUEUED MemoryJob")
        if self.failed_job.source_identity != self.reopened_job.source_identity:
            raise ValueError("retried MemoryJob source identity changed")
        if self.reopened_job.attempts != 0 or self.reopened_job.last_error is not None:
            raise ValueError("reopened MemoryJob must reset its retry state")
        if not isinstance(self.worker_restarted, bool):
            raise TypeError("worker_restarted must be boolean")


@dataclass(frozen=True)
class MemoryJobAbandonResult:
    """一次显式永久失败处置及其 Worker 恢复结果。"""

    failed_job: MemoryJob
    abandonment: MemoryJobAbandonment
    worker_restarted: bool


@dataclass(frozen=True)
class RuntimeConversationProtocolIngestResult:
    """外部协议转换事实与实际写入结果。"""

    adaptation: ConversationAdaptation
    ingest: ConversationMemoryIngestResult
    effective_after_turn: bool

    def __post_init__(self) -> None:
        if not isinstance(self.adaptation, ConversationAdaptation):
            raise TypeError("adaptation must be ConversationAdaptation")
        if not isinstance(self.ingest, ConversationMemoryIngestResult):
            raise TypeError("ingest must be ConversationMemoryIngestResult")
        if not isinstance(self.effective_after_turn, bool):
            raise TypeError("effective_after_turn must be boolean")

    @property
    def next_sequence(self) -> int:
        """返回写入锁内确认的 Conversation 权威下一序号。"""

        return self.ingest.append.next_sequence


# TODO(memory-tree-public-read): 前端树形展示协议确定后，再增加独立公共读取服务；当前不得泄露内部存储布局。
class Runtime:
    """只管理组装、初始化顺序和 Job 执行入口，不承载领域规则。"""

    def __init__(
        self,
        config: M2BOSConfig,
        components: RuntimeComponents,
        *,
        conversation_adapters: ConversationAdapterRegistry | None = None,
    ) -> None:
        if not isinstance(config, M2BOSConfig):
            raise TypeError("config must be M2BOSConfig")
        if not isinstance(components, RuntimeComponents):
            raise TypeError("components must be RuntimeComponents")
        if conversation_adapters is not None and not isinstance(
            conversation_adapters, ConversationAdapterRegistry
        ):
            raise TypeError("conversation_adapters must be ConversationAdapterRegistry or None")
        if components.memory.tree.root != config.memory_root:
            raise ValueError("runtime components are bound to another memory root")
        if components.behavior.database.root != config.behavior_root:
            raise ValueError("runtime components are bound to another behavior root")
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
            conversation_adapters
            if conversation_adapters is not None
            else ConversationAdapterRegistry.with_builtins()
        )
        self._health = RuntimeHealthService(components)

    @property
    def state(self) -> RuntimeState:
        return self._state

    @property
    def initialized(self) -> bool:
        return self._state in {RuntimeState.READY, RuntimeState.RUNNING}

    def initialize(self) -> RuntimeInitialization:
        """初始化树；无任务时恢复孤儿事务，有任务时留给租约持有者恢复。"""

        started = time.monotonic()
        if self._state is RuntimeState.CLOSED:
            raise RuntimeStateError("closed runtime cannot be initialized")
        if self._initialization is not None:
            return self._initialization
        try:
            self._ensure_storage_root()
            self.components.infrastructure.initialize()
            memory_root = self.components.memory.tree.initialize()
            self.components.behavior.database.initialize()
            self.components.workflow.jobs.initialize()
            self.components.memory.lifecycle.initialize()
            self.components.conversation.summary_use.initialize()
            self.components.workflow.lifecycle.retirement_store.initialize()
            oldest_job = self.components.workflow.jobs.oldest_uncommitted()
            recovered = (
                self.components.workflow.runner.transaction_recovery.recover_pending()
                if oldest_job is None
                else ()
            )
            initialization = RuntimeInitialization(
                memory_root=memory_root,
                behavior_root=self.components.behavior.database.root,
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

    async def ingest_behavior_semantic(
        self,
        adapter_name: str,
        payload: object,
        delivery_id: str,
    ) -> BehaviorEvidenceIngressResult:
        """通过已注册强类型 Adapter 原子写入 Evidence；不自动规范化 Claim。"""

        self._require_initialized("Behavior Evidence ingest")
        return await self.components.behavior.evidence_ingress.ingest(
            adapter_name,
            payload,
            delivery_id=delivery_id,
        )

    async def normalize_behavior_evidence(
        self,
        evidence_record_id: str,
    ) -> ClaimNormalizationResult:
        """独立处理一条已落盘 Evidence 的 Core 与可选 Enhancement。"""

        self._require_initialized("Behavior Claim normalization")
        return await self.components.behavior.claim_normalization.normalize(evidence_record_id)

    async def retry_behavior_enhancement(
        self,
        evidence_record_id: str,
        normalizer_name: str,
    ) -> ClaimNormalizationResult:
        """只重试指定 Evidence 的指定 Enhancement 路由。"""

        self._require_initialized("Behavior Enhancement retry")
        return await self.components.behavior.claim_normalization.retry_enhancement(
            evidence_record_id,
            normalizer_name,
        )

    async def start(self) -> None:
        """完成初始化后依次启动 Job Worker 和生命周期维护 Worker。"""

        started = time.monotonic()
        if self._state is RuntimeState.CLOSED:
            raise RuntimeStateError("closed runtime cannot be started")
        try:
            self.initialize()
            if self._state is RuntimeState.RUNNING:
                return
            await self.components.memory.vector_index.ensure_ready()
            await self.components.conversation.summary_vector_index.ensure_ready()
            await self.components.workflow.worker.start()
            try:
                await self.components.workflow.lifecycle_worker.start()
            except BaseException:
                try:
                    await self.components.workflow.lifecycle_worker.stop()
                finally:
                    await self.components.workflow.worker.stop()
                raise
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
        if self._state is RuntimeState.CREATED:
            return
        try:
            await self.components.workflow.lifecycle_worker.stop()
        finally:
            try:
                await self.components.workflow.worker.stop()
            finally:
                self._state = RuntimeState.READY

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
    ) -> ConversationMemoryIngestResult:
        """追加规范化消息并返回可用于一致性等待的耐久 Job 句柄。"""

        self._require_initialized("conversation append")
        appended = await asyncio.to_thread(
            self.components.workflow.enqueuer.append,
            address,
            batch,
            omit_tool_call_ids=omit_tool_call_ids,
            ingress=ingress,
        )
        boundary_hints = None
        preview = await asyncio.to_thread(
            self.components.workflow.enqueuer.preview_retention,
            address,
            after_turn=after_turn,
        )
        if preview.should_seal:
            live = await asyncio.to_thread(
                self.components.conversation.journal.read_live,
                address,
            )
            boundary_hints = await self.components.conversation.boundary_scorer.score(live)
        jobs, retention = await asyncio.to_thread(
            self.components.workflow.enqueuer.enqueue_ready_segments,
            address,
            after_turn=after_turn,
            flush=False,
            boundary_hints=boundary_hints,
        )
        result = ConversationMemoryIngestResult(
            append=appended,
            jobs=jobs,
            retention=retention,
        )
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
        if not isinstance(resolved_after_turn, bool):
            raise TypeError("after_turn must be boolean or None")
        ingress = None
        if delivery_id is not None:
            request_digest = canonical_digest(
                {
                    "conversation_id": address.conversation_id,
                    "started_on": address.started_on,
                    "protocol": protocol,
                    "batch": adaptation.batch.to_dict(),
                    "after_turn": resolved_after_turn,
                    "omit_tool_call_ids": omit_tool_call_ids,
                }
            )
            ingress = ConversationIngressRequest(delivery_id, request_digest)
        ingest = await self.append_conversation(
            address,
            adaptation.batch,
            after_turn=resolved_after_turn,
            omit_tool_call_ids=omit_tool_call_ids,
            ingress=ingress,
        )
        return RuntimeConversationProtocolIngestResult(adaptation, ingest, resolved_after_turn)

    def conversation_protocols(self) -> tuple[str, ...]:
        """返回当前显式注册的协议名，供 HTTP/Agent 接入层发现能力。"""

        return self._conversation_adapters.protocols()

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

        if not isinstance(failed_job, MemoryJob):
            raise TypeError("failed_job must be a MemoryJob")
        if failed_job.status is not MemoryJobStatus.FAILED:
            raise ValueError("failed_job must be a FAILED MemoryJob snapshot")
        self._require_initialized("failed memory job retry")

        started = time.monotonic()
        with bind_observation_context(
            memory_sequence=failed_job.memory_sequence,
            transaction_id=failed_job.transaction_id,
        ):
            try:
                worker = self.components.workflow.worker
                restart_worker = self._state is RuntimeState.RUNNING
                if restart_worker:
                    if worker.state is not MemoryWorkerState.BLOCKED or worker.busy:
                        raise RuntimeStateError(
                            "running runtime can only retry a failed job after its memory worker is blocked and idle"
                        )
                    await worker.wait_stopped()

                reopened = await asyncio.to_thread(
                    self.components.workflow.jobs.retry_failed,
                    failed_job,
                )
                if restart_worker:
                    await worker.start()
                result = MemoryJobRetryResult(
                    failed_job=failed_job,
                    reopened_job=reopened,
                    worker_restarted=restart_worker,
                )
            except Exception as exc:
                self._observe(
                    "operations",
                    "retry_job",
                    ObservationStatus.FAILURE,
                    started,
                    {"error_type": type(exc).__name__, "job_status": failed_job.status.value},
                )
                raise
            self._observe(
                "operations",
                "retry_job",
                ObservationStatus.SUCCESS,
                started,
                {"job_status": reopened.status.value},
            )
            return result

    async def abandon_failed_memory_job(
        self,
        failed_job: MemoryJob,
        *,
        reason: str,
    ) -> MemoryJobAbandonResult:
        """显式放弃无法修复的最早 Job；有任何未决或已提交副作用时拒绝执行。"""

        if not isinstance(failed_job, MemoryJob) or failed_job.status is not MemoryJobStatus.FAILED:
            raise ValueError("failed_job must be a FAILED MemoryJob snapshot")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("reason must be non-empty text")
        self._require_initialized("failed memory job abandonment")
        source = MemoryChangeSource.from_job(failed_job)
        receipt = await asyncio.to_thread(self.components.workflow.receipts.try_read, source)
        if receipt is not None:
            raise RuntimeStateError("a failed job with a prepared or committed receipt cannot be abandoned")
        journal = await asyncio.to_thread(
            self.components.memory.editor.transaction.journal.try_read,
            failed_job.transaction_id,
        )
        if journal is not None and journal.state is not MemoryTransactionJournalState.ROLLED_BACK:
            raise RuntimeStateError("a failed job with a pending or committed transaction cannot be abandoned")
        worker = self.components.workflow.worker
        restart_worker = self._state is RuntimeState.RUNNING
        if restart_worker:
            if worker.state is not MemoryWorkerState.BLOCKED or worker.busy:
                raise RuntimeStateError(
                    "running runtime can only abandon a failed job after its memory worker is blocked and idle"
                )
            await worker.wait_stopped()
        abandonment = await asyncio.to_thread(
            self.components.workflow.jobs.abandon_failed,
            failed_job,
            reason=reason,
        )
        if restart_worker:
            await worker.start()
        return MemoryJobAbandonResult(failed_job, abandonment, restart_worker)

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

    async def record_memory_use(
        self,
        *,
        memory_uris: tuple[MemoryURI | str, ...] = (),
        summary_references: tuple[ConversationSummaryReference, ...] = (),
        used_at: datetime | None = None,
    ) -> MemoryUseReceipt:
        """手动兼容入口；标准检索链已按最终模型可见 Context 自动计入，勿重复上报。"""

        self._require_initialized("memory use receipt")
        if not isinstance(memory_uris, tuple):
            raise TypeError("memory_uris must be a tuple")
        parsed_uris = tuple(MemoryURI.parse(uri) for uri in memory_uris)
        for uri in parsed_uris:
            uri.to_address()
        if len(parsed_uris) != len(set(parsed_uris)):
            raise ValueError("memory_uris must be unique")
        if not isinstance(summary_references, tuple) or any(
            not isinstance(reference, ConversationSummaryReference)
            for reference in summary_references
        ):
            raise TypeError("summary_references must contain ConversationSummaryReference values")
        if len({reference.identity for reference in summary_references}) != len(summary_references):
            raise ValueError("summary_references must be unique")
        timestamp = used_at or datetime.now(timezone.utc)
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ValueError("used_at must be timezone-aware")
        if parsed_uris:
            await self.components.memory.lifecycle.record_use(parsed_uris, used_at=timestamp)
        if summary_references:
            await asyncio.to_thread(
                self.components.conversation.summary_use.record_use,
                summary_references,
                used_at=timestamp,
            )
        return MemoryUseReceipt(parsed_uris, summary_references, timestamp)

    async def maintain_conversation(
        self,
        address: ConversationAddress,
        *,
        now: datetime | None = None,
    ) -> ConversationLifecycleMaintenanceResult:
        """显式执行一次单 Conversation 维护；自动批量维护由 LifecycleWorker 负责。"""

        if not isinstance(address, ConversationAddress):
            raise TypeError("address must be ConversationAddress")
        if not self.initialized:
            raise RuntimeStateError("conversation maintenance requires an initialized runtime")
        return await self.components.workflow.lifecycle.maintain_once(address, now=now)

    async def health(self, *, deep: bool = False) -> RuntimeHealthReport:
        """返回无副作用健康报告；deep 额外探测模型与两套派生索引。"""

        return await self._health.report(self._state.value, deep=deep)

    async def refresh_observability(self) -> ObservabilitySnapshot:
        """只读刷新队列、Worker 和锁聚合值；采集失败不阻断业务或指标端点。"""

        attributes: dict[str, str | int | float | bool] = {}
        status = ObservationStatus.SUCCESS
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
            status = ObservationStatus.DEGRADED
            attributes["queue_error_type"] = type(exc).__name__
        snapshotter = getattr(self.components.infrastructure.path_lock.lock_store, "observability_snapshot", None)
        if callable(snapshotter):
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
                status = ObservationStatus.DEGRADED
                attributes["lock_error_type"] = type(exc).__name__
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
        """先停止生命周期维护，再停止 Job Worker 并关闭 Runtime。"""

        if self._state is RuntimeState.CLOSED:
            return
        try:
            if self._state is not RuntimeState.CREATED:
                try:
                    await self.components.workflow.lifecycle_worker.stop()
                finally:
                    await self.components.workflow.worker.stop()
        finally:
            try:
                await self.components.conversation.summary_vector_index.close()
            finally:
                try:
                    await self.components.memory.vector_index.close()
                finally:
                    try:
                        await self.components.models.aclose()
                    finally:
                        self.components.behavior.database.close()
                        manager = self.components.infrastructure.managed_observability
                        if manager is not None:
                            manager.close()
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
            if current.is_symlink():
                raise RuntimeInitializationError("runtime storage path cannot traverse a symbolic link")
            missing.append(current)
            parent = current.parent
            if parent == current:
                raise RuntimeInitializationError("runtime storage root has no existing parent")
            current = parent
        if current.is_symlink() or not current.is_dir():
            raise RuntimeInitializationError("runtime storage ancestor must be a real directory")
        for directory in reversed(missing):
            if directory.is_symlink():
                raise RuntimeInitializationError("runtime storage path cannot traverse a symbolic link")
            try:
                directory.mkdir(mode=0o700)
            except FileExistsError:
                pass
            if directory.is_symlink() or not directory.is_dir():
                raise RuntimeInitializationError("runtime storage path is not a real directory")
            try:
                directory.chmod(0o700)
            except OSError:
                pass

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
        try:
            observer.record(
                ObservationEvent(
                    category=category,
                    operation=operation,
                    status=status,
                    duration_seconds=max(0.0, time.monotonic() - started),
                    attributes=attributes,
                )
            )
        except Exception:
            pass


__all__ = [
    "MemoryJobAbandonResult",
    "RuntimeConversationProtocolIngestResult",
    "Runtime",
    "RuntimeInitialization",
    "RuntimeInitializationError",
    "RuntimeState",
    "RuntimeStateError",
]
