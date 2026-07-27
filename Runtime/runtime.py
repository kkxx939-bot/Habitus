"""记忆主链唯一组合根的显式生命周期。"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path

from Config import M2BOSConfig
from memory.conversation import ConversationAddress
from memory.intention import MemoryIntentionRecallScope
from memory.model import MemoryKind
from memory.retrieval import MemorySearchResult
from memory.uri import MemoryURI
from memory.workflow import (
    ConversationLifecycleMaintenanceResult,
    MemoryJob,
    MemoryJobRunResult,
    MemoryJobStatus,
)
from Runtime.components import RuntimeComponents
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
    recovered_transaction_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.memory_root, Path) or not self.memory_root.is_absolute():
            raise ValueError("memory_root must be an absolute Path")
        if not isinstance(self.recovered_transaction_ids, tuple) or any(
            not isinstance(identifier, str) or not identifier for identifier in self.recovered_transaction_ids
        ):
            raise TypeError("recovered_transaction_ids must contain non-empty strings")


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


class Runtime:
    """只管理组装、初始化顺序和 Job 执行入口，不承载领域规则。"""

    def __init__(
        self,
        config: M2BOSConfig,
        components: RuntimeComponents,
    ) -> None:
        if not isinstance(config, M2BOSConfig):
            raise TypeError("config must be M2BOSConfig")
        if not isinstance(components, RuntimeComponents):
            raise TypeError("components must be RuntimeComponents")
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

    @property
    def state(self) -> RuntimeState:
        return self._state

    @property
    def initialized(self) -> bool:
        return self._state in {RuntimeState.READY, RuntimeState.RUNNING}

    def initialize(self) -> RuntimeInitialization:
        """初始化树；无任务时恢复孤儿事务，有任务时留给租约持有者恢复。"""

        if self._state is RuntimeState.CLOSED:
            raise RuntimeStateError("closed runtime cannot be initialized")
        if self._initialization is not None:
            return self._initialization
        self._ensure_storage_root()
        self.components.infrastructure.initialize()
        memory_root = self.components.memory.tree.initialize()
        self.components.workflow.jobs.initialize()
        oldest_job = self.components.workflow.jobs.oldest_uncommitted()
        recovered = self.components.workflow.runner.transaction_recovery.recover_pending() if oldest_job is None else ()
        initialization = RuntimeInitialization(
            memory_root=memory_root,
            recovered_transaction_ids=recovered,
        )
        self._initialization = initialization
        self._state = RuntimeState.READY
        return initialization

    async def start(self) -> None:
        """完成初始化后依次启动 Job Worker 和生命周期维护 Worker。"""

        if self._state is RuntimeState.CLOSED:
            raise RuntimeStateError("closed runtime cannot be started")
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
        return MemoryJobRetryResult(
            failed_job=failed_job,
            reopened_job=reopened,
            worker_restarted=restart_worker,
        )

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


__all__ = [
    "Runtime",
    "RuntimeInitialization",
    "RuntimeInitializationError",
    "RuntimeState",
    "RuntimeStateError",
]
