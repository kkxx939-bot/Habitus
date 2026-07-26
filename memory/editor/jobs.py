"""按 memory-root 严格排序的跨 Conversation 耐久记忆任务。"""

from __future__ import annotations

import hashlib
import json
from asyncio import to_thread
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date, datetime, timezone
from enum import Enum
from pathlib import Path
from uuid import uuid4

from foundation.integrity import canonical_json
from infrastructure.store.contracts import PathLock
from infrastructure.store.filesystem import (
    atomic_create_bytes,
    atomic_replace_bytes,
    atomic_temporary_destination,
    read_regular_bytes,
)
from memory.conversation import ConversationAddress, ConversationMessageJournal
from memory.editor.engine import MemoryEditor
from memory.editor.transaction import MemoryCommitResult
from memory.editor.transaction_log import (
    MemoryTransactionJournalError,
    MemoryTransactionJournalState,
)
from memory.semantic import MemorySemanticRefresher
from pre.conversation import ConversationSegment

_JOB_SCHEMA = "memory_job_v2"
_MAX_JOB_BYTES = 64 * 1024
_MAX_JOB_FILES = 100_000
_MAX_ERROR_CHARS = 2_000


class MemoryJobError(RuntimeError):
    """MemoryJob 无法安全排队、认领、执行或推进状态。"""


class MemoryJobBlockedError(MemoryJobError):
    """最早任务失败或仍在运行，后续序号不能越过。"""


class MemoryJobExecutionError(MemoryJobError):
    """一次后台记忆任务执行失败并已记录重试状态。"""


class MemoryJobStatus(str, Enum):
    """后台记忆任务的耐久状态。"""

    STAGED = "staged"
    QUEUED = "queued"
    RUNNING = "running"
    FAILED = "failed"
    COMMITTED = "committed"


@dataclass(frozen=True)
class MemoryJobConfig:
    """跨 Conversation 队列的显式重试与锁边界。"""

    max_attempts: int = 3
    lock_ttl_seconds: int = 30
    lock_wait_timeout_seconds: float = 5.0
    lock_retry_delay_seconds: float = 0.01

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_attempts, bool)
            or not isinstance(self.max_attempts, int)
            or not 1 <= self.max_attempts <= 20
        ):
            raise ValueError("max_attempts must be between 1 and 20")
        if (
            isinstance(self.lock_ttl_seconds, bool)
            or not isinstance(self.lock_ttl_seconds, int)
            or not 1 <= self.lock_ttl_seconds <= 3_600
        ):
            raise ValueError("lock_ttl_seconds must be between 1 and 3600")
        for name, value, maximum in (
            ("lock_wait_timeout_seconds", self.lock_wait_timeout_seconds, 60.0),
            ("lock_retry_delay_seconds", self.lock_retry_delay_seconds, 1.0),
        ):
            if isinstance(value, bool) or not isinstance(value, int | float) or not 0 < float(value) <= maximum:
                raise ValueError(f"{name} must be greater than zero and at most {maximum:g}")


@dataclass(frozen=True)
class MemoryJob:
    """一个不可变 ConversationSegment 对应的有序后台工作记录。"""

    memory_sequence: int
    conversation_id: str
    started_on: date
    segment_id: str
    source_segment_digest: str
    transaction_id: str
    status: MemoryJobStatus
    attempts: int
    claim_id: str | None
    last_error: str | None
    created_at: datetime
    updated_at: datetime

    def __post_init__(self) -> None:
        if (
            isinstance(self.memory_sequence, bool)
            or not isinstance(self.memory_sequence, int)
            or self.memory_sequence <= 0
        ):
            raise ValueError("memory_sequence must be a positive integer")
        for name in ("conversation_id", "segment_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value or value != value.strip():
                raise ValueError(f"{name} must be non-empty normalized text")
        if isinstance(self.started_on, datetime) or not isinstance(self.started_on, date):
            raise ValueError("started_on must be a calendar date")
        if not self._hex(self.source_segment_digest, 64):
            raise ValueError("source_segment_digest must be lowercase SHA-256 text")
        if not self._hex(self.transaction_id, 32):
            raise ValueError("transaction_id must be 32 lowercase hexadecimal characters")
        try:
            status = MemoryJobStatus(self.status)
        except ValueError as exc:
            raise ValueError("memory job contains an unsupported status") from exc
        object.__setattr__(self, "status", status)
        if isinstance(self.attempts, bool) or not isinstance(self.attempts, int) or self.attempts < 0:
            raise ValueError("memory job attempts must be non-negative")
        if status is MemoryJobStatus.RUNNING:
            if not isinstance(self.claim_id, str) or not self._hex(self.claim_id, 32):
                raise ValueError("running memory job requires a claim_id")
        elif self.claim_id is not None:
            raise ValueError("non-running memory job cannot retain a claim_id")
        if status is MemoryJobStatus.STAGED and (self.attempts != 0 or self.last_error is not None):
            raise ValueError("staged memory job cannot contain attempts or an execution error")
        if self.last_error is not None and (
            not isinstance(self.last_error, str) or not self.last_error or len(self.last_error) > _MAX_ERROR_CHARS
        ):
            raise ValueError("memory job last_error is invalid")
        for name in ("created_at", "updated_at"):
            value = getattr(self, name)
            if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
                raise ValueError(f"memory job {name} must be timezone-aware")
            object.__setattr__(self, name, value.astimezone(timezone.utc))
        if self.updated_at < self.created_at:
            raise ValueError("memory job updated_at cannot precede created_at")

    @property
    def source_identity(self) -> tuple[str, date, str, str]:
        return (
            self.conversation_id,
            self.started_on,
            self.segment_id,
            self.source_segment_digest,
        )

    @staticmethod
    def _hex(value: object, length: int) -> bool:
        if not isinstance(value, str) or len(value) != length or value != value.lower():
            return False
        try:
            int(value, 16)
        except ValueError:
            return False
        return True


class MemoryJobStore:
    """用一个 memory-root 队列锁维护全局单调序号和任务状态。"""

    def __init__(
        self,
        root: str | Path,
        path_lock: PathLock,
        *,
        memory_root: str | Path,
        config: MemoryJobConfig | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        requested = Path(root).expanduser().absolute()
        if requested.is_symlink():
            raise MemoryJobError("memory job root cannot be a symbolic link")
        requested_memory_root = Path(memory_root).expanduser().absolute()
        if requested_memory_root.is_symlink():
            raise MemoryJobError("bound memory root cannot be a symbolic link")
        if not isinstance(path_lock, PathLock):
            raise TypeError("path_lock must be a PathLock")
        if config is not None and not isinstance(config, MemoryJobConfig):
            raise TypeError("config must be a MemoryJobConfig")
        if clock is not None and not callable(clock):
            raise TypeError("clock must be callable")
        self.root = requested.resolve(strict=False)
        self.memory_root = requested_memory_root.resolve(strict=False)
        self.jobs_root = self.root / "jobs"
        try:
            self.jobs_root.relative_to(self.memory_root)
        except ValueError:
            pass
        else:
            raise MemoryJobError("memory jobs must be stored outside the L2 memory tree")
        self.path_lock = path_lock
        self.config = config or MemoryJobConfig()
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        digest = hashlib.sha256(str(self.memory_root).encode("utf-8")).hexdigest()
        self.lock_key = f"memory-job-root:{digest}"

    def stage(
        self,
        address: ConversationAddress,
        segment: ConversationSegment,
    ) -> MemoryJob:
        """在 history 发布前幂等建立不可执行 outbox 并分配全局序号。"""

        if not isinstance(address, ConversationAddress):
            raise TypeError("address must be a ConversationAddress")
        if not isinstance(segment, ConversationSegment):
            raise TypeError("segment must be a ConversationSegment")
        if segment.conversation_id != address.conversation_id:
            raise ValueError("conversation address does not match the sealed segment")
        source_key = self._source_key(address, segment)
        with self.path_lock.acquire(
            self.lock_key,
            ttl_seconds=self.config.lock_ttl_seconds,
            wait_timeout_seconds=self.config.lock_wait_timeout_seconds,
            retry_delay_seconds=self.config.lock_retry_delay_seconds,
        ) as guard:
            with guard.fenced():
                existing = self._try_read_path(self._path(source_key))
                if existing is not None:
                    self._require_source(existing, address, segment)
                    return existing
                jobs = self._read_all()
                sequence = max((job.memory_sequence for job in jobs), default=0) + 1
                timestamp = self._timestamp()
                transaction_id = hashlib.sha256(f"{sequence}\0{source_key}".encode()).hexdigest()[:32]
                job = MemoryJob(
                    memory_sequence=sequence,
                    conversation_id=segment.conversation_id,
                    started_on=address.started_on,
                    segment_id=segment.segment_id,
                    source_segment_digest=segment.digest,
                    transaction_id=transaction_id,
                    status=MemoryJobStatus.STAGED,
                    attempts=0,
                    claim_id=None,
                    last_error=None,
                    created_at=timestamp,
                    updated_at=timestamp,
                )
                atomic_create_bytes(
                    self._path(source_key),
                    self._encode(job),
                    artifact_root=self.root,
                )
                return job

    def activate(self, job: MemoryJob) -> MemoryJob:
        """仅在对应 history 已耐久发布后把 STAGED 任务开放给 Worker。"""

        if not isinstance(job, MemoryJob):
            raise TypeError("job must be a MemoryJob")
        with self.path_lock.acquire(
            self.lock_key,
            ttl_seconds=self.config.lock_ttl_seconds,
            wait_timeout_seconds=self.config.lock_wait_timeout_seconds,
            retry_delay_seconds=self.config.lock_retry_delay_seconds,
        ) as guard:
            with guard.fenced():
                current = self._try_read_path(self._path(self._source_key_from_job(job)))
                if current is None:
                    raise MemoryJobError("staged memory job disappeared before activation")
                if current.source_identity != job.source_identity:
                    raise MemoryJobError("memory job source changed before activation")
                if current.status is not MemoryJobStatus.STAGED:
                    return current
                activated = self._replace(
                    current,
                    status=MemoryJobStatus.QUEUED,
                    claim_id=None,
                    last_error=None,
                )
                self._write(activated)
                return activated

    def oldest_uncommitted(self) -> MemoryJob | None:
        """返回最早未完成任务，不允许调用者越过它。"""

        with self.path_lock.acquire(
            self.lock_key,
            ttl_seconds=self.config.lock_ttl_seconds,
            wait_timeout_seconds=self.config.lock_wait_timeout_seconds,
            retry_delay_seconds=self.config.lock_retry_delay_seconds,
        ) as guard:
            with guard.fenced():
                return next(
                    (job for job in self._read_all() if job.status is not MemoryJobStatus.COMMITTED),
                    None,
                )

    def claim(self, job: MemoryJob) -> MemoryJob:
        """只认领当前最早 QUEUED 任务。"""

        if not isinstance(job, MemoryJob):
            raise TypeError("job must be a MemoryJob")
        with self.path_lock.acquire(
            self.lock_key,
            ttl_seconds=self.config.lock_ttl_seconds,
            wait_timeout_seconds=self.config.lock_wait_timeout_seconds,
            retry_delay_seconds=self.config.lock_retry_delay_seconds,
        ) as guard:
            with guard.fenced():
                current = self._oldest_required(job)
                if current.status is MemoryJobStatus.FAILED:
                    raise MemoryJobBlockedError("oldest memory job exhausted retries and blocks later jobs")
                if current.status is MemoryJobStatus.RUNNING:
                    raise MemoryJobBlockedError("oldest memory job is already running")
                if current.status is MemoryJobStatus.COMMITTED:
                    raise MemoryJobError("committed memory job cannot be claimed")
                if current.status is MemoryJobStatus.STAGED:
                    raise MemoryJobBlockedError("oldest memory job has not published its history source")
                claimed = self._replace(
                    current,
                    status=MemoryJobStatus.RUNNING,
                    attempts=current.attempts + 1,
                    claim_id=uuid4().hex,
                    last_error=None,
                )
                self._write(claimed)
                return claimed

    def complete(self, job: MemoryJob) -> MemoryJob:
        """将当前 claim 对应的最早任务标记 COMMITTED。"""

        return self._finish(job, committed=True, error=None)

    def fail(self, job: MemoryJob, error: BaseException) -> MemoryJob:
        """记录失败；达到配置次数后保持 FAILED 并阻塞后项。"""

        if not isinstance(error, BaseException):
            raise TypeError("error must be an exception")
        message = " ".join(str(error).split()) or type(error).__name__
        return self._finish(job, committed=False, error=message[:_MAX_ERROR_CHARS])

    def requeue_running(self, job: MemoryJob) -> MemoryJob:
        """仅供确认原 Worker 已停止后的崩溃恢复使用。"""

        if job.status is not MemoryJobStatus.RUNNING:
            raise ValueError("only a running memory job can be requeued")
        with self.path_lock.acquire(
            self.lock_key,
            ttl_seconds=self.config.lock_ttl_seconds,
            wait_timeout_seconds=self.config.lock_wait_timeout_seconds,
            retry_delay_seconds=self.config.lock_retry_delay_seconds,
        ) as guard:
            with guard.fenced():
                current = self._oldest_required(job)
                if current.status is MemoryJobStatus.QUEUED:
                    return current
                if current.status is not MemoryJobStatus.RUNNING:
                    raise MemoryJobError("memory job is no longer in a recoverable running state")
                if current.claim_id != job.claim_id:
                    raise MemoryJobBlockedError(
                        "memory job is owned by a newer worker claim and cannot be requeued"
                    )
                queued = self._replace(
                    current,
                    status=MemoryJobStatus.QUEUED,
                    claim_id=None,
                    last_error="worker stopped before completing the memory job",
                )
                self._write(queued)
                return queued

    def retry_failed(self, job: MemoryJob) -> MemoryJob:
        """显式人工处置后重新开放最早 FAILED 任务。"""

        if job.status is not MemoryJobStatus.FAILED:
            raise ValueError("only a failed memory job can be retried")
        with self.path_lock.acquire(
            self.lock_key,
            ttl_seconds=self.config.lock_ttl_seconds,
            wait_timeout_seconds=self.config.lock_wait_timeout_seconds,
            retry_delay_seconds=self.config.lock_retry_delay_seconds,
        ) as guard:
            with guard.fenced():
                current = self._oldest_required(job)
                if current.status is MemoryJobStatus.QUEUED:
                    return current
                if current.status is not MemoryJobStatus.FAILED:
                    raise MemoryJobError("memory job is no longer in a retryable failed state")
                if current != job:
                    raise MemoryJobBlockedError(
                        "failed memory job changed after this retry decision was made"
                    )
                queued = self._replace(
                    current,
                    status=MemoryJobStatus.QUEUED,
                    attempts=0,
                    claim_id=None,
                    last_error=None,
                )
                self._write(queued)
                return queued

    def _finish(
        self,
        job: MemoryJob,
        *,
        committed: bool,
        error: str | None,
    ) -> MemoryJob:
        if not isinstance(job, MemoryJob):
            raise TypeError("job must be a MemoryJob")
        with self.path_lock.acquire(
            self.lock_key,
            ttl_seconds=self.config.lock_ttl_seconds,
            wait_timeout_seconds=self.config.lock_wait_timeout_seconds,
            retry_delay_seconds=self.config.lock_retry_delay_seconds,
        ) as guard:
            with guard.fenced():
                current = self._oldest_required(job)
                if current.status is not MemoryJobStatus.RUNNING or current.claim_id != job.claim_id:
                    raise MemoryJobError("memory job claim is no longer current")
                if committed:
                    status = MemoryJobStatus.COMMITTED
                elif current.attempts >= self.config.max_attempts:
                    status = MemoryJobStatus.FAILED
                else:
                    status = MemoryJobStatus.QUEUED
                finished = self._replace(
                    current,
                    status=status,
                    claim_id=None,
                    last_error=error,
                )
                self._write(finished)
                return finished

    def _oldest_required(self, expected: MemoryJob) -> MemoryJob:
        oldest = next(
            (item for item in self._read_all() if item.status is not MemoryJobStatus.COMMITTED),
            None,
        )
        if oldest is None or oldest.source_identity != expected.source_identity:
            raise MemoryJobError("memory job is not the oldest uncommitted sequence")
        return oldest

    def _read_all(self) -> tuple[MemoryJob, ...]:
        if not self.jobs_root.exists():
            return ()
        if self.jobs_root.is_symlink() or not self.jobs_root.is_dir():
            raise MemoryJobError("memory jobs path is not a safe directory")
        jobs: list[MemoryJob] = []
        for entry_count, child in enumerate(self.jobs_root.iterdir(), start=1):
            if entry_count > _MAX_JOB_FILES:
                raise MemoryJobError("memory job entry count exceeds its safety bound")
            if child.is_symlink() or not child.is_file():
                raise MemoryJobError("memory jobs directory contains an unsupported entry")
            temporary_destination = atomic_temporary_destination(child.name)
            if temporary_destination is not None and MemoryJob._hex(
                Path(temporary_destination).stem, 64
            ) and Path(temporary_destination).suffix == ".json":
                continue
            if child.suffix != ".json":
                raise MemoryJobError("memory jobs directory contains an unsupported entry")
            jobs.append(self._read_path(child))
        jobs.sort(key=lambda job: job.memory_sequence)
        sequences = tuple(job.memory_sequence for job in jobs)
        if len(sequences) != len(set(sequences)):
            raise MemoryJobError("memory jobs contain duplicate memory_sequence values")
        return tuple(jobs)

    def _read_path(self, path: Path) -> MemoryJob:
        try:
            raw = json.loads(
                read_regular_bytes(
                    path,
                    artifact_root=self.root,
                    max_bytes=_MAX_JOB_BYTES,
                )
            )
            return self._parse(raw)
        except Exception as exc:
            if isinstance(exc, MemoryJobError):
                raise
            raise MemoryJobError("failed to read memory job") from exc

    def _try_read_path(self, path: Path) -> MemoryJob | None:
        try:
            return self._read_path(path)
        except MemoryJobError as exc:
            if isinstance(exc.__cause__, FileNotFoundError):
                return None
            raise

    def _write(self, job: MemoryJob) -> None:
        atomic_replace_bytes(
            self._path(self._source_key_from_job(job)),
            self._encode(job),
            artifact_root=self.root,
        )

    def _path(self, source_key: str) -> Path:
        if not MemoryJob._hex(source_key, 64):
            raise ValueError("memory job source key must be lowercase SHA-256 text")
        return self.jobs_root / f"{source_key}.json"

    @staticmethod
    def _source_key(
        address: ConversationAddress,
        segment: ConversationSegment,
    ) -> str:
        return hashlib.sha256(
            (
                f"{address.conversation_id}\0{address.started_on.isoformat()}\0{segment.segment_id}\0{segment.digest}"
            ).encode()
        ).hexdigest()

    @staticmethod
    def _source_key_from_job(job: MemoryJob) -> str:
        address = ConversationAddress(job.conversation_id, job.started_on)
        return hashlib.sha256(
            (
                f"{address.conversation_id}\0{address.started_on.isoformat()}\0"
                f"{job.segment_id}\0{job.source_segment_digest}"
            ).encode()
        ).hexdigest()

    @staticmethod
    def _require_source(
        job: MemoryJob,
        address: ConversationAddress,
        segment: ConversationSegment,
    ) -> None:
        if (
            job.conversation_id != address.conversation_id
            or job.started_on != address.started_on
            or job.segment_id != segment.segment_id
            or job.source_segment_digest != segment.digest
        ):
            raise MemoryJobError("memory job source key conflicts with another segment")

    def _replace(
        self,
        job: MemoryJob,
        *,
        status: MemoryJobStatus,
        attempts: int | None = None,
        claim_id: str | None,
        last_error: str | None,
    ) -> MemoryJob:
        return MemoryJob(
            memory_sequence=job.memory_sequence,
            conversation_id=job.conversation_id,
            started_on=job.started_on,
            segment_id=job.segment_id,
            source_segment_digest=job.source_segment_digest,
            transaction_id=job.transaction_id,
            status=status,
            attempts=job.attempts if attempts is None else attempts,
            claim_id=claim_id,
            last_error=last_error,
            created_at=job.created_at,
            updated_at=self._timestamp(),
        )

    @staticmethod
    def _encode(job: MemoryJob) -> bytes:
        return canonical_json(
            {
                "schema": _JOB_SCHEMA,
                "memory_sequence": job.memory_sequence,
                "conversation_id": job.conversation_id,
                "started_on": job.started_on.isoformat(),
                "segment_id": job.segment_id,
                "source_segment_digest": job.source_segment_digest,
                "transaction_id": job.transaction_id,
                "status": job.status.value,
                "attempts": job.attempts,
                "claim_id": job.claim_id,
                "last_error": job.last_error,
                "created_at": MemoryJobStore._format_time(job.created_at),
                "updated_at": MemoryJobStore._format_time(job.updated_at),
            }
        ).encode("utf-8")

    @staticmethod
    def _parse(value: object) -> MemoryJob:
        expected = {
            "schema",
            "memory_sequence",
            "conversation_id",
            "started_on",
            "segment_id",
            "source_segment_digest",
            "transaction_id",
            "status",
            "attempts",
            "claim_id",
            "last_error",
            "created_at",
            "updated_at",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise MemoryJobError("memory job has an invalid shape")
        if value["schema"] != _JOB_SCHEMA:
            raise MemoryJobError("memory job has an unsupported schema")
        try:
            return MemoryJob(
                memory_sequence=value["memory_sequence"],
                conversation_id=value["conversation_id"],
                started_on=date.fromisoformat(value["started_on"]),
                segment_id=value["segment_id"],
                source_segment_digest=value["source_segment_digest"],
                transaction_id=value["transaction_id"],
                status=MemoryJobStatus(value["status"]),
                attempts=value["attempts"],
                claim_id=value["claim_id"],
                last_error=value["last_error"],
                created_at=MemoryJobStore._parse_time(value["created_at"]),
                updated_at=MemoryJobStore._parse_time(value["updated_at"]),
            )
        except (TypeError, ValueError) as exc:
            raise MemoryJobError("memory job fields are invalid") from exc

    @staticmethod
    def _format_time(value: datetime) -> str:
        return value.astimezone(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")

    @staticmethod
    def _parse_time(value: object) -> datetime:
        if not isinstance(value, str):
            raise ValueError("memory job timestamp must be a string")
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError("memory job timestamp must include a timezone")
        return parsed.astimezone(timezone.utc)

    def _timestamp(self) -> datetime:
        value = self.clock()
        if not isinstance(value, datetime):
            raise TypeError("memory job clock must return a datetime")
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("memory job clock must return a timezone-aware datetime")
        return value.astimezone(timezone.utc)


@dataclass(frozen=True)
class MemoryJobRunResult:
    """一次 Worker 调用处理的任务和记忆事务结果。"""

    job: MemoryJob | None
    commit: MemoryCommitResult | None
    recovered: bool = False
    semantic_refreshed: bool = False
    journal_cleaned: bool = True


class MemoryJobRunner:
    """异步执行最早任务，绝不越过同一 memory-root 的前序任务。"""

    def __init__(
        self,
        store: MemoryJobStore,
        conversations: ConversationMessageJournal,
        editor: MemoryEditor,
        semantic_refresher: MemorySemanticRefresher,
    ) -> None:
        if not isinstance(store, MemoryJobStore):
            raise TypeError("store must be a MemoryJobStore")
        if not isinstance(conversations, ConversationMessageJournal):
            raise TypeError("conversations must be a ConversationMessageJournal")
        if not isinstance(editor, MemoryEditor):
            raise TypeError("editor must be a MemoryEditor")
        if not isinstance(semantic_refresher, MemorySemanticRefresher):
            raise TypeError("semantic_refresher must be a MemorySemanticRefresher")
        if store.memory_root != editor.transaction.tree.root:
            raise ValueError("MemoryJobStore is bound to another memory root")
        if semantic_refresher.tree.root != editor.transaction.tree.root:
            raise ValueError("semantic_refresher and MemoryEditor must use the same memory root")
        self.store = store
        self.conversations = conversations
        self.editor = editor
        self.semantic_refresher = semantic_refresher

    async def run_next(self) -> MemoryJobRunResult:
        """恢复未完成事务后，只处理当前最早的一个任务。"""

        self.editor.transaction.recover_pending(discard_terminal=False)
        job = self.store.oldest_uncommitted()
        if job is None:
            return MemoryJobRunResult(job=None, commit=None)

        if job.status is MemoryJobStatus.STAGED:
            try:
                address = ConversationAddress(job.conversation_id, job.started_on)
                _start_sequence, end_sequence = self.conversations.layout.segment_range(job.segment_id)
                sealed = self.conversations.seal(address, through_sequence=end_sequence)
                if sealed.segment.segment_id != job.segment_id or sealed.segment.digest != job.source_segment_digest:
                    raise MemoryJobError("recovered ConversationSegment does not match its staged MemoryJob")
                job = self.store.activate(job)
            except Exception as exc:
                raise MemoryJobExecutionError(
                    "staged memory job could not publish and verify its ConversationSegment"
                ) from exc

        receipt = self.editor.transaction.journal.try_read(job.transaction_id)
        if receipt is not None and receipt.state is MemoryTransactionJournalState.COMMITTED:
            if job.status is MemoryJobStatus.FAILED:
                raise MemoryJobBlockedError("oldest memory job exhausted semantic refresh retries")
            if job.status is MemoryJobStatus.RUNNING:
                raise MemoryJobBlockedError(
                    "running memory job has a COMMITTED receipt; confirm worker death before requeue"
                )
            claimed = self.store.claim(job)
            try:
                await to_thread(self._refresh_receipt, job.transaction_id)
                committed = self.store.complete(claimed)
            except Exception as exc:
                failed = self.store.fail(claimed, exc)
                raise MemoryJobExecutionError(
                    "committed L2 memory remains durable, but semantic refresh "
                    f"failed with job status {failed.status.value}"
                ) from exc
            cleaned = self._discard_receipt(job.transaction_id)
            return MemoryJobRunResult(
                job=committed,
                commit=None,
                recovered=True,
                semantic_refreshed=True,
                journal_cleaned=cleaned,
            )
        if receipt is not None and receipt.state is MemoryTransactionJournalState.ROLLED_BACK:
            self.editor.transaction.journal.discard_terminal(job.transaction_id)

        if job.status is MemoryJobStatus.RUNNING:
            raise MemoryJobBlockedError(
                "running memory job has no terminal transaction receipt; confirm worker death before requeue"
            )
        claimed = self.store.claim(job)
        try:
            address = ConversationAddress(claimed.conversation_id, claimed.started_on)
            segment = self.conversations.read_segment(address, claimed.segment_id)
            if segment.digest != claimed.source_segment_digest:
                raise MemoryJobError("sealed ConversationSegment digest does not match its MemoryJob")
            commit = await self.editor.edit(
                segment,
                transaction_id=claimed.transaction_id,
                retain_transaction_journal=True,
            )
            await to_thread(self._refresh_receipt, claimed.transaction_id)
            completed = self.store.complete(claimed)
        except Exception as exc:
            failed = self.store.fail(claimed, exc)
            raise MemoryJobExecutionError(
                f"memory job {failed.memory_sequence} failed with status {failed.status.value}"
            ) from exc
        cleaned = self._discard_receipt(claimed.transaction_id)
        return MemoryJobRunResult(
            job=completed,
            commit=commit,
            semantic_refreshed=True,
            journal_cleaned=cleaned,
        )

    def _refresh_receipt(self, transaction_id: str) -> None:
        receipt = self.editor.transaction.journal.read(transaction_id)
        if receipt.state is not MemoryTransactionJournalState.COMMITTED:
            raise MemoryJobError("semantic refresh requires a COMMITTED memory transaction receipt")
        addresses = tuple(entry.uri.to_address() for entry in receipt.entries)
        self.semantic_refresher.refresh_for_many(addresses)

    def _discard_receipt(self, transaction_id: str) -> bool:
        try:
            self.editor.transaction.journal.discard_terminal(transaction_id)
        except MemoryTransactionJournalError:
            return False
        return True


class ConversationMemoryEnqueuer:
    """把 history 封存结果幂等接入 memory-root 顺序队列。"""

    def __init__(
        self,
        conversations: ConversationMessageJournal,
        jobs: MemoryJobStore,
    ) -> None:
        if not isinstance(conversations, ConversationMessageJournal):
            raise TypeError("conversations must be a ConversationMessageJournal")
        if not isinstance(jobs, MemoryJobStore):
            raise TypeError("jobs must be a MemoryJobStore")
        self.conversations = conversations
        self.jobs = jobs

    def seal_and_enqueue(
        self,
        address: ConversationAddress,
        *,
        through_sequence: int,
    ) -> MemoryJob:
        """先建立 STAGED outbox，发布原文后再激活后台任务。"""

        def stage_before_publish(segment: ConversationSegment) -> None:
            self.jobs.stage(address, segment)

        sealed = self.conversations.seal(
            address,
            through_sequence=through_sequence,
            before_history_publish=stage_before_publish,
        )
        staged = self.jobs.stage(address, sealed.segment)
        return self.jobs.activate(staged)


__all__ = [
    "ConversationMemoryEnqueuer",
    "MemoryJob",
    "MemoryJobBlockedError",
    "MemoryJobConfig",
    "MemoryJobError",
    "MemoryJobExecutionError",
    "MemoryJobRunResult",
    "MemoryJobRunner",
    "MemoryJobStatus",
    "MemoryJobStore",
]
