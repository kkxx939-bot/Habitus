"""按 memory-root 严格排序且具有持久化租约的记忆任务文件队列。"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

from foundation.integrity import canonical_json
from infrastructure.store.contracts import PathLock
from infrastructure.store.filesystem import (
    atomic_create_bytes,
    atomic_replace_bytes,
    atomic_temporary_destination,
    durable_unlink,
    read_regular_bytes,
)
from memory.conversation import ConversationAddress, ConversationLayout
from memory.workflow.jobs.model import (
    MEMORY_JOB_ERROR_MAX_CHARS,
    MemoryJob,
    MemoryJobAbandonment,
    MemoryJobBlockedError,
    MemoryJobConfig,
    MemoryJobError,
    MemoryJobLease,
    MemoryJobLeaseLostError,
    MemoryJobNotReadyError,
    MemoryJobStatus,
)
from pre.conversation import ConversationSegment

_JOB_SCHEMA = "memory_job_v3"
_JOB_STATE_SCHEMA = "memory_job_sequence_state_v1"


@dataclass(frozen=True)
class _MemoryJobSequenceState:
    """在 Job 清理后仍永久保存的全局序号高水位。"""

    last_allocated_memory_sequence: int

    def __post_init__(self) -> None:
        value = self.last_allocated_memory_sequence
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise MemoryJobError("memory job sequence high-watermark must be non-negative")


class MemoryJobStore:
    """在持久化队列锁内维护全局序号、重试和 Worker fencing。"""

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
        self.abandonments_root = self.root / "job_abandonments"
        self.state_path = self.jobs_root / "state.json"
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

    def initialize(self) -> int:
        """建立并校验永久序号高水位，返回最后一次已经分配的序号。"""

        with self._queue_fence():
            jobs = self._read_all()
            state = self._load_or_initialize_state(jobs)
            return state.last_allocated_memory_sequence

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
        with self._queue_fence():
            existing = self._try_read_path(self._path(source_key))
            if existing is not None:
                self._require_source(existing, address, segment)
                return existing
            jobs = self._read_all()
            state = self._load_or_initialize_state(jobs)
            sequence = state.last_allocated_memory_sequence + 1
            self._write_state(_MemoryJobSequenceState(sequence))
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
                claim_generation=0,
                worker_id=None,
                lease_expires_at=None,
                next_attempt_at=None,
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

    def high_watermark(self) -> int:
        """返回永不回退且不因 Job 清理而丢失的全局序号高水位。"""

        with self._queue_fence():
            jobs = self._read_all()
            return self._load_or_initialize_state(jobs).last_allocated_memory_sequence

    def list_for_conversation(
        self,
        address: ConversationAddress,
    ) -> tuple[MemoryJob, ...]:
        """按全局序号返回属于一个 Conversation 的全部仍保留 Job。"""

        if not isinstance(address, ConversationAddress):
            raise TypeError("address must be a ConversationAddress")
        with self._queue_fence():
            return tuple(
                job
                for job in self._read_all()
                if job.conversation_id == address.conversation_id and job.started_on == address.started_on
            )

    def discard_committed(self, job: MemoryJob) -> bool:
        """在队列栅栏内耐久删除一条已经满足外部安全门槛的终态 Job。"""

        self._require_job(job)
        if job.status is not MemoryJobStatus.COMMITTED:
            raise MemoryJobError("only a COMMITTED memory job can be discarded")
        with self._queue_fence():
            state = self._load_or_initialize_state(self._read_all())
            if state.last_allocated_memory_sequence < job.memory_sequence:
                raise MemoryJobError("memory job sequence high-watermark is behind the discarded job")
            path = self._path(self._source_key_from_job(job))
            current = self._try_read_path(path)
            if current is None:
                return False
            if current != job:
                raise MemoryJobBlockedError("memory job changed before lifecycle cleanup")
            return durable_unlink(path, artifact_root=self.root)

    def activate(self, job: MemoryJob) -> MemoryJob:
        """仅在对应 history 已耐久发布后把 STAGED 任务开放给 Worker。"""

        self._require_job(job)
        with self._queue_fence():
            current = self._try_read_path(self._path(self._source_key_from_job(job)))
            if current is None:
                raise MemoryJobError("staged memory job disappeared before activation")
            if current.source_identity != job.source_identity:
                raise MemoryJobError("memory job source changed before activation")
            if current.status is not MemoryJobStatus.STAGED:
                return current
            activated = replace(
                current,
                status=MemoryJobStatus.QUEUED,
                next_attempt_at=None,
                last_error=None,
                updated_at=self._timestamp(),
            )
            self._write(activated)
            return activated

    def oldest_uncommitted(self) -> MemoryJob | None:
        """返回最早未完成任务，不允许调用者越过它。"""

        with self._queue_fence():
            return self._oldest_uncommitted_unlocked()

    def try_read_source(
        self,
        address: ConversationAddress,
        segment_id: str,
        source_segment_digest: str,
    ) -> MemoryJob | None:
        """按不可变 ConversationSegment 身份读取 Job，不枚举或猜测来源。"""

        if not isinstance(address, ConversationAddress):
            raise TypeError("address must be a ConversationAddress")
        ConversationLayout.segment_range(segment_id)
        if not MemoryJob._hex(source_segment_digest, 64):
            raise ValueError("source_segment_digest must be lowercase SHA-256 text")
        source_key = self._source_key_values(address, segment_id, source_segment_digest)
        return self._try_read_path(self._path(source_key))

    def require_staged_ready(self, job: MemoryJob) -> MemoryJob:
        """校验最早 STAGED Job 已到耐久重试时间。"""

        self._require_job(job)
        with self._queue_fence():
            current = self._oldest_required(job)
            if current.status is not MemoryJobStatus.STAGED:
                return current
            now = self._timestamp()
            if current.next_attempt_at is not None and current.next_attempt_at > now:
                raise MemoryJobNotReadyError(current.next_attempt_at)
            return current

    def record_staged_failure(
        self,
        job: MemoryJob,
        error: BaseException,
    ) -> MemoryJob:
        """在 history 恢复失败时耐久记录退避，耗尽后阻塞全队列。"""

        self._require_job(job)
        if not isinstance(error, BaseException):
            raise TypeError("error must be an exception")
        message = (" ".join(str(error).split()) or type(error).__name__)[:MEMORY_JOB_ERROR_MAX_CHARS]
        with self._queue_fence():
            current = self._oldest_required(job)
            if current.status is not MemoryJobStatus.STAGED:
                raise MemoryJobError("memory job is no longer staged")
            if current != job:
                raise MemoryJobBlockedError("staged memory job changed before its failure was recorded")
            now = self._timestamp()
            attempts = current.attempts + 1
            exhausted = attempts >= self.config.max_attempts
            failed = replace(
                current,
                status=(MemoryJobStatus.FAILED if exhausted else MemoryJobStatus.STAGED),
                attempts=attempts,
                next_attempt_at=(
                    None if exhausted else now + timedelta(seconds=self.config.retry_delay_seconds(attempts))
                ),
                last_error=message,
                updated_at=now,
            )
            self._write(failed)
            return failed

    def claim(
        self,
        job: MemoryJob,
        worker_id: str,
    ) -> MemoryJobLease:
        """认领最早可执行任务，或在租约过期后以新 generation 接管。"""

        self._require_job(job)
        if not MemoryJob.valid_worker_id(worker_id):
            raise ValueError("worker_id must be normalized stable text")
        with self._queue_fence():
            current = self._oldest_required(job)
            now = self._timestamp()
            if current.status is MemoryJobStatus.FAILED:
                raise MemoryJobBlockedError("oldest memory job exhausted retries and blocks later jobs")
            if current.status is MemoryJobStatus.COMMITTED:
                raise MemoryJobError("committed memory job cannot be claimed")
            if current.status is MemoryJobStatus.STAGED:
                raise MemoryJobBlockedError("oldest memory job has not published its history source")
            last_error: str | None
            if current.status is MemoryJobStatus.RUNNING:
                assert current.lease_expires_at is not None
                if current.lease_expires_at > now:
                    raise MemoryJobBlockedError("oldest memory job is held by an active worker lease")
                previous_owner = current.worker_id or "unknown"
                previous_generation = current.claim_generation
                last_error = (
                    f"worker lease expired; reclaimed from {previous_owner} generation {previous_generation}"
                )[:MEMORY_JOB_ERROR_MAX_CHARS]
            else:
                if current.next_attempt_at is not None and current.next_attempt_at > now:
                    raise MemoryJobNotReadyError(current.next_attempt_at)
                last_error = current.last_error
            claimed = replace(
                current,
                status=MemoryJobStatus.RUNNING,
                attempts=current.attempts + 1,
                claim_id=uuid4().hex,
                claim_generation=current.claim_generation + 1,
                worker_id=worker_id,
                lease_expires_at=now + timedelta(seconds=self.config.lease_ttl_seconds),
                next_attempt_at=None,
                last_error=last_error,
                updated_at=now,
            )
            self._write(claimed)
            return MemoryJobLease(claimed)

    def renew(self, lease: MemoryJobLease) -> MemoryJobLease:
        """仅为仍然有效且身份完全匹配的 Worker 租约续期。"""

        self._require_lease(lease)
        with self._queue_fence():
            now = self._timestamp()
            current = self._current_lease_job(lease, now=now)
            renewed = replace(
                current,
                lease_expires_at=now + timedelta(seconds=self.config.lease_ttl_seconds),
                updated_at=now,
            )
            self._write(renewed)
            return MemoryJobLease(renewed)

    def assert_current(self, lease: MemoryJobLease) -> MemoryJob:
        """在发布边界确认租约尚未过期且未被更新 generation 接管。"""

        self._require_lease(lease)
        with self._queue_fence():
            return self._current_lease_job(lease, now=self._timestamp())

    def is_settled(self, lease: MemoryJobLease) -> bool:
        """判断同一 generation 是否已由执行方完成合法终态结算。"""

        self._require_lease(lease)
        with self._queue_fence():
            current = self._try_read_path(self._path(self._source_key_from_job(lease.job)))
            if current is None or current.source_identity != lease.source_identity:
                raise MemoryJobError("memory job disappeared while checking settlement")
            return current.claim_generation == lease.claim_generation and current.status is not MemoryJobStatus.RUNNING

    def complete(self, lease: MemoryJobLease) -> MemoryJob:
        """以 fencing 租约把最早任务标记为 COMMITTED。"""

        return self._finish(lease, committed=True, error=None, retryable=False)

    def fail(
        self,
        lease: MemoryJobLease,
        error: BaseException,
        *,
        retryable: bool = True,
        before_settlement: Callable[[], None] | None = None,
    ) -> MemoryJob:
        """在同一租约栅栏内清理未提交副作用、记录失败和退避。"""

        if not isinstance(error, BaseException):
            raise TypeError("error must be an exception")
        if not isinstance(retryable, bool):
            raise TypeError("retryable must be boolean")
        if before_settlement is not None and not callable(before_settlement):
            raise TypeError("before_settlement must be callable or None")
        message = " ".join(str(error).split()) or type(error).__name__
        return self._finish(
            lease,
            committed=False,
            error=message[:MEMORY_JOB_ERROR_MAX_CHARS],
            retryable=retryable,
            before_settlement=before_settlement,
        )

    def retry_failed(self, job: MemoryJob) -> MemoryJob:
        """显式人工处置后重新开放最早 FAILED 任务。"""

        self._require_job(job)
        if job.status is not MemoryJobStatus.FAILED:
            raise ValueError("only a failed memory job can be retried")
        with self._queue_fence():
            if self._try_read_abandonment_path(
                self._abandonment_path(self._source_key_from_job(job))
            ) is not None:
                raise MemoryJobBlockedError(
                    "failed memory job already has an abandonment decision; finish abandonment instead of retrying"
                )
            current = self._oldest_required(job)
            if current.status in {MemoryJobStatus.STAGED, MemoryJobStatus.QUEUED}:
                if current.attempts == 0 and current.last_error is None:
                    return current
                raise MemoryJobBlockedError(
                    "failed memory job has advanced into another retry cycle"
                )
            if current.status is not MemoryJobStatus.FAILED:
                raise MemoryJobError("memory job is no longer in a retryable failed state")
            if current != job:
                raise MemoryJobBlockedError("failed memory job changed after this retry decision was made")
            next_status = MemoryJobStatus.STAGED if current.claim_generation == 0 else MemoryJobStatus.QUEUED
            queued = replace(
                current,
                status=next_status,
                attempts=0,
                claim_id=None,
                worker_id=None,
                lease_expires_at=None,
                next_attempt_at=None,
                last_error=None,
                updated_at=self._timestamp(),
            )
            self._write(queued)
            return queued

    def abandon_failed(self, job: MemoryJob, *, reason: str) -> MemoryJobAbandonment:
        """记录显式人工决定后删除最早 FAILED Job，使后续序号可以继续。"""

        self._require_job(job)
        if job.status is not MemoryJobStatus.FAILED:
            raise ValueError("only a failed memory job can be abandoned")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("abandonment reason must be non-empty text")
        with self._queue_fence():
            path = self._abandonment_path(self._source_key_from_job(job))
            existing = self._try_read_abandonment_path(path)
            current_path = self._path(self._source_key_from_job(job))
            current = self._try_read_path(current_path)
            if existing is not None:
                if existing.source_identity != job.source_identity or existing.transaction_id != job.transaction_id:
                    raise MemoryJobError("memory job abandonment conflicts with the requested source")
                if current is not None:
                    if current != job:
                        raise MemoryJobBlockedError("failed memory job changed after abandonment was recorded")
                    durable_unlink(current_path, artifact_root=self.root)
                return existing
            current = self._oldest_required(job)
            if current.status is not MemoryJobStatus.FAILED:
                raise MemoryJobError("memory job is no longer in a failed state")
            if current != job:
                raise MemoryJobBlockedError("failed memory job changed after this abandonment decision")
            record = MemoryJobAbandonment.from_failed_job(
                current,
                reason=reason,
                abandoned_at=self._timestamp(),
            )
            atomic_create_bytes(path, self._encode_abandonment(record), artifact_root=self.root)
            if not durable_unlink(current_path, artifact_root=self.root):
                raise MemoryJobError("failed memory job disappeared after abandonment was recorded")
            return self._read_abandonment_path(path)

    def try_read_abandonment(self, job: MemoryJob) -> MemoryJobAbandonment | None:
        """按 Job 的不可变来源身份读取人工放弃记录。"""

        self._require_job(job)
        return self._try_read_abandonment_path(
            self._abandonment_path(self._source_key_from_job(job))
        )

    def list_abandonments(self) -> tuple[MemoryJobAbandonment, ...]:
        """按全局序号返回全部仍保留的人工失败处置记录。"""

        if not self.abandonments_root.exists():
            return ()
        if self.abandonments_root.is_symlink() or not self.abandonments_root.is_dir():
            raise MemoryJobError("memory job abandonments path is not a safe directory")
        records: list[MemoryJobAbandonment] = []
        for entry_count, child in enumerate(self.abandonments_root.iterdir(), start=1):
            if entry_count > self.config.max_files:
                raise MemoryJobError("memory job abandonment count exceeds its configured bound")
            if child.suffix != ".json" or child.is_symlink() or not child.is_file():
                raise MemoryJobError("memory job abandonments directory contains an unsupported entry")
            records.append(self._read_abandonment_path(child))
        return tuple(sorted(records, key=lambda item: item.memory_sequence))

    def _finish(
        self,
        lease: MemoryJobLease,
        *,
        committed: bool,
        error: str | None,
        retryable: bool,
        before_settlement: Callable[[], None] | None = None,
    ) -> MemoryJob:
        self._require_lease(lease)
        with self._queue_fence():
            now = self._timestamp()
            current = self._current_lease_job(lease, now=now)
            if before_settlement is not None:
                before_settlement()
            if committed:
                status = MemoryJobStatus.COMMITTED
                next_attempt_at = None
                last_error = current.last_error
            elif not retryable or current.attempts >= self.config.max_attempts:
                status = MemoryJobStatus.FAILED
                next_attempt_at = None
                last_error = error
            else:
                status = MemoryJobStatus.QUEUED
                next_attempt_at = now + timedelta(seconds=self.config.retry_delay_seconds(current.attempts))
                last_error = error
            finished = replace(
                current,
                status=status,
                claim_id=None,
                worker_id=None,
                lease_expires_at=None,
                next_attempt_at=next_attempt_at,
                last_error=last_error,
                updated_at=now,
            )
            self._write(finished)
            return finished

    def _current_lease_job(
        self,
        lease: MemoryJobLease,
        *,
        now: datetime,
    ) -> MemoryJob:
        current = self._oldest_required(lease.job)
        if (
            current.status is not MemoryJobStatus.RUNNING
            or current.claim_id != lease.claim_id
            or current.claim_generation != lease.claim_generation
            or current.worker_id != lease.worker_id
        ):
            raise MemoryJobLeaseLostError("memory job lease identity is no longer current")
        assert current.lease_expires_at is not None
        if current.lease_expires_at <= now:
            raise MemoryJobLeaseLostError("memory job lease has expired")
        return current

    def _oldest_required(self, expected: MemoryJob) -> MemoryJob:
        oldest = self._oldest_uncommitted_unlocked()
        if oldest is None or oldest.source_identity != expected.source_identity:
            raise MemoryJobError("memory job is not the oldest uncommitted sequence")
        return oldest

    def _oldest_uncommitted_unlocked(self) -> MemoryJob | None:
        return next(
            (item for item in self._read_all() if item.status is not MemoryJobStatus.COMMITTED),
            None,
        )

    @contextmanager
    def _queue_fence(self) -> Iterator[None]:
        """取得短期跨进程队列锁；租约本身另行耐久保存。"""

        config = self.config
        with self.path_lock.acquire(
            self.lock_key,
            ttl_seconds=config.lock_ttl_seconds,
            wait_timeout_seconds=config.lock_wait_timeout_seconds,
            retry_delay_seconds=config.lock_retry_delay_seconds,
        ) as guard:
            with guard.fenced():
                yield

    def _read_all(self) -> tuple[MemoryJob, ...]:
        if not self.jobs_root.exists():
            return ()
        if self.jobs_root.is_symlink() or not self.jobs_root.is_dir():
            raise MemoryJobError("memory jobs path is not a safe directory")
        jobs: list[MemoryJob] = []
        for entry_count, child in enumerate(self.jobs_root.iterdir(), start=1):
            if entry_count > self.config.max_files:
                raise MemoryJobError("memory job entry count exceeds its safety bound")
            if child.is_symlink() or not child.is_file():
                raise MemoryJobError("memory jobs directory contains an unsupported entry")
            if child.name == self.state_path.name:
                continue
            temporary_destination = atomic_temporary_destination(child.name)
            if temporary_destination == self.state_path.name:
                continue
            if (
                temporary_destination is not None
                and MemoryJob._hex(Path(temporary_destination).stem, 64)
                and Path(temporary_destination).suffix == ".json"
            ):
                continue
            if child.suffix != ".json":
                raise MemoryJobError("memory jobs directory contains an unsupported entry")
            jobs.append(self._read_path(child))
        jobs.sort(key=lambda item: item.memory_sequence)
        sequences = tuple(item.memory_sequence for item in jobs)
        if len(sequences) != len(set(sequences)):
            raise MemoryJobError("memory jobs contain duplicate memory_sequence values")
        return tuple(jobs)

    def _load_or_initialize_state(
        self,
        jobs: tuple[MemoryJob, ...],
    ) -> _MemoryJobSequenceState:
        try:
            encoded = read_regular_bytes(
                self.state_path,
                artifact_root=self.root,
                max_bytes=self.config.max_file_bytes,
            )
        except FileNotFoundError:
            initial = _MemoryJobSequenceState(max((job.memory_sequence for job in jobs), default=0))
            atomic_create_bytes(
                self.state_path,
                self._encode_state(initial),
                artifact_root=self.root,
            )
            return initial
        try:
            raw = json.loads(encoded)
        except json.JSONDecodeError as exc:
            raise MemoryJobError("memory job sequence state is invalid JSON") from exc
        expected = {"schema", "last_allocated_memory_sequence"}
        if not isinstance(raw, Mapping) or set(raw) != expected:
            raise MemoryJobError("memory job sequence state has an invalid shape")
        if raw["schema"] != _JOB_STATE_SCHEMA:
            raise MemoryJobError("memory job sequence state has an unsupported schema")
        state = _MemoryJobSequenceState(raw["last_allocated_memory_sequence"])
        if encoded != self._encode_state(state):
            raise MemoryJobError("memory job sequence state is not canonically encoded")
        maximum_existing = max((job.memory_sequence for job in jobs), default=0)
        if state.last_allocated_memory_sequence < maximum_existing:
            raise MemoryJobError("memory job sequence state is behind existing jobs")
        return state

    def _write_state(self, state: _MemoryJobSequenceState) -> None:
        atomic_replace_bytes(
            self.state_path,
            self._encode_state(state),
            artifact_root=self.root,
        )

    def _encode_state(self, state: _MemoryJobSequenceState) -> bytes:
        encoded = (
            canonical_json(
                {
                    "schema": _JOB_STATE_SCHEMA,
                    "last_allocated_memory_sequence": state.last_allocated_memory_sequence,
                }
            )
            + "\n"
        ).encode("utf-8")
        if len(encoded) > self.config.max_file_bytes:
            raise MemoryJobError("memory job sequence state exceeds its configured file bound")
        return encoded

    def _read_path(self, path: Path) -> MemoryJob:
        try:
            raw = json.loads(
                read_regular_bytes(
                    path,
                    artifact_root=self.root,
                    max_bytes=self.config.max_file_bytes,
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

    def _abandonment_path(self, source_key: str) -> Path:
        if not MemoryJob._hex(source_key, 64):
            raise ValueError("memory job source key must be lowercase SHA-256 text")
        return self.abandonments_root / f"{source_key}.json"

    def _encode_abandonment(self, record: MemoryJobAbandonment) -> bytes:
        encoded = canonical_json(
            {
                "schema": record.SCHEMA_VERSION,
                "memory_sequence": record.memory_sequence,
                "conversation_id": record.conversation_id,
                "started_on": record.started_on.isoformat(),
                "segment_id": record.segment_id,
                "source_segment_digest": record.source_segment_digest,
                "transaction_id": record.transaction_id,
                "attempts": record.attempts,
                "last_error": record.last_error,
                "reason": record.reason,
                "abandoned_at": self._format_time(record.abandoned_at),
            }
        ).encode("utf-8")
        if len(encoded) > self.config.max_file_bytes:
            raise MemoryJobError("memory job abandonment exceeds its configured file bound")
        return encoded

    def _read_abandonment_path(self, path: Path) -> MemoryJobAbandonment:
        try:
            encoded = read_regular_bytes(
                path,
                artifact_root=self.root,
                max_bytes=self.config.max_file_bytes,
            )
            value = json.loads(encoded)
            expected = {
                "schema", "memory_sequence", "conversation_id", "started_on", "segment_id",
                "source_segment_digest", "transaction_id", "attempts", "last_error", "reason",
                "abandoned_at",
            }
            if not isinstance(value, Mapping) or set(value) != expected:
                raise MemoryJobError("memory job abandonment has an invalid shape")
            if value["schema"] != MemoryJobAbandonment.SCHEMA_VERSION:
                raise MemoryJobError("memory job abandonment has an unsupported schema")
            record = MemoryJobAbandonment(
                memory_sequence=value["memory_sequence"],
                conversation_id=value["conversation_id"],
                started_on=date.fromisoformat(value["started_on"]),
                segment_id=value["segment_id"],
                source_segment_digest=value["source_segment_digest"],
                transaction_id=value["transaction_id"],
                attempts=value["attempts"],
                last_error=value["last_error"],
                reason=value["reason"],
                abandoned_at=self._parse_time(value["abandoned_at"]),
            )
            if encoded != self._encode_abandonment(record):
                raise MemoryJobError("memory job abandonment is not canonically encoded")
            return record
        except Exception as exc:
            if isinstance(exc, MemoryJobError):
                raise
            raise MemoryJobError("failed to read memory job abandonment") from exc

    def _try_read_abandonment_path(self, path: Path) -> MemoryJobAbandonment | None:
        try:
            return self._read_abandonment_path(path)
        except MemoryJobError as exc:
            if isinstance(exc.__cause__, FileNotFoundError):
                return None
            raise

    @staticmethod
    def _source_key(
        address: ConversationAddress,
        segment: ConversationSegment,
    ) -> str:
        return MemoryJobStore._source_key_values(address, segment.segment_id, segment.digest)

    @staticmethod
    def _source_key_values(
        address: ConversationAddress,
        segment_id: str,
        source_segment_digest: str,
    ) -> str:
        return hashlib.sha256(
            (
                f"{address.conversation_id}\0{address.started_on.isoformat()}\0{segment_id}\0{source_segment_digest}"
            ).encode()
        ).hexdigest()

    @staticmethod
    def _source_key_from_job(job: MemoryJob) -> str:
        address = ConversationAddress(job.conversation_id, job.started_on)
        return MemoryJobStore._source_key_values(
            address,
            job.segment_id,
            job.source_segment_digest,
        )

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

    @staticmethod
    def _require_job(job: object) -> None:
        if not isinstance(job, MemoryJob):
            raise TypeError("job must be a MemoryJob")

    @staticmethod
    def _require_lease(lease: object) -> None:
        if not isinstance(lease, MemoryJobLease):
            raise TypeError("lease must be a MemoryJobLease")

    def _encode(self, job: MemoryJob) -> bytes:
        encoded = canonical_json(
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
                "claim_generation": job.claim_generation,
                "worker_id": job.worker_id,
                "lease_expires_at": MemoryJobStore._format_optional_time(job.lease_expires_at),
                "next_attempt_at": MemoryJobStore._format_optional_time(job.next_attempt_at),
                "last_error": job.last_error,
                "created_at": MemoryJobStore._format_time(job.created_at),
                "updated_at": MemoryJobStore._format_time(job.updated_at),
            }
        ).encode("utf-8")
        if len(encoded) > self.config.max_file_bytes:
            raise MemoryJobError("memory job exceeds its configured file bound")
        return encoded

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
            "claim_generation",
            "worker_id",
            "lease_expires_at",
            "next_attempt_at",
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
                claim_generation=value["claim_generation"],
                worker_id=value["worker_id"],
                lease_expires_at=MemoryJobStore._parse_optional_time(value["lease_expires_at"]),
                next_attempt_at=MemoryJobStore._parse_optional_time(value["next_attempt_at"]),
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
    def _format_optional_time(value: datetime | None) -> str | None:
        return None if value is None else MemoryJobStore._format_time(value)

    @staticmethod
    def _parse_time(value: object) -> datetime:
        if not isinstance(value, str):
            raise ValueError("memory job timestamp must be a string")
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError("memory job timestamp must include a timezone")
        return parsed.astimezone(timezone.utc)

    @staticmethod
    def _parse_optional_time(value: object) -> datetime | None:
        return None if value is None else MemoryJobStore._parse_time(value)

    def _timestamp(self) -> datetime:
        value = self.clock()
        if not isinstance(value, datetime):
            raise TypeError("memory job clock must return a datetime")
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("memory job clock must return a timezone-aware datetime")
        return value.astimezone(timezone.utc)


__all__ = ["MemoryJobStore"]
