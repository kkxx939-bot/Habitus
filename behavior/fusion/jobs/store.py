"""按 behavior-root 严格排序、带耐久租约的融合作业队列。

## 为什么严格串行

判断之间有 ``continues`` / ``supersedes`` / ``concurrent_with`` 三种关系，而它们要跨融合窗口
成立，后一段就必须看得见前一段**已经落盘的判断**——一次做饭被切段切开时，后半段要能指回前半段。
上下文是在构造 prompt 时读的，所以并行两个作业时，后一段调模型的那一刻还看不到前一段的产物，
只能把同一件事判成两件互不相干的事。

这条收益已经用真模型验证过：一次洗衣被切段拦腰切断时，后半段确实产出了 ``continues`` 指回前半段
的真实身份；而备菜与炒菜这类"两件相继的事"，模型判为独立——它是按语义判的，不是见到上下文就乱指。

代价很小：一段覆盖十几分钟到半小时，而一次融合是秒级，队列平时是空的；唯一会积压的场景是停机后
补算，而那恰恰最需要顺序正确。
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

from behavior.fusion.jobs.model import (
    FUSION_JOB_ERROR_MAX_CHARS,
    BehaviorFusionJob,
    BehaviorFusionJobBlockedError,
    BehaviorFusionJobConfig,
    BehaviorFusionJobError,
    BehaviorFusionJobLease,
    BehaviorFusionJobLeaseLostError,
    BehaviorFusionJobNotReadyError,
    BehaviorFusionJobStatus,
    BehaviorFusionQueueSnapshot,
    StagedFusion,
    lease_deadline,
)
from behavior.fusion.receipt import receipt_identity
from foundation.integrity import canonical_json
from infrastructure.store.contracts import PathLock
from infrastructure.store.filesystem import (
    atomic_create_bytes,
    atomic_replace_bytes,
    atomic_temporary_destination,
    durable_unlink,
    read_regular_bytes,
)

_STATE_SCHEMA = "behavior_fusion_job_sequence_state_v1"
_STATE_FILE = "state.json"


def _is_job_filename(name: str) -> bool:
    """作业文件恒为 ``<64 位十六进制>.json``——与 ``_path`` 的构造规则同源。"""

    stem, _, suffix = name.rpartition(".")
    return (
        suffix == "json"
        and len(stem) == 64
        and all(character in "0123456789abcdef" for character in stem)
    )


@dataclass(frozen=True)
class _SequenceState:
    """作业被清理之后仍然永久保留的序号高水位。"""

    last_allocated_fusion_sequence: int

    def __post_init__(self) -> None:
        value = self.last_allocated_fusion_sequence
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise BehaviorFusionJobError("fusion job sequence high-watermark must be non-negative")


class BehaviorFusionJobStore:
    """在跨进程队列锁内维护全局序号、租约与重试。"""

    def __init__(
        self,
        root: str | Path,
        path_lock: PathLock,
        *,
        config: BehaviorFusionJobConfig | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        requested = Path(root).expanduser().absolute()
        if requested.is_symlink():
            raise BehaviorFusionJobError("fusion job root cannot be a symbolic link")
        if not isinstance(path_lock, PathLock):
            raise TypeError("path_lock must be a PathLock")
        if config is not None and not isinstance(config, BehaviorFusionJobConfig):
            raise TypeError("config must be BehaviorFusionJobConfig")
        if clock is not None and not callable(clock):
            raise TypeError("clock must be callable")
        self.root = requested.resolve(strict=False)
        self.jobs_root = self.root / "fusion" / "jobs"
        self.state_path = self.jobs_root / _STATE_FILE
        self.path_lock = path_lock
        self.config = config or BehaviorFusionJobConfig()
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.lock_key = f"behavior-fusion-jobs:{self.root}"
        self._fence_depth = 0

    # --- 排队 --------------------------------------------------------------------------

    def enqueue(
        self,
        *,
        segment_digest: str,
        observation_ids: tuple[str, ...],
        source_refs: tuple[str, ...],
        fusion_version: str,
        prompt_version: str,
    ) -> BehaviorFusionJob:
        """幂等登记一段观测；同一片段同版本重复排队返回既有作业。

        作业身份直接用**未来那份回执的身份**，于是"这段观测在这个版本下融合过没有"是一次
        文件存在性判断，不需要扫描行为树反推。
        """

        job_id = receipt_identity(segment_digest, fusion_version, prompt_version)
        with self._reentrant_fence():
            existing = self._try_read(self._path(job_id))
            if existing is not None:
                if existing.segment_digest != segment_digest:
                    raise BehaviorFusionJobError("fusion job identity collides with another segment")
                return existing
            jobs = self._read_all()
            state = self._load_or_initialize_state(jobs)
            sequence = state.last_allocated_fusion_sequence + 1
            self._write_state(_SequenceState(sequence))
            now = self._timestamp()
            job = BehaviorFusionJob(
                fusion_sequence=sequence,
                job_id=job_id,
                segment_digest=segment_digest,
                observation_ids=tuple(observation_ids),
                source_refs=tuple(source_refs),
                fusion_version=fusion_version,
                prompt_version=prompt_version,
                status=BehaviorFusionJobStatus.QUEUED,
                attempts=0,
                claim_id=None,
                claim_generation=0,
                worker_id=None,
                lease_expires_at=None,
                next_attempt_at=None,
                last_error=None,
                staged=None,
                created_at=now,
                updated_at=now,
            )
            atomic_create_bytes(self._path(job_id), self._encode(job), artifact_root=self.root)
            return job

    def retarget(
        self,
        job: BehaviorFusionJob,
        *,
        fusion_version: str,
        prompt_version: str,
    ) -> BehaviorFusionJob | None:
        """把一条尚未到检查点的作业改挂到当前版本身份下，**保留它的队列位置**。

        作业身份在排队时由片段与版本钉死，而队列是耐久的——改一次提示词再重启（升级本来就
        是停服务、重装、再跑），盘上那批作业记的还是旧版本。执行方要等到"回执身份与作业身份
        不符"才发现，而那一步在**调完模型之后**：一次白烧的调用、退避重试若干次、最后 FAILED
        卡住整条串行队列，`retry_failed` 重开后照样再来一轮。版本在排队时就是已知的，所以在
        认领之前纠正过来，一次调用都不用浪费。

        已经到检查点（``staged`` 非空）的作业不动：它的判断是在旧版本下真做出来的，按它自己的
        版本提交才诚实。持有活跃租约的也不动，交给 ``claim`` 去报"被占用"。

        返回改挂后的作业；若当前版本下这段已经另有作业，则删掉这条残留并返回 ``None``，由调用
        方重新取队首。
        """

        self._require_job(job)
        with self._queue_fence():
            current = self._try_read(self._path(job.job_id))
            if current is None or current.staged is not None:
                return current
            now = self._timestamp()
            if (
                current.status is BehaviorFusionJobStatus.RUNNING
                and current.lease_expires_at is not None
                and current.lease_expires_at > now
            ):
                return current
            if (
                current.fusion_version == fusion_version
                and current.prompt_version == prompt_version
            ):
                return current
            replacement_id = receipt_identity(current.segment_digest, fusion_version, prompt_version)
            existing = self._try_read(self._path(replacement_id))
            # 先删旧再写新。反过来会在崩溃窗口里留下两条共用同一 ``fusion_sequence`` 的记录，
            # 而 ``_read_all`` 见到重复序号会整库硬失败——那正是这次修复要消灭的那类永久卡死。
            # 崩在中间只会让这段回到"没排过队"，由入队扫描重建。
            self._path(current.job_id).unlink(missing_ok=True)
            if existing is not None:
                return None
            replacement = replace(
                current,
                job_id=replacement_id,
                fusion_version=fusion_version,
                prompt_version=prompt_version,
                status=BehaviorFusionJobStatus.QUEUED,
                attempts=0,
                claim_id=None,
                claim_generation=0,
                worker_id=None,
                lease_expires_at=None,
                next_attempt_at=None,
                last_error=None,
                updated_at=now,
            )
            atomic_create_bytes(
                self._path(replacement_id), self._encode(replacement), artifact_root=self.root
            )
            return replacement

    def initialize(self) -> int:
        with self._queue_fence():
            return self._load_or_initialize_state(self._read_all()).last_allocated_fusion_sequence

    def high_watermark(self) -> int:
        with self._queue_fence():
            return self._load_or_initialize_state(self._read_all()).last_allocated_fusion_sequence

    def oldest_uncommitted(self) -> BehaviorFusionJob | None:
        """返回最早未完成的作业；串行纪律要求调用方不得越过它。"""

        with self._queue_fence():
            return self._oldest_unlocked()

    def try_read(self, job_id: str) -> BehaviorFusionJob | None:
        return self._try_read(self._path(job_id))

    @contextmanager
    def scan_fence(self) -> Iterator[None]:
        """把"读覆盖 → 读观测 → 登记"整段圈进同一个栅栏。

        ``enqueue`` 自己有栅栏，但排队决策**依据的快照**是在栅栏外读的：两次扫描交错时，后读
        快照的那次先登记，先读快照的那次再登记一个更小的片段——两段的 ``segment_digest`` 不同，
        ``job_id`` 也就不同，幂等完全不起作用，同一批观测被融合两次。
        """

        with self._queue_fence():
            yield

    def covered_observation_ids(self, *, fenced: bool = True) -> frozenset[str]:
        """所有仍保留的作业已经覆盖的观测；排队时据此跳过已处理的片段。"""

        if not fenced:
            return frozenset(
                observation_id for job in self._read_all() for observation_id in job.observation_ids
            )
        with self._queue_fence():
            return frozenset(
                observation_id for job in self._read_all() for observation_id in job.observation_ids
            )

    def observability_snapshot(self) -> BehaviorFusionQueueSnapshot:
        with self._queue_fence():
            jobs = self._read_all()
            state = self._load_or_initialize_state(jobs)
            counts = {status: 0 for status in BehaviorFusionJobStatus}
            for job in jobs:
                counts[job.status] += 1
            oldest = next((item for item in jobs if not item.is_terminal), None)
            age = 0.0 if oldest is None else max(0.0, (self._timestamp() - oldest.created_at).total_seconds())
            return BehaviorFusionQueueSnapshot(
                queued=counts[BehaviorFusionJobStatus.QUEUED],
                running=counts[BehaviorFusionJobStatus.RUNNING],
                staged=counts[BehaviorFusionJobStatus.STAGED],
                committed=counts[BehaviorFusionJobStatus.COMMITTED],
                failed=counts[BehaviorFusionJobStatus.FAILED],
                high_watermark=state.last_allocated_fusion_sequence,
                oldest_age_seconds=age,
            )

    # --- 认领与租约 --------------------------------------------------------------------

    def claim(self, job: BehaviorFusionJob, worker_id: str) -> BehaviorFusionJobLease:
        """认领最早可执行的作业，或在租约过期后以更新的 generation 接管。

        接管一个已到检查点（``STAGED``）的作业不会重跑模型：``staged`` 原样保留，执行方据此
        只重放发布。
        """

        self._require_job(job)
        if not BehaviorFusionJob.valid_worker_id(worker_id):
            raise ValueError("worker_id must be normalized stable text")
        with self._queue_fence():
            current = self._oldest_required(job)
            now = self._timestamp()
            if current.status is BehaviorFusionJobStatus.FAILED:
                raise BehaviorFusionJobBlockedError("oldest fusion job exhausted retries and blocks later jobs")
            if current.status is BehaviorFusionJobStatus.COMMITTED:
                raise BehaviorFusionJobError("committed fusion job cannot be claimed")
            if current.attempts >= self.config.max_attempts:
                # Worker 硬崩（OOM、被 kill、断电）永远走不到 ``fail()``，于是作业在"接管 → 崩 →
                # 租约过期 → 接管"里无限循环，而且每一轮都因为 ``staged is None`` 重调一次模型。
                # 结算路径判了次数，认领路径也必须判。
                exhausted = replace(
                    current,
                    status=BehaviorFusionJobStatus.FAILED,
                    claim_id=None,
                    worker_id=None,
                    lease_expires_at=None,
                    next_attempt_at=None,
                    last_error=(current.last_error or "worker died before settling its attempt"),
                    updated_at=now,
                )
                self._write(exhausted)
                raise BehaviorFusionJobBlockedError(
                    "oldest fusion job exhausted its attempts without settling and blocks later jobs"
                )
            if current.status is BehaviorFusionJobStatus.RUNNING:
                assert current.lease_expires_at is not None
                if current.lease_expires_at > now:
                    raise BehaviorFusionJobBlockedError("oldest fusion job is held by an active worker lease")
                previous = current.worker_id or "unknown"
                reclaimed = (
                    f"worker lease expired; reclaimed from {previous} "
                    f"generation {current.claim_generation}"
                )
                last_error: str | None = reclaimed[:FUSION_JOB_ERROR_MAX_CHARS]
            else:
                if current.next_attempt_at is not None and current.next_attempt_at > now:
                    raise BehaviorFusionJobNotReadyError(current.next_attempt_at)
                last_error = current.last_error
            claimed = replace(
                current,
                status=BehaviorFusionJobStatus.RUNNING,
                attempts=current.attempts + 1,
                claim_id=uuid4().hex,
                claim_generation=current.claim_generation + 1,
                worker_id=worker_id,
                lease_expires_at=lease_deadline(now, self.config),
                next_attempt_at=None,
                last_error=last_error,
                updated_at=now,
            )
            self._write(claimed)
            return BehaviorFusionJobLease(claimed)

    def renew(self, lease: BehaviorFusionJobLease) -> BehaviorFusionJobLease:
        self._require_lease(lease)
        with self._queue_fence():
            now = self._timestamp()
            current = self._current_lease_job(lease, now=now)
            renewed = replace(current, lease_expires_at=lease_deadline(now, self.config), updated_at=now)
            self._write(renewed)
            return BehaviorFusionJobLease(renewed)

    def assert_current(self, lease: BehaviorFusionJobLease) -> BehaviorFusionJob:
        """在发布边界确认租约仍然有效且未被接管。"""

        self._require_lease(lease)
        with self._queue_fence():
            return self._current_lease_job(lease, now=self._timestamp())

    # --- 结算 --------------------------------------------------------------------------

    def stage(self, lease: BehaviorFusionJobLease, staged: StagedFusion) -> BehaviorFusionJobLease:
        """记录检查点：payload 与回执耐久落盘，此后不再调用模型。

        租约**继续持有**——暂存与发布是同一次认领里的两步，中间没有必要放开队列。作业若在这里
        之后崩溃，接管方看到 ``staged`` 已存在，直接从它重放发布。
        """

        self._require_lease(lease)
        if not isinstance(staged, StagedFusion):
            raise TypeError("staged must be StagedFusion")
        with self._queue_fence():
            now = self._timestamp()
            current = self._current_lease_job(lease, now=now)
            if current.staged is not None:
                raise BehaviorFusionJobError("fusion job already recorded its staged fusion")
            if staged.receipt.receipt_id != current.job_id:
                raise BehaviorFusionJobError("staged receipt identity does not match its job")
            if staged.receipt.segment_digest != current.segment_digest:
                raise BehaviorFusionJobError("staged receipt covers a different observation segment")
            checkpointed = replace(current, staged=staged, updated_at=now)
            self._write(checkpointed)
            return BehaviorFusionJobLease(checkpointed)

    def commit(self, lease: BehaviorFusionJobLease) -> BehaviorFusionJob:
        self._require_lease(lease)
        with self._queue_fence():
            current = self._current_lease_job(lease, now=self._timestamp())
            if current.staged is None:
                raise BehaviorFusionJobError("a fusion job cannot commit before it stages its fusion")
        return self._finish(lease, committed=True, error=None, retryable=False)

    def fail(
        self,
        lease: BehaviorFusionJobLease,
        error: BaseException,
        *,
        retryable: bool = True,
    ) -> BehaviorFusionJob:
        if not isinstance(error, BaseException):
            raise TypeError("error must be an exception")
        if not isinstance(retryable, bool):
            raise TypeError("retryable must be boolean")
        message = " ".join(str(error).split()) or type(error).__name__
        return self._finish(
            lease, committed=False, error=message[:FUSION_JOB_ERROR_MAX_CHARS], retryable=retryable
        )

    def retry_failed(self, job: BehaviorFusionJob) -> BehaviorFusionJob:
        """人工处置后重新开放最早的 FAILED 作业。"""

        self._require_job(job)
        if job.status is not BehaviorFusionJobStatus.FAILED:
            raise ValueError("only a failed fusion job can be retried")
        with self._queue_fence():
            current = self._oldest_required(job)
            if current.status is not BehaviorFusionJobStatus.FAILED:
                raise BehaviorFusionJobError("fusion job is no longer in a retryable failed state")
            if current != job:
                raise BehaviorFusionJobBlockedError("failed fusion job changed after this retry decision")
            reopened = replace(
                current,
                status=self._resume_status(current),
                attempts=0,
                claim_id=None,
                claim_generation=current.claim_generation,
                worker_id=None,
                lease_expires_at=None,
                next_attempt_at=None,
                last_error=None,
                updated_at=self._timestamp(),
            )
            self._write(reopened)
            return reopened

    def discard_committed(self, job: BehaviorFusionJob) -> bool:
        """删除一条已完成的作业；回执本身另有生命周期，不随作业消失。"""

        self._require_job(job)
        if job.status is not BehaviorFusionJobStatus.COMMITTED:
            raise BehaviorFusionJobError("only a COMMITTED fusion job can be discarded")
        with self._queue_fence():
            state = self._load_or_initialize_state(self._read_all())
            if state.last_allocated_fusion_sequence < job.fusion_sequence:
                raise BehaviorFusionJobError("fusion job sequence high-watermark is behind the discarded job")
            path = self._path(job.job_id)
            current = self._try_read(path)
            if current is None:
                return False
            if current != job:
                raise BehaviorFusionJobBlockedError("fusion job changed before lifecycle cleanup")
            return durable_unlink(path, artifact_root=self.root)

    # --- 内部 --------------------------------------------------------------------------

    def _finish(
        self,
        lease: BehaviorFusionJobLease,
        *,
        committed: bool,
        error: str | None,
        retryable: bool,
    ) -> BehaviorFusionJob:
        self._require_lease(lease)
        with self._queue_fence():
            now = self._timestamp()
            current = self._current_lease_job(lease, now=now)
            if committed:
                status = BehaviorFusionJobStatus.COMMITTED
                next_attempt_at = None
                last_error = current.last_error
            elif not retryable or current.attempts >= self.config.max_attempts:
                status = BehaviorFusionJobStatus.FAILED
                next_attempt_at = None
                last_error = error
            else:
                # 回退到哪里由**有没有暂存**决定：已过检查点的重试只重放发布，绝不再调模型。
                status = self._resume_status(current)
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

    @staticmethod
    def _resume_status(job: BehaviorFusionJob) -> BehaviorFusionJobStatus:
        return BehaviorFusionJobStatus.QUEUED if job.needs_fusion else BehaviorFusionJobStatus.STAGED

    def _current_lease_job(
        self, lease: BehaviorFusionJobLease, *, now: datetime
    ) -> BehaviorFusionJob:
        current = self._oldest_required(lease.job)
        if (
            current.status is not BehaviorFusionJobStatus.RUNNING
            or current.claim_id != lease.claim_id
            or current.claim_generation != lease.claim_generation
            or current.worker_id != lease.worker_id
        ):
            raise BehaviorFusionJobLeaseLostError("fusion job lease identity is no longer current")
        assert current.lease_expires_at is not None
        if current.lease_expires_at <= now:
            raise BehaviorFusionJobLeaseLostError("fusion job lease has expired")
        return current

    def _oldest_required(self, expected: BehaviorFusionJob) -> BehaviorFusionJob:
        oldest = self._oldest_unlocked()
        if oldest is None or oldest.job_id != expected.job_id:
            raise BehaviorFusionJobBlockedError("fusion job is not the oldest uncommitted sequence")
        return oldest

    def _oldest_unlocked(self) -> BehaviorFusionJob | None:
        return next((item for item in self._read_all() if not item.is_terminal), None)

    @staticmethod
    def _require_job(job: object) -> None:
        if not isinstance(job, BehaviorFusionJob):
            raise TypeError("job must be BehaviorFusionJob")

    @staticmethod
    def _require_lease(lease: object) -> None:
        if not isinstance(lease, BehaviorFusionJobLease):
            raise TypeError("lease must be BehaviorFusionJobLease")

    @contextmanager
    def _reentrant_fence(self) -> Iterator[None]:
        """已经在 ``scan_fence`` 里就不再取一次锁；路径锁不可重入。"""

        if self._fence_depth:
            yield
            return
        with self._queue_fence():
            yield

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
                self._fence_depth += 1
                try:
                    yield
                finally:
                    self._fence_depth -= 1

    def _timestamp(self) -> datetime:
        now = self.clock()
        if not isinstance(now, datetime) or now.tzinfo is None:
            raise BehaviorFusionJobError("fusion job clock must return timezone-aware datetimes")
        return now.astimezone(timezone.utc)

    def _read_all(self) -> tuple[BehaviorFusionJob, ...]:
        if not self.jobs_root.exists():
            return ()
        if self.jobs_root.is_symlink() or not self.jobs_root.is_dir():
            raise BehaviorFusionJobError("fusion jobs path is not a safe directory")
        jobs: list[BehaviorFusionJob] = []
        for count, child in enumerate(self.jobs_root.iterdir(), start=1):
            if count > self.config.max_files:
                raise BehaviorFusionJobError("fusion job entry count exceeds its safety bound")
            if child.name == _STATE_FILE:
                continue
            temporary = atomic_temporary_destination(child.name)
            if temporary is not None:
                # 原子替换的中间文件；它属于本存储，跳过而不是当成损坏。
                continue
            if child.is_symlink() or not child.is_file() or not _is_job_filename(child.name):
                # 不符合本存储命名规则的条目一定不是本存储写的（``.DS_Store``、同步副本、
                # 顺手放的笔记）。为它们整库硬失败，等于让一次偶发污染永久瘫痪**整条**流水线：
                # 排队、认领、结算、可观测性全都要枚举这个目录。判断与回执两个兄弟存储早就是
                # 跳过的，这里必须同口径。名字合规但内容损坏的仍然照旧硬失败——那才是我们的东西。
                continue
            jobs.append(self._read(child))
        jobs.sort(key=lambda item: item.fusion_sequence)
        sequences = tuple(item.fusion_sequence for item in jobs)
        if len(sequences) != len(set(sequences)):
            raise BehaviorFusionJobError("fusion jobs contain duplicate fusion_sequence values")
        return tuple(jobs)

    def _load_or_initialize_state(self, jobs: tuple[BehaviorFusionJob, ...]) -> _SequenceState:
        try:
            encoded = read_regular_bytes(
                self.state_path, artifact_root=self.root, max_bytes=self.config.max_file_bytes
            )
        except FileNotFoundError:
            initial = _SequenceState(max((job.fusion_sequence for job in jobs), default=0))
            atomic_create_bytes(self.state_path, self._encode_state(initial), artifact_root=self.root)
            return initial
        try:
            raw = json.loads(encoded)
        except json.JSONDecodeError as exc:
            raise BehaviorFusionJobError("fusion job sequence state is invalid JSON") from exc
        if not isinstance(raw, Mapping) or set(raw) != {"schema", "last_allocated_fusion_sequence"}:
            raise BehaviorFusionJobError("fusion job sequence state has an invalid shape")
        if raw["schema"] != _STATE_SCHEMA:
            raise BehaviorFusionJobError("fusion job sequence state has an unsupported schema")
        state = _SequenceState(raw["last_allocated_fusion_sequence"])
        if encoded != self._encode_state(state):
            raise BehaviorFusionJobError("fusion job sequence state is not canonically encoded")
        highest = max((job.fusion_sequence for job in jobs), default=0)
        if state.last_allocated_fusion_sequence < highest:
            raise BehaviorFusionJobError("fusion job sequence state is behind existing jobs")
        return state

    def _write_state(self, state: _SequenceState) -> None:
        atomic_replace_bytes(self.state_path, self._encode_state(state), artifact_root=self.root)

    def _encode_state(self, state: _SequenceState) -> bytes:
        encoded = (
            canonical_json(
                {
                    "schema": _STATE_SCHEMA,
                    "last_allocated_fusion_sequence": state.last_allocated_fusion_sequence,
                }
            )
            + "\n"
        ).encode("utf-8")
        if len(encoded) > self.config.max_file_bytes:
            raise BehaviorFusionJobError("fusion job sequence state exceeds its configured file bound")
        return encoded

    def _read(self, path: Path) -> BehaviorFusionJob:
        try:
            raw = json.loads(
                read_regular_bytes(path, artifact_root=self.root, max_bytes=self.config.max_file_bytes)
            )
            return BehaviorFusionJob.from_dict(raw)
        except Exception as exc:
            if isinstance(exc, BehaviorFusionJobError):
                raise
            raise BehaviorFusionJobError("failed to read fusion job") from exc

    def _try_read(self, path: Path) -> BehaviorFusionJob | None:
        try:
            return self._read(path)
        except BehaviorFusionJobError as exc:
            if isinstance(exc.__cause__, FileNotFoundError):
                return None
            raise

    def _write(self, job: BehaviorFusionJob) -> None:
        atomic_replace_bytes(self._path(job.job_id), self._encode(job), artifact_root=self.root)

    def _path(self, job_id: str) -> Path:
        if not isinstance(job_id, str) or len(job_id) != 64 or any(
            character not in "0123456789abcdef" for character in job_id
        ):
            raise BehaviorFusionJobError("job_id must be lowercase SHA-256 text")
        return self.jobs_root / f"{job_id}.json"

    def _encode(self, job: BehaviorFusionJob) -> bytes:
        # 作业记录**不能**整体走 canonical_json：它会把暂存判断里的 ``started_at`` /
        # ``last_observed_at`` 折成 UTC，东八区凌晨的行为就掉到前一天，归约层按"人的一天"做的
        # 聚合会全错。这里用固定的键序 + 保留偏移的时间文本自行序列化。
        encoded = (
            json.dumps(job.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n"
        ).encode("utf-8")
        if len(encoded) > self.config.max_file_bytes:
            raise BehaviorFusionJobError("fusion job exceeds its configured file bound")
        return encoded


__all__ = ["BehaviorFusionJobStore"]
