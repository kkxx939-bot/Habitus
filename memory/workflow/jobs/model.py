"""耐久记忆任务的状态、租约、配置和不可变记录模型。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, timezone
from enum import Enum

MEMORY_JOB_ERROR_MAX_CHARS = 2_000
_WORKER_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


class MemoryJobError(RuntimeError):
    """MemoryJob 无法安全排队、认领、执行或推进状态。"""


class MemoryJobBlockedError(MemoryJobError):
    """最早任务失败、尚未就绪或由另一有效租约持有。"""


class MemoryJobNotReadyError(MemoryJobBlockedError):
    """最早任务仍处于耐久重试退避期。"""

    def __init__(self, available_at: datetime) -> None:
        if not isinstance(available_at, datetime) or available_at.tzinfo is None:
            raise TypeError("available_at must be a timezone-aware datetime")
        self.available_at = available_at.astimezone(timezone.utc)
        super().__init__(f"oldest memory job is not ready before {self.available_at.isoformat()}")


class MemoryJobLeaseError(MemoryJobError):
    """MemoryJob 租约不可用于当前操作。"""


class MemoryJobLeaseLostError(MemoryJobLeaseError):
    """Worker 的持久化租约已经过期或被更新 generation 接管。"""


class MemoryJobExecutionError(MemoryJobError):
    """一次后台记忆任务执行失败并已记录重试状态。"""

    def __init__(self, message: str, *, job: MemoryJob | None = None) -> None:
        super().__init__(message)
        self.job = job


class MemoryJobStatus(str, Enum):
    """后台记忆任务的耐久状态。"""

    STAGED = "staged"
    QUEUED = "queued"
    RUNNING = "running"
    FAILED = "failed"
    COMMITTED = "committed"


@dataclass(frozen=True)
class MemoryJobConfig:
    """跨 Conversation 队列的显式重试、租约与锁边界。"""

    max_attempts: int = 3
    max_file_bytes: int = 64 * 1024
    max_files: int = 100_000
    lease_ttl_seconds: int = 120
    retry_base_delay_seconds: float = 1.0
    retry_max_delay_seconds: float = 60.0
    lock_ttl_seconds: int = 30
    lock_wait_timeout_seconds: float = 5.0
    lock_retry_delay_seconds: float = 0.01

    def __post_init__(self) -> None:
        for name, value, maximum in (
            ("max_attempts", self.max_attempts, 20),
            ("max_files", self.max_files, 10_000_000),
            ("lease_ttl_seconds", self.lease_ttl_seconds, 3_600),
            ("lock_ttl_seconds", self.lock_ttl_seconds, 3_600),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
                raise ValueError(f"{name} must be between 1 and {maximum}")
        if (
            isinstance(self.max_file_bytes, bool)
            or not isinstance(self.max_file_bytes, int)
            or not 16_384 <= self.max_file_bytes <= 16 * 1024 * 1024
        ):
            raise ValueError("max_file_bytes must be between 16384 and 16777216")
        for float_name, float_value, float_maximum in (
            ("retry_base_delay_seconds", self.retry_base_delay_seconds, 3_600.0),
            ("retry_max_delay_seconds", self.retry_max_delay_seconds, 86_400.0),
            ("lock_wait_timeout_seconds", self.lock_wait_timeout_seconds, 60.0),
            ("lock_retry_delay_seconds", self.lock_retry_delay_seconds, 1.0),
        ):
            if (
                isinstance(float_value, bool)
                or not isinstance(float_value, int | float)
                or not 0 < float(float_value) <= float_maximum
            ):
                raise ValueError(f"{float_name} must be greater than zero and at most {float_maximum:g}")
        if self.retry_max_delay_seconds < self.retry_base_delay_seconds:
            raise ValueError("retry_max_delay_seconds cannot be less than retry_base_delay_seconds")

    def retry_delay_seconds(self, attempts: int) -> float:
        """按已经发生的尝试次数返回有上限的指数退避。"""

        if isinstance(attempts, bool) or not isinstance(attempts, int) or attempts <= 0:
            raise ValueError("attempts must be a positive integer")
        return min(
            float(self.retry_max_delay_seconds),
            float(self.retry_base_delay_seconds) * (2 ** (attempts - 1)),
        )


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
    claim_generation: int
    worker_id: str | None
    lease_expires_at: datetime | None
    next_attempt_at: datetime | None
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
        if (
            isinstance(self.claim_generation, bool)
            or not isinstance(self.claim_generation, int)
            or self.claim_generation < 0
        ):
            raise ValueError("memory job claim_generation must be non-negative")
        self._normalize_optional_time("lease_expires_at")
        self._normalize_optional_time("next_attempt_at")
        if status is MemoryJobStatus.RUNNING:
            if not isinstance(self.claim_id, str) or not self._hex(self.claim_id, 32):
                raise ValueError("running memory job requires a claim_id")
            if not isinstance(self.worker_id, str) or _WORKER_ID.fullmatch(self.worker_id) is None:
                raise ValueError("running memory job requires a normalized worker_id")
            if self.claim_generation <= 0:
                raise ValueError("running memory job requires a positive claim_generation")
            if self.lease_expires_at is None:
                raise ValueError("running memory job requires lease_expires_at")
            if self.next_attempt_at is not None:
                raise ValueError("running memory job cannot retain next_attempt_at")
        elif any(value is not None for value in (self.claim_id, self.worker_id, self.lease_expires_at)):
            raise ValueError("non-running memory job cannot retain lease ownership")
        if status not in {MemoryJobStatus.STAGED, MemoryJobStatus.QUEUED} and self.next_attempt_at is not None:
            raise ValueError("only staged or queued memory jobs can contain next_attempt_at")
        if status is MemoryJobStatus.STAGED:
            if self.claim_generation != 0:
                raise ValueError("staged memory job cannot contain a claim generation")
            if self.attempts == 0 and (self.last_error is not None or self.next_attempt_at is not None):
                raise ValueError("new staged memory job cannot contain retry state")
            if self.attempts > 0 and self.last_error is None:
                raise ValueError("retried staged memory job requires its last error")
        if self.last_error is not None and (
            not isinstance(self.last_error, str)
            or not self.last_error
            or len(self.last_error) > MEMORY_JOB_ERROR_MAX_CHARS
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

    def _normalize_optional_time(self, name: str) -> None:
        value = getattr(self, name)
        if value is None:
            return
        if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(f"memory job {name} must be timezone-aware")
        object.__setattr__(self, name, value.astimezone(timezone.utc))

    @staticmethod
    def valid_worker_id(value: object) -> bool:
        return isinstance(value, str) and _WORKER_ID.fullmatch(value) is not None

    @staticmethod
    def _hex(value: object, length: int) -> bool:
        if not isinstance(value, str) or len(value) != length or value != value.lower():
            return False
        try:
            int(value, 16)
        except ValueError:
            return False
        return True


@dataclass(frozen=True)
class MemoryJobLease:
    """Worker 认领任务时得到的稳定 fencing 身份。"""

    job: MemoryJob

    def __post_init__(self) -> None:
        if not isinstance(self.job, MemoryJob):
            raise TypeError("job must be MemoryJob")
        if self.job.status is not MemoryJobStatus.RUNNING:
            raise ValueError("memory job lease requires a RUNNING job")

    @property
    def claim_id(self) -> str:
        assert self.job.claim_id is not None
        return self.job.claim_id

    @property
    def claim_generation(self) -> int:
        return self.job.claim_generation

    @property
    def worker_id(self) -> str:
        assert self.job.worker_id is not None
        return self.job.worker_id

    @property
    def lease_expires_at(self) -> datetime:
        assert self.job.lease_expires_at is not None
        return self.job.lease_expires_at

    @property
    def source_identity(self) -> tuple[str, date, str, str]:
        return self.job.source_identity


__all__ = [
    "MemoryJob",
    "MemoryJobBlockedError",
    "MemoryJobConfig",
    "MemoryJobError",
    "MemoryJobExecutionError",
    "MemoryJobLease",
    "MemoryJobLeaseError",
    "MemoryJobLeaseLostError",
    "MemoryJobNotReadyError",
    "MemoryJobStatus",
]
