"""融合作业的耐久状态、暂存产物与租约模型。

## 为什么必须两阶段

融合不是纯函数：同一段观测重跑一次，模型换个说法就是**另一批判断**，落盘之后就是两套互不相认
的记录。所以模型调用必须恰好一次，派生出的判断必须在**任何一次落盘之前**先耐久暂存。
``STAGED`` 就是这个检查点——到达它之后，无论崩溃多少次都只从暂存的判断重放，不再碰模型。

  QUEUED ──claim──> RUNNING ──stage──> STAGED ──commit──> COMMITTED
                       │                  │
                       └── fail ──────────┴──> QUEUED / STAGED（退避重试）或 FAILED（次数用尽）

失败回退到哪里由**有没有暂存**决定，而不是由失败发生在哪一步决定：已暂存就回 ``STAGED``，
重试只重放发布；未暂存才回 ``QUEUED``，重试从模型开始。

## 暂存判断不能走 canonical_json

``canonicalize`` 会把 datetime 折成 UTC，而判断里的 ``started_at`` / ``last_observed_at`` 带的是
**本地偏移**：``2026-08-14T00:30+08:00`` 折完变成前一天，归约层按"人的一天"做的聚合就全错了。
所以暂存与判断存储一样，用固定键序 + 保留偏移的时间文本自行序列化（见 ``store.py``）。
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any

from habitus.behavior.fusion.errors import BehaviorFusionError
from habitus.behavior.fusion.receipt import BehaviorFusionReceipt, receipt_identity

FUSION_JOB_ERROR_MAX_CHARS = 2_000


def _positive_int(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value > 0


def _non_negative_int(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value >= 0


_WORKER_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class BehaviorFusionJobError(BehaviorFusionError):
    """融合作业无法安全排队、认领、推进或结算。"""


class BehaviorFusionJobBlockedError(BehaviorFusionJobError):
    """最早的作业失败、被他人持有，或调用方越过了它。"""


class BehaviorFusionJobNotReadyError(BehaviorFusionJobBlockedError):
    """最早的作业仍处于耐久重试退避期。"""

    def __init__(self, available_at: datetime) -> None:
        if not isinstance(available_at, datetime) or available_at.tzinfo is None:
            raise TypeError("available_at must be a timezone-aware datetime")
        self.available_at = available_at.astimezone(UTC)
        super().__init__(f"oldest fusion job is not ready before {self.available_at.isoformat()}")


class BehaviorFusionJobLeaseLostError(BehaviorFusionJobError):
    """Worker 的耐久租约已过期或被更新的 generation 接管。"""


class BehaviorFusionJobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    STAGED = "staged"
    COMMITTED = "committed"
    FAILED = "failed"


_REQUIRES_CHECKPOINT = frozenset({"staged", "committed"})


@dataclass(frozen=True)
class BehaviorFusionJobConfig:
    """作业队列的运维参数；不含任何语义判据。"""

    lease_ttl_seconds: int = 300
    max_attempts: int = 5
    retry_base_delay_seconds: float = 30.0
    retry_max_delay_seconds: float = 1_800.0
    lock_ttl_seconds: int = 30
    lock_wait_timeout_seconds: float = 5.0
    lock_retry_delay_seconds: float = 0.01
    max_files: int = 100_000
    # 作业携带完整的判断记录与回执，比观测交付大一个量级，单独给一档。
    max_file_bytes: int = 4_194_304

    def __post_init__(self) -> None:
        for name in ("retry_base_delay_seconds", "retry_max_delay_seconds",
                     "lock_wait_timeout_seconds", "lock_retry_delay_seconds"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int | float) or value <= 0:
                raise ValueError(f"{name} must be a positive number")
        for name in ("lease_ttl_seconds", "lock_ttl_seconds", "max_attempts", "max_files", "max_file_bytes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.retry_max_delay_seconds < self.retry_base_delay_seconds:
            raise ValueError("retry_max_delay_seconds must not be below retry_base_delay_seconds")

    def retry_delay_seconds(self, attempts: int) -> float:
        """指数退避并封顶；attempts 是本次失败前已经用掉的次数。"""

        if isinstance(attempts, bool) or not isinstance(attempts, int) or attempts < 0:
            raise ValueError("attempts must be a non-negative integer")
        scaled = self.retry_base_delay_seconds * (2 ** min(attempts, 16))
        return min(scaled, self.retry_max_delay_seconds)


@dataclass(frozen=True)
class StagedFusion:
    """融合完成后的耐久检查点：已派生的判断与它们的回执。

    回执在这里就完全成形，commit 时只是落盘。这样判断与回执由构造保证一一对应，不会出现
    "落了三条判断、回执只记了两条"。
    """

    receipt: BehaviorFusionReceipt
    judgements: tuple[Mapping[str, Any], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.receipt, BehaviorFusionReceipt):
            raise TypeError("receipt must be BehaviorFusionReceipt")
        if isinstance(self.judgements, (str, bytes)) or not isinstance(self.judgements, Sequence):
            raise BehaviorFusionJobError("staged judgements must be a sequence")
        object.__setattr__(self, "judgements", tuple(dict(item) for item in self.judgements))
        identities = [item["judgement_id"] for item in self.judgements]
        if len(identities) != len(set(identities)):
            raise BehaviorFusionJobError("staged judgements contain duplicate identities")
        if tuple(identities) != tuple(self.receipt.judgement_ids):
            raise BehaviorFusionJobError("staged judgements do not match the receipt")

    def to_dict(self) -> dict[str, Any]:
        return {
            "receipt": self.receipt.to_dict(),
            "judgements": [dict(item) for item in self.judgements],
        }

    @classmethod
    def from_dict(cls, value: object, label: str) -> StagedFusion:
        if not isinstance(value, Mapping) or set(value) != {"receipt", "judgements"}:
            raise BehaviorFusionJobError(f"{label} schema is invalid")
        raw = value["judgements"]
        if not isinstance(raw, list) or any(not isinstance(item, Mapping) for item in raw):
            raise BehaviorFusionJobError(f"{label}.judgements must be a list of objects")
        return cls(
            receipt=BehaviorFusionReceipt.from_dict(value["receipt"]),
            judgements=tuple(dict(item) for item in raw),
        )


@dataclass(frozen=True)
class BehaviorFusionJob:
    """一段观测的融合作业；作业身份与它最终产出的回执身份是同一个。"""

    fusion_sequence: int
    job_id: str
    segment_digest: str
    observation_ids: tuple[str, ...]
    source_refs: tuple[str, ...]
    fusion_version: str
    prompt_version: str
    status: BehaviorFusionJobStatus
    attempts: int
    claim_id: str | None
    claim_generation: int
    worker_id: str | None
    lease_expires_at: datetime | None
    next_attempt_at: datetime | None
    last_error: str | None
    staged: StagedFusion | None
    created_at: datetime
    updated_at: datetime

    def __post_init__(self) -> None:
        if not _positive_int(self.fusion_sequence):
            raise BehaviorFusionJobError("fusion_sequence must be a positive integer")
        for name in ("job_id", "segment_digest"):
            if not _SHA256.fullmatch(str(getattr(self, name))):
                raise BehaviorFusionJobError(f"{name} must be lowercase SHA-256 text")
        if not self.observation_ids or any(
            not _SHA256.fullmatch(item) for item in self.observation_ids
        ):
            raise BehaviorFusionJobError("observation_ids must be non-empty SHA-256 text")
        if not self.source_refs:
            raise BehaviorFusionJobError("a fusion job must record its source deliveries")
        if not isinstance(self.status, BehaviorFusionJobStatus):
            raise TypeError("status must be BehaviorFusionJobStatus")
        for name in ("attempts", "claim_generation"):
            if not _non_negative_int(getattr(self, name)):
                raise BehaviorFusionJobError(f"{name} must be a non-negative integer")
        expected_id = receipt_identity(self.segment_digest, self.fusion_version, self.prompt_version)
        if self.job_id != expected_id:
            raise BehaviorFusionJobError("job_id does not match its segment and versions")
        for name in ("lease_expires_at", "next_attempt_at", "created_at", "updated_at"):
            value = getattr(self, name)
            if value is None:
                continue
            if not isinstance(value, datetime) or value.tzinfo is None:
                raise BehaviorFusionJobError(f"{name} must be timezone-aware")
            object.__setattr__(self, name, value.astimezone(UTC))
        if self.created_at is None or self.updated_at is None:
            raise BehaviorFusionJobError("created_at and updated_at are required")
        if self.last_error is not None:
            if not isinstance(self.last_error, str) or not self.last_error.strip():
                raise BehaviorFusionJobError("last_error must be non-empty text or null")
            normalized = " ".join(self.last_error.split())
            object.__setattr__(self, "last_error", normalized[:FUSION_JOB_ERROR_MAX_CHARS])
        if self.worker_id is not None and not _WORKER_ID.fullmatch(self.worker_id):
            raise BehaviorFusionJobError("worker_id must be normalized stable text")
        if (self.status is BehaviorFusionJobStatus.RUNNING) != (self.claim_id is not None):
            raise BehaviorFusionJobError("only a RUNNING fusion job carries a claim")
        if (self.claim_id is None) != (self.lease_expires_at is None):
            raise BehaviorFusionJobError("a claim and its lease expiry must appear together")
        if (self.claim_id is None) != (self.worker_id is None):
            raise BehaviorFusionJobError("a claim and its worker must appear together")
        if self.status in _REQUIRES_CHECKPOINT and self.staged is None:
            raise BehaviorFusionJobError(f"a {self.status.value} fusion job must carry its checkpoint")
        if self.staged is not None:
            if not isinstance(self.staged, StagedFusion):
                raise TypeError("staged must be StagedFusion")
            if self.staged.receipt.receipt_id != self.job_id:
                raise BehaviorFusionJobError("staged receipt identity does not match its job")

    @property
    def is_terminal(self) -> bool:
        return self.status is BehaviorFusionJobStatus.COMMITTED

    @property
    def needs_fusion(self) -> bool:
        """还没到检查点——重试要从模型开始；到了就只重放发布。"""

        return self.staged is None

    @staticmethod
    def valid_worker_id(value: object) -> bool:
        return isinstance(value, str) and _WORKER_ID.fullmatch(value) is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": FUSION_JOB_SCHEMA,
            "fusion_sequence": self.fusion_sequence,
            "job_id": self.job_id,
            "segment_digest": self.segment_digest,
            "observation_ids": list(self.observation_ids),
            "source_refs": list(self.source_refs),
            "fusion_version": self.fusion_version,
            "prompt_version": self.prompt_version,
            "status": self.status.value,
            "attempts": self.attempts,
            "claim_id": self.claim_id,
            "claim_generation": self.claim_generation,
            "worker_id": self.worker_id,
            "lease_expires_at": _timestamp(self.lease_expires_at),
            "next_attempt_at": _timestamp(self.next_attempt_at),
            "last_error": self.last_error,
            "staged": None if self.staged is None else self.staged.to_dict(),
            "created_at": _timestamp(self.created_at),
            "updated_at": _timestamp(self.updated_at),
        }

    @classmethod
    def from_dict(cls, value: object) -> BehaviorFusionJob:
        if not isinstance(value, Mapping) or set(value) != _JOB_KEYS:
            raise BehaviorFusionJobError("fusion job schema is invalid")
        if value["schema"] != FUSION_JOB_SCHEMA:
            raise BehaviorFusionJobError("fusion job schema version is unsupported")
        try:
            status = BehaviorFusionJobStatus(value["status"])
        except ValueError as exc:
            raise BehaviorFusionJobError("fusion job status is invalid") from exc
        raw_staged = value["staged"]
        return cls(
            fusion_sequence=value["fusion_sequence"],
            job_id=value["job_id"],
            segment_digest=value["segment_digest"],
            observation_ids=tuple(value["observation_ids"]),
            source_refs=tuple(value["source_refs"]),
            fusion_version=value["fusion_version"],
            prompt_version=value["prompt_version"],
            status=status,
            attempts=value["attempts"],
            claim_id=value["claim_id"],
            claim_generation=value["claim_generation"],
            worker_id=value["worker_id"],
            lease_expires_at=_parse_timestamp(value["lease_expires_at"], "lease_expires_at"),
            next_attempt_at=_parse_timestamp(value["next_attempt_at"], "next_attempt_at"),
            last_error=value["last_error"],
            staged=None if raw_staged is None else StagedFusion.from_dict(raw_staged, "staged"),
            created_at=_required_timestamp(value["created_at"], "created_at"),
            updated_at=_required_timestamp(value["updated_at"], "updated_at"),
        )


FUSION_JOB_SCHEMA = "behavior_fusion_job_v1"
_JOB_KEYS = {
    "schema",
    "fusion_sequence",
    "job_id",
    "segment_digest",
    "observation_ids",
    "source_refs",
    "fusion_version",
    "prompt_version",
    "status",
    "attempts",
    "claim_id",
    "claim_generation",
    "worker_id",
    "lease_expires_at",
    "next_attempt_at",
    "last_error",
    "staged",
    "created_at",
    "updated_at",
}


@dataclass(frozen=True)
class BehaviorFusionJobLease:
    """一次认领的 fencing 凭据；结算时必须与磁盘上的身份完全一致。"""

    job: BehaviorFusionJob

    def __post_init__(self) -> None:
        if not isinstance(self.job, BehaviorFusionJob):
            raise TypeError("job must be BehaviorFusionJob")
        if self.job.status is not BehaviorFusionJobStatus.RUNNING or self.job.claim_id is None:
            raise BehaviorFusionJobError("a lease requires a RUNNING fusion job")

    @property
    def job_id(self) -> str:
        return self.job.job_id

    @property
    def claim_id(self) -> str:
        assert self.job.claim_id is not None
        return self.job.claim_id

    @property
    def claim_generation(self) -> int:
        return self.job.claim_generation

    @property
    def worker_id(self) -> str | None:
        return self.job.worker_id


@dataclass(frozen=True)
class BehaviorFusionQueueSnapshot:
    """不含错误正文与来源身份的队列聚合视图。"""

    queued: int
    running: int
    staged: int
    committed: int
    failed: int
    high_watermark: int
    oldest_age_seconds: float


def _timestamp(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _parse_timestamp(value: object, label: str) -> datetime | None:
    if value is None:
        return None
    return _required_timestamp(value, label)


def _required_timestamp(value: object, label: str) -> datetime:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise BehaviorFusionJobError(f"{label} must be timezone-aware")
        return value.astimezone(UTC)
    if not isinstance(value, str):
        raise BehaviorFusionJobError(f"{label} must be an ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise BehaviorFusionJobError(f"{label} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise BehaviorFusionJobError(f"{label} must be timezone-aware")
    return parsed.astimezone(UTC)


def lease_deadline(now: datetime, config: BehaviorFusionJobConfig) -> datetime:
    return now + timedelta(seconds=config.lease_ttl_seconds)


__all__ = [
    "FUSION_JOB_ERROR_MAX_CHARS",
    "FUSION_JOB_SCHEMA",
    "BehaviorFusionJob",
    "BehaviorFusionJobBlockedError",
    "BehaviorFusionJobConfig",
    "BehaviorFusionJobError",
    "BehaviorFusionJobLease",
    "BehaviorFusionJobLeaseLostError",
    "BehaviorFusionJobNotReadyError",
    "BehaviorFusionJobStatus",
    "BehaviorFusionQueueSnapshot",
    "StagedFusion",
    "lease_deadline",
]
