"""夜线的 Job 账：每一晚 5 段各一个耐久 Job，落在 ``<scene_root>/nightly/jobs/<这一晚>.json``（一晚一份，原子替换）。

一个 Job 记：状态（排队 / 在跑 / 成了 / 停下 / 失败）、尝试次数、下一次能试的时刻（停下之后有界退避）、认领（``claim_id`` +
``claim_generation`` + ``worker_id`` + ``lease_expires_at``）、停下或没准备好的原因、起止时刻、这一段原有的报告摘要，以及前置处理
里重试那一支等锁时留下的那一份（``RetryCheckpoint``）。这份账取代了原来的 ``last_run.json``（旧文件不读、不迁移）。

照 MemoryJob 的写法、不复用它的本体（那一套绑死在记忆领域）：

- **认领**：只有执行器拿着全局租约（``SQLiteLockStore``）时才写账，每次写都在 ``LockStore.fenced`` 里先校验租约；认领时
  ``claim_generation`` 加一、换新的 ``claim_id``；在跑的 Job 租约没过期不让接管，过期了只能递增 generation 接管。
- **结算**（成了 / 没准备好 / 停下）和续约都要对上同一份认领：陈旧的认领（被别人接管了）写不进来（``NightClaimLostError``）。
- **"这一晚的线跑完了没有"**（``finished``）三值照旧：5 段都成了为真；有账、没全成（停下了、没跑到、过了那天没跑完）或这一晚
  根本没有账（没开机）为假；比留存里第一次跑完的那一晚还早（或一晚都没跑完过）是 None——认不出，调用方各按各的保守口径。
"""

from __future__ import annotations

import json
import threading
import uuid
from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field, replace
from datetime import date, datetime, timedelta
from pathlib import Path

from habitus.behavior.retry import RetryCheckpoint
from habitus.foundation.lifecycle.backoff import BackoffPolicy
from habitus.infrastructure.store.filesystem import (
    atomic_replace_bytes,
    durable_unlink,
    list_real_directory,
    read_regular_bytes,
)
from habitus.runtime.nightly import DONE, HELD, STEPS, StepResult
from habitus.runtime.nightly_errors import NightJobsError

QUEUED, RUNNING, SUCCEEDED, STOPPED, FAILED = "queued", "running", "done", "stopped", "failed"
STATES = (QUEUED, RUNNING, SUCCEEDED, STOPPED, FAILED)
_SCHEMA = "nightly_jobs_v1"
_MAX_BYTES = 1_048_576
_MAX_FILES = 4_096
#: 过了那一晚还没跑完的段记成失败时的原因。
NIGHT_PASSED = "这一晚过去了，没跑完"
_JOB_KEYS = frozenset(
    {
        "step",
        "state",
        "attempts",
        "next_attempt_at",
        "claim_id",
        "claim_generation",
        "worker_id",
        "lease_expires_at",
        "reason",
        "started_at",
        "finished_at",
        "duration_seconds",
        "report",
        "retry_checkpoint",
    }
)


class NightClaimLostError(RuntimeError):
    """这份认领已经不是账上的那一份（被接管、或这一段已经结算）：陈旧的执行者不许再写。"""


@dataclass(frozen=True)
class NightJob:
    step: str
    state: str = QUEUED
    attempts: int = 0
    next_attempt_at: datetime | None = None
    claim_id: str | None = None
    claim_generation: int = 0
    worker_id: str | None = None
    lease_expires_at: datetime | None = None
    #: 停下、失败或还没准备好的原因（成了是 None）。
    reason: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    #: 最近一次跑这一段用了多久。
    duration_seconds: float = 0.0
    report: Mapping[str, object] = field(default_factory=dict)
    retry_checkpoint: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        if self.step not in STEPS or self.state not in STATES:
            raise NightJobsError(f"unknown nightly job step or state: {self.step!r} {self.state!r}")
        if self.attempts < 0 or self.claim_generation < 0 or self.duration_seconds < 0:
            raise NightJobsError("nightly job counters must be non-negative")
        for value in (self.next_attempt_at, self.lease_expires_at, self.started_at, self.finished_at):
            if value is not None and value.utcoffset() is None:
                raise NightJobsError("nightly job times must be timezone-aware")

    def runnable(self, now: datetime) -> bool:
        """现在能不能认领：排队的、退避到点的停下的、租约过期的在跑的。"""

        if self.state == QUEUED:
            return True
        if self.state == STOPPED:
            return self.next_attempt_at is None or self.next_attempt_at <= now
        if self.state == RUNNING:
            return self.lease_expires_at is None or self.lease_expires_at <= now
        return False


@dataclass(frozen=True)
class NightLedger:
    night: date
    created_at: datetime
    jobs: tuple[NightJob, ...]

    def __post_init__(self) -> None:
        if tuple(job.step for job in self.jobs) != STEPS:
            raise NightJobsError("a night ledger holds one job per step, in the ruled order")

    def job(self, step: str) -> NightJob:
        return self.jobs[STEPS.index(step)]

    @property
    def complete(self) -> bool:
        return all(job.state == SUCCEEDED for job in self.jobs)

    @property
    def failed(self) -> NightJob | None:
        return next((job for job in self.jobs if job.state == FAILED), None)

    @property
    def closed(self) -> bool:
        """这一晚不会再跑了：5 段都成了，或有一段失败了。"""

        return self.complete or self.failed is not None

    def next_job(self) -> NightJob | None:
        """按先后第一个没成的段（失败的也算：它挡着后面）；全成了是 None。"""

        return next((job for job in self.jobs if job.state != SUCCEEDED), None)

    def replaced(self, job: NightJob) -> NightLedger:
        jobs = list(self.jobs)
        jobs[STEPS.index(job.step)] = job
        return replace(self, jobs=tuple(jobs))

    @classmethod
    def fresh(cls, night: date, now: datetime) -> NightLedger:
        return cls(night=night, created_at=now, jobs=tuple(NightJob(step) for step in STEPS))


@dataclass(frozen=True)
class NightClaim:
    night: date
    step: str
    claim_id: str
    claim_generation: int
    worker_id: str
    attempt: int


def claim(ledger: NightLedger, step: str, *, worker_id: str, now: datetime) -> NightClaim:
    """认领这一段：generation 加一、换新的 claim_id。不是按先后该跑的那一段、或现在还不能认领，抛 ``NightClaimLostError``。"""

    job = ledger.next_job()
    if job is None or job.step != step or not job.runnable(now):
        raise NightClaimLostError(f"nightly job {ledger.night}/{step} is not claimable now")
    return NightClaim(ledger.night, step, uuid.uuid4().hex, job.claim_generation + 1, worker_id, job.attempts + 1)


def claimed(ledger: NightLedger, token: NightClaim, *, now: datetime, ttl_seconds: float) -> NightLedger:
    job = ledger.job(token.step)
    return ledger.replaced(
        replace(
            job,
            state=RUNNING,
            attempts=token.attempt,
            claim_id=token.claim_id,
            claim_generation=token.claim_generation,
            worker_id=token.worker_id,
            lease_expires_at=now + timedelta(seconds=ttl_seconds),
            started_at=job.started_at or now,
        )
    )


def require_current(ledger: NightLedger, token: NightClaim) -> NightJob:
    job = ledger.job(token.step)
    if (
        job.state != RUNNING
        or job.claim_id != token.claim_id
        or job.claim_generation != token.claim_generation
        or job.worker_id != token.worker_id
    ):
        raise NightClaimLostError(f"nightly job {ledger.night}/{token.step} is no longer held by this claim")
    return job


def renewed(ledger: NightLedger, token: NightClaim, *, now: datetime, ttl_seconds: float) -> NightLedger:
    job = require_current(ledger, token)
    return ledger.replaced(replace(job, lease_expires_at=now + timedelta(seconds=ttl_seconds)))


def settled(
    ledger: NightLedger,
    token: NightClaim,
    result: StepResult,
    *,
    now: datetime,
    duration_seconds: float,
    backoff: BackoffPolicy,
    max_attempts: int,
) -> NightLedger:
    """按这一段的结局结算：成了 → 成了；还没准备好 → 回到排队（不算一次尝试）；停下 → 退避后再试，次数用完 → 失败。"""

    job = require_current(ledger, token)
    base = replace(
        job,
        claim_id=None,
        worker_id=None,
        lease_expires_at=None,
        duration_seconds=max(0.0, duration_seconds),
        report=dict(result.report),
        reason=result.reason,
        retry_checkpoint=None if result.retry_checkpoint is None else encode_checkpoint(result.retry_checkpoint),
    )
    if result.outcome == DONE:
        return ledger.replaced(replace(base, state=SUCCEEDED, finished_at=now, next_attempt_at=None))
    if result.outcome == HELD:
        return ledger.replaced(replace(base, state=QUEUED, attempts=max(0, job.attempts - 1), next_attempt_at=None))
    if job.attempts >= max_attempts:
        return ledger.replaced(replace(base, state=FAILED, finished_at=now, next_attempt_at=None))
    retry_at = now + timedelta(seconds=backoff.delay(job.attempts))
    return ledger.replaced(replace(base, state=STOPPED, next_attempt_at=retry_at))


def released(ledger: NightLedger, token: NightClaim) -> NightLedger:
    """执行被取消（停机超时、租约丢了）：放回排队，不算一次尝试，下一次安全接着跑。"""

    job = require_current(ledger, token)
    return ledger.replaced(
        replace(
            job, state=QUEUED, attempts=max(0, job.attempts - 1), claim_id=None, worker_id=None, lease_expires_at=None
        )
    )


def expired(ledger: NightLedger, *, now: datetime) -> NightLedger:
    """这一晚过去了还没跑完：没成的段记失败（"这一晚过去了，没跑完"），不在新的日子上按新日子的时钟补算（那会算成别的日子）。"""

    jobs = tuple(
        job
        if job.state in (SUCCEEDED, FAILED)
        else replace(
            job,
            state=FAILED,
            reason=NIGHT_PASSED if job.reason is None or job.state == QUEUED else f"{NIGHT_PASSED}（{job.reason}）",
            claim_id=None,
            worker_id=None,
            lease_expires_at=None,
            next_attempt_at=None,
            finished_at=now,
        )
        for job in ledger.jobs
    )
    return replace(ledger, jobs=jobs)


class NightJobs:
    """Job 账的落盘：一晚一份文件。``fence`` 是"写之前先校验租约"的上下文（执行器交 ``LockStore.fenced``）；读不需要。"""

    def __init__(self, root: str | Path, *, artifact_root: str | Path, kept_nights: int) -> None:
        if kept_nights < 1:
            raise ValueError("kept_nights must be a positive integer")
        self.root = Path(root)
        self.artifact_root = Path(artifact_root)
        self.kept_nights = kept_nights
        self._cache: dict[date, tuple[tuple[int, int], NightLedger]] = {}
        self._lock = threading.Lock()

    def path(self, night: date) -> Path:
        return self.root / f"{night.isoformat()}.json"

    def nights(self) -> tuple[date, ...]:
        found: list[date] = []
        for entry in list_real_directory(self.root, artifact_root=self.artifact_root, max_entries=_MAX_FILES):
            if entry.is_file() and entry.name.endswith(".json") and not entry.name.startswith("."):
                try:
                    found.append(date.fromisoformat(entry.name[:-5]))
                except ValueError:
                    continue  # 不是按日子命名的文件不是账
        return tuple(sorted(found))

    def read(self, night: date) -> NightLedger | None:
        path = self.path(night)
        try:
            info = path.stat()
        except FileNotFoundError:
            return None
        stamp = (info.st_mtime_ns, info.st_size)
        with self._lock:
            cached = self._cache.get(night)
            if cached is not None and cached[0] == stamp:
                return cached[1]
        try:
            raw = read_regular_bytes(path, artifact_root=self.artifact_root, max_bytes=_MAX_BYTES)
        except FileNotFoundError:
            return None
        ledger = decode_ledger(raw, night=night)
        with self._lock:
            self._cache[night] = (stamp, ledger)
        return ledger

    def latest(self) -> NightLedger | None:
        nights = self.nights()
        return None if not nights else self.read(nights[-1])

    def write(self, ledger: NightLedger, *, fence: Callable[[], AbstractContextManager[object]] | None = None) -> None:
        with (fence or nullcontext)():
            atomic_replace_bytes(self.path(ledger.night), encode_ledger(ledger), artifact_root=self.artifact_root)

    def update(
        self,
        night: date,
        change: Callable[[NightLedger | None], NightLedger],
        *,
        fence: Callable[[], AbstractContextManager[object]] | None = None,
    ) -> NightLedger:
        """在租约校验之内读、改、写这一晚的账（``change`` 抛出就什么都不写）。"""

        with (fence or nullcontext)():
            ledger = change(self.read(night))
            atomic_replace_bytes(self.path(night), encode_ledger(ledger), artifact_root=self.artifact_root)
            return ledger

    def prune(self, newest: date) -> int:
        """比 ``newest`` 早 ``kept_nights`` 晚以上的账删掉。"""

        floor = newest - timedelta(days=self.kept_nights)
        removed = 0
        for night in self.nights():
            if night < floor and durable_unlink(self.path(night), artifact_root=self.artifact_root):
                removed += 1
                with self._lock:
                    self._cache.pop(night, None)
        return removed

    def finished(self, night: date) -> bool | None:
        """这一晚的线整条跑完了没有（见模块文档串的三值口径）。一份账都没有、或这一晚早于留存里第一次跑完的那一晚：None。
        要读的账（第一次跑完之前的那几份、以及这一晚自己的）读不了时抛 ``NightJobsError``——不吞成"没跑完"：那会让结算把它
        认作停掉的那天、截断后补（codex R2）。"""

        try:
            nights = self.nights()
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise NightJobsError(f"nightly job ledgers are unreadable: {type(exc).__name__}") from exc
        since = next((item for item in nights if item <= night and self._complete(item)), None)
        if since is None:
            return None
        return night == since or self._complete(night)

    def _complete(self, night: date) -> bool:
        """这一晚的账全成了（没有账是没跑完）；读不了抛 ``NightJobsError``。"""

        try:
            ledger = self.read(night)
        except OSError as exc:
            raise NightJobsError(f"nightly job ledger of {night} is unreadable: {type(exc).__name__}") from exc
        return ledger is not None and ledger.complete


def encode_ledger(ledger: NightLedger) -> bytes:
    raw = {
        "schema": _SCHEMA,
        "night": ledger.night.isoformat(),
        "created_at": ledger.created_at.isoformat(),
        "jobs": [_encode_job(job) for job in ledger.jobs],
    }
    encoded = json.dumps(raw, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8")
    if len(encoded) > _MAX_BYTES:
        raise NightJobsError("nightly job ledger exceeds its byte bound")
    return encoded


def _encode_job(job: NightJob) -> dict[str, object]:
    return {
        "step": job.step,
        "state": job.state,
        "attempts": job.attempts,
        "next_attempt_at": _time(job.next_attempt_at),
        "claim_id": job.claim_id,
        "claim_generation": job.claim_generation,
        "worker_id": job.worker_id,
        "lease_expires_at": _time(job.lease_expires_at),
        "reason": job.reason,
        "started_at": _time(job.started_at),
        "finished_at": _time(job.finished_at),
        "duration_seconds": round(job.duration_seconds, 3),
        "report": dict(job.report),
        "retry_checkpoint": None if job.retry_checkpoint is None else dict(job.retry_checkpoint),
    }


def decode_ledger(raw: bytes, *, night: date) -> NightLedger:
    try:
        data = json.loads(raw.decode("utf-8"))
        if not isinstance(data, dict) or set(data) != {"schema", "night", "created_at", "jobs"}:
            raise NightJobsError("nightly job ledger has unexpected fields")
        if data["schema"] != _SCHEMA or date.fromisoformat(data["night"]) != night:
            raise NightJobsError("nightly job ledger belongs to another schema or night")
        jobs = tuple(_decode_job(item) for item in data["jobs"])
        return NightLedger(night=night, created_at=datetime.fromisoformat(data["created_at"]), jobs=jobs)
    except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, NightJobsError):
            raise
        raise NightJobsError(f"nightly job ledger of {night} is corrupt: {exc}") from exc


def _decode_job(item: object) -> NightJob:
    if not isinstance(item, dict) or set(item) != _JOB_KEYS:
        raise NightJobsError("nightly job has unexpected fields")
    report = item["report"]
    checkpoint = item["retry_checkpoint"]
    if not isinstance(report, dict) or not (checkpoint is None or isinstance(checkpoint, dict)):
        raise NightJobsError("nightly job report and retry checkpoint must be objects")
    return NightJob(
        step=str(item["step"]),
        state=str(item["state"]),
        attempts=int(item["attempts"]),
        next_attempt_at=_parse(item["next_attempt_at"]),
        claim_id=None if item["claim_id"] is None else str(item["claim_id"]),
        claim_generation=int(item["claim_generation"]),
        worker_id=None if item["worker_id"] is None else str(item["worker_id"]),
        lease_expires_at=_parse(item["lease_expires_at"]),
        reason=None if item["reason"] is None else str(item["reason"]),
        started_at=_parse(item["started_at"]),
        finished_at=_parse(item["finished_at"]),
        duration_seconds=float(item["duration_seconds"]),
        report=report,
        retry_checkpoint=checkpoint,
    )


def encode_checkpoint(checkpoint: RetryCheckpoint) -> dict[str, object]:
    """重试那一支等锁时留下的那一份，存进前置处理那个 Job。"""

    return {
        "through": checkpoint.through.isoformat(),
        "waiting": {
            name: {"written": list(written), "busy": list(busy)} for name, (written, busy) in checkpoint.waiting.items()
        },
        "counts": dict(checkpoint.counts),
    }


def decode_checkpoint(raw: Mapping[str, object]) -> RetryCheckpoint:
    """从 Job 账读回重试那一支等锁时留下的那一份；格式不对抛 ``NightJobsError``。"""

    waiting, counts = raw.get("waiting"), raw.get("counts")
    if not isinstance(waiting, dict) or not isinstance(counts, dict):
        raise NightJobsError("retry checkpoint needs waiting and counts objects")
    try:
        return RetryCheckpoint(
            through=date.fromisoformat(str(raw["through"])),
            waiting={
                str(name): (tuple(str(item) for item in value["written"]), tuple(str(item) for item in value["busy"]))
                for name, value in waiting.items()
            },
            counts={str(name): int(value) for name, value in counts.items()},
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise NightJobsError(f"retry checkpoint is corrupt: {exc}") from exc


def _time(value: datetime | None) -> str | None:
    return None if value is None else value.isoformat()


def _parse(value: object) -> datetime | None:
    return None if value is None else datetime.fromisoformat(str(value))


__all__ = [
    "FAILED",
    "NIGHT_PASSED",
    "QUEUED",
    "RUNNING",
    "STATES",
    "STOPPED",
    "SUCCEEDED",
    "NightClaim",
    "NightClaimLostError",
    "NightJob",
    "NightJobs",
    "NightJobsError",
    "NightLedger",
    "claim",
    "decode_checkpoint",
    "encode_checkpoint",
    "claimed",
    "expired",
    "released",
    "renewed",
    "require_current",
    "settled",
]
