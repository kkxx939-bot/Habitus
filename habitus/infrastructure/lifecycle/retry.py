"""重试账：替掉各处"失败了就下一轮原样再来、没有上限"的重排。

- 可重试的（传输、超时）按退避推迟，次数用完转确定性；
- 确定性的（标了 ``DeterministicFailure``）第一次就不再重试，交调用方处置（会话源：放弃、写成没读懂、进每晚重试的堆）；
- 一个 owner 一份文件；写是"读整份、改、整份写回"，同一进程里几个线程（并发恢复、生命周期的回收）会同时写，所以
  按文件路径串起来（L10.10）。跨进程不共用同一个 owner。

账本身也有退出：成功即 ``clear``；对象已经不在的键由 ``reap`` 回收；放弃了太久的由 ``expire_given_up`` 回收
（键里带口径的，口径一变旧键就再也不会被清）。条目有上限，满了先挤掉最早到期的
可重试条目（它们被挤掉只意味着下一轮会再试一次），全是确定性的才抛错。
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path

from habitus.foundation.ids import require_safe_path_segment
from habitus.foundation.integrity import canonical_json
from habitus.foundation.lifecycle import BackoffPolicy, DeterministicFailure
from habitus.infrastructure.store.filesystem import atomic_replace_bytes, read_regular_bytes

_SCHEMA = "lifecycle_retry_v1"
_MESSAGE_LIMIT = 500


_LOCKS: dict[Path, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()


def _lock_for(path: Path) -> threading.Lock:
    with _LOCKS_GUARD:
        return _LOCKS.setdefault(path, threading.Lock())


class FailureKind(str, Enum):
    RETRYABLE = "retryable"
    DETERMINISTIC = "deterministic"


class RetryLedgerFullError(RuntimeError):
    """条目全是确定性失败、又撞上了上限：放弃了的条目回收得不够快，要人去看。"""


@dataclass(frozen=True)
class RetryEntry:
    key: str
    attempts: int
    next_at: datetime
    kind: FailureKind
    error_type: str
    message: str


@dataclass(frozen=True)
class RetryDecision:
    entry: RetryEntry

    @property
    def give_up(self) -> bool:
        """不再重试：调用方该放弃这一份（写成没读懂）、发 signal。"""

        return self.entry.kind is FailureKind.DETERMINISTIC


class RetryLedger:
    def __init__(
        self,
        root: str | Path,
        *,
        owner: str,
        policy: BackoffPolicy,
        max_attempts: int,
        clock: Callable[[], datetime],
        max_entries: int = 10_000,
        max_file_bytes: int = 8 * 1024 * 1024,
    ) -> None:
        if not isinstance(policy, BackoffPolicy):
            raise TypeError("policy must be BackoffPolicy")
        for label, value in (
            ("max_attempts", max_attempts),
            ("max_entries", max_entries),
            ("max_file_bytes", max_file_bytes),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{label} must be a positive integer")
        self.root = Path(root).expanduser().resolve(strict=False)
        self.owner = require_safe_path_segment(owner, "owner")
        self.path = self.root / "lifecycle" / "retry" / f"{self.owner}.json"
        self.policy = policy
        self.max_attempts = max_attempts
        self.max_entries = max_entries
        self.max_file_bytes = max_file_bytes
        self._clock = clock
        self._lock = _lock_for(self.path)

    def record_failure(self, key: str, error: BaseException) -> RetryDecision:
        """记一次失败，返回这次之后该怎么办。"""

        with self._lock:
            entries = self._read()
            previous = entries.get(key)
            attempts = 1 if previous is None else previous.attempts + 1
            deterministic = isinstance(error, DeterministicFailure) or attempts >= self.max_attempts
            now = self._now()
            entry = RetryEntry(
                key=_require_key(key),
                attempts=attempts,
                next_at=now if deterministic else now + timedelta(seconds=self.policy.delay(attempts)),
                kind=FailureKind.DETERMINISTIC if deterministic else FailureKind.RETRYABLE,
                error_type=type(error).__name__,
                message=(str(error).strip() or type(error).__name__)[:_MESSAGE_LIMIT],
            )
            entries[key] = entry
            self._write(self._bounded(entries))
            return RetryDecision(entry)

    def entry(self, key: str) -> RetryEntry | None:
        return self._read().get(key)

    def blocked(self, key: str) -> bool:
        """现在不该试：已放弃，或者还在退避期内。"""

        entry = self._read().get(key)
        if entry is None:
            return False
        return entry.kind is FailureKind.DETERMINISTIC or entry.next_at > self._now()

    def due(self) -> tuple[str, ...]:
        """退避期已过、可以再试的键。"""

        now = self._now()
        return tuple(
            key
            for key, entry in sorted(self._read().items())
            if entry.kind is FailureKind.RETRYABLE and entry.next_at <= now
        )

    def given_up(self) -> tuple[RetryEntry, ...]:
        return tuple(entry for _key, entry in sorted(self._read().items()) if entry.kind is FailureKind.DETERMINISTIC)

    def clear(self, key: str) -> bool:
        """成功了，或者对象被重放 / 放弃了。"""

        with self._lock:
            entries = self._read()
            if entries.pop(key, None) is None:
                return False
            self._write(entries)
            return True

    def reap(self, alive: Callable[[str], bool]) -> int:
        """回收对象已经不在的键。"""

        return self._drop(lambda key, _entry: not alive(key))

    def expire_given_up(self, before: datetime) -> int:
        """放弃时刻早于 ``before`` 的确定性条目回收（统一寿命 ``lifecycle.retry.given_up_days``，由组合根给）。"""

        return self._drop(lambda _key, entry: entry.kind is FailureKind.DETERMINISTIC and entry.next_at < before)

    def _drop(self, doomed: Callable[[str, RetryEntry], bool]) -> int:
        with self._lock:
            entries = self._read()
            dead = [key for key, entry in entries.items() if doomed(key, entry)]
            for key in dead:
                del entries[key]
            if dead:
                self._write(entries)
            return len(dead)

    def size(self) -> int:
        return len(self._read())

    # ── 内部 ──────────────────────────────────────────────────────────────────────────

    def _now(self) -> datetime:
        now = self._clock()
        if now.utcoffset() is None:
            raise ValueError("clock must return timezone-aware datetimes")
        return now

    def _bounded(self, entries: dict[str, RetryEntry]) -> dict[str, RetryEntry]:
        if len(entries) <= self.max_entries:
            return entries
        retryable = sorted(
            (entry for entry in entries.values() if entry.kind is FailureKind.RETRYABLE),
            key=lambda entry: (entry.next_at, entry.key),
        )
        excess = len(entries) - self.max_entries
        if len(retryable) < excess:
            raise RetryLedgerFullError(f"retry ledger {self.owner!r} is full of deterministic failures")
        for entry in retryable[:excess]:
            del entries[entry.key]
        return entries

    def _read(self) -> dict[str, RetryEntry]:
        try:
            encoded = read_regular_bytes(self.path, artifact_root=self.root, max_bytes=self.max_file_bytes)
        except FileNotFoundError:
            return {}
        payload = json.loads(encoded.decode("utf-8"))
        if payload.get("schema") != _SCHEMA or payload.get("owner") != self.owner:
            raise ValueError(f"retry ledger {self.path} has an unexpected schema or owner")
        return {item["key"]: _decode(item) for item in payload["entries"]}

    def _write(self, entries: dict[str, RetryEntry]) -> None:
        payload = {
            "schema": _SCHEMA,
            "owner": self.owner,
            "entries": [_encode(entry) for _key, entry in sorted(entries.items())],
        }
        atomic_replace_bytes(self.path, canonical_json(payload).encode("utf-8"), artifact_root=self.root)


def _require_key(key: object) -> str:
    if not isinstance(key, str) or not key or "\n" in key:
        raise ValueError("retry key must be non-empty single-line text")
    return key


def _encode(entry: RetryEntry) -> dict[str, object]:
    return {
        "key": entry.key,
        "attempts": entry.attempts,
        "next_at": entry.next_at.isoformat(),
        "kind": entry.kind.value,
        "error_type": entry.error_type,
        "message": entry.message,
    }


def _decode(item: dict[str, object]) -> RetryEntry:
    attempts = item["attempts"]
    if isinstance(attempts, bool) or not isinstance(attempts, int) or attempts < 1:
        raise ValueError("retry attempts must be a positive integer")
    return RetryEntry(
        key=_require_key(item["key"]),
        attempts=attempts,
        next_at=datetime.fromisoformat(str(item["next_at"])),
        kind=FailureKind(item["kind"]),
        error_type=str(item["error_type"]),
        message=str(item["message"]),
    )


__all__ = ["FailureKind", "RetryDecision", "RetryEntry", "RetryLedger", "RetryLedgerFullError"]
