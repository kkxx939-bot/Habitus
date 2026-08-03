"""保存 L2 生命周期扫描游标和单节点退避状态。"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from foundation.integrity import canonical_json
from infrastructure.store.contracts import PathLock
from infrastructure.store.filesystem import atomic_replace_bytes, read_regular_bytes
from memory.model import MemoryAddress
from memory.tree import MemoryTree
from memory.uri import MemoryURI


class MemoryLifecycleScanStateError(RuntimeError):
    """L2 生命周期扫描状态损坏或无法耐久更新。"""


@dataclass(frozen=True)
class MemoryLifecycleScanFailure:
    uri: MemoryURI
    attempts: int
    retry_at: datetime
    error_type: str
    message: str

    def __post_init__(self) -> None:
        parsed = MemoryURI.parse(self.uri)
        parsed.to_address()
        object.__setattr__(self, "uri", parsed)
        if isinstance(self.attempts, bool) or not isinstance(self.attempts, int) or self.attempts <= 0:
            raise ValueError("memory lifecycle failure attempts must be positive")
        object.__setattr__(self, "retry_at", _timestamp(self.retry_at))
        for name in ("error_type", "message"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"memory lifecycle failure {name} must be non-empty")
        if len(self.message) > 1_000:
            raise ValueError("memory lifecycle failure message exceeds its bound")

    def to_dict(self) -> dict[str, object]:
        return {
            "uri": str(self.uri),
            "attempts": self.attempts,
            "retry_at": _format_timestamp(self.retry_at),
            "error_type": self.error_type,
            "message": self.message,
        }


class MemoryLifecycleScanStore:
    """用一个小型规范 JSON 保存跨重启游标和有界失败集合。"""

    def __init__(
        self,
        tree: MemoryTree,
        path_lock: PathLock,
        *,
        max_failures: int = 10_000,
        max_file_bytes: int = 2 * 1024 * 1024,
        base_retry_seconds: float = 60.0,
        max_retry_seconds: float = 86_400.0,
    ) -> None:
        if not isinstance(tree, MemoryTree):
            raise TypeError("tree must be MemoryTree")
        if not isinstance(path_lock, PathLock):
            raise TypeError("path_lock must be PathLock")
        if isinstance(max_failures, bool) or not isinstance(max_failures, int) or max_failures <= 0:
            raise ValueError("max_failures must be positive")
        if isinstance(max_file_bytes, bool) or not isinstance(max_file_bytes, int) or max_file_bytes <= 0:
            raise ValueError("max_file_bytes must be positive")
        for name, value in {
            "base_retry_seconds": base_retry_seconds,
            "max_retry_seconds": max_retry_seconds,
        }.items():
            if isinstance(value, bool) or not isinstance(value, int | float) or float(value) <= 0:
                raise ValueError(f"{name} must be positive numeric data")
        if float(max_retry_seconds) < float(base_retry_seconds):
            raise ValueError("max_retry_seconds cannot be lower than base_retry_seconds")
        self.root = (tree.root.parent / "workflow").resolve(strict=False)
        self.path = self.root / "lifecycle" / "l2_scan_state.json"
        self.path_lock = path_lock
        self.max_failures = max_failures
        self.max_file_bytes = max_file_bytes
        self.base_retry_seconds = float(base_retry_seconds)
        self.max_retry_seconds = float(max_retry_seconds)
        digest = hashlib.sha256(str(tree.root).encode("utf-8")).hexdigest()[:24]
        self.lock_key = f"memory-lifecycle-scan:{digest}"

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._read()

    def cursor(self) -> MemoryAddress | None:
        after, _failures = self._read()
        return None if after is None else MemoryURI.parse(after).to_address()

    def eligible(self, uri: MemoryURI, *, now: datetime) -> bool:
        _after, failures = self._read()
        failure = failures.get(str(uri))
        return failure is None or _timestamp(now) >= failure.retry_at

    def advance(self, address: MemoryAddress) -> None:
        uri = MemoryURI.from_address(address)
        with self.path_lock.acquire(self.lock_key, ttl_seconds=30):
            _after, failures = self._read()
            self._write(str(uri), failures)

    def reset_cursor(self) -> None:
        with self.path_lock.acquire(self.lock_key, ttl_seconds=30):
            _after, failures = self._read()
            self._write(None, failures)

    def clear_failure(self, uri: MemoryURI) -> None:
        with self.path_lock.acquire(self.lock_key, ttl_seconds=30):
            after, failures = self._read()
            failures.pop(str(uri), None)
            self._write(after, failures)

    def record_failure(
        self,
        uri: MemoryURI,
        error: Exception,
        *,
        failed_at: datetime,
    ) -> MemoryLifecycleScanFailure:
        current_time = _timestamp(failed_at)
        with self.path_lock.acquire(self.lock_key, ttl_seconds=30):
            after, failures = self._read()
            previous = failures.get(str(uri))
            attempts = 1 if previous is None else previous.attempts + 1
            delay = min(self.max_retry_seconds, self.base_retry_seconds * (2 ** min(20, attempts - 1)))
            message = str(error).strip() or type(error).__name__
            failure = MemoryLifecycleScanFailure(
                uri=uri,
                attempts=attempts,
                retry_at=current_time + timedelta(seconds=delay),
                error_type=type(error).__name__,
                message=message[:1_000],
            )
            failures[str(uri)] = failure
            if len(failures) > self.max_failures:
                oldest = min(failures.values(), key=lambda item: (item.retry_at, str(item.uri)))
                failures.pop(str(oldest.uri), None)
            self._write(after, failures)
            return failure

    def _read(self) -> tuple[str | None, dict[str, MemoryLifecycleScanFailure]]:
        try:
            encoded = read_regular_bytes(
                self.path,
                artifact_root=self.root,
                max_bytes=self.max_file_bytes,
            )
        except FileNotFoundError:
            return None, {}
        try:
            raw = json.loads(encoded)
            if not isinstance(raw, Mapping) or set(raw) != {"schema", "after_uri", "failures"}:
                raise ValueError("invalid fields")
            if raw["schema"] != "memory_lifecycle_scan_v1":
                raise ValueError("unsupported schema")
            after = raw["after_uri"]
            if after is not None:
                MemoryURI.parse(after).to_address()
            values = raw["failures"]
            if not isinstance(values, list) or len(values) > self.max_failures:
                raise ValueError("invalid failures")
            failures: dict[str, MemoryLifecycleScanFailure] = {}
            for value in values:
                if not isinstance(value, Mapping) or set(value) != {
                    "uri",
                    "attempts",
                    "retry_at",
                    "error_type",
                    "message",
                }:
                    raise ValueError("invalid failure")
                failure = MemoryLifecycleScanFailure(
                    uri=MemoryURI.parse(value["uri"]),
                    attempts=value["attempts"],
                    retry_at=_parse_timestamp(value["retry_at"]),
                    error_type=value["error_type"],
                    message=value["message"],
                )
                if str(failure.uri) in failures:
                    raise ValueError("duplicate failure")
                failures[str(failure.uri)] = failure
            if encoded != self._encode(after, failures):
                raise ValueError("non-canonical state")
            return after, failures
        except Exception as exc:
            raise MemoryLifecycleScanStateError("failed to read L2 lifecycle scan state") from exc

    def _write(
        self,
        after: str | None,
        failures: dict[str, MemoryLifecycleScanFailure],
    ) -> None:
        encoded = self._encode(after, failures)
        if len(encoded) > self.max_file_bytes:
            raise MemoryLifecycleScanStateError("L2 lifecycle scan state exceeds its file bound")
        atomic_replace_bytes(self.path, encoded, artifact_root=self.root)

    @staticmethod
    def _encode(after: str | None, failures: dict[str, MemoryLifecycleScanFailure]) -> bytes:
        return (
            canonical_json(
                {
                    "schema": "memory_lifecycle_scan_v1",
                    "after_uri": after,
                    "failures": [failures[key].to_dict() for key in sorted(failures)],
                }
            )
            + "\n"
        ).encode("utf-8")


def _timestamp(value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("memory lifecycle scan timestamp must be timezone-aware")
    return value.astimezone(timezone.utc)


def _format_timestamp(value: datetime) -> str:
    return _timestamp(value).isoformat().replace("+00:00", "Z")


def _parse_timestamp(value: object) -> datetime:
    if not isinstance(value, str) or not value:
        raise ValueError("memory lifecycle scan timestamp is invalid")
    return _timestamp(datetime.fromisoformat(value.replace("Z", "+00:00")))


__all__ = [
    "MemoryLifecycleScanFailure",
    "MemoryLifecycleScanStateError",
    "MemoryLifecycleScanStore",
]
