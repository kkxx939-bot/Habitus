"""保存 L2 生命周期扫描游标、单节点退避状态和压缩模型判过 KEEP 的内容版本。"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from habitus.foundation.integrity import canonical_json
from habitus.foundation.lifecycle import BackoffPolicy
from habitus.foundation.validation import is_int
from habitus.infrastructure.store.contracts import PathLock
from habitus.infrastructure.store.filesystem import atomic_replace_bytes, read_regular_bytes
from habitus.memory.model import MemoryAddress
from habitus.memory.tree import MemoryTree, memory_root_lock_prefix
from habitus.memory.uri import MemoryURI, decode_memory_uri

_SCHEMA = "memory_lifecycle_scan_v2"
# 旧版扫描账的 schema 标记：只用来认出旧文件并明确报错，不迁移、不读其中内容。
_STALE_SCHEMA = "memory_lifecycle_scan_v1"


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
        if self.attempts <= 0:
            raise ValueError("memory lifecycle failure attempts must be positive")
        object.__setattr__(self, "retry_at", _timestamp(self.retry_at))
        for name in ("error_type", "message"):
            value = getattr(self, name)
            if not value:
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


@dataclass(frozen=True)
class MemoryCompactionKeep:
    """压缩模型对某一版 L2 答 KEEP 的记录。

    键是内容版本（revision 与快照摘要）加最近一次被用到的时刻：内容变了、或被用到过（回到热区再冷下来），
    记录就不再匹配，下一次冷下来会重新送模型判。
    """

    uri: MemoryURI
    revision: int
    source_digest: str
    last_used_at: datetime | None
    judged_at: datetime

    def __post_init__(self) -> None:
        parsed = MemoryURI.parse(self.uri)
        parsed.to_address()
        object.__setattr__(self, "uri", parsed)
        if self.revision <= 0:
            raise ValueError("memory compaction keep revision must be positive")
        if not self.source_digest:
            raise ValueError("memory compaction keep source_digest must be non-empty")
        if self.last_used_at is not None:
            object.__setattr__(self, "last_used_at", _timestamp(self.last_used_at))
        object.__setattr__(self, "judged_at", _timestamp(self.judged_at))

    def same_version(self, other: MemoryCompactionKeep) -> bool:
        return (
            self.uri == other.uri
            and self.revision == other.revision
            and self.source_digest == other.source_digest
            and self.last_used_at == other.last_used_at
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "uri": str(self.uri),
            "revision": self.revision,
            "source_digest": self.source_digest,
            "last_used_at": None if self.last_used_at is None else _format_timestamp(self.last_used_at),
            "judged_at": _format_timestamp(self.judged_at),
        }


class MemoryLifecycleScanStore:
    """用一个小型规范 JSON 保存跨重启游标、有界失败集合和有界 KEEP 记录。"""

    def __init__(
        self,
        tree: MemoryTree,
        path_lock: PathLock,
        workflow_root: str | Path,
        *,
        max_failures: int = 10_000,
        max_compaction_keeps: int = 10_000,
        max_file_bytes: int = 2 * 1024 * 1024,
        base_retry_seconds: float = 60.0,
        max_retry_seconds: float = 86_400.0,
    ) -> None:
        if max_failures <= 0:
            raise ValueError("max_failures must be positive")
        if max_compaction_keeps <= 0:
            raise ValueError("max_compaction_keeps must be positive")
        if max_file_bytes <= 0:
            raise ValueError("max_file_bytes must be positive")
        for name, value in {
            "base_retry_seconds": base_retry_seconds,
            "max_retry_seconds": max_retry_seconds,
        }.items():
            if float(value) <= 0:
                raise ValueError(f"{name} must be positive numeric data")
        if float(max_retry_seconds) < float(base_retry_seconds):
            raise ValueError("max_retry_seconds cannot be lower than base_retry_seconds")
        self.root = tree.require_outside(workflow_root)
        self.path = self.root / "lifecycle" / "l2_scan_state.json"
        self.path_lock = path_lock
        self.max_failures = max_failures
        self.max_compaction_keeps = max_compaction_keeps
        self.max_file_bytes = max_file_bytes
        self.base_retry_seconds = float(base_retry_seconds)
        self.max_retry_seconds = float(max_retry_seconds)
        self.backoff = BackoffPolicy(base_seconds=self.base_retry_seconds, cap_seconds=self.max_retry_seconds)
        self.lock_key = memory_root_lock_prefix("memory-lifecycle-scan", tree.root)

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._read()

    def cursor(self) -> MemoryAddress | None:
        state = self._read()
        return None if state.after is None else MemoryURI.parse(state.after).to_address()

    def eligible(self, uri: MemoryURI, *, now: datetime) -> bool:
        failure = self._read().failures.get(str(uri))
        return failure is None or _timestamp(now) >= failure.retry_at

    def advance(self, address: MemoryAddress) -> None:
        uri = MemoryURI.from_address(address)
        with self.path_lock.acquire(self.lock_key, ttl_seconds=30):
            state = self._read()
            state.after = str(uri)
            self._write(state)

    def reset_cursor(self) -> None:
        with self.path_lock.acquire(self.lock_key, ttl_seconds=30):
            state = self._read()
            state.after = None
            self._write(state)

    def clear_failure(self, uri: MemoryURI) -> None:
        with self.path_lock.acquire(self.lock_key, ttl_seconds=30):
            state = self._read()
            state.failures.pop(str(uri), None)
            self._write(state)

    def compaction_kept(self, version: MemoryCompactionKeep) -> bool:
        """这一版内容（含最近被用到的时刻）是否已被压缩模型判过 KEEP。"""

        recorded = self._read().keeps.get(str(version.uri))
        return recorded is not None and recorded.same_version(version)

    def record_compaction_keep(self, version: MemoryCompactionKeep) -> None:
        with self.path_lock.acquire(self.lock_key, ttl_seconds=30):
            state = self._read()
            state.keeps[str(version.uri)] = version
            if len(state.keeps) > self.max_compaction_keeps:
                oldest = min(state.keeps.values(), key=lambda item: (item.judged_at, str(item.uri)))
                state.keeps.pop(str(oldest.uri), None)
            self._write(state)

    def record_failure(
        self,
        uri: MemoryURI,
        error: Exception,
        *,
        failed_at: datetime,
    ) -> MemoryLifecycleScanFailure:
        current_time = _timestamp(failed_at)
        with self.path_lock.acquire(self.lock_key, ttl_seconds=30):
            state = self._read()
            failures = state.failures
            previous = failures.get(str(uri))
            attempts = 1 if previous is None else previous.attempts + 1
            delay = self.backoff.delay(attempts)
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
            self._write(state)
            return failure

    def _read(self) -> _ScanState:
        try:
            encoded = read_regular_bytes(
                self.path,
                artifact_root=self.root,
                max_bytes=self.max_file_bytes,
            )
        except FileNotFoundError:
            return _ScanState(None, {}, {})
        try:
            raw = json.loads(encoded)
            if isinstance(raw, Mapping) and raw.get("schema") == _STALE_SCHEMA:
                raise MemoryLifecycleScanStateError(
                    f"这是旧格式扫描账（{_STALE_SCHEMA}），删除 {self.path} 后重启即可（只存游标与退避）"
                )
            if not isinstance(raw, Mapping) or set(raw) != {"schema", "after_uri", "failures", "compaction_keeps"}:
                raise ValueError("invalid fields")
            if raw["schema"] != _SCHEMA:
                raise ValueError("unsupported schema")
            after = _decode_after_uri(raw["after_uri"])
            values = raw["failures"]
            if not isinstance(values, list) or len(values) > self.max_failures:
                raise ValueError("invalid failures")
            failures: dict[str, MemoryLifecycleScanFailure] = {}
            for value in values:
                failure = _decode_failure(value)
                if str(failure.uri) in failures:
                    raise ValueError("duplicate failure")
                failures[str(failure.uri)] = failure
            keep_values = raw["compaction_keeps"]
            if not isinstance(keep_values, list) or len(keep_values) > self.max_compaction_keeps:
                raise ValueError("invalid compaction keeps")
            keeps: dict[str, MemoryCompactionKeep] = {}
            for value in keep_values:
                keep = _decode_compaction_keep(value)
                if str(keep.uri) in keeps:
                    raise ValueError("duplicate compaction keep")
                keeps[str(keep.uri)] = keep
            state = _ScanState(after, failures, keeps)
            if encoded != self._encode(state):
                raise ValueError("non-canonical state")
            return state
        except MemoryLifecycleScanStateError:
            raise
        except Exception as exc:
            raise MemoryLifecycleScanStateError("failed to read L2 lifecycle scan state") from exc

    def _write(self, state: _ScanState) -> None:
        encoded = self._encode(state)
        if len(encoded) > self.max_file_bytes:
            raise MemoryLifecycleScanStateError("L2 lifecycle scan state exceeds its file bound")
        atomic_replace_bytes(self.path, encoded, artifact_root=self.root)

    @staticmethod
    def _encode(state: _ScanState) -> bytes:
        return (
            canonical_json(
                {
                    "schema": _SCHEMA,
                    "after_uri": state.after,
                    "failures": [state.failures[key].to_dict() for key in sorted(state.failures)],
                    "compaction_keeps": [state.keeps[key].to_dict() for key in sorted(state.keeps)],
                }
            )
            + "\n"
        ).encode("utf-8")


@dataclass
class _ScanState:
    after: str | None
    failures: dict[str, MemoryLifecycleScanFailure]
    keeps: dict[str, MemoryCompactionKeep]


def _decode_after_uri(value: object) -> str | None:
    """扫描游标：``null`` 或一个 L2 URI；类型错的文字照抄 ``MemoryURI.__init__``。"""

    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError("memory URI must be a string")
    MemoryURI(value).to_address()
    return value


def _decode_failure(value: object) -> MemoryLifecycleScanFailure:
    """磁盘读回的一条失败退避记录；字段类型错的文字照抄 ``MemoryLifecycleScanFailure.__post_init__``。"""

    if not isinstance(value, Mapping) or set(value) != {"uri", "attempts", "retry_at", "error_type", "message"}:
        raise ValueError("invalid failure")
    attempts, error_type, message = value["attempts"], value["error_type"], value["message"]
    if not is_int(attempts):
        raise ValueError("memory lifecycle failure attempts must be positive")
    if not isinstance(error_type, str):
        raise ValueError("memory lifecycle failure error_type must be non-empty")
    if not isinstance(message, str):
        raise ValueError("memory lifecycle failure message must be non-empty")
    return MemoryLifecycleScanFailure(
        uri=decode_memory_uri(value["uri"]),
        attempts=attempts,
        retry_at=_parse_timestamp(value["retry_at"]),
        error_type=error_type,
        message=message,
    )


def _decode_compaction_keep(value: object) -> MemoryCompactionKeep:
    """磁盘读回的一条 KEEP 记录；字段类型错的文字照抄 ``MemoryCompactionKeep.__post_init__``。"""

    if not isinstance(value, Mapping) or set(value) != {
        "uri",
        "revision",
        "source_digest",
        "last_used_at",
        "judged_at",
    }:
        raise ValueError("invalid compaction keep")
    revision, source_digest = value["revision"], value["source_digest"]
    if not is_int(revision):
        raise ValueError("memory compaction keep revision must be positive")
    if not isinstance(source_digest, str):
        raise ValueError("memory compaction keep source_digest must be non-empty")
    return MemoryCompactionKeep(
        uri=decode_memory_uri(value["uri"]),
        revision=revision,
        source_digest=source_digest,
        last_used_at=None if value["last_used_at"] is None else _parse_timestamp(value["last_used_at"]),
        judged_at=_parse_timestamp(value["judged_at"]),
    )


def _timestamp(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("memory lifecycle scan timestamp must be timezone-aware")
    return value.astimezone(UTC)


def _format_timestamp(value: datetime) -> str:
    return _timestamp(value).isoformat().replace("+00:00", "Z")


def _parse_timestamp(value: object) -> datetime:
    if not isinstance(value, str) or not value:
        raise ValueError("memory lifecycle scan timestamp is invalid")
    return _timestamp(datetime.fromisoformat(value.replace("Z", "+00:00")))


__all__ = [
    "MemoryCompactionKeep",
    "MemoryLifecycleScanFailure",
    "MemoryLifecycleScanStateError",
    "MemoryLifecycleScanStore",
]
