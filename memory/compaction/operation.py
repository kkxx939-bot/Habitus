"""为 L2 生命周期保存可恢复、可校验的耐久操作状态。"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from collections.abc import Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from foundation.integrity import canonical_json
from infrastructure.store.contracts import LeaseGuard, PathLock
from infrastructure.store.filesystem import atomic_replace_bytes, durable_unlink, read_regular_bytes
from memory.snapshot import MemorySnapshot
from memory.tree import MemoryTree
from memory.uri import MemoryURI


class MemoryLifecycleOperationError(RuntimeError):
    """生命周期耐久操作损坏、冲突或超过资源边界。"""


class MemoryLifecycleOperationKind(str, Enum):
    COMPACT = "compact"
    RESTORE = "restore"
    RETIRE = "retire"


class MemoryLifecycleOperationPhase(str, Enum):
    PREPARED = "prepared"
    L2_COMMITTED = "l2_committed"
    STATE_COMMITTED = "state_committed"
    DERIVED_PENDING = "derived_pending"


@dataclass(frozen=True)
class MemoryLifecycleOperation:
    operation_id: str
    kind: MemoryLifecycleOperationKind
    uri: MemoryURI
    source_revision: int
    source_digest: str
    source_created_at: datetime
    expected_state_version: int
    planned_fields: Mapping[str, Any] | None
    phase: MemoryLifecycleOperationPhase
    target_revision: int | None
    target_digest: str | None
    prepared_at: datetime
    updated_at: datetime

    def __post_init__(self) -> None:
        if not isinstance(self.operation_id, str) or len(self.operation_id) != 64:
            raise ValueError("memory lifecycle operation_id must be SHA-256")
        if any(character not in "0123456789abcdef" for character in self.operation_id):
            raise ValueError("memory lifecycle operation_id must be lowercase SHA-256")
        object.__setattr__(self, "kind", MemoryLifecycleOperationKind(self.kind))
        uri = MemoryURI.parse(self.uri)
        uri.to_address()
        object.__setattr__(self, "uri", uri)
        if (
            isinstance(self.source_revision, bool)
            or not isinstance(self.source_revision, int)
            or self.source_revision <= 0
        ):
            raise ValueError("memory lifecycle source_revision must be positive")
        _require_digest(self.source_digest, "source_digest")
        object.__setattr__(self, "source_created_at", _timestamp(self.source_created_at))
        if (
            isinstance(self.expected_state_version, bool)
            or not isinstance(self.expected_state_version, int)
            or self.expected_state_version < 0
        ):
            raise ValueError("memory lifecycle expected_state_version must be non-negative")
        if self.planned_fields is not None:
            if not isinstance(self.planned_fields, Mapping):
                raise TypeError("memory lifecycle planned_fields must be a mapping or None")
            object.__setattr__(self, "planned_fields", dict(self.planned_fields))
        object.__setattr__(self, "phase", MemoryLifecycleOperationPhase(self.phase))
        if (self.target_revision is None) != (self.target_digest is None):
            raise ValueError("memory lifecycle target revision and digest must appear together")
        if self.target_revision is not None:
            if (
                isinstance(self.target_revision, bool)
                or not isinstance(self.target_revision, int)
                or self.target_revision <= self.source_revision
            ):
                raise ValueError("memory lifecycle target_revision must advance source_revision")
            assert self.target_digest is not None
            _require_digest(self.target_digest, "target_digest")
        if (
            self.kind is not MemoryLifecycleOperationKind.RETIRE
            and self.phase is not MemoryLifecycleOperationPhase.PREPARED
            and self.target_revision is None
        ):
            raise ValueError("advanced memory lifecycle operation requires a target fingerprint")
        object.__setattr__(self, "prepared_at", _timestamp(self.prepared_at))
        object.__setattr__(self, "updated_at", _timestamp(self.updated_at))
        if self.updated_at < self.prepared_at:
            raise ValueError("memory lifecycle operation updated_at precedes prepared_at")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": "memory_lifecycle_operation_v1",
            "operation_id": self.operation_id,
            "kind": self.kind.value,
            "uri": str(self.uri),
            "source_revision": self.source_revision,
            "source_digest": self.source_digest,
            "source_created_at": _format_timestamp(self.source_created_at),
            "expected_state_version": self.expected_state_version,
            "planned_fields": self.planned_fields,
            "phase": self.phase.value,
            "target_revision": self.target_revision,
            "target_digest": self.target_digest,
            "prepared_at": _format_timestamp(self.prepared_at),
            "updated_at": _format_timestamp(self.updated_at),
        }

    @classmethod
    def from_dict(cls, value: object) -> MemoryLifecycleOperation:
        expected = {
            "schema",
            "operation_id",
            "kind",
            "uri",
            "source_revision",
            "source_digest",
            "source_created_at",
            "expected_state_version",
            "planned_fields",
            "phase",
            "target_revision",
            "target_digest",
            "prepared_at",
            "updated_at",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ValueError("memory lifecycle operation fields are invalid")
        if value["schema"] != "memory_lifecycle_operation_v1":
            raise ValueError("memory lifecycle operation schema is unsupported")
        return cls(
            operation_id=value["operation_id"],
            kind=MemoryLifecycleOperationKind(value["kind"]),
            uri=MemoryURI.parse(value["uri"]),
            source_revision=value["source_revision"],
            source_digest=value["source_digest"],
            source_created_at=_parse_timestamp(value["source_created_at"]),
            expected_state_version=value["expected_state_version"],
            planned_fields=value["planned_fields"],
            phase=MemoryLifecycleOperationPhase(value["phase"]),
            target_revision=value["target_revision"],
            target_digest=value["target_digest"],
            prepared_at=_parse_timestamp(value["prepared_at"]),
            updated_at=_parse_timestamp(value["updated_at"]),
        )


class MemoryLifecycleOperationStore:
    """每个 L2 URI 最多保存一个未完成操作，并用独立短锁保护状态迁移。"""

    def __init__(
        self,
        tree: MemoryTree,
        path_lock: PathLock,
        *,
        root: str | Path | None = None,
        max_operations: int = 100_000,
        max_file_bytes: int = 512 * 1024,
    ) -> None:
        if not isinstance(tree, MemoryTree):
            raise TypeError("tree must be MemoryTree")
        if not isinstance(path_lock, PathLock):
            raise TypeError("path_lock must be PathLock")
        for name, value in {
            "max_operations": max_operations,
            "max_file_bytes": max_file_bytes,
        }.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        canonical = (tree.root.parent / "workflow" / "lifecycle" / "l2_operations").resolve(
            strict=False
        )
        resolved = canonical if root is None else Path(root).expanduser().resolve(strict=False)
        if resolved != canonical:
            raise ValueError("memory lifecycle operations must use the canonical workflow root")
        self.tree = tree
        self.path_lock = path_lock
        self.root = resolved
        self.max_operations = max_operations
        self.max_file_bytes = max_file_bytes
        root_digest = hashlib.sha256(str(tree.root).encode("utf-8")).hexdigest()[:24]
        self._lock_prefix = f"memory-lifecycle-operation:{root_digest}"

    def initialize(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.root, 0o700)
        self.pending()

    def prepare(
        self,
        kind: MemoryLifecycleOperationKind,
        snapshot: MemorySnapshot,
        *,
        planned_fields: Mapping[str, Any] | None,
        expected_state_version: int,
        prepared_at: datetime,
    ) -> MemoryLifecycleOperation:
        if not snapshot.exists or snapshot.value is None:
            raise ValueError("memory lifecycle operation requires an existing source snapshot")
        uri = MemoryURI.parse(snapshot.identity)
        uri.to_address()
        if snapshot.revision is None or snapshot.source_digest is None:
            raise ValueError("memory lifecycle source snapshot lacks a fingerprint")
        timestamp = _timestamp(prepared_at)
        operation_id = _operation_id(kind, uri, snapshot.revision, snapshot.source_digest)
        operation = MemoryLifecycleOperation(
            operation_id=operation_id,
            kind=kind,
            uri=uri,
            source_revision=snapshot.revision,
            source_digest=snapshot.source_digest,
            source_created_at=snapshot.value.metadata.created_at,
            expected_state_version=expected_state_version,
            planned_fields=planned_fields,
            phase=MemoryLifecycleOperationPhase.PREPARED,
            target_revision=None,
            target_digest=None,
            prepared_at=timestamp,
            updated_at=timestamp,
        )
        with self._guard(uri):
            current = self.try_read(uri)
            if current is not None:
                if current == operation:
                    return current
                if (
                    current.operation_id == operation.operation_id
                    and current.kind is operation.kind
                    and current.planned_fields == operation.planned_fields
                ):
                    return current
                raise MemoryLifecycleOperationError(
                    "another unfinished lifecycle operation already owns this memory"
                )
            self._write(operation)
            return self.read(uri)

    def advance(
        self,
        operation: MemoryLifecycleOperation,
        phase: MemoryLifecycleOperationPhase,
        *,
        updated_at: datetime,
        target_revision: int | None = None,
        target_digest: str | None = None,
    ) -> MemoryLifecycleOperation:
        if not isinstance(operation, MemoryLifecycleOperation):
            raise TypeError("operation must be MemoryLifecycleOperation")
        selected = MemoryLifecycleOperationPhase(phase)
        order = tuple(MemoryLifecycleOperationPhase)
        if order.index(selected) < order.index(operation.phase):
            raise ValueError("memory lifecycle operation phase cannot move backwards")
        with self._guard(operation.uri):
            return self.advance_owned(
                operation,
                selected,
                updated_at=updated_at,
                target_revision=target_revision,
                target_digest=target_digest,
            )

    def advance_owned(
        self,
        operation: MemoryLifecycleOperation,
        phase: MemoryLifecycleOperationPhase,
        *,
        updated_at: datetime,
        target_revision: int | None = None,
        target_digest: str | None = None,
    ) -> MemoryLifecycleOperation:
        """由已持有同一 URI operation lock 的协调路径推进操作。"""

        if not isinstance(operation, MemoryLifecycleOperation):
            raise TypeError("operation must be MemoryLifecycleOperation")
        selected = MemoryLifecycleOperationPhase(phase)
        order = tuple(MemoryLifecycleOperationPhase)
        if order.index(selected) < order.index(operation.phase):
            raise ValueError("memory lifecycle operation phase cannot move backwards")
        current = self.read(operation.uri)
        if current.operation_id != operation.operation_id:
            raise MemoryLifecycleOperationError("memory lifecycle operation ownership changed")
        revision = current.target_revision if target_revision is None else target_revision
        digest = current.target_digest if target_digest is None else target_digest
        advanced = replace(
            current,
            phase=selected,
            target_revision=revision,
            target_digest=digest,
            updated_at=_timestamp(updated_at),
        )
        self._write(advanced)
        return self.read(operation.uri)

    def read(self, uri: MemoryURI | str) -> MemoryLifecycleOperation:
        parsed = MemoryURI.parse(uri)
        parsed.to_address()
        operation = self._read_path(self._path(parsed))
        if operation.uri != parsed:
            raise MemoryLifecycleOperationError("memory lifecycle operation identity changed")
        return operation

    def acquire(
        self,
        uri: MemoryURI | str,
        *,
        wait_timeout_seconds: float = 0.0,
    ) -> AbstractContextManager[LeaseGuard]:
        """串行化同一 URI 的生命周期占位检查与普通 use 状态提交。"""

        parsed = MemoryURI.parse(uri)
        parsed.to_address()
        return self._guard(parsed, wait_timeout_seconds=wait_timeout_seconds)

    def _read_path(self, path: Path) -> MemoryLifecycleOperation:
        try:
            encoded = read_regular_bytes(
                path,
                artifact_root=self.root,
                max_bytes=self.max_file_bytes,
            )
            raw = json.loads(encoded)
            operation = MemoryLifecycleOperation.from_dict(raw)
        except Exception as exc:
            if isinstance(exc, MemoryLifecycleOperationError):
                raise
            raise MemoryLifecycleOperationError("failed to read memory lifecycle operation") from exc
        if encoded != self._encode(operation) or self._path(operation.uri) != path:
            raise MemoryLifecycleOperationError("memory lifecycle operation failed canonical validation")
        return operation

    def try_read(self, uri: MemoryURI | str) -> MemoryLifecycleOperation | None:
        try:
            return self.read(uri)
        except MemoryLifecycleOperationError as exc:
            if isinstance(exc.__cause__, FileNotFoundError):
                return None
            raise

    def pending(self) -> tuple[MemoryLifecycleOperation, ...]:
        try:
            metadata = self.root.stat(follow_symlinks=False)
        except FileNotFoundError:
            return ()
        if not stat.S_ISDIR(metadata.st_mode) or self.root.is_symlink():
            raise MemoryLifecycleOperationError("memory lifecycle operation root is unsafe")
        paths: list[Path] = []
        with os.scandir(self.root) as entries:
            for entry in entries:
                if entry.is_symlink() or not entry.is_file(follow_symlinks=False):
                    raise MemoryLifecycleOperationError("memory lifecycle operation root has an unknown entry")
                if not entry.name.endswith(".json") or len(entry.name) != 69:
                    raise MemoryLifecycleOperationError("memory lifecycle operation filename is invalid")
                paths.append(Path(entry.path))
                if len(paths) > self.max_operations:
                    raise MemoryLifecycleOperationError("memory lifecycle operations exceed their bound")
        operations = tuple(self._read_path(path) for path in sorted(paths))
        if len({item.uri for item in operations}) != len(operations):
            raise MemoryLifecycleOperationError("memory lifecycle operation identities are duplicated")
        return operations

    def complete(self, operation: MemoryLifecycleOperation) -> bool:
        if not isinstance(operation, MemoryLifecycleOperation):
            raise TypeError("operation must be MemoryLifecycleOperation")
        with self._guard(operation.uri):
            return self.complete_owned(operation)

    def complete_owned(self, operation: MemoryLifecycleOperation) -> bool:
        """由已持有同一 URI operation lock 的协调路径完成操作。"""

        if not isinstance(operation, MemoryLifecycleOperation):
            raise TypeError("operation must be MemoryLifecycleOperation")
        current = self.read(operation.uri)
        if current.operation_id != operation.operation_id:
            raise MemoryLifecycleOperationError("memory lifecycle operation ownership changed")
        return durable_unlink(self._path(operation.uri), artifact_root=self.root)

    def _write(self, operation: MemoryLifecycleOperation) -> None:
        encoded = self._encode(operation)
        if len(encoded) > self.max_file_bytes:
            raise MemoryLifecycleOperationError("memory lifecycle operation exceeds its file bound")
        atomic_replace_bytes(self._path(operation.uri), encoded, artifact_root=self.root)

    def _path(self, uri: MemoryURI) -> Path:
        identity = hashlib.sha256(str(uri).encode("utf-8")).hexdigest()
        return self.root / f"{identity}.json"

    def _guard(self, uri: MemoryURI, *, wait_timeout_seconds: float = 0.0):
        key = f"{self._lock_prefix}:{hashlib.sha256(str(uri).encode('utf-8')).hexdigest()}"
        return self.path_lock.acquire(
            key,
            ttl_seconds=30,
            wait_timeout_seconds=wait_timeout_seconds,
        )

    @staticmethod
    def _encode(operation: MemoryLifecycleOperation) -> bytes:
        return (canonical_json(operation.to_dict()) + "\n").encode("utf-8")


def _operation_id(
    kind: MemoryLifecycleOperationKind,
    uri: MemoryURI,
    revision: int,
    digest: str,
) -> str:
    value = f"{kind.value}\n{uri}\n{revision}\n{digest}".encode()
    return hashlib.sha256(value).hexdigest()


def _require_digest(value: object, label: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"memory lifecycle {label} must be lowercase SHA-256")


def _timestamp(value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("memory lifecycle operation timestamp must be timezone-aware")
    return value.astimezone(UTC)


def _format_timestamp(value: datetime) -> str:
    return _timestamp(value).isoformat().replace("+00:00", "Z")


def _parse_timestamp(value: object) -> datetime:
    if not isinstance(value, str) or not value:
        raise ValueError("memory lifecycle operation timestamp is invalid")
    return _timestamp(datetime.fromisoformat(value.replace("Z", "+00:00")))


__all__ = [
    "MemoryLifecycleOperation",
    "MemoryLifecycleOperationError",
    "MemoryLifecycleOperationKind",
    "MemoryLifecycleOperationPhase",
    "MemoryLifecycleOperationStore",
]
