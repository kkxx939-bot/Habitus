"""Prediction Pattern generation 的完整性清单与原子激活存储。"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, NoReturn

from foundation.integrity import canonical_digest, canonical_json, canonicalize
from infrastructure.store.contracts.path_lock import PathLock
from infrastructure.store.filesystem import (
    DurablePathIntegrityError,
    ImmutableArtifactConflictError,
    atomic_create_bytes,
    atomic_replace_bytes,
    ensure_real_directory,
    read_regular_bytes,
    regular_file_exists,
)
from infrastructure.store.sqlite.lock_store import SQLiteLockStore
from prediction.model import PredictionPatternKind, prediction_sample_id, prediction_shard
from prediction.uri import PredictionURI

_GENERATION_SCHEMA = "prediction-pattern-generation-v1"
_ACTIVE_SCHEMA = "prediction-pattern-active-v2"
_ACTIVE_DIRECTORY = "active"
_GENERATION_DIRECTORY = "generations"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class PredictionPatternGenerationState(str, Enum):
    PREPARED = "prepared"
    READY = "ready"


@dataclass(frozen=True, order=True)
class PredictionPatternGenerationEntry:
    """一个 generation 必须完整包含的 Pattern 文档。"""

    uri: str
    digest: str

    def __post_init__(self) -> None:
        parsed = PredictionURI.parse(self.uri)
        if not parsed.is_pattern_document or str(parsed) != self.uri:
            raise ValueError("pattern generation entry must use a canonical Pattern URI")
        if not isinstance(self.digest, str) or _SHA256.fullmatch(self.digest) is None:
            raise ValueError("pattern generation entry digest must be lowercase SHA-256")

    def to_dict(self) -> dict[str, str]:
        return {"uri": self.uri, "digest": self.digest}


@dataclass(frozen=True)
class PredictionPatternGenerationManifest:
    """一个逻辑 State 及其全部 Branch 的预期文档集合；READY 后内容不可改变。

    一致性单元是单个 State：条件概率只在同一 State 的分支之间归一，
    ``statistics`` 的四个字段也都只依赖该 State 自己的样本。
    因此一代只描述一个 State，增量更新的成本随受影响的 State 数而不是总量增长。
    """

    generation_id: str
    projection_version: str
    created_at: datetime
    logical_state_key: str
    entries: tuple[PredictionPatternGenerationEntry, ...]
    learning_identity: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "generation_id", prediction_sample_id(self.generation_id))
        if not isinstance(self.projection_version, str) or not self.projection_version.strip():
            raise ValueError("pattern generation projection_version must be non-empty text")
        object.__setattr__(self, "projection_version", self.projection_version.strip())
        object.__setattr__(self, "created_at", _datetime(self.created_at, "pattern generation created_at"))
        object.__setattr__(self, "logical_state_key", prediction_sample_id(self.logical_state_key))
        entries = tuple(self.entries)
        if not entries or any(not isinstance(item, PredictionPatternGenerationEntry) for item in entries):
            raise ValueError("pattern generation manifest requires entries")
        if entries != tuple(sorted(entries)) or len(entries) != len({item.uri for item in entries}):
            raise ValueError("pattern generation entries must be unique and sorted")
        addresses = [PredictionURI.parse(item.uri).to_pattern_address() for item in entries]
        states = [address for address in addresses if address.kind is PredictionPatternKind.STATE]
        if len(states) != 1:
            raise ValueError("one pattern generation must describe exactly one StatePattern")
        state_key = states[0].state_key
        if any(
            address.state_key != state_key
            for address in addresses
            if address.kind is PredictionPatternKind.BRANCH
        ):
            raise ValueError("pattern generation Branch must belong to its own StatePattern")
        object.__setattr__(self, "entries", entries)
        if self.learning_identity is not None:
            if not isinstance(self.learning_identity, Mapping) or not self.learning_identity:
                raise ValueError("pattern generation learning_identity must be a non-empty mapping")
            normalized_identity = canonicalize(dict(self.learning_identity))
            assert isinstance(normalized_identity, dict)
            object.__setattr__(self, "learning_identity", normalized_identity)

    @property
    def state_key(self) -> str:
        """本代物化出的物理 state_key；跨代会变，逻辑身份由 logical_state_key 保持。"""

        return PredictionURI.parse(self.state_entries[0].uri).to_pattern_address().state_key

    @property
    def manifest_digest(self) -> str:
        return canonical_digest(self.to_dict())

    @property
    def state_entries(self) -> tuple[PredictionPatternGenerationEntry, ...]:
        return tuple(
            item
            for item in self.entries
            if PredictionURI.parse(item.uri).to_pattern_address().kind is PredictionPatternKind.STATE
        )

    @property
    def branch_entries(self) -> tuple[PredictionPatternGenerationEntry, ...]:
        return tuple(
            item
            for item in self.entries
            if PredictionURI.parse(item.uri).to_pattern_address().kind is PredictionPatternKind.BRANCH
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "generation_id": self.generation_id,
            "projection_version": self.projection_version,
            "created_at": self.created_at,
            "logical_state_key": self.logical_state_key,
            "entries": [item.to_dict() for item in self.entries],
            "learning_identity": self.learning_identity,
        }

    @classmethod
    def from_dict(cls, value: object) -> PredictionPatternGenerationManifest:
        payload = _exact_mapping(
            value,
            {
                "generation_id",
                "projection_version",
                "created_at",
                "logical_state_key",
                "entries",
                "learning_identity",
            },
            "pattern generation manifest",
        )
        entries = tuple(
            PredictionPatternGenerationEntry(
                **_exact_mapping(item, {"uri", "digest"}, "pattern generation entry")
            )
            for item in _sequence(payload["entries"], "pattern generation entries")
        )
        return cls(
            generation_id=payload["generation_id"],
            projection_version=payload["projection_version"],
            created_at=_datetime(payload["created_at"], "pattern generation created_at"),
            logical_state_key=payload["logical_state_key"],
            entries=entries,
            learning_identity=payload["learning_identity"],
        )


@dataclass(frozen=True)
class PredictionPatternGenerationRecord:
    manifest: PredictionPatternGenerationManifest
    state: PredictionPatternGenerationState

    def __post_init__(self) -> None:
        if not isinstance(self.manifest, PredictionPatternGenerationManifest):
            raise TypeError("manifest must be PredictionPatternGenerationManifest")
        object.__setattr__(self, "state", PredictionPatternGenerationState(self.state))


class PredictionPatternGenerationStoreError(RuntimeError):
    """generation 清单或 active 指针损坏、冲突或不可安全访问。"""


class PredictionPatternGenerationStore:
    """PREPARED/READY generation 与当前 active 指针的耐久存储。"""

    def __init__(self, root: str | Path, *, max_record_bytes: int = 32_000_000) -> None:
        requested = Path(root).expanduser().absolute()
        if requested.is_symlink():
            raise PredictionPatternGenerationStoreError("pattern generation root cannot be a symbolic link")
        if isinstance(max_record_bytes, bool) or not isinstance(max_record_bytes, int) or max_record_bytes <= 0:
            raise ValueError("max_record_bytes must be a positive integer")
        self.root = requested.resolve(strict=False)
        self.max_record_bytes = max_record_bytes
        lock_path = self.root.parent / f".{self.root.name}.locks.sqlite3"
        self._activation_lock = PathLock(SQLiteLockStore(lock_path, initialize=False))
        self._activation_lock_key = f"prediction-pattern-active:{canonical_digest({'root': str(self.root)})}"

    def prepare(
        self,
        manifest: PredictionPatternGenerationManifest,
    ) -> PredictionPatternGenerationRecord:
        record = PredictionPatternGenerationRecord(manifest, PredictionPatternGenerationState.PREPARED)
        encoded = self._encode_record(record)
        if len(encoded) > self.max_record_bytes:
            raise PredictionPatternGenerationStoreError(
                "pattern generation exceeds its configured size limit"
            )
        self._initialize()
        path = self._generation_path(manifest.logical_state_key, manifest.generation_id)
        try:
            ensure_real_directory(path.parent, artifact_root=self.root)
            atomic_create_bytes(path, encoded, artifact_root=self.root)
        except ImmutableArtifactConflictError as exc:
            current = self.read(manifest.logical_state_key, manifest.generation_id)
            if (
                current.manifest.generation_id != manifest.generation_id
                or current.manifest.projection_version != manifest.projection_version
                or current.manifest.entries != manifest.entries
                or current.manifest.learning_identity != manifest.learning_identity
            ):
                raise PredictionPatternGenerationStoreError(
                    "pattern generation identity conflicts with another manifest"
                ) from exc
            return current
        return self.read(manifest.logical_state_key, manifest.generation_id)

    def _mark_ready(self, logical_state_key: str, generation_id: str) -> PredictionPatternGenerationRecord:
        current = self.read(logical_state_key, generation_id)
        if current.state is PredictionPatternGenerationState.READY:
            return current
        ready = PredictionPatternGenerationRecord(
            current.manifest,
            PredictionPatternGenerationState.READY,
        )
        try:
            atomic_replace_bytes(
                self._generation_path(logical_state_key, generation_id),
                self._encode_record(ready),
                artifact_root=self.root,
            )
        except DurablePathIntegrityError as exc:
            raise PredictionPatternGenerationStoreError(
                "pattern generation cannot be marked READY safely"
            ) from exc
        return self.read(logical_state_key, generation_id)

    def _activate(self, logical_state_key: str, generation_id: str) -> PredictionPatternGenerationRecord:
        """只替换该逻辑 State 的激活指针；其余 State 完全不受影响。"""

        record = self.read(logical_state_key, generation_id)
        if record.state is not PredictionPatternGenerationState.READY:
            raise PredictionPatternGenerationStoreError("only a READY pattern generation can be activated")
        shard_path = self._shard_path(logical_state_key)
        with self._activation_lock.acquire(
            self._activation_lock_key,
            ttl_seconds=30,
            wait_timeout_seconds=5.0,
        ) as guard:
            with guard.fenced():
                states = self._read_shard(shard_path)
                current = states.get(logical_state_key)
                if current is not None:
                    current_record = self.read(logical_state_key, current["generation_id"])
                    target_order = (record.manifest.created_at, record.manifest.generation_id)
                    current_order = (
                        current_record.manifest.created_at,
                        current_record.manifest.generation_id,
                    )
                    if target_order <= current_order:
                        return current_record
                states[logical_state_key] = {
                    "generation_id": record.manifest.generation_id,
                    "state_key": record.manifest.state_key,
                    "manifest_digest": record.manifest.manifest_digest,
                }
                payload = {
                    "schema": _ACTIVE_SCHEMA,
                    "states": {key: states[key] for key in sorted(states)},
                }
                try:
                    self._initialize()
                    ensure_real_directory(shard_path.parent, artifact_root=self.root)
                    atomic_replace_bytes(
                        shard_path,
                        (canonical_json(payload) + "\n").encode("utf-8"),
                        artifact_root=self.root,
                    )
                except DurablePathIntegrityError as exc:
                    raise PredictionPatternGenerationStoreError(
                        "active pattern generation cannot be replaced safely"
                    ) from exc
        return self.read_active_state(logical_state_key)

    def read(self, logical_state_key: str, generation_id: str) -> PredictionPatternGenerationRecord:
        normalized = prediction_sample_id(generation_id)
        value, raw = self._read_json(
            self._generation_path(logical_state_key, normalized),
            "pattern generation",
        )
        payload = _exact_mapping(value, {"schema", "state", "manifest"}, "pattern generation record")
        if payload["schema"] != _GENERATION_SCHEMA:
            raise PredictionPatternGenerationStoreError("pattern generation schema is unsupported")
        try:
            record = PredictionPatternGenerationRecord(
                PredictionPatternGenerationManifest.from_dict(payload["manifest"]),
                PredictionPatternGenerationState(payload["state"]),
            )
        except (TypeError, ValueError) as exc:
            raise PredictionPatternGenerationStoreError("pattern generation record is invalid") from exc
        if record.manifest.generation_id != normalized:
            raise PredictionPatternGenerationStoreError("pattern generation does not match its physical address")
        if self._encode_record(record) != raw:
            raise PredictionPatternGenerationStoreError("pattern generation is not canonically serialized")
        return record

    def read_active_state(self, logical_state_key: str) -> PredictionPatternGenerationRecord:
        """解析一个逻辑 State 当前激活的那一代；只读它所在的分片。"""

        normalized = prediction_sample_id(logical_state_key)
        states = self._read_shard(self._shard_path(normalized))
        entry = states.get(normalized)
        if entry is None:
            raise PredictionPatternGenerationStoreError("logical StatePattern has no active generation")
        record = self.read(normalized, entry["generation_id"])
        if record.state is not PredictionPatternGenerationState.READY:
            raise PredictionPatternGenerationStoreError("active pattern generation is not READY")
        if entry["manifest_digest"] != record.manifest.manifest_digest:
            raise PredictionPatternGenerationStoreError("active pattern manifest digest does not match")
        if record.manifest.logical_state_key != normalized or entry["state_key"] != record.manifest.state_key:
            raise PredictionPatternGenerationStoreError("active pattern entry does not match its manifest")
        return record

    def active_logical_state_keys(self) -> tuple[str, ...]:
        """有界枚举全部激活的逻辑 State；只需读 256 个分片文件。"""

        keys: list[str] = []
        active_root = self.root / _ACTIVE_DIRECTORY
        for shard in _shard_names():
            path = active_root / f"{shard}.json"
            try:
                exists = regular_file_exists(path, artifact_root=self.root)
            except DurablePathIntegrityError as exc:
                raise PredictionPatternGenerationStoreError(
                    "active pattern shard cannot be inspected safely"
                ) from exc
            if exists:
                keys.extend(self._read_shard(path))
        return tuple(sorted(keys))

    def _shard_path(self, logical_state_key: str) -> Path:
        shard = prediction_shard(prediction_sample_id(logical_state_key))
        return self.root / _ACTIVE_DIRECTORY / f"{shard}.json"

    def _read_shard(self, path: Path) -> dict[str, dict[str, Any]]:
        try:
            if not regular_file_exists(path, artifact_root=self.root):
                return {}
        except DurablePathIntegrityError as exc:
            raise PredictionPatternGenerationStoreError(
                "active pattern shard cannot be inspected safely"
            ) from exc
        value, raw = self._read_json(path, "active pattern shard")
        payload = _exact_mapping(value, {"schema", "states"}, "active pattern shard")
        if payload["schema"] != _ACTIVE_SCHEMA:
            raise PredictionPatternGenerationStoreError("active pattern shard schema is unsupported")
        if (canonical_json(value) + "\n").encode("utf-8") != raw:
            raise PredictionPatternGenerationStoreError("active pattern shard is not canonically serialized")
        states = _mapping(payload["states"], "active pattern states")
        return {
            prediction_sample_id(key): _exact_mapping(
                item,
                {"generation_id", "state_key", "manifest_digest"},
                "active pattern state entry",
            )
            for key, item in states.items()
        }

    def _initialize(self) -> None:
        try:
            ensure_real_directory(self.root, artifact_root=self.root)
        except DurablePathIntegrityError as exc:
            raise PredictionPatternGenerationStoreError(
                "pattern generation store cannot be initialized safely"
            ) from exc

    def _generation_path(self, logical_state_key: str, generation_id: str) -> Path:
        """generation 记录按逻辑 State 分开存放。

        ``pattern_generation`` 表示第几轮建图，多个 State 会共用同一个轮次 ID，
        因此记录路径必须再带上逻辑 State 才能唯一。
        """

        normalized = prediction_sample_id(logical_state_key)
        return (
            self.root
            / _GENERATION_DIRECTORY
            / prediction_shard(normalized)
            / normalized
            / f"{prediction_sample_id(generation_id)}.json"
        )

    def _read_json(self, path: Path, label: str) -> tuple[Any, bytes]:
        try:
            raw = read_regular_bytes(path, artifact_root=self.root, max_bytes=self.max_record_bytes)
            value = json.loads(
                raw.decode("utf-8"),
                object_pairs_hook=self._unique_object,
                parse_constant=self._reject_constant,
            )
            return value, raw
        except (OSError, DurablePathIntegrityError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PredictionPatternGenerationStoreError(f"{label} cannot be read safely") from exc

    @staticmethod
    def _encode_record(record: PredictionPatternGenerationRecord) -> bytes:
        return (
            canonical_json(
                {
                    "schema": _GENERATION_SCHEMA,
                    "state": record.state.value,
                    "manifest": record.manifest.to_dict(),
                }
            )
            + "\n"
        ).encode("utf-8")

    @staticmethod
    def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise PredictionPatternGenerationStoreError(
                    "pattern generation JSON contains duplicate keys"
                )
            result[key] = value
        return result

    @staticmethod
    def _reject_constant(value: str) -> NoReturn:
        raise PredictionPatternGenerationStoreError(
            f"pattern generation JSON contains invalid constant: {value}"
        )


def _shard_names() -> tuple[str, ...]:
    return tuple(f"{value:02x}" for value in range(256))


def default_generation_store_root(prediction_root: Path) -> Path:
    return prediction_root.parent / f".{prediction_root.name}.prediction-pattern-generations"


def _datetime(value: object, label: str) -> datetime:
    parsed = value
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(f"{label} must be an ISO timestamp") from exc
    if not isinstance(parsed, datetime) or parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{label} must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{label} must be a mapping with string keys")
    return dict(value)


def _exact_mapping(value: object, expected: set[str], label: str) -> dict[str, Any]:
    payload = _mapping(value, label)
    if set(payload) != expected:
        raise ValueError(f"{label} must contain exactly {sorted(expected)}")
    return payload


def _sequence(value: object, label: str) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{label} must be a sequence")
    return tuple(value)


__all__ = [
    "PredictionPatternGenerationEntry",
    "PredictionPatternGenerationManifest",
    "PredictionPatternGenerationRecord",
    "PredictionPatternGenerationState",
    "PredictionPatternGenerationStore",
    "PredictionPatternGenerationStoreError",
    "default_generation_store_root",
]
