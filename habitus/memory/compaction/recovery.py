"""保存当前压缩代唯一有效的 L2 详细内容基线。"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

from habitus.foundation.integrity import canonical_json, text_digest
from habitus.infrastructure.store.filesystem import atomic_replace_bytes, durable_unlink, read_regular_bytes
from habitus.memory.document import MemoryDocument, MemoryDocumentCodec
from habitus.memory.snapshot import MemorySnapshot
from habitus.memory.tree import MemoryTree
from habitus.memory.uri import MemoryURI


class MemoryRecoveryError(RuntimeError):
    """活动恢复基线无法被完整验证或安全保存。"""


@dataclass(frozen=True)
class MemoryRecoveryRecord:
    uri: MemoryURI
    source_revision: int
    source_created_at: datetime
    source_digest: str
    encoded_document: str
    saved_at: datetime
    compacted_revision: int | None = None
    compacted_digest: str | None = None

    def __post_init__(self) -> None:
        parsed = MemoryURI.parse(self.uri)
        parsed.to_address()
        object.__setattr__(self, "uri", parsed)
        if (
            isinstance(self.source_revision, bool)
            or not isinstance(self.source_revision, int)
            or self.source_revision <= 0
        ):
            raise ValueError("memory recovery source_revision must be positive")
        object.__setattr__(self, "source_created_at", _timestamp(self.source_created_at))
        object.__setattr__(self, "saved_at", _timestamp(self.saved_at))
        if not _sha256(self.source_digest):
            raise ValueError("memory recovery source_digest must be SHA-256")
        if not isinstance(self.encoded_document, str) or not self.encoded_document:
            raise ValueError("memory recovery encoded_document must be non-empty")
        if text_digest(self.encoded_document) != self.source_digest:
            raise ValueError("memory recovery document digest does not match")
        if (self.compacted_revision is None) != (self.compacted_digest is None):
            raise ValueError("memory recovery compacted revision and digest must appear together")
        if self.compacted_revision is not None:
            if (
                isinstance(self.compacted_revision, bool)
                or not isinstance(self.compacted_revision, int)
                or self.compacted_revision <= self.source_revision
            ):
                raise ValueError("memory recovery compacted_revision must advance source_revision")
            if not _sha256(self.compacted_digest):
                raise ValueError("memory recovery compacted_digest must be SHA-256")

    @property
    def active(self) -> bool:
        return self.compacted_revision is not None

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": "memory_recovery_active_v2",
            "uri": str(self.uri),
            "source_revision": self.source_revision,
            "source_created_at": _format_timestamp(self.source_created_at),
            "source_digest": self.source_digest,
            "encoded_document": self.encoded_document,
            "saved_at": _format_timestamp(self.saved_at),
            "compacted_revision": self.compacted_revision,
            "compacted_digest": self.compacted_digest,
        }

    @classmethod
    def from_dict(cls, value: object) -> MemoryRecoveryRecord:
        expected = {
            "schema",
            "uri",
            "source_revision",
            "source_created_at",
            "source_digest",
            "encoded_document",
            "saved_at",
            "compacted_revision",
            "compacted_digest",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ValueError("memory recovery record fields are invalid")
        if value["schema"] != "memory_recovery_active_v2":
            raise ValueError("memory recovery record schema is unsupported")
        return cls(
            uri=MemoryURI.parse(value["uri"]),
            source_revision=value["source_revision"],
            source_created_at=_parse_timestamp(value["source_created_at"]),
            source_digest=value["source_digest"],
            encoded_document=value["encoded_document"],
            saved_at=_parse_timestamp(value["saved_at"]),
            compacted_revision=value["compacted_revision"],
            compacted_digest=value["compacted_digest"],
        )


class MemoryRecoveryStore:
    """每个 URI 只保留当前压缩代的一份详细内容基线。"""

    def __init__(
        self,
        tree: MemoryTree,
        *,
        root: str | Path | None = None,
        max_records_per_memory: int = 1,
        max_file_bytes: int = 4 * 1024 * 1024,
    ) -> None:
        if not isinstance(tree, MemoryTree):
            raise TypeError("tree must be MemoryTree")
        if max_records_per_memory != 1:
            raise ValueError("memory recovery keeps exactly one active baseline per memory")
        if isinstance(max_file_bytes, bool) or not isinstance(max_file_bytes, int) or max_file_bytes <= 0:
            raise ValueError("max_file_bytes must be a positive integer")
        canonical = (tree.root.parent / "workflow" / "lifecycle" / "l2_recovery").resolve(
            strict=False
        )
        resolved = canonical if root is None else Path(root).expanduser().resolve(strict=False)
        if resolved != canonical:
            raise ValueError("memory recovery store must use the canonical sibling workflow root")
        self.tree = tree
        self.root = resolved
        self.codec: MemoryDocumentCodec = tree.document_codec
        self.max_records_per_memory = 1
        self.max_file_bytes = max_file_bytes

    def initialize(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.root, 0o700)

    def save(self, document: MemoryDocument, *, saved_at: datetime) -> MemoryRecoveryRecord:
        """保存待压缩详细基线；相同内容重放不受 saved_at 差异影响。"""

        if not isinstance(document, MemoryDocument):
            raise TypeError("document must be MemoryDocument")
        timestamp = _timestamp(saved_at)
        encoded = self.codec.encode(document)
        record = MemoryRecoveryRecord(
            uri=MemoryURI.from_address(document.address),
            source_revision=document.metadata.revision,
            source_created_at=document.metadata.created_at,
            source_digest=text_digest(encoded),
            encoded_document=encoded,
            saved_at=timestamp,
        )
        current = self.latest(record.uri, created_at=record.source_created_at)
        if current is not None:
            same_source = (
                current.source_revision == record.source_revision
                and current.source_digest == record.source_digest
                and current.encoded_document == record.encoded_document
            )
            if same_source:
                return current
            if current.active:
                raise MemoryRecoveryError("active recovery baseline cannot be overwritten")
        self._write(record)
        verified = self._read(record.uri)
        if (
            verified.source_revision != record.source_revision
            or verified.source_digest != record.source_digest
            or verified.encoded_document != record.encoded_document
        ):
            raise MemoryRecoveryError("memory recovery baseline failed read-back verification")
        return verified

    def activate(
        self,
        record: MemoryRecoveryRecord,
        compacted: MemorySnapshot,
    ) -> MemoryRecoveryRecord:
        """把详细基线精确绑定到提交后的压缩 L2 指纹。"""

        if not isinstance(record, MemoryRecoveryRecord):
            raise TypeError("record must be MemoryRecoveryRecord")
        if not compacted.exists or compacted.value is None:
            raise ValueError("compacted snapshot must exist")
        if compacted.identity != str(record.uri):
            raise ValueError("compacted snapshot identity does not match recovery baseline")
        if compacted.revision is None or compacted.source_digest is None:
            raise ValueError("compacted snapshot lacks a fingerprint")
        current = self._read(record.uri)
        if current.source_revision != record.source_revision or current.source_digest != record.source_digest:
            raise MemoryRecoveryError("recovery baseline changed before activation")
        active = replace(
            current,
            compacted_revision=compacted.revision,
            compacted_digest=compacted.source_digest,
        )
        self._write(active)
        verified = self._read(record.uri)
        if verified != active:
            raise MemoryRecoveryError("active recovery baseline failed read-back verification")
        return verified

    def for_compacted(self, snapshot: MemorySnapshot) -> MemoryRecoveryRecord | None:
        if not snapshot.exists or snapshot.value is None:
            return None
        record = self.latest(
            MemoryURI.parse(snapshot.identity),
            created_at=snapshot.value.metadata.created_at,
        )
        if record is None or not record.active:
            return None
        if (
            record.compacted_revision != snapshot.revision
            or record.compacted_digest != snapshot.source_digest
        ):
            return None
        return record

    def latest(self, uri: MemoryURI | str, *, created_at: datetime) -> MemoryRecoveryRecord | None:
        parsed = MemoryURI.parse(uri)
        parsed.to_address()
        try:
            record = self._read(parsed)
        except MemoryRecoveryError as exc:
            if isinstance(exc.__cause__, FileNotFoundError):
                return None
            raise
        return record if record.source_created_at == _timestamp(created_at) else None

    def restore(
        self,
        uri: MemoryURI | str,
        *,
        created_at: datetime,
        compacted_snapshot: MemorySnapshot | None = None,
    ) -> MemoryDocument:
        record = self.latest(uri, created_at=created_at)
        if record is None:
            raise MemoryRecoveryError("memory recovery record does not exist")
        if compacted_snapshot is not None and self.for_compacted(compacted_snapshot) != record:
            raise MemoryRecoveryError("memory recovery baseline is stale for compacted L2")
        try:
            document = self.codec.decode(
                record.encoded_document,
                expected_address=record.uri.to_address(),
            )
        except Exception as exc:
            raise MemoryRecoveryError("memory recovery document failed codec validation") from exc
        if (
            document.metadata.revision != record.source_revision
            or document.metadata.created_at != record.source_created_at
        ):
            raise MemoryRecoveryError("memory recovery document metadata does not match its manifest")
        return document

    def delete(self, uri: MemoryURI | str) -> int:
        parsed = MemoryURI.parse(uri)
        parsed.to_address()
        path = self._path(parsed)
        deleted = int(durable_unlink(path, artifact_root=self.root))
        directory = path.parent
        try:
            directory.rmdir()
            directory.parent.rmdir()
        except FileNotFoundError:
            pass
        except OSError:
            # 同一分片仍可包含其他 URI 的活动基线。
            pass
        return deleted

    def _read(self, uri: MemoryURI) -> MemoryRecoveryRecord:
        path = self._path(uri)
        try:
            encoded = read_regular_bytes(
                path,
                artifact_root=self.root,
                max_bytes=self.max_file_bytes,
            )
            record = MemoryRecoveryRecord.from_dict(json.loads(encoded))
        except Exception as exc:
            raise MemoryRecoveryError("failed to read a valid memory recovery baseline") from exc
        if record.uri != uri or encoded != self._encode(record):
            raise MemoryRecoveryError("memory recovery baseline failed canonical validation")
        return record

    def _write(self, record: MemoryRecoveryRecord) -> None:
        encoded = self._encode(record)
        if len(encoded) > self.max_file_bytes:
            raise MemoryRecoveryError("memory recovery baseline exceeds its file bound")
        path = self._path(record.uri)
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        for directory in (self.root, path.parent.parent, path.parent):
            os.chmod(directory, 0o700)
        atomic_replace_bytes(path, encoded, artifact_root=self.root)

    def _path(self, uri: MemoryURI) -> Path:
        identity = hashlib.sha256(str(uri).encode("utf-8")).hexdigest()
        return self.root / identity[:2] / identity / "active.json"

    @staticmethod
    def _encode(record: MemoryRecoveryRecord) -> bytes:
        return (canonical_json(record.to_dict()) + "\n").encode("utf-8")


def _timestamp(value: datetime) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError("memory recovery timestamp must be datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("memory recovery timestamp must include a timezone")
    return value.astimezone(UTC)


def _format_timestamp(value: datetime) -> str:
    return _timestamp(value).isoformat().replace("+00:00", "Z")


def _parse_timestamp(value: object) -> datetime:
    if not isinstance(value, str) or not value:
        raise ValueError("memory recovery timestamp is invalid")
    return _timestamp(datetime.fromisoformat(value.replace("Z", "+00:00")))


def _sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


__all__ = ["MemoryRecoveryError", "MemoryRecoveryRecord", "MemoryRecoveryStore"]
