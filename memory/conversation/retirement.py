"""保存 Archive Summary 先下线后清理所需的精确耐久清单。"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from foundation.integrity import canonical_json
from infrastructure.store.filesystem import atomic_replace_bytes, durable_unlink, read_regular_bytes
from memory.conversation.indexing.model import ConversationSummaryReference, summary_reference
from memory.conversation.layout import ConversationAddress
from pre.conversation import (
    ConversationRangeSummary,
    ConversationSegmentSummary,
)


class ConversationSummaryRetirementError(RuntimeError):
    """Archive 退休清单损坏、冲突或超过边界。"""


class ConversationSummaryRetirementPhase(str, Enum):
    RETIRING = "retiring"
    INDEX_REMOVED = "index_removed"
    HISTORY_RELEASED = "history_released"
    SOURCES_DELETED = "sources_deleted"


@dataclass(frozen=True)
class ConversationSummaryRetirementSource:
    reference: ConversationSummaryReference
    digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.reference, ConversationSummaryReference):
            raise TypeError("retirement source reference is invalid")
        _require_digest(self.digest)

    def to_dict(self) -> dict[str, str]:
        return {"identity": self.reference.identity, "digest": self.digest}


@dataclass(frozen=True)
class ConversationSummaryRetirementManifest:
    archive: ConversationSummaryRetirementSource
    start_sequence: int
    end_sequence: int
    ranges: tuple[ConversationSummaryRetirementSource, ...]
    segments: tuple[ConversationSummaryRetirementSource, ...]
    expected_use_version: int
    phase: ConversationSummaryRetirementPhase
    prepared_at: datetime
    updated_at: datetime

    def __post_init__(self) -> None:
        if self.archive.reference.stage.value != "archive":
            raise ValueError("retirement manifest archive reference has the wrong stage")
        if (
            isinstance(self.start_sequence, bool)
            or not isinstance(self.start_sequence, int)
            or self.start_sequence < 0
        ):
            raise ValueError("retirement manifest start_sequence must be non-negative")
        if (
            isinstance(self.end_sequence, bool)
            or not isinstance(self.end_sequence, int)
            or self.end_sequence <= 0
        ):
            raise ValueError("retirement manifest end_sequence must be positive")
        if self.end_sequence < self.start_sequence:
            raise ValueError("retirement manifest sequence range is invalid")
        for name, values, stage in (
            ("ranges", self.ranges, "range"),
            ("segments", self.segments, "segment"),
        ):
            if not isinstance(values, tuple) or any(
                not isinstance(item, ConversationSummaryRetirementSource) for item in values
            ):
                raise TypeError(f"retirement manifest {name} are invalid")
            if any(item.reference.stage.value != stage for item in values):
                raise ValueError(f"retirement manifest {name} have the wrong stage")
            if len({item.reference.identity for item in values}) != len(values):
                raise ValueError(f"retirement manifest {name} contain duplicate identities")
        addresses = {
            self.archive.reference.address,
            *(item.reference.address for item in self.ranges),
            *(item.reference.address for item in self.segments),
        }
        if len(addresses) != 1:
            raise ValueError("retirement manifest sources belong to different Conversations")
        if (
            isinstance(self.expected_use_version, bool)
            or not isinstance(self.expected_use_version, int)
            or self.expected_use_version < 0
        ):
            raise ValueError("retirement manifest expected_use_version must be non-negative")
        object.__setattr__(self, "phase", ConversationSummaryRetirementPhase(self.phase))
        object.__setattr__(self, "prepared_at", _timestamp(self.prepared_at))
        object.__setattr__(self, "updated_at", _timestamp(self.updated_at))
        if self.updated_at < self.prepared_at:
            raise ValueError("retirement manifest updated_at precedes prepared_at")

    @property
    def address(self) -> ConversationAddress:
        return self.archive.reference.address

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": "conversation_summary_retirement_v1",
            "archive": self.archive.to_dict(),
            "start_sequence": self.start_sequence,
            "end_sequence": self.end_sequence,
            "ranges": [item.to_dict() for item in self.ranges],
            "segments": [item.to_dict() for item in self.segments],
            "expected_use_version": self.expected_use_version,
            "phase": self.phase.value,
            "prepared_at": _format_timestamp(self.prepared_at),
            "updated_at": _format_timestamp(self.updated_at),
        }

    @classmethod
    def from_dict(cls, value: object) -> ConversationSummaryRetirementManifest:
        expected = {
            "schema",
            "archive",
            "start_sequence",
            "end_sequence",
            "ranges",
            "segments",
            "expected_use_version",
            "phase",
            "prepared_at",
            "updated_at",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ValueError("retirement manifest fields are invalid")
        if value["schema"] != "conversation_summary_retirement_v1":
            raise ValueError("retirement manifest schema is unsupported")
        return cls(
            archive=_source(value["archive"]),
            start_sequence=value["start_sequence"],
            end_sequence=value["end_sequence"],
            ranges=_sources(value["ranges"]),
            segments=_sources(value["segments"]),
            expected_use_version=value["expected_use_version"],
            phase=ConversationSummaryRetirementPhase(value["phase"]),
            prepared_at=_parse_timestamp(value["prepared_at"]),
            updated_at=_parse_timestamp(value["updated_at"]),
        )


class ConversationSummaryRetirementStore:
    """按 Archive 身份保存未完成退休清单，并为索引源提供隐藏判断。"""

    def __init__(
        self,
        workflow_root: str | Path,
        *,
        max_manifests: int = 100_000,
        max_file_bytes: int = 2 * 1024 * 1024,
    ) -> None:
        root = Path(workflow_root).expanduser().resolve(strict=False)
        if isinstance(max_manifests, bool) or not isinstance(max_manifests, int) or max_manifests <= 0:
            raise ValueError("max_manifests must be positive")
        if isinstance(max_file_bytes, bool) or not isinstance(max_file_bytes, int) or max_file_bytes <= 0:
            raise ValueError("max_file_bytes must be positive")
        self.workflow_root = root
        self.root = root / "lifecycle" / "summary_retirements"
        self.max_manifests = max_manifests
        self.max_file_bytes = max_file_bytes

    def initialize(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.root, 0o700)
        self.pending()

    def prepare(
        self,
        address: ConversationAddress,
        archive: ConversationRangeSummary,
        ranges: tuple[ConversationRangeSummary, ...],
        segments: tuple[ConversationSegmentSummary, ...],
        *,
        expected_use_version: int,
        prepared_at: datetime,
    ) -> ConversationSummaryRetirementManifest:
        archive_reference = summary_reference(address, archive)
        manifest = ConversationSummaryRetirementManifest(
            archive=ConversationSummaryRetirementSource(archive_reference, archive.digest),
            start_sequence=archive.start_sequence,
            end_sequence=archive.end_sequence,
            ranges=tuple(
                ConversationSummaryRetirementSource(summary_reference(address, item), item.digest)
                for item in ranges
            ),
            segments=tuple(
                ConversationSummaryRetirementSource(summary_reference(address, item), item.digest)
                for item in segments
            ),
            expected_use_version=expected_use_version,
            phase=ConversationSummaryRetirementPhase.RETIRING,
            prepared_at=_timestamp(prepared_at),
            updated_at=_timestamp(prepared_at),
        )
        current = self.try_read(archive_reference)
        if current is not None:
            if current == manifest:
                return current
            raise ConversationSummaryRetirementError("Archive already has another retirement manifest")
        self._write(manifest)
        return self.read(archive_reference)

    def advance(
        self,
        manifest: ConversationSummaryRetirementManifest,
        phase: ConversationSummaryRetirementPhase,
        *,
        updated_at: datetime,
    ) -> ConversationSummaryRetirementManifest:
        current = self.read(manifest.archive.reference)
        if current.archive != manifest.archive:
            raise ConversationSummaryRetirementError("Archive retirement manifest ownership changed")
        order = tuple(ConversationSummaryRetirementPhase)
        selected = ConversationSummaryRetirementPhase(phase)
        if order.index(selected) < order.index(current.phase):
            raise ValueError("Archive retirement phase cannot move backwards")
        advanced = replace(current, phase=selected, updated_at=_timestamp(updated_at))
        self._write(advanced)
        return self.read(advanced.archive.reference)

    def hidden(self, reference: ConversationSummaryReference) -> bool:
        if reference.stage.value != "archive":
            return False
        return self.try_read(reference) is not None

    def for_address(
        self,
        address: ConversationAddress,
    ) -> tuple[ConversationSummaryRetirementManifest, ...]:
        return tuple(item for item in self.pending() if item.address == address)

    def pending(self) -> tuple[ConversationSummaryRetirementManifest, ...]:
        try:
            metadata = self.root.stat(follow_symlinks=False)
        except FileNotFoundError:
            return ()
        if not stat.S_ISDIR(metadata.st_mode) or self.root.is_symlink():
            raise ConversationSummaryRetirementError("summary retirement root is unsafe")
        manifests: list[ConversationSummaryRetirementManifest] = []
        with os.scandir(self.root) as entries:
            for entry in entries:
                if entry.is_symlink() or not entry.is_file(follow_symlinks=False):
                    raise ConversationSummaryRetirementError("summary retirement root has an unknown entry")
                if not entry.name.endswith(".json") or len(entry.name) != 69:
                    raise ConversationSummaryRetirementError("summary retirement filename is invalid")
                encoded = read_regular_bytes(
                    Path(entry.path),
                    artifact_root=self.root,
                    max_bytes=self.max_file_bytes,
                )
                manifest = ConversationSummaryRetirementManifest.from_dict(json.loads(encoded))
                if encoded != self._encode(manifest) or self._path(manifest.archive.reference) != Path(entry.path):
                    raise ConversationSummaryRetirementError("summary retirement manifest is not canonical")
                manifests.append(manifest)
                if len(manifests) > self.max_manifests:
                    raise ConversationSummaryRetirementError("summary retirement manifests exceed their bound")
        return tuple(sorted(manifests, key=lambda item: item.archive.reference.identity))

    def read(
        self,
        reference: ConversationSummaryReference,
    ) -> ConversationSummaryRetirementManifest:
        try:
            encoded = read_regular_bytes(
                self._path(reference),
                artifact_root=self.root,
                max_bytes=self.max_file_bytes,
            )
            manifest = ConversationSummaryRetirementManifest.from_dict(json.loads(encoded))
        except Exception as exc:
            raise ConversationSummaryRetirementError("failed to read summary retirement manifest") from exc
        if manifest.archive.reference != reference or encoded != self._encode(manifest):
            raise ConversationSummaryRetirementError("summary retirement manifest failed validation")
        return manifest

    def try_read(
        self,
        reference: ConversationSummaryReference,
    ) -> ConversationSummaryRetirementManifest | None:
        try:
            return self.read(reference)
        except ConversationSummaryRetirementError as exc:
            if isinstance(exc.__cause__, FileNotFoundError):
                return None
            raise

    def complete(self, manifest: ConversationSummaryRetirementManifest) -> bool:
        current = self.read(manifest.archive.reference)
        if current.archive != manifest.archive:
            raise ConversationSummaryRetirementError("summary retirement manifest ownership changed")
        return durable_unlink(
            self._path(manifest.archive.reference),
            artifact_root=self.root,
        )

    def _write(self, manifest: ConversationSummaryRetirementManifest) -> None:
        encoded = self._encode(manifest)
        if len(encoded) > self.max_file_bytes:
            raise ConversationSummaryRetirementError("summary retirement manifest exceeds its file bound")
        atomic_replace_bytes(
            self._path(manifest.archive.reference),
            encoded,
            artifact_root=self.root,
        )

    def _path(self, reference: ConversationSummaryReference) -> Path:
        digest = hashlib.sha256(reference.identity.encode("utf-8")).hexdigest()
        return self.root / f"{digest}.json"

    @staticmethod
    def _encode(manifest: ConversationSummaryRetirementManifest) -> bytes:
        return (canonical_json(manifest.to_dict()) + "\n").encode("utf-8")


def _source(value: object) -> ConversationSummaryRetirementSource:
    if not isinstance(value, Mapping) or set(value) != {"identity", "digest"}:
        raise ValueError("retirement source fields are invalid")
    return ConversationSummaryRetirementSource(
        ConversationSummaryReference.parse(value["identity"]),
        value["digest"],
    )


def _sources(value: object) -> tuple[ConversationSummaryRetirementSource, ...]:
    if not isinstance(value, list):
        raise ValueError("retirement sources must be a list")
    return tuple(_source(item) for item in value)


def _require_digest(value: object) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError("summary retirement digest must be lowercase SHA-256")


def _timestamp(value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("summary retirement timestamp must be timezone-aware")
    return value.astimezone(timezone.utc)


def _format_timestamp(value: datetime) -> str:
    return _timestamp(value).isoformat().replace("+00:00", "Z")


def _parse_timestamp(value: object) -> datetime:
    if not isinstance(value, str) or not value:
        raise ValueError("summary retirement timestamp is invalid")
    return _timestamp(datetime.fromisoformat(value.replace("Z", "+00:00")))


__all__ = [
    "ConversationSummaryRetirementError",
    "ConversationSummaryRetirementManifest",
    "ConversationSummaryRetirementPhase",
    "ConversationSummaryRetirementSource",
    "ConversationSummaryRetirementStore",
]
