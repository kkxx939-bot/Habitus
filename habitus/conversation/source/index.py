"""按地址的源索引：每份源一个只创建的标记文件，有序检查与释放只看本地址的标记，不再全量列信封。

标记先于信封创建、最后于信封删除——所以"有标记无信封"只有两种情形：put 还没写完（地址水位没盖住它），
或者释放删到一半（水位已盖住它）。读侧据此分辨，不需要额外状态。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from habitus.conversation.source.address import SourceAddress, entry_count, remove_if_empty
from habitus.conversation.source.model import (
    ConversationSourceCapacityError,
    ConversationSourceEnvelope,
    ConversationSourceError,
    encode_durable_record,
    require_sha256,
)
from habitus.infrastructure.store.filesystem import (
    DurablePathIntegrityError,
    ImmutableArtifactConflictError,
    atomic_create_bytes,
    atomic_temporary_destination,
    durable_unlink,
    list_real_directory,
    read_regular_bytes,
)

_SCHEMA = "conversation_source_marker_v1"
_SEQUENCE_DIGITS = 12
_MARKER_FILE = re.compile(r"^(?P<start>\d{12})-(?P<end>\d{12})-(?P<source_id>[0-9a-f]{64})$")
_ADDRESS_DIRECTORY = re.compile(r"^[0-9a-f]{32}$")
_MAX_MARKER_BYTES = 4 * 1024


@dataclass(frozen=True)
class SourceMarker:
    source_id: str
    address: SourceAddress
    start_sequence: int
    end_sequence: int

    @property
    def order(self) -> tuple[int, int, str]:
        """与交付的排序键一致：start、end、source_id。"""

        return (self.start_sequence, self.end_sequence, self.source_id)

    @classmethod
    def of(cls, envelope: ConversationSourceEnvelope) -> SourceMarker:
        return cls(
            envelope.source_id,
            SourceAddress.of(envelope),
            envelope.batch.start_sequence,
            envelope.batch.end_sequence,
        )


class SourceIndex:
    def __init__(self, conversation_root: str | Path, *, max_addresses: int, max_markers_per_address: int) -> None:
        for label, value in (("max_addresses", max_addresses), ("max_markers_per_address", max_markers_per_address)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{label} must be a positive integer")
        self.root = Path(conversation_root).expanduser().resolve(strict=False)
        self.index_root = self.root / "source" / "index"
        self.max_addresses = max_addresses
        self.max_markers_per_address = max_markers_per_address

    def mark(self, envelope: ConversationSourceEnvelope) -> SourceMarker:
        """建标记；同一份源重复建是幂等的。新地址、新标记到了上限就拒收——枚举那一侧按上限读，写进去的不能比它多。"""

        marker = SourceMarker.of(envelope)
        directory = self.index_root / marker.address.digest
        if not self._path(marker).exists():
            if not directory.exists() and entry_count(self.index_root) >= self.max_addresses:
                raise ConversationSourceCapacityError("source index reached its address bound")
            if entry_count(directory) >= self.max_markers_per_address:
                raise ConversationSourceCapacityError("source index address reached its marker bound")
        try:
            atomic_create_bytes(self._path(marker), self._encode(marker), artifact_root=self.root)
        except ImmutableArtifactConflictError as exc:
            raise ConversationSourceError("source marker conflicts with a different record") from exc
        return marker

    def markers(self, address: SourceAddress) -> tuple[SourceMarker, ...]:
        """本地址的全部标记，按交付顺序。"""

        return self._markers_in(self.index_root / address.digest)

    def address_directories(self) -> tuple[str, ...]:
        """有标记的地址（目录名），给释放按地址轮转用。"""

        try:
            entries = list_real_directory(self.index_root, artifact_root=self.root, max_entries=self.max_addresses)
        except DurablePathIntegrityError as exc:
            raise ConversationSourceError("source index exceeds its address bound") from exc
        return tuple(entry.name for entry in entries if entry.is_dir() and _ADDRESS_DIRECTORY.fullmatch(entry.name))

    def markers_in(self, directory_name: str) -> tuple[SourceMarker, ...]:
        if _ADDRESS_DIRECTORY.fullmatch(directory_name) is None:
            raise ConversationSourceError("source index directory name is invalid")
        return self._markers_in(self.index_root / directory_name)

    def remove(self, marker: SourceMarker) -> bool:
        removed = durable_unlink(self._path(marker), artifact_root=self.root)
        remove_if_empty(self.index_root / marker.address.digest, artifact_root=self.root)
        return removed

    def count(self) -> int:
        return len(self.address_directories())

    # ── 内部 ──────────────────────────────────────────────────────────────────────────

    def _markers_in(self, directory: Path) -> tuple[SourceMarker, ...]:
        try:
            entries = list_real_directory(directory, artifact_root=self.root, max_entries=self.max_markers_per_address)
        except DurablePathIntegrityError as exc:
            raise ConversationSourceError("source index address exceeds its marker bound") from exc
        markers = []
        for entry in entries:
            temporary = atomic_temporary_destination(entry.name)
            if entry.is_file() and temporary is not None and _MARKER_FILE.fullmatch(temporary) is not None:
                continue
            if not entry.is_file() or _MARKER_FILE.fullmatch(entry.name) is None:
                raise ConversationSourceError("source index contains an unsupported entry")
            markers.append(self._read(directory / entry.name))
        return tuple(sorted(markers, key=lambda item: item.order))

    def _read(self, path: Path) -> SourceMarker:
        encoded = read_regular_bytes(path, artifact_root=self.root, max_bytes=_MAX_MARKER_BYTES)
        try:
            payload = json.loads(encoded)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ConversationSourceError("source marker is corrupt") from exc
        if not isinstance(payload, dict) or payload.get("schema") != _SCHEMA:
            raise ConversationSourceError("source marker has an unexpected schema")
        marker = SourceMarker(
            require_sha256(payload["source_id"], "source_id"),
            SourceAddress.from_dict(payload["address"]),
            int(payload["start_sequence"]),
            int(payload["end_sequence"]),
        )
        if path != self._path(marker) or encoded != self._encode(marker):
            raise ConversationSourceError("source marker does not match its path or canonical form")
        return marker

    def _path(self, marker: SourceMarker) -> Path:
        limit = 10**_SEQUENCE_DIGITS
        if not 0 <= marker.start_sequence <= marker.end_sequence < limit:
            raise ConversationSourceError("source sequence range is outside the index bound")
        name = (
            f"{marker.start_sequence:0{_SEQUENCE_DIGITS}d}-{marker.end_sequence:0{_SEQUENCE_DIGITS}d}-"
            f"{require_sha256(marker.source_id, 'source_id')}"
        )
        return self.index_root / marker.address.digest / name

    @staticmethod
    def _encode(marker: SourceMarker) -> bytes:
        record = {
            "schema": _SCHEMA,
            "source_id": marker.source_id,
            "address": marker.address.to_dict(),
            "start_sequence": marker.start_sequence,
            "end_sequence": marker.end_sequence,
        }
        return encode_durable_record(record, max_bytes=_MAX_MARKER_BYTES, label="source marker")


__all__ = ["SourceIndex", "SourceMarker"]
