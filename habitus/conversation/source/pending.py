"""还没终态的源：put 时打上，恢复只看这里，不再全量列信封。

清除是惰性的：恢复检查一份标记过的源，发现两个消费者都已终态，就把标记删掉。旧标记多留一会儿的代价只是
恢复时多看一眼，所以不在交付热路径上清。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from habitus.conversation.source.address import SourceAddress, entry_count
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

_SCHEMA = "conversation_source_pending_v1"
_PENDING_FILE = re.compile(r"^(?P<source_id>[0-9a-f]{64})\.json$")
_MAX_PENDING_BYTES = 4 * 1024


@dataclass(frozen=True)
class PendingListing:
    """列一次"还没终态"目录的结果。坏条目只挡它自己（裁定 60 第 6 条）：读不出来的标记按源号记进 ``corrupt``，
    认不出的文件（``.DS_Store``、编辑器备份等）按文件名记进 ``unsupported``，都不吞掉，也不拖住别的源。"""

    sources: tuple[tuple[str, SourceAddress], ...]
    corrupt: tuple[str, ...] = ()
    unsupported: tuple[str, ...] = ()


class PendingSources:
    def __init__(self, conversation_root: str | Path, *, max_entries: int) -> None:
        if isinstance(max_entries, bool) or not isinstance(max_entries, int) or max_entries <= 0:
            raise ValueError("max_entries must be a positive integer")
        self.root = Path(conversation_root).expanduser().resolve(strict=False)
        self.pending_root = self.root / "source" / "pending"
        self.max_entries = max_entries

    def mark(self, envelope: ConversationSourceEnvelope) -> None:
        if not self._path(envelope.source_id).exists() and entry_count(self.pending_root) >= self.max_entries:
            raise ConversationSourceCapacityError("pending sources reached their bound")
        record = {"schema": _SCHEMA, "source_id": envelope.source_id, "address": SourceAddress.of(envelope).to_dict()}
        encoded = encode_durable_record(record, max_bytes=_MAX_PENDING_BYTES, label="pending source")
        try:
            atomic_create_bytes(self._path(envelope.source_id), encoded, artifact_root=self.root)
        except ImmutableArtifactConflictError as exc:
            raise ConversationSourceError("pending source conflicts with a different record") from exc

    def clear(self, source_id: str) -> bool:
        return durable_unlink(self._path(source_id), artifact_root=self.root)

    def listing(self) -> PendingListing:
        try:
            entries = list_real_directory(self.pending_root, artifact_root=self.root, max_entries=self.max_entries)
        except DurablePathIntegrityError as exc:
            raise ConversationSourceError("pending sources exceed their bound") from exc
        found: list[tuple[str, SourceAddress]] = []
        corrupt: list[str] = []
        unsupported: list[str] = []
        for entry in entries:
            temporary = atomic_temporary_destination(entry.name)
            if entry.is_file() and temporary is not None and _PENDING_FILE.fullmatch(temporary) is not None:
                continue
            match = _PENDING_FILE.fullmatch(entry.name)
            if not entry.is_file() or match is None:
                unsupported.append(entry.name)
                continue
            try:
                found.append((match.group("source_id"), self._address(self.pending_root / entry.name)))
            except ConversationSourceError:
                corrupt.append(match.group("source_id"))
        return PendingListing(tuple(sorted(found)), tuple(corrupt), tuple(unsupported))

    def count(self) -> int:
        """占了多少条目（容量读数）：与写入闸一样只数个数，不解码，坏条目也占位。"""

        return entry_count(self.pending_root)

    def marked_at(self, source_id: str) -> datetime | None:
        """标记打上的时刻：标记只建一次、从不改写，文件的修改时刻就是它。标记已不在时为 None。"""

        try:
            modified = self._path(source_id).lstat().st_mtime
        except FileNotFoundError:
            return None
        return datetime.fromtimestamp(modified, UTC)

    def _address(self, path: Path) -> SourceAddress:
        try:
            encoded = read_regular_bytes(path, artifact_root=self.root, max_bytes=_MAX_PENDING_BYTES)
        except DurablePathIntegrityError as exc:
            raise ConversationSourceError("pending source record exceeds its read bound") from exc
        try:
            payload = json.loads(encoded)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ConversationSourceError("pending source record is corrupt") from exc
        if not isinstance(payload, dict) or payload.get("schema") != _SCHEMA:
            raise ConversationSourceError("pending source record has an unexpected schema")
        return SourceAddress.from_dict(payload.get("address"))

    def _path(self, source_id: str) -> Path:
        return self.pending_root / f"{require_sha256(source_id, 'source_id')}.json"


__all__ = ["PendingListing", "PendingSources"]
