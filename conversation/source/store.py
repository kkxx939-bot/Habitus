"""ConversationSourceEnvelope 的仅创建文件存储。"""

from __future__ import annotations

import json
import re
from pathlib import Path

from conversation.source.model import (
    ConversationSourceEnvelope,
    ConversationSourceError,
    encode_durable_record,
)
from infrastructure.store.filesystem import (
    DurablePathIntegrityError,
    ImmutableArtifactConflictError,
    atomic_create_bytes,
    atomic_temporary_destination,
    list_real_directory,
    read_regular_bytes,
)

_SOURCE_FILE = re.compile(r"^(?P<source_id>[0-9a-f]{64})\.json$")

# TODO: 等 Behavior 下游 ACK 与完整生命周期一起确定 Source、Outcome、Output 的释放门槛；
# 当前保留所有不可变交付记录，避免提前清理破坏重放和冲突检测。


class ConversationSourceStore:
    """在 conversation_root/source/envelopes 下保存不可变 Source Record。"""

    def __init__(self, conversation_root: str | Path, *, max_files: int, max_file_bytes: int) -> None:
        self.root = Path(conversation_root).expanduser().resolve(strict=False)
        if isinstance(max_files, bool) or not isinstance(max_files, int) or max_files <= 0:
            raise ValueError("max_files must be a positive integer")
        if isinstance(max_file_bytes, bool) or not isinstance(max_file_bytes, int) or max_file_bytes <= 0:
            raise ValueError("max_file_bytes must be a positive integer")
        self.max_files = max_files
        self.max_file_bytes = max_file_bytes
        self.envelope_root = self.root / "source" / "envelopes"

    def put(self, envelope: ConversationSourceEnvelope) -> ConversationSourceEnvelope:
        if not isinstance(envelope, ConversationSourceEnvelope):
            raise TypeError("envelope must be ConversationSourceEnvelope")
        encoded = self._encode(envelope)
        try:
            atomic_create_bytes(self._path(envelope.source_id), encoded, artifact_root=self.root)
        except ImmutableArtifactConflictError as exc:
            try:
                current = self.read(envelope.source_id)
            except Exception:
                raise ConversationSourceError("source identity collides with an unreadable artifact") from exc
            if current is None or current.source_payload_digest != envelope.source_payload_digest:
                raise ConversationSourceError("source_id conflicts with different source payload") from exc
            return current
        stored = self.read(envelope.source_id)
        if stored is None or stored.source_record_digest != envelope.source_record_digest:
            raise ConversationSourceError("source record was not durably read back")
        return stored

    def read(self, source_id: str) -> ConversationSourceEnvelope | None:
        try:
            encoded = read_regular_bytes(
                self._path(source_id), artifact_root=self.root, max_bytes=self.max_file_bytes
            )
        except FileNotFoundError:
            return None
        try:
            envelope = ConversationSourceEnvelope.from_dict(json.loads(encoded))
        except (UnicodeDecodeError, json.JSONDecodeError, ConversationSourceError) as exc:
            raise ConversationSourceError("source record is corrupt") from exc
        if encoded != self._encode(envelope):
            raise ConversationSourceError("source record is not canonically encoded")
        if envelope.source_id != source_id:
            raise ConversationSourceError("source filename does not match source_id")
        return envelope

    def list(self) -> tuple[ConversationSourceEnvelope, ...]:
        try:
            entries = list_real_directory(
                self.envelope_root, artifact_root=self.root, max_entries=self.max_files
            )
        except DurablePathIntegrityError as exc:
            raise ConversationSourceError("source envelope directory is invalid or exceeds its bound") from exc
        source_ids: list[str] = []
        for entry in entries:
            temporary = atomic_temporary_destination(entry.name)
            if entry.is_file() and temporary is not None and _SOURCE_FILE.fullmatch(temporary) is not None:
                continue
            match = _SOURCE_FILE.fullmatch(entry.name)
            if not entry.is_file() or match is None:
                raise ConversationSourceError("source envelope directory contains an unsupported entry")
            source_ids.append(match.group("source_id"))
        envelopes = tuple(self._required_read(source_id) for source_id in sorted(source_ids))
        return tuple(sorted(envelopes, key=lambda item: (item.recorded_at, item.source_id)))

    def _required_read(self, source_id: str) -> ConversationSourceEnvelope:
        envelope = self.read(source_id)
        if envelope is None:
            raise ConversationSourceError("source record disappeared during enumeration")
        return envelope

    def _path(self, source_id: str) -> Path:
        if not isinstance(source_id, str) or re.fullmatch(r"[0-9a-f]{64}", source_id) is None:
            raise ConversationSourceError("source_id must be lowercase SHA-256 text")
        return self.envelope_root / f"{source_id}.json"

    def _encode(self, envelope: ConversationSourceEnvelope) -> bytes:
        return encode_durable_record(
            envelope.to_dict(), max_bytes=self.max_file_bytes, label="source record"
        )


__all__ = ["ConversationSourceStore"]
