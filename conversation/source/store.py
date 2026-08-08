"""ConversationSourceEnvelope 的仅创建文件存储。"""

from __future__ import annotations

import json
import re
from pathlib import Path

from conversation.source.model import ConversationSourceEnvelope, ConversationSourceError
from foundation.integrity import canonical_json
from infrastructure.store.filesystem import (
    DurablePathIntegrityError,
    ImmutableArtifactConflictError,
    atomic_create_bytes,
    atomic_temporary_destination,
    list_real_directory,
    read_regular_bytes,
)

_SOURCE_FILE = re.compile(r"^(?P<source_id>[0-9a-f]{64})\.json$")
_SOURCE_DIRECTORY = re.compile(r"^[0-9a-f]{64}$")

# TODO: Behavior Projection 下游 ACK 与保留契约确定后，在 Conversation 生命周期中统一设计
# SourceEnvelope、Consumer Receipt 和 ProjectionBatch 的释放与清理；在此之前保留这些不可变文件，
# 避免单独清理 Source 导致重放、重新投影或幂等冲突信息丢失。


class ConversationSourceStore:
    """在 conversation_root/source 下保存不可变来源文件。"""

    def __init__(self, conversation_root: str | Path, *, max_entries: int, max_file_bytes: int) -> None:
        self.root = Path(conversation_root).expanduser().resolve(strict=False)
        if isinstance(max_entries, bool) or not isinstance(max_entries, int) or max_entries <= 0:
            raise ValueError("max_entries must be a positive integer")
        if isinstance(max_file_bytes, bool) or not isinstance(max_file_bytes, int) or max_file_bytes <= 0:
            raise ValueError("max_file_bytes must be a positive integer")
        self.max_entries = max_entries
        self.max_file_bytes = max_file_bytes
        self.source_root = self.root / "source"

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
            if current is None or current.content_digest != envelope.content_digest:
                raise ConversationSourceError("source_id conflicts with different source content") from exc
            return current
        stored = self.read(envelope.source_id)
        if stored is None:
            raise ConversationSourceError("persisted source envelope cannot be read back")
        if stored.content_digest != envelope.content_digest:
            raise ConversationSourceError("persisted source envelope differs from requested content")
        return stored

    def read(self, source_id: str) -> ConversationSourceEnvelope | None:
        path = self._path(source_id)
        try:
            encoded = read_regular_bytes(path, artifact_root=self.root, max_bytes=self.max_file_bytes)
        except FileNotFoundError:
            return None
        try:
            value = json.loads(encoded)
            envelope = ConversationSourceEnvelope.from_dict(value)
        except (UnicodeDecodeError, json.JSONDecodeError, ConversationSourceError) as exc:
            raise ConversationSourceError("source envelope is corrupt") from exc
        if encoded != self._encode(envelope):
            raise ConversationSourceError("source envelope is not canonically encoded")
        if envelope.source_id != source_id:
            raise ConversationSourceError("source envelope filename does not match source_id")
        return envelope

    def list(self) -> tuple[ConversationSourceEnvelope, ...]:
        try:
            entries = list_real_directory(
                self.source_root,
                artifact_root=self.root,
                max_entries=self.max_entries,
            )
        except DurablePathIntegrityError as exc:
            raise ConversationSourceError("source directory is invalid or exceeds its bound") from exc
        source_ids: list[str] = []
        receipt_source_ids: set[str] = set()
        for entry in entries:
            temporary = atomic_temporary_destination(entry.name)
            if entry.is_file() and temporary is not None and _SOURCE_FILE.fullmatch(temporary) is not None:
                continue
            match = _SOURCE_FILE.fullmatch(entry.name)
            if entry.is_file() and match is not None:
                source_ids.append(match.group("source_id"))
                continue
            if entry.is_dir() and _SOURCE_DIRECTORY.fullmatch(entry.name) is not None:
                receipt_source_ids.add(entry.name)
                continue
            raise ConversationSourceError("source directory contains an unsupported entry")
        if not receipt_source_ids.issubset(source_ids):
            raise ConversationSourceError("consumer receipt directory has no matching source envelope")
        envelopes = tuple(self._required_read(source_id) for source_id in sorted(source_ids))
        return tuple(sorted(envelopes, key=lambda item: (item.created_at, item.source_id)))

    def _required_read(self, source_id: str) -> ConversationSourceEnvelope:
        envelope = self.read(source_id)
        if envelope is None:
            raise ConversationSourceError("source envelope disappeared during enumeration")
        return envelope

    def _path(self, source_id: str) -> Path:
        if not isinstance(source_id, str) or re.fullmatch(r"[0-9a-f]{64}", source_id) is None:
            raise ConversationSourceError("source_id must be lowercase SHA-256 text")
        return self.source_root / f"{source_id}.json"

    def _encode(self, envelope: ConversationSourceEnvelope) -> bytes:
        encoded = (canonical_json(envelope.to_dict()) + "\n").encode("utf-8")
        if len(encoded) > self.max_file_bytes:
            raise ConversationSourceError("source envelope exceeds its configured file bound")
        return encoded
