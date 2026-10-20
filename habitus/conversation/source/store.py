"""ConversationSourceEnvelope 的仅创建文件存储。"""

from __future__ import annotations

import json
import re
from pathlib import Path

from habitus.conversation.source.ledger import ConversationSourceLedger
from habitus.conversation.source.model import (
    ConversationSourceEnvelope,
    ConversationSourceError,
    encode_durable_record,
)
from habitus.infrastructure.store.filesystem import (
    ImmutableArtifactConflictError,
    atomic_create_bytes,
    durable_unlink,
    read_regular_bytes,
)

# 释放：两个消费者都终态、且被同地址后一份已提交的交付超越之后，由 ``release.py`` 删（裁定 60，26 号清单 2.1）。
# 不再提供全量 ``list()``：有序检查读按地址的标记，恢复读"还没终态"的标记。


class ConversationSourceStore:
    """在 conversation_root/source/envelopes 下保存不可变 Source Record。"""

    def __init__(self, conversation_root: str | Path, *, max_file_bytes: int, ledger: ConversationSourceLedger) -> None:
        self.root = Path(conversation_root).expanduser().resolve(strict=False)
        if isinstance(max_file_bytes, bool) or not isinstance(max_file_bytes, int) or max_file_bytes <= 0:
            raise ValueError("max_file_bytes must be a positive integer")
        if not isinstance(ledger, ConversationSourceLedger):
            raise TypeError("ledger must be ConversationSourceLedger")
        if ledger.index.root != self.root:
            raise ValueError("source store and ledger must share one conversation root")
        self.ledger = ledger
        self.max_file_bytes = max_file_bytes
        self.envelope_root = self.root / "source" / "envelopes"

    def put(self, envelope: ConversationSourceEnvelope) -> ConversationSourceEnvelope:
        if not isinstance(envelope, ConversationSourceEnvelope):
            raise TypeError("envelope must be ConversationSourceEnvelope")
        encoded = self._encode(envelope)
        # 标记先于信封，且只在这一扇写入门里打：有序检查与释放靠"标记在、信封不在"分辨 put 进行中与已释放
        # （index.py）；重放撞车时同样先补标记，幂等。
        self.ledger.admit(envelope)
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
            encoded = read_regular_bytes(self._path(source_id), artifact_root=self.root, max_bytes=self.max_file_bytes)
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

    def discard(self, source_id: str) -> bool:
        """释放：删信封。"""

        return durable_unlink(self._path(source_id), artifact_root=self.root)

    def _path(self, source_id: str) -> Path:
        if not isinstance(source_id, str) or re.fullmatch(r"[0-9a-f]{64}", source_id) is None:
            raise ConversationSourceError("source_id must be lowercase SHA-256 text")
        return self.envelope_root / f"{source_id}.json"

    def _encode(self, envelope: ConversationSourceEnvelope) -> bytes:
        return encode_durable_record(envelope.to_dict(), max_bytes=self.max_file_bytes, label="source record")


__all__ = ["ConversationSourceStore"]
