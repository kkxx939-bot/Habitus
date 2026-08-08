"""从 SourceEnvelope.batch 生成无模型、无 Claim 的确定性 Behavior 投影。"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from conversation.source.coordinator import ConversationConsumerExecution
from conversation.source.model import ConversationSourceEnvelope, ConversationSourceError
from conversation.source.receipt import (
    ConversationConsumerReceipt,
    ConversationConsumerReceiptState,
    ConversationSourceConsumer,
)
from foundation.integrity import canonical_digest, canonical_json, canonicalize, immutable_snapshot
from infrastructure.store.filesystem import ImmutableArtifactConflictError, atomic_create_bytes, read_regular_bytes
from pre.conversation import ConversationMessage, ConversationMessageRole
from pre.conversation.messages.model import conversation_datetime

CONVERSATION_BEHAVIOR_PROJECTOR_VERSION = "conversation_behavior_projector_v1"
_BATCH_SCHEMA = "conversation_behavior_projection_batch_v1"
_ITEM_SCHEMA = "conversation_behavior_projection_item_v1"


class ConversationBehaviorProjectionKind(str, Enum):
    USER_CONVERSATION_INPUT = "user_conversation_input"
    AGENT_TOOL_CALL = "agent_tool_call"
    TOOL_EXECUTION_RESULT = "tool_execution_result"


def _sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ConversationSourceError(f"{label} must be lowercase SHA-256 text")
    return value


@dataclass(frozen=True)
class ConversationBehaviorProjectionItem:
    projection_item_id: str
    source_id: str
    source_message_id: str
    source_message_digest: str
    occurred_at: datetime
    projection_kind: ConversationBehaviorProjectionKind
    payload: Any

    def __post_init__(self) -> None:
        _sha256(self.projection_item_id, "projection_item_id")
        _sha256(self.source_id, "projection item source_id")
        if not isinstance(self.source_message_id, str) or not self.source_message_id:
            raise ConversationSourceError("source_message_id must be non-empty text")
        _sha256(self.source_message_digest, "source_message_digest")
        object.__setattr__(self, "occurred_at", conversation_datetime(self.occurred_at, "occurred_at"))
        try:
            object.__setattr__(self, "projection_kind", ConversationBehaviorProjectionKind(self.projection_kind))
        except ValueError as exc:
            raise ConversationSourceError("projection_kind is invalid") from exc
        object.__setattr__(self, "payload", immutable_snapshot(canonicalize(self.payload)))
        expected = canonical_digest(self._identity_payload())
        if self.projection_item_id != expected:
            raise ConversationSourceError("projection_item_id does not match item content")

    @classmethod
    def create(
        cls,
        *,
        source_id: str,
        message: ConversationMessage,
        projection_kind: ConversationBehaviorProjectionKind,
        payload: Any,
    ) -> ConversationBehaviorProjectionItem:
        _sha256(source_id, "source_id")
        source_message_digest = canonical_digest(message.to_dict())
        identity = {
            "schema_version": _ITEM_SCHEMA,
            "source_id": source_id,
            "source_message_id": message.message_id,
            "source_message_digest": source_message_digest,
            "occurred_at": message.occurred_at,
            "projection_kind": projection_kind.value,
            "payload": payload,
        }
        return cls(
            projection_item_id=canonical_digest(identity),
            source_id=source_id,
            source_message_id=message.message_id,
            source_message_digest=source_message_digest,
            occurred_at=message.occurred_at,
            projection_kind=projection_kind,
            payload=payload,
        )

    def _identity_payload(self) -> dict[str, Any]:
        payload = {
            "schema_version": _ITEM_SCHEMA,
            "source_id": self.source_id,
            "source_message_id": self.source_message_id,
            "source_message_digest": self.source_message_digest,
            "occurred_at": self.occurred_at,
            "projection_kind": self.projection_kind.value,
            "payload": canonicalize(self.payload),
        }
        return canonicalize(payload)

    def to_dict(self) -> dict[str, Any]:
        return canonicalize(
            {
                "schema_version": _ITEM_SCHEMA,
                "projection_item_id": self.projection_item_id,
                "source_id": self.source_id,
                "source_message_id": self.source_message_id,
                "source_message_digest": self.source_message_digest,
                "occurred_at": self.occurred_at,
                "projection_kind": self.projection_kind.value,
                "payload": canonicalize(self.payload),
            }
        )

    @classmethod
    def from_dict(
        cls,
        value: object,
        *,
        source_id: str,
    ) -> ConversationBehaviorProjectionItem:
        if not isinstance(value, Mapping):
            raise ConversationSourceError("projection item must be an object")
        expected = {
            "schema_version",
            "projection_item_id",
            "source_id",
            "source_message_id",
            "source_message_digest",
            "occurred_at",
            "projection_kind",
            "payload",
        }
        if set(value) != expected or value.get("schema_version") != _ITEM_SCHEMA:
            raise ConversationSourceError("projection item schema is invalid")
        try:
            kind = ConversationBehaviorProjectionKind(value["projection_kind"])
        except (TypeError, ValueError) as exc:
            raise ConversationSourceError("projection item kind is invalid") from exc
        _sha256(source_id, "source_id")
        item = cls(
            projection_item_id=value["projection_item_id"],
            source_id=value["source_id"],
            source_message_id=value["source_message_id"],
            source_message_digest=value["source_message_digest"],
            occurred_at=conversation_datetime(value["occurred_at"], "occurred_at"),
            projection_kind=kind,
            payload=value["payload"],
        )
        if item.source_id != source_id:
            raise ConversationSourceError("projection item belongs to another source")
        return item


@dataclass(frozen=True)
class ConversationBehaviorProjectionBatch:
    projection_id: str
    source_id: str
    source_digest: str
    projector_version: str
    items: tuple[ConversationBehaviorProjectionItem, ...]
    created_at: datetime
    content_digest: str

    def __post_init__(self) -> None:
        _sha256(self.projection_id, "projection_id")
        _sha256(self.source_id, "projection source_id")
        _sha256(self.source_digest, "projection source_digest")
        if not isinstance(self.projector_version, str) or not self.projector_version:
            raise ConversationSourceError("projector_version must be non-empty text")
        if not isinstance(self.items, tuple) or not self.items:
            raise ConversationSourceError("projection batch must contain items")
        if any(not isinstance(item, ConversationBehaviorProjectionItem) for item in self.items):
            raise TypeError("projection items must be ConversationBehaviorProjectionItem values")
        if any(item.source_id != self.source_id for item in self.items):
            raise ConversationSourceError("projection batch contains an item from another source")
        if len({item.projection_item_id for item in self.items}) != len(self.items):
            raise ConversationSourceError("projection item IDs must be unique")
        object.__setattr__(self, "created_at", conversation_datetime(self.created_at, "created_at"))
        _sha256(self.content_digest, "projection content_digest")
        if self.projection_id != canonical_digest(self._identity_payload()):
            raise ConversationSourceError("projection_id does not match projection content")
        if self.content_digest != canonical_digest(self._content_payload()):
            raise ConversationSourceError("projection content_digest does not match content")

    @classmethod
    def create(
        cls,
        *,
        source_id: str,
        source_digest: str,
        projector_version: str,
        items: tuple[ConversationBehaviorProjectionItem, ...],
        created_at: datetime,
    ) -> ConversationBehaviorProjectionBatch:
        identity = {
            "schema_version": _BATCH_SCHEMA,
            "source_id": source_id,
            "source_digest": source_digest,
            "projector_version": projector_version,
            "items": [item.to_dict() for item in items],
        }
        projection_id = canonical_digest(identity)
        content = {**identity, "projection_id": projection_id}
        return cls(
            projection_id=projection_id,
            source_id=source_id,
            source_digest=source_digest,
            projector_version=projector_version,
            items=items,
            created_at=created_at,
            content_digest=canonical_digest(content),
        )

    def _identity_payload(self) -> dict[str, Any]:
        return canonicalize(
            {
                "schema_version": _BATCH_SCHEMA,
                "source_id": self.source_id,
                "source_digest": self.source_digest,
                "projector_version": self.projector_version,
                "items": [item.to_dict() for item in self.items],
            }
        )

    def _content_payload(self) -> dict[str, Any]:
        return canonicalize({**self._identity_payload(), "projection_id": self.projection_id})

    def to_dict(self) -> dict[str, Any]:
        return canonicalize(
            {
                **self._content_payload(),
                "created_at": self.created_at,
                "content_digest": self.content_digest,
            }
        )

    @classmethod
    def from_dict(cls, value: object) -> ConversationBehaviorProjectionBatch:
        if not isinstance(value, Mapping):
            raise ConversationSourceError("projection batch must be an object")
        expected = {
            "schema_version",
            "projection_id",
            "source_id",
            "source_digest",
            "projector_version",
            "items",
            "created_at",
            "content_digest",
        }
        if set(value) != expected or value.get("schema_version") != _BATCH_SCHEMA:
            raise ConversationSourceError("projection batch schema is invalid")
        raw_items = value["items"]
        if not isinstance(raw_items, list):
            raise ConversationSourceError("projection batch items must be a list")
        return cls(
            projection_id=value["projection_id"],
            source_id=value["source_id"],
            source_digest=value["source_digest"],
            projector_version=value["projector_version"],
            items=tuple(
                ConversationBehaviorProjectionItem.from_dict(item, source_id=value["source_id"])
                for item in raw_items
            ),
            created_at=conversation_datetime(value["created_at"], "created_at"),
            content_digest=value["content_digest"],
        )


class ConversationBehaviorProjector:
    """只读取 SourceEnvelope.batch；COMPLETION 在第一版明确跳过。"""

    def __init__(self, *, clock: Callable[[], datetime] | None = None) -> None:
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.projector_version = CONVERSATION_BEHAVIOR_PROJECTOR_VERSION

    def project(self, envelope: ConversationSourceEnvelope) -> ConversationBehaviorProjectionBatch | None:
        if not isinstance(envelope, ConversationSourceEnvelope):
            raise TypeError("envelope must be ConversationSourceEnvelope")
        items = tuple(
            projected
            for message in envelope.batch.messages
            if (projected := self._project_message(envelope.source_id, message)) is not None
        )
        if not items:
            return None
        return ConversationBehaviorProjectionBatch.create(
            source_id=envelope.source_id,
            source_digest=envelope.content_digest,
            projector_version=self.projector_version,
            items=items,
            created_at=self.clock(),
        )

    def _project_message(
        self,
        source_id: str,
        message: ConversationMessage,
    ) -> ConversationBehaviorProjectionItem | None:
        if message.role is ConversationMessageRole.COMPLETION:
            return None
        if message.role is ConversationMessageRole.PROMPT:
            kind = ConversationBehaviorProjectionKind.USER_CONVERSATION_INPUT
            payload = {"content": message.content}
        elif message.role is ConversationMessageRole.TOOL_CALL:
            kind = ConversationBehaviorProjectionKind.AGENT_TOOL_CALL
            payload = {
                "tool_call_id": message.tool_call_id,
                "tool_name": message.tool_name,
                "arguments_digest": canonical_digest(message.content),
            }
        elif message.role is ConversationMessageRole.TOOL_RESULT:
            kind = ConversationBehaviorProjectionKind.TOOL_EXECUTION_RESULT
            payload = {
                "tool_call_id": message.tool_call_id,
                "tool_name": message.tool_name,
                "tool_status": message.tool_status.value if message.tool_status is not None else None,
                "content_mode": message.content_mode.value if message.content_mode is not None else None,
                "result_digest": canonical_digest(message.content),
                "source_ref": message.source_ref,
                "original_size_bytes": message.original_size_bytes,
                "original_sha256": message.original_sha256,
            }
        else:  # pragma: no cover - ConversationMessageRole 是封闭枚举。
            raise ConversationSourceError("unsupported Conversation role")
        return ConversationBehaviorProjectionItem.create(
            source_id=source_id,
            message=message,
            projection_kind=kind,
            payload=payload,
        )


class ConversationBehaviorProjectionStore:
    """独立于旧 Behavior Evidence/Claim 的不可变文件 Outbox。"""

    def __init__(self, conversation_root: str | Path, *, max_file_bytes: int) -> None:
        self.root = Path(conversation_root).expanduser().resolve(strict=False)
        if isinstance(max_file_bytes, bool) or not isinstance(max_file_bytes, int) or max_file_bytes <= 0:
            raise ValueError("max_file_bytes must be a positive integer")
        self.max_file_bytes = max_file_bytes
        self.projection_root = self.root / "projections" / "behavior"

    def put(self, batch: ConversationBehaviorProjectionBatch) -> ConversationBehaviorProjectionBatch:
        if not isinstance(batch, ConversationBehaviorProjectionBatch):
            raise TypeError("batch must be ConversationBehaviorProjectionBatch")
        encoded = self._encode(batch)
        try:
            atomic_create_bytes(self._path(batch.projection_id), encoded, artifact_root=self.root)
        except ImmutableArtifactConflictError as exc:
            try:
                current = self.read(batch.projection_id)
            except Exception:
                raise ConversationSourceError("projection identity collides with an unreadable artifact") from exc
            if current is None or current.content_digest != batch.content_digest:
                raise ConversationSourceError("projection_id conflicts with different projection content") from exc
            return current
        stored = self.read(batch.projection_id)
        if stored is None or stored.content_digest != batch.content_digest:
            raise ConversationSourceError("projection batch was not durably read back")
        return stored

    def read(self, projection_id: str) -> ConversationBehaviorProjectionBatch | None:
        path = self._path(projection_id)
        try:
            encoded = read_regular_bytes(path, artifact_root=self.root, max_bytes=self.max_file_bytes)
        except FileNotFoundError:
            return None
        try:
            batch = ConversationBehaviorProjectionBatch.from_dict(json.loads(encoded))
        except (UnicodeDecodeError, json.JSONDecodeError, ConversationSourceError) as exc:
            raise ConversationSourceError("behavior projection is corrupt") from exc
        if encoded != self._encode(batch):
            raise ConversationSourceError("behavior projection is not canonically encoded")
        if batch.projection_id != projection_id:
            raise ConversationSourceError("projection filename does not match projection_id")
        return batch

    def _path(self, projection_id: str) -> Path:
        _sha256(projection_id, "projection_id")
        return self.projection_root / f"{projection_id}.json"

    def _encode(self, batch: ConversationBehaviorProjectionBatch) -> bytes:
        encoded = (canonical_json(batch.to_dict()) + "\n").encode("utf-8")
        if len(encoded) > self.max_file_bytes:
            raise ConversationSourceError("behavior projection exceeds its configured file bound")
        return encoded


class ConversationBehaviorProjectionConsumer:
    consumer = ConversationSourceConsumer.BEHAVIOR_PROJECTION

    def __init__(
        self,
        projector: ConversationBehaviorProjector,
        store: ConversationBehaviorProjectionStore,
    ) -> None:
        self.projector = projector
        self.store = store

    async def consume(self, envelope: ConversationSourceEnvelope) -> ConversationConsumerExecution:
        projected = self.projector.project(envelope)
        if projected is None:
            return ConversationConsumerExecution(
                ConversationConsumerReceiptState.SKIPPED,
                None,
                None,
                None,
            )
        stored = self.store.put(projected)
        return ConversationConsumerExecution(
            ConversationConsumerReceiptState.SUCCEEDED,
            stored,
            stored.projection_id,
            stored.content_digest,
        )

    async def completed(
        self,
        envelope: ConversationSourceEnvelope,
        receipt: ConversationConsumerReceipt,
    ) -> ConversationBehaviorProjectionBatch | None:
        if receipt.state is ConversationConsumerReceiptState.SKIPPED:
            return None
        assert receipt.result_id is not None
        stored = self.store.read(receipt.result_id)
        if stored is None:
            raise ConversationSourceError("successful projection receipt points to missing outbox data")
        if stored.source_id != envelope.source_id or stored.source_digest != envelope.content_digest:
            raise ConversationSourceError("successful projection receipt points to another source")
        if stored.content_digest != receipt.result_digest:
            raise ConversationSourceError("projection receipt digest differs from outbox data")
        return stored
