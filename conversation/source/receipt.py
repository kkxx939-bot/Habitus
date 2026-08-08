"""Conversation Source 两个 Consumer 的独立终态回执。"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from conversation.source.model import ConversationSourceError
from foundation.integrity import canonical_digest, canonical_json, canonicalize
from infrastructure.store.filesystem import ImmutableArtifactConflictError, atomic_create_bytes, read_regular_bytes

_SCHEMA = "conversation_consumer_receipt_v1"


class ConversationSourceConsumer(str, Enum):
    MEMORY = "memory"
    BEHAVIOR_PROJECTION = "behavior_projection"


class ConversationConsumerReceiptState(str, Enum):
    SUCCEEDED = "succeeded"
    SKIPPED = "skipped"


def _sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ConversationSourceError(f"{label} must be lowercase SHA-256 text")
    return value


def _time(value: datetime | str) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ConversationSourceError("receipt completed_at must be ISO-8601") from exc
    else:
        raise ConversationSourceError("receipt completed_at must be a datetime")
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ConversationSourceError("receipt completed_at must be timezone-aware")
    return parsed.astimezone(timezone.utc)


@dataclass(frozen=True)
class ConversationConsumerReceipt:
    source_id: str
    source_digest: str
    consumer: ConversationSourceConsumer
    state: ConversationConsumerReceiptState
    result_id: str | None
    result_digest: str | None
    completed_at: datetime
    content_digest: str

    def __post_init__(self) -> None:
        _sha256(self.source_id, "receipt source_id")
        _sha256(self.source_digest, "receipt source_digest")
        try:
            object.__setattr__(self, "consumer", ConversationSourceConsumer(self.consumer))
            object.__setattr__(self, "state", ConversationConsumerReceiptState(self.state))
        except ValueError as exc:
            raise ConversationSourceError("receipt consumer or state is invalid") from exc
        if self.state is ConversationConsumerReceiptState.SUCCEEDED:
            _sha256(self.result_id, "receipt result_id")
            _sha256(self.result_digest, "receipt result_digest")
        elif self.result_id is not None or self.result_digest is not None:
            raise ConversationSourceError("skipped receipt cannot reference a result")
        object.__setattr__(self, "completed_at", _time(self.completed_at))
        _sha256(self.content_digest, "receipt content_digest")
        if self.content_digest != canonical_digest(self._content_payload()):
            raise ConversationSourceError("receipt content_digest does not match content")

    @classmethod
    def create(
        cls,
        *,
        source_id: str,
        source_digest: str,
        consumer: ConversationSourceConsumer,
        state: ConversationConsumerReceiptState,
        result_id: str | None,
        result_digest: str | None,
        completed_at: datetime,
    ) -> ConversationConsumerReceipt:
        content = {
            "schema_version": _SCHEMA,
            "source_id": source_id,
            "source_digest": source_digest,
            "consumer": consumer.value,
            "state": state.value,
            "result_id": result_id,
            "result_digest": result_digest,
        }
        return cls(
            source_id=source_id,
            source_digest=source_digest,
            consumer=consumer,
            state=state,
            result_id=result_id,
            result_digest=result_digest,
            completed_at=completed_at,
            content_digest=canonical_digest(content),
        )

    def _content_payload(self) -> dict[str, Any]:
        return canonicalize(
            {
                "schema_version": _SCHEMA,
                "source_id": self.source_id,
                "source_digest": self.source_digest,
                "consumer": self.consumer.value,
                "state": self.state.value,
                "result_id": self.result_id,
                "result_digest": self.result_digest,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return canonicalize(
            {
                **self._content_payload(),
                "completed_at": self.completed_at,
                "content_digest": self.content_digest,
            }
        )

    @classmethod
    def from_dict(cls, value: object) -> ConversationConsumerReceipt:
        if not isinstance(value, Mapping):
            raise ConversationSourceError("consumer receipt must be an object")
        expected = {
            "schema_version",
            "source_id",
            "source_digest",
            "consumer",
            "state",
            "result_id",
            "result_digest",
            "completed_at",
            "content_digest",
        }
        if set(value) != expected or value.get("schema_version") != _SCHEMA:
            raise ConversationSourceError("consumer receipt schema is invalid")
        try:
            consumer = ConversationSourceConsumer(value["consumer"])
            state = ConversationConsumerReceiptState(value["state"])
        except (TypeError, ValueError) as exc:
            raise ConversationSourceError("consumer receipt state is invalid") from exc
        return cls(
            source_id=value["source_id"],
            source_digest=value["source_digest"],
            consumer=consumer,
            state=state,
            result_id=value["result_id"],
            result_digest=value["result_digest"],
            completed_at=_time(value["completed_at"]),
            content_digest=value["content_digest"],
        )


class ConversationSourceReceiptStore:
    """每个 Source/Consumer 只允许创建一张不可变终态回执。"""

    def __init__(self, conversation_root: str | Path, *, max_file_bytes: int) -> None:
        self.root = Path(conversation_root).expanduser().resolve(strict=False)
        if isinstance(max_file_bytes, bool) or not isinstance(max_file_bytes, int) or max_file_bytes <= 0:
            raise ValueError("max_file_bytes must be a positive integer")
        self.max_file_bytes = max_file_bytes
        self.source_root = self.root / "source"

    def put(self, receipt: ConversationConsumerReceipt) -> ConversationConsumerReceipt:
        if not isinstance(receipt, ConversationConsumerReceipt):
            raise TypeError("receipt must be ConversationConsumerReceipt")
        encoded = self._encode(receipt)
        try:
            atomic_create_bytes(self._path(receipt.source_id, receipt.consumer), encoded, artifact_root=self.root)
        except ImmutableArtifactConflictError as exc:
            try:
                current = self.read(receipt.source_id, receipt.consumer)
            except Exception:
                raise ConversationSourceError("consumer receipt collides with an unreadable artifact") from exc
            if current is None or current.content_digest != receipt.content_digest:
                raise ConversationSourceError("consumer receipt conflicts with a different terminal result") from exc
            return current
        stored = self.read(receipt.source_id, receipt.consumer)
        if stored is None or stored.content_digest != receipt.content_digest:
            raise ConversationSourceError("consumer receipt was not durably read back")
        return stored

    def read(
        self,
        source_id: str,
        consumer: ConversationSourceConsumer,
    ) -> ConversationConsumerReceipt | None:
        path = self._path(source_id, consumer)
        try:
            encoded = read_regular_bytes(path, artifact_root=self.root, max_bytes=self.max_file_bytes)
        except FileNotFoundError:
            return None
        try:
            receipt = ConversationConsumerReceipt.from_dict(json.loads(encoded))
        except (UnicodeDecodeError, json.JSONDecodeError, ConversationSourceError) as exc:
            raise ConversationSourceError("consumer receipt is corrupt") from exc
        if encoded != self._encode(receipt):
            raise ConversationSourceError("consumer receipt is not canonically encoded")
        if receipt.source_id != source_id or receipt.consumer is not ConversationSourceConsumer(consumer):
            raise ConversationSourceError("consumer receipt path does not match its identity")
        return receipt

    def _path(self, source_id: str, consumer: ConversationSourceConsumer) -> Path:
        _sha256(source_id, "source_id")
        try:
            resolved_consumer = ConversationSourceConsumer(consumer)
        except ValueError as exc:
            raise ConversationSourceError("unsupported source consumer") from exc
        return self.source_root / source_id / f"{resolved_consumer.value}.json"

    def _encode(self, receipt: ConversationConsumerReceipt) -> bytes:
        encoded = (canonical_json(receipt.to_dict()) + "\n").encode("utf-8")
        if len(encoded) > self.max_file_bytes:
            raise ConversationSourceError("consumer receipt exceeds its configured file bound")
        return encoded
