"""Adapter 输出之后、任何 Consumer 处理之前的不可变来源事实。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any

from foundation.integrity import canonical_digest, canonicalize
from pre.conversation import ConversationBatch

_SCHEMA = "conversation_source_envelope_v1"
_IDENTITY_SCHEMA = "conversation_source_identity_v1"


class ConversationSourceError(ValueError):
    """Conversation Source 身份、内容或耐久文件不满足契约。"""


def _sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ConversationSourceError(f"{label} must be lowercase SHA-256 text")
    return value


def _protocol(value: object) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ConversationSourceError("protocol must be non-empty normalized text")
    if any(ord(character) < 32 for character in value):
        raise ConversationSourceError("protocol contains control characters")
    return value


def _started_on(value: date | str) -> date:
    if isinstance(value, datetime):
        raise ConversationSourceError("started_on must be a calendar date")
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError as exc:
            raise ConversationSourceError("started_on must be an ISO calendar date") from exc
    raise ConversationSourceError("started_on must be a calendar date")


def _timestamp(value: datetime | str, label: str) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ConversationSourceError(f"{label} must be ISO-8601") from exc
    else:
        raise ConversationSourceError(f"{label} must be a datetime")
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ConversationSourceError(f"{label} must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _omit_tool_call_ids(value: object) -> frozenset[str]:
    if not isinstance(value, frozenset | set | tuple | list):
        raise ConversationSourceError("omit_tool_call_ids must be a collection")
    resolved = frozenset(value)
    if any(
        not isinstance(identifier, str)
        or not identifier
        or identifier != identifier.strip()
        or any(ord(character) < 32 for character in identifier)
        for identifier in resolved
    ):
        raise ConversationSourceError("omit_tool_call_ids must contain normalized non-empty strings")
    return resolved


def conversation_source_request_digest(
    *,
    conversation_id: str,
    started_on: date,
    protocol: str,
    batch: ConversationBatch,
    after_turn: bool,
    omit_tool_call_ids: frozenset[str],
) -> str:
    """对一次规范来源请求生成与创建时间无关的稳定摘要。"""

    if not isinstance(batch, ConversationBatch):
        raise TypeError("batch must be ConversationBatch")
    if batch.conversation_id != conversation_id:
        raise ConversationSourceError("batch belongs to another conversation")
    resolved_date = _started_on(started_on)
    resolved_protocol = _protocol(protocol)
    if not isinstance(after_turn, bool):
        raise TypeError("after_turn must be boolean")
    omitted = _omit_tool_call_ids(omit_tool_call_ids)
    return canonical_digest(
        {
            "conversation_id": conversation_id,
            "started_on": resolved_date.isoformat(),
            "protocol": resolved_protocol,
            "batch": batch.to_dict(),
            "after_turn": after_turn,
            "omit_tool_call_ids": sorted(omitted),
        }
    )


def _source_content_payload(
    *,
    source_id: str,
    conversation_id: str,
    started_on: date,
    protocol: str,
    batch: ConversationBatch,
    after_turn: bool,
    omit_tool_call_ids: frozenset[str],
    delivery_id: str,
    request_digest: str,
) -> dict[str, Any]:
    return canonicalize(
        {
            "schema_version": _SCHEMA,
            "source_id": source_id,
            "conversation_id": conversation_id,
            "started_on": started_on.isoformat(),
            "protocol": protocol,
            "batch": batch.to_dict(),
            "after_turn": after_turn,
            "omit_tool_call_ids": sorted(omit_tool_call_ids),
            "delivery_id": delivery_id,
            "request_digest": request_digest,
        }
    )


@dataclass(frozen=True)
class ConversationSourceEnvelope:
    """两个 Consumer 共同读取、永不原地修改的规范 Conversation 来源。"""

    source_id: str
    conversation_id: str
    started_on: date
    protocol: str
    batch: ConversationBatch
    after_turn: bool
    omit_tool_call_ids: frozenset[str]
    delivery_id: str
    request_digest: str
    created_at: datetime
    content_digest: str

    def __post_init__(self) -> None:
        _sha256(self.source_id, "source_id")
        if not isinstance(self.batch, ConversationBatch):
            raise TypeError("batch must be ConversationBatch")
        if self.batch.conversation_id != self.conversation_id:
            raise ConversationSourceError("batch belongs to another conversation")
        object.__setattr__(self, "started_on", _started_on(self.started_on))
        object.__setattr__(self, "protocol", _protocol(self.protocol))
        if not isinstance(self.after_turn, bool):
            raise TypeError("after_turn must be boolean")
        object.__setattr__(self, "omit_tool_call_ids", _omit_tool_call_ids(self.omit_tool_call_ids))
        _sha256(self.delivery_id, "delivery_id")
        _sha256(self.request_digest, "request_digest")
        object.__setattr__(self, "created_at", _timestamp(self.created_at, "created_at"))
        _sha256(self.content_digest, "content_digest")
        expected_source_id = self.source_identity(
            self.conversation_id,
            self.started_on,
            self.delivery_id,
        )
        if self.source_id != expected_source_id:
            raise ConversationSourceError("source_id does not match Conversation delivery identity")
        if self.content_digest != canonical_digest(self._content_payload()):
            raise ConversationSourceError("content_digest does not match source content")

    @classmethod
    def create(
        cls,
        *,
        conversation_id: str,
        started_on: date,
        protocol: str,
        batch: ConversationBatch,
        after_turn: bool,
        omit_tool_call_ids: frozenset[str],
        delivery_id: str,
        request_digest: str,
        created_at: datetime,
    ) -> ConversationSourceEnvelope:
        if not isinstance(batch, ConversationBatch):
            raise TypeError("batch must be ConversationBatch")
        if not isinstance(after_turn, bool):
            raise TypeError("after_turn must be boolean")
        resolved_date = _started_on(started_on)
        resolved_protocol = _protocol(protocol)
        resolved_omitted = _omit_tool_call_ids(omit_tool_call_ids)
        resolved_delivery = _sha256(delivery_id, "delivery_id")
        resolved_request = _sha256(request_digest, "request_digest")
        source_id = cls.source_identity(conversation_id, resolved_date, resolved_delivery)
        content_digest = canonical_digest(
            _source_content_payload(
                source_id=source_id,
                conversation_id=conversation_id,
                started_on=resolved_date,
                protocol=resolved_protocol,
                batch=batch,
                after_turn=after_turn,
                omit_tool_call_ids=resolved_omitted,
                delivery_id=resolved_delivery,
                request_digest=resolved_request,
            )
        )
        return cls(
            source_id=source_id,
            conversation_id=conversation_id,
            started_on=resolved_date,
            protocol=resolved_protocol,
            batch=batch,
            after_turn=after_turn,
            omit_tool_call_ids=resolved_omitted,
            delivery_id=resolved_delivery,
            request_digest=resolved_request,
            created_at=_timestamp(created_at, "created_at"),
            content_digest=content_digest,
        )

    @staticmethod
    def source_identity(conversation_id: str, started_on: date, delivery_id: str) -> str:
        if not isinstance(conversation_id, str) or not conversation_id:
            raise ConversationSourceError("conversation_id must be non-empty text")
        return canonical_digest(
            {
                "schema_version": _IDENTITY_SCHEMA,
                "conversation_id": conversation_id,
                "started_on": _started_on(started_on).isoformat(),
                "delivery_id": _sha256(delivery_id, "delivery_id"),
            }
        )

    def _content_payload(self) -> dict[str, Any]:
        return _source_content_payload(
            source_id=self.source_id,
            conversation_id=self.conversation_id,
            started_on=self.started_on,
            protocol=self.protocol,
            batch=self.batch,
            after_turn=self.after_turn,
            omit_tool_call_ids=self.omit_tool_call_ids,
            delivery_id=self.delivery_id,
            request_digest=self.request_digest,
        )

    def to_dict(self) -> dict[str, Any]:
        return canonicalize(
            {
                **self._content_payload(),
                "created_at": self.created_at,
                "content_digest": self.content_digest,
            }
        )

    @classmethod
    def from_dict(cls, value: object) -> ConversationSourceEnvelope:
        if not isinstance(value, Mapping):
            raise ConversationSourceError("source envelope must be an object")
        expected = {
            "schema_version",
            "source_id",
            "conversation_id",
            "started_on",
            "protocol",
            "batch",
            "after_turn",
            "omit_tool_call_ids",
            "delivery_id",
            "request_digest",
            "created_at",
            "content_digest",
        }
        if set(value) != expected or value.get("schema_version") != _SCHEMA:
            raise ConversationSourceError("source envelope schema is invalid")
        batch_value = value["batch"]
        if not isinstance(batch_value, Mapping):
            raise ConversationSourceError("source envelope batch must be an object")
        return cls(
            source_id=value["source_id"],
            conversation_id=value["conversation_id"],
            started_on=_started_on(value["started_on"]),
            protocol=value["protocol"],
            batch=ConversationBatch.from_dict(batch_value),
            after_turn=value["after_turn"],
            omit_tool_call_ids=_omit_tool_call_ids(value["omit_tool_call_ids"]),
            delivery_id=value["delivery_id"],
            request_digest=value["request_digest"],
            created_at=_timestamp(value["created_at"], "created_at"),
            content_digest=value["content_digest"],
        )
