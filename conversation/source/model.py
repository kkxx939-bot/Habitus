"""Adapter 输出之后、任何 Consumer 处理之前的不可变来源事实。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any

from foundation.integrity import canonical_digest, canonical_json, canonicalize
from pre.conversation import ConversationBatch

SOURCE_SCHEMA_VERSION = "conversation_source_envelope_v2"
_IDENTITY_SCHEMA = "conversation_source_identity_v1"


class ConversationSourceError(ValueError):
    """Conversation Source 身份、内容或耐久文件不满足契约。"""


def require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ConversationSourceError(f"{label} must be lowercase SHA-256 text")
    return value


def require_record(
    value: object,
    *,
    expected: set[str],
    label: str,
    schema_version: str | None = None,
) -> Mapping[str, Any]:
    """解码耐久记录前的统一守卫：必须是对象、键集合完全相等、Schema 版本吻合。

    键集合用相等而不是包含关系判断，未知字段和缺失字段一律拒绝；这一条是
    所有 Source、Outcome 与 Consumer Output 记录共享的严格解码前提，集中在
    这里可以避免每个 ``from_dict`` 各写一份而逐渐漂移。
    """

    if not isinstance(value, Mapping) or set(value) != expected:
        raise ConversationSourceError(f"{label} schema is invalid")
    if schema_version is not None and value.get("schema_version") != schema_version:
        raise ConversationSourceError(f"{label} schema version is invalid")
    return value


def encode_durable_record(record: Mapping[str, Any], *, max_bytes: int, label: str) -> bytes:
    """把一条记录编码成带换行的规范 UTF-8 JSON，并执行统一的文件上限检查。

    读路径会用同一函数重新编码并与磁盘内容逐字节比对，规范编码因此必须只有
    这一个实现；容量上限由各 Store 从 Config 传入，不在此处设默认值。
    """

    encoded = (canonical_json(record) + "\n").encode("utf-8")
    if len(encoded) > max_bytes:
        raise ConversationSourceError(f"{label} exceeds its configured file bound")
    return encoded


def source_timestamp(value: datetime | str, label: str) -> datetime:
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
    """保持 Journal Ingress 已有公式不变的规范请求摘要。"""

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


def _source_payload(
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
            "schema_version": SOURCE_SCHEMA_VERSION,
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
    source_payload_digest: str
    recorded_at: datetime
    source_record_digest: str

    def __post_init__(self) -> None:
        require_sha256(self.source_id, "source_id")
        if not isinstance(self.conversation_id, str) or not self.conversation_id:
            raise ConversationSourceError("conversation_id must be non-empty text")
        if not isinstance(self.batch, ConversationBatch):
            raise TypeError("batch must be ConversationBatch")
        if self.batch.conversation_id != self.conversation_id:
            raise ConversationSourceError("batch belongs to another conversation")
        object.__setattr__(self, "started_on", _started_on(self.started_on))
        object.__setattr__(self, "protocol", _protocol(self.protocol))
        if not isinstance(self.after_turn, bool):
            raise TypeError("after_turn must be boolean")
        object.__setattr__(self, "omit_tool_call_ids", _omit_tool_call_ids(self.omit_tool_call_ids))
        require_sha256(self.delivery_id, "delivery_id")
        require_sha256(self.request_digest, "request_digest")
        expected_request_digest = conversation_source_request_digest(
            conversation_id=self.conversation_id,
            started_on=self.started_on,
            protocol=self.protocol,
            batch=self.batch,
            after_turn=self.after_turn,
            omit_tool_call_ids=self.omit_tool_call_ids,
        )
        if self.request_digest != expected_request_digest:
            raise ConversationSourceError("request_digest does not match the canonical source request")
        require_sha256(self.source_payload_digest, "source_payload_digest")
        object.__setattr__(self, "recorded_at", source_timestamp(self.recorded_at, "recorded_at"))
        require_sha256(self.source_record_digest, "source_record_digest")
        if self.source_id != self.source_identity(self.conversation_id, self.started_on, self.delivery_id):
            raise ConversationSourceError("source_id does not match Conversation delivery identity")
        if self.source_payload_digest != canonical_digest(self._payload()):
            raise ConversationSourceError("source_payload_digest does not match source payload")
        if self.source_record_digest != canonical_digest(self._record_without_digest()):
            raise ConversationSourceError("source_record_digest does not match source record")

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
        recorded_at: datetime,
    ) -> ConversationSourceEnvelope:
        resolved_date = _started_on(started_on)
        resolved_protocol = _protocol(protocol)
        resolved_omitted = _omit_tool_call_ids(omit_tool_call_ids)
        resolved_delivery = require_sha256(delivery_id, "delivery_id")
        resolved_request = require_sha256(request_digest, "request_digest")
        resolved_recorded_at = source_timestamp(recorded_at, "recorded_at")
        source_id = cls.source_identity(conversation_id, resolved_date, resolved_delivery)
        payload = _source_payload(
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
        payload_digest = canonical_digest(payload)
        record = canonicalize(
            {**payload, "source_payload_digest": payload_digest, "recorded_at": resolved_recorded_at}
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
            source_payload_digest=payload_digest,
            recorded_at=resolved_recorded_at,
            source_record_digest=canonical_digest(record),
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
                "delivery_id": require_sha256(delivery_id, "delivery_id"),
            }
        )

    def _payload(self) -> dict[str, Any]:
        return _source_payload(
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

    def _record_without_digest(self) -> dict[str, Any]:
        return canonicalize(
            {
                **self._payload(),
                "source_payload_digest": self.source_payload_digest,
                "recorded_at": self.recorded_at,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return canonicalize({**self._record_without_digest(), "source_record_digest": self.source_record_digest})

    @classmethod
    def from_dict(cls, value: object) -> ConversationSourceEnvelope:
        record = require_record(
            value,
            expected={
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
                "source_payload_digest",
                "recorded_at",
                "source_record_digest",
            },
            label="source envelope",
            schema_version=SOURCE_SCHEMA_VERSION,
        )
        batch_value = record["batch"]
        if not isinstance(batch_value, Mapping):
            raise ConversationSourceError("source envelope batch must be an object")
        return cls(
            source_id=record["source_id"],
            conversation_id=record["conversation_id"],
            started_on=_started_on(record["started_on"]),
            protocol=record["protocol"],
            batch=ConversationBatch.from_dict(batch_value),
            after_turn=record["after_turn"],
            omit_tool_call_ids=_omit_tool_call_ids(record["omit_tool_call_ids"]),
            delivery_id=record["delivery_id"],
            request_digest=record["request_digest"],
            source_payload_digest=record["source_payload_digest"],
            recorded_at=source_timestamp(record["recorded_at"], "recorded_at"),
            source_record_digest=record["source_record_digest"],
        )


__all__ = [
    "ConversationSourceEnvelope",
    "ConversationSourceError",
    "SOURCE_SCHEMA_VERSION",
    "conversation_source_request_digest",
    "encode_durable_record",
    "require_record",
    "require_sha256",
    "source_timestamp",
]
