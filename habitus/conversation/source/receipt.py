"""Conversation Source Consumer 的输出引用、运行结果与首次耐久 Outcome。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from habitus.conversation.source.model import (
    ConversationSourceError,
    encode_durable_record,
    require_record,
    require_sha256,
    source_timestamp,
)
from habitus.foundation.integrity import canonical_digest, canonicalize
from habitus.infrastructure.store.filesystem import (
    ImmutableArtifactConflictError,
    atomic_create_bytes,
    read_regular_bytes,
)

OUTCOME_SCHEMA_VERSION = "conversation_consumer_outcome_v1"
OUTPUT_IDENTITY_SCHEMA = "conversation_consumer_output_identity_v1"


class ConversationSourceConsumer(str, Enum):
    MEMORY = "memory"
    BEHAVIOR_PROJECTION = "behavior_projection"


class ConversationConsumerOutcomeState(str, Enum):
    COMMITTED = "committed"
    SKIPPED = "skipped"


class ConversationConsumerRunDisposition(str, Enum):
    """一次 Consumer 执行的处置；复用既有 Output 不经过这里。

    既有 Output 的复用发生在执行之前——Inspector 判定为 OUTPUT_READY 后，交付层
    直接采用并建立 Outcome，根本不会调用 Consumer，因此不存在"复用"这种运行处置。
    """

    OUTPUT_WRITTEN = "output_written"
    SKIPPED = "skipped"


def _normalized(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or any(ord(character) < 32 for character in value)
    ):
        raise ConversationSourceError(f"{label} must be non-empty normalized text")
    return value


def conversation_consumer_output_id(
    *,
    source_id: str,
    source_payload_digest: str,
    consumer: ConversationSourceConsumer,
    processor_fingerprint: str,
    output_schema_version: str,
) -> str:
    """从 Source 与 Processor 契约直接计算耐久 Output 身份。"""

    return canonical_digest(
        {
            "schema_version": OUTPUT_IDENTITY_SCHEMA,
            "source_id": require_sha256(source_id, "source_id"),
            "source_payload_digest": require_sha256(source_payload_digest, "source_payload_digest"),
            "consumer": ConversationSourceConsumer(consumer).value,
            "processor_fingerprint": require_sha256(processor_fingerprint, "processor_fingerprint"),
            "output_schema_version": _normalized(output_schema_version, "output_schema_version"),
        }
    )


@dataclass(frozen=True)
class ConsumerOutputRef:
    output_kind: str
    output_id: str
    output_record_digest: str
    processor_fingerprint: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_kind", _normalized(self.output_kind, "output_kind"))
        require_sha256(self.output_id, "output_id")
        require_sha256(self.output_record_digest, "output_record_digest")
        require_sha256(self.processor_fingerprint, "processor_fingerprint")

    def to_dict(self) -> dict[str, str]:
        return {
            "output_kind": self.output_kind,
            "output_id": self.output_id,
            "output_record_digest": self.output_record_digest,
            "processor_fingerprint": self.processor_fingerprint,
        }

    @classmethod
    def from_dict(cls, value: object) -> ConsumerOutputRef:
        record = require_record(
            value,
            expected={
                "output_kind",
                "output_id",
                "output_record_digest",
                "processor_fingerprint",
            },
            label="consumer output reference",
        )
        return cls(
            output_kind=record["output_kind"],
            output_id=record["output_id"],
            output_record_digest=record["output_record_digest"],
            processor_fingerprint=record["processor_fingerprint"],
        )


@dataclass(frozen=True)
class ConversationConsumerRunResult:
    """仅描述本次调用；runtime_result 永不进入耐久 Outcome。"""

    disposition: ConversationConsumerRunDisposition
    output_ref: ConsumerOutputRef | None
    skip_reason: str | None
    runtime_result: object | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "disposition", ConversationConsumerRunDisposition(self.disposition))
        if self.disposition is ConversationConsumerRunDisposition.SKIPPED:
            if self.output_ref is not None or self.runtime_result is not None:
                raise ConversationSourceError("skipped run cannot contain output or runtime result")
            object.__setattr__(self, "skip_reason", _normalized(self.skip_reason, "skip_reason"))
        else:
            if not isinstance(self.output_ref, ConsumerOutputRef):
                raise ConversationSourceError("output run must contain ConsumerOutputRef")
            if self.skip_reason is not None:
                raise ConversationSourceError("output run cannot contain skip_reason")


@dataclass(frozen=True)
class ConversationConsumerOutcome:
    """固定在 (source_id, consumer) 上、首次原子创建的唯一耐久终态。"""

    source_id: str
    source_payload_digest: str
    consumer: ConversationSourceConsumer
    processor_fingerprint: str
    state: ConversationConsumerOutcomeState
    output_ref: ConsumerOutputRef | None
    skip_reason: str | None
    completed_at: datetime
    outcome_record_digest: str

    def __post_init__(self) -> None:
        require_sha256(self.source_id, "outcome source_id")
        require_sha256(self.source_payload_digest, "outcome source_payload_digest")
        object.__setattr__(self, "consumer", ConversationSourceConsumer(self.consumer))
        require_sha256(self.processor_fingerprint, "outcome processor_fingerprint")
        object.__setattr__(self, "state", ConversationConsumerOutcomeState(self.state))
        if self.state is ConversationConsumerOutcomeState.COMMITTED:
            if not isinstance(self.output_ref, ConsumerOutputRef):
                raise ConversationSourceError("committed outcome requires output_ref")
            if self.output_ref.processor_fingerprint != self.processor_fingerprint:
                raise ConversationSourceError("outcome and output processor fingerprints differ")
            if self.skip_reason is not None:
                raise ConversationSourceError("committed outcome cannot contain skip_reason")
        else:
            if self.output_ref is not None:
                raise ConversationSourceError("skipped outcome cannot reference output")
            object.__setattr__(self, "skip_reason", _normalized(self.skip_reason, "skip_reason"))
        object.__setattr__(self, "completed_at", source_timestamp(self.completed_at, "completed_at"))
        require_sha256(self.outcome_record_digest, "outcome_record_digest")
        if self.outcome_record_digest != canonical_digest(self._record_without_digest()):
            raise ConversationSourceError("outcome_record_digest does not match outcome record")

    @classmethod
    def create(
        cls,
        *,
        source_id: str,
        source_payload_digest: str,
        consumer: ConversationSourceConsumer,
        processor_fingerprint: str,
        state: ConversationConsumerOutcomeState,
        output_ref: ConsumerOutputRef | None,
        skip_reason: str | None,
        completed_at: datetime,
    ) -> ConversationConsumerOutcome:
        record = canonicalize(
            {
                "schema_version": OUTCOME_SCHEMA_VERSION,
                "source_id": source_id,
                "source_payload_digest": source_payload_digest,
                "consumer": ConversationSourceConsumer(consumer).value,
                "processor_fingerprint": processor_fingerprint,
                "state": ConversationConsumerOutcomeState(state).value,
                "output_ref": None if output_ref is None else output_ref.to_dict(),
                "skip_reason": skip_reason,
                "completed_at": source_timestamp(completed_at, "completed_at"),
            }
        )
        return cls(
            source_id=source_id,
            source_payload_digest=source_payload_digest,
            consumer=consumer,
            processor_fingerprint=processor_fingerprint,
            state=state,
            output_ref=output_ref,
            skip_reason=skip_reason,
            completed_at=completed_at,
            outcome_record_digest=canonical_digest(record),
        )

    def _record_without_digest(self) -> dict[str, Any]:
        return canonicalize(
            {
                "schema_version": OUTCOME_SCHEMA_VERSION,
                "source_id": self.source_id,
                "source_payload_digest": self.source_payload_digest,
                "consumer": self.consumer.value,
                "processor_fingerprint": self.processor_fingerprint,
                "state": self.state.value,
                "output_ref": None if self.output_ref is None else self.output_ref.to_dict(),
                "skip_reason": self.skip_reason,
                "completed_at": self.completed_at,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return canonicalize({**self._record_without_digest(), "outcome_record_digest": self.outcome_record_digest})

    @classmethod
    def from_dict(cls, value: object) -> ConversationConsumerOutcome:
        record = require_record(
            value,
            expected={
                "schema_version",
                "source_id",
                "source_payload_digest",
                "consumer",
                "processor_fingerprint",
                "state",
                "output_ref",
                "skip_reason",
                "completed_at",
                "outcome_record_digest",
            },
            label="consumer outcome",
            schema_version=OUTCOME_SCHEMA_VERSION,
        )
        try:
            consumer = ConversationSourceConsumer(record["consumer"])
            state = ConversationConsumerOutcomeState(record["state"])
        except (TypeError, ValueError) as exc:
            raise ConversationSourceError("consumer outcome state is invalid") from exc
        output_ref = None if record["output_ref"] is None else ConsumerOutputRef.from_dict(record["output_ref"])
        return cls(
            source_id=record["source_id"],
            source_payload_digest=record["source_payload_digest"],
            consumer=consumer,
            processor_fingerprint=record["processor_fingerprint"],
            state=state,
            output_ref=output_ref,
            skip_reason=record["skip_reason"],
            completed_at=source_timestamp(record["completed_at"], "completed_at"),
            outcome_record_digest=record["outcome_record_digest"],
        )


class ConversationConsumerOutcomeStore:
    """以 atomic create 固定每个 Source/Consumer 的第一份合法 Outcome。"""

    def __init__(self, conversation_root: str | Path, *, max_file_bytes: int) -> None:
        self.root = Path(conversation_root).expanduser().resolve(strict=False)
        if isinstance(max_file_bytes, bool) or not isinstance(max_file_bytes, int) or max_file_bytes <= 0:
            raise ValueError("max_file_bytes must be a positive integer")
        self.max_file_bytes = max_file_bytes
        self.outcome_root = self.root / "source" / "outcomes"

    def create_first(self, outcome: ConversationConsumerOutcome) -> ConversationConsumerOutcome:
        if not isinstance(outcome, ConversationConsumerOutcome):
            raise TypeError("outcome must be ConversationConsumerOutcome")
        try:
            atomic_create_bytes(
                self._path(outcome.source_id, outcome.consumer), self._encode(outcome), artifact_root=self.root
            )
        except ImmutableArtifactConflictError as exc:
            try:
                current = self.read(outcome.source_id, outcome.consumer)
            except Exception:
                raise ConversationSourceError("consumer outcome collides with an unreadable artifact") from exc
            if current is None:
                raise ConversationSourceError("consumer outcome disappeared after create conflict") from exc
            return current
        stored = self.read(outcome.source_id, outcome.consumer)
        if stored is None or stored.outcome_record_digest != outcome.outcome_record_digest:
            raise ConversationSourceError("consumer outcome was not durably read back")
        return stored

    def read(
        self, source_id: str, consumer: ConversationSourceConsumer
    ) -> ConversationConsumerOutcome | None:
        try:
            encoded = read_regular_bytes(
                self._path(source_id, consumer), artifact_root=self.root, max_bytes=self.max_file_bytes
            )
        except FileNotFoundError:
            return None
        try:
            outcome = ConversationConsumerOutcome.from_dict(json.loads(encoded))
        except (UnicodeDecodeError, json.JSONDecodeError, ConversationSourceError) as exc:
            raise ConversationSourceError("consumer outcome is corrupt") from exc
        if encoded != self._encode(outcome):
            raise ConversationSourceError("consumer outcome is not canonically encoded")
        if outcome.source_id != source_id or outcome.consumer is not ConversationSourceConsumer(consumer):
            raise ConversationSourceError("consumer outcome path does not match its identity")
        return outcome

    def _path(self, source_id: str, consumer: ConversationSourceConsumer) -> Path:
        require_sha256(source_id, "source_id")
        return self.outcome_root / source_id / f"{ConversationSourceConsumer(consumer).value}.json"

    def _encode(self, outcome: ConversationConsumerOutcome) -> bytes:
        return encode_durable_record(
            outcome.to_dict(), max_bytes=self.max_file_bytes, label="consumer outcome"
        )


__all__ = [
    "ConsumerOutputRef",
    "ConversationConsumerOutcome",
    "ConversationConsumerOutcomeState",
    "ConversationConsumerOutcomeStore",
    "ConversationConsumerRunDisposition",
    "ConversationConsumerRunResult",
    "ConversationSourceConsumer",
    "conversation_consumer_output_id",
]
