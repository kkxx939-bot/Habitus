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

from conversation.source.fence import ConversationConsumerExecutionLease
from conversation.source.model import ConversationSourceEnvelope, ConversationSourceError, require_sha256
from conversation.source.receipt import (
    ConsumerOutputRef,
    ConversationConsumerRunDisposition,
    ConversationConsumerRunResult,
    ConversationSourceConsumer,
    conversation_consumer_output_id,
)
from foundation.integrity import canonical_digest, canonical_json, canonicalize, immutable_snapshot
from infrastructure.store.filesystem import (
    DurablePathIntegrityError,
    ImmutableArtifactConflictError,
    atomic_create_bytes,
    atomic_temporary_destination,
    list_real_directory,
    read_regular_bytes,
)
from pre.conversation import ConversationMessage, ConversationMessageRole
from pre.conversation.messages.model import conversation_datetime

# TODO(conversation-source): 修改 _project_message 映射或 payload 摘要语义时，必须同步提升
# Projector 版本；该版本当前由人工维护。
CONVERSATION_BEHAVIOR_PROJECTOR_VERSION = "conversation_behavior_projector_v1"
BEHAVIOR_PROJECTION_OUTPUT_SCHEMA_VERSION = "conversation_behavior_projection_output_v2"
BEHAVIOR_PROJECTION_OUTPUT_KIND = "conversation_behavior_projection"
_PROJECTION_ID_SCHEMA = "conversation_behavior_projection_identity_v1"
_ITEM_SCHEMA = "conversation_behavior_projection_item_v1"
_OUTPUT_FILE = re.compile(r"^(?P<output_id>[0-9a-f]{64})\.json$")


class ConversationBehaviorProjectionKind(str, Enum):
    USER_CONVERSATION_INPUT = "user_conversation_input"
    AGENT_TOOL_CALL = "agent_tool_call"
    TOOL_EXECUTION_RESULT = "tool_execution_result"


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
        require_sha256(self.projection_item_id, "projection_item_id")
        require_sha256(self.source_id, "projection item source_id")
        if not isinstance(self.source_message_id, str) or not self.source_message_id:
            raise ConversationSourceError("source_message_id must be non-empty text")
        require_sha256(self.source_message_digest, "source_message_digest")
        object.__setattr__(self, "occurred_at", conversation_datetime(self.occurred_at, "occurred_at"))
        try:
            object.__setattr__(self, "projection_kind", ConversationBehaviorProjectionKind(self.projection_kind))
        except ValueError as exc:
            raise ConversationSourceError("projection_kind is invalid") from exc
        object.__setattr__(self, "payload", immutable_snapshot(canonicalize(self.payload)))
        if self.projection_item_id != canonical_digest(self._identity_payload()):
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
        require_sha256(source_id, "source_id")
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
        return canonicalize(
            {
                "schema_version": _ITEM_SCHEMA,
                "source_id": self.source_id,
                "source_message_id": self.source_message_id,
                "source_message_digest": self.source_message_digest,
                "occurred_at": self.occurred_at,
                "projection_kind": self.projection_kind.value,
                "payload": canonicalize(self.payload),
            }
        )

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
        cls, value: object, *, source_id: str
    ) -> ConversationBehaviorProjectionItem:
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
        if not isinstance(value, Mapping) or set(value) != expected or value.get("schema_version") != _ITEM_SCHEMA:
            raise ConversationSourceError("projection item schema is invalid")
        try:
            kind = ConversationBehaviorProjectionKind(value["projection_kind"])
        except (TypeError, ValueError) as exc:
            raise ConversationSourceError("projection item kind is invalid") from exc
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
    output_id: str
    projection_id: str
    source_id: str
    source_payload_digest: str
    consumer: ConversationSourceConsumer
    processor_fingerprint: str
    projector_version: str
    items: tuple[ConversationBehaviorProjectionItem, ...]
    recorded_at: datetime
    output_record_digest: str

    def __post_init__(self) -> None:
        for label, value in (
            ("projection output_id", self.output_id),
            ("projection_id", self.projection_id),
            ("projection source_id", self.source_id),
            ("projection source_payload_digest", self.source_payload_digest),
            ("projection processor_fingerprint", self.processor_fingerprint),
            ("projection output_record_digest", self.output_record_digest),
        ):
            require_sha256(value, label)
        object.__setattr__(self, "consumer", ConversationSourceConsumer(self.consumer))
        if self.consumer is not ConversationSourceConsumer.BEHAVIOR_PROJECTION:
            raise ConversationSourceError("projection output has the wrong consumer")
        if not isinstance(self.projector_version, str) or not self.projector_version:
            raise ConversationSourceError("projector_version must be non-empty text")
        if not isinstance(self.items, tuple) or not self.items:
            raise ConversationSourceError("projection output must contain items")
        if any(not isinstance(item, ConversationBehaviorProjectionItem) for item in self.items):
            raise TypeError("projection items must be ConversationBehaviorProjectionItem values")
        if any(item.source_id != self.source_id for item in self.items):
            raise ConversationSourceError("projection output contains an item from another source")
        if len({item.projection_item_id for item in self.items}) != len(self.items):
            raise ConversationSourceError("projection item IDs must be unique")
        object.__setattr__(self, "recorded_at", conversation_datetime(self.recorded_at, "recorded_at"))
        if self.output_id != conversation_consumer_output_id(
            source_id=self.source_id,
            source_payload_digest=self.source_payload_digest,
            consumer=self.consumer,
            processor_fingerprint=self.processor_fingerprint,
            output_schema_version=BEHAVIOR_PROJECTION_OUTPUT_SCHEMA_VERSION,
        ):
            raise ConversationSourceError("projection output_id does not match output identity")
        if self.projection_id != canonical_digest(self._projection_identity_payload()):
            raise ConversationSourceError("projection_id does not match projection content")
        if self.output_record_digest != canonical_digest(self._record_without_digest()):
            raise ConversationSourceError("projection output_record_digest does not match output record")

    @classmethod
    def create(
        cls,
        *,
        source: ConversationSourceEnvelope,
        processor_fingerprint: str,
        projector_version: str,
        items: tuple[ConversationBehaviorProjectionItem, ...],
        recorded_at: datetime,
    ) -> ConversationBehaviorProjectionBatch:
        output_id = conversation_consumer_output_id(
            source_id=source.source_id,
            source_payload_digest=source.source_payload_digest,
            consumer=ConversationSourceConsumer.BEHAVIOR_PROJECTION,
            processor_fingerprint=processor_fingerprint,
            output_schema_version=BEHAVIOR_PROJECTION_OUTPUT_SCHEMA_VERSION,
        )
        identity = canonicalize(
            {
                "schema_version": _PROJECTION_ID_SCHEMA,
                "source_id": source.source_id,
                "source_payload_digest": source.source_payload_digest,
                "projector_version": projector_version,
                "items": [item.to_dict() for item in items],
            }
        )
        projection_id = canonical_digest(identity)
        record = canonicalize(
            {
                "schema_version": BEHAVIOR_PROJECTION_OUTPUT_SCHEMA_VERSION,
                "output_id": output_id,
                "projection_id": projection_id,
                "source_id": source.source_id,
                "source_payload_digest": source.source_payload_digest,
                "consumer": ConversationSourceConsumer.BEHAVIOR_PROJECTION.value,
                "processor_fingerprint": processor_fingerprint,
                "projector_version": projector_version,
                "items": [item.to_dict() for item in items],
                "recorded_at": conversation_datetime(recorded_at, "recorded_at"),
            }
        )
        return cls(
            output_id=output_id,
            projection_id=projection_id,
            source_id=source.source_id,
            source_payload_digest=source.source_payload_digest,
            consumer=ConversationSourceConsumer.BEHAVIOR_PROJECTION,
            processor_fingerprint=processor_fingerprint,
            projector_version=projector_version,
            items=items,
            recorded_at=recorded_at,
            output_record_digest=canonical_digest(record),
        )

    def _projection_identity_payload(self) -> dict[str, Any]:
        return canonicalize(
            {
                "schema_version": _PROJECTION_ID_SCHEMA,
                "source_id": self.source_id,
                "source_payload_digest": self.source_payload_digest,
                "projector_version": self.projector_version,
                "items": [item.to_dict() for item in self.items],
            }
        )

    def _record_without_digest(self) -> dict[str, Any]:
        return canonicalize(
            {
                "schema_version": BEHAVIOR_PROJECTION_OUTPUT_SCHEMA_VERSION,
                "output_id": self.output_id,
                "projection_id": self.projection_id,
                "source_id": self.source_id,
                "source_payload_digest": self.source_payload_digest,
                "consumer": self.consumer.value,
                "processor_fingerprint": self.processor_fingerprint,
                "projector_version": self.projector_version,
                "items": [item.to_dict() for item in self.items],
                "recorded_at": self.recorded_at,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return canonicalize({**self._record_without_digest(), "output_record_digest": self.output_record_digest})

    @classmethod
    def from_dict(cls, value: object) -> ConversationBehaviorProjectionBatch:
        expected = {
            "schema_version",
            "output_id",
            "projection_id",
            "source_id",
            "source_payload_digest",
            "consumer",
            "processor_fingerprint",
            "projector_version",
            "items",
            "recorded_at",
            "output_record_digest",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ConversationSourceError("projection output schema is invalid")
        if value.get("schema_version") != BEHAVIOR_PROJECTION_OUTPUT_SCHEMA_VERSION:
            raise ConversationSourceError("projection output schema version is invalid")
        raw_items = value["items"]
        if not isinstance(raw_items, list):
            raise ConversationSourceError("projection output items must be a list")
        try:
            consumer = ConversationSourceConsumer(value["consumer"])
        except (TypeError, ValueError) as exc:
            raise ConversationSourceError("projection output consumer is invalid") from exc
        return cls(
            output_id=value["output_id"],
            projection_id=value["projection_id"],
            source_id=value["source_id"],
            source_payload_digest=value["source_payload_digest"],
            consumer=consumer,
            processor_fingerprint=value["processor_fingerprint"],
            projector_version=value["projector_version"],
            items=tuple(
                ConversationBehaviorProjectionItem.from_dict(item, source_id=value["source_id"])
                for item in raw_items
            ),
            recorded_at=conversation_datetime(value["recorded_at"], "recorded_at"),
            output_record_digest=value["output_record_digest"],
        )


class ConversationBehaviorProjector:
    """只读取 SourceEnvelope.batch；COMPLETION 在第一版明确跳过。"""

    def __init__(self, *, clock: Callable[[], datetime] | None = None) -> None:
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.projector_version = CONVERSATION_BEHAVIOR_PROJECTOR_VERSION
        self.processor_fingerprint = canonical_digest(
            {
                "schema_version": "conversation_behavior_processor_fingerprint_v1",
                "projector_version": self.projector_version,
                "output_schema_version": BEHAVIOR_PROJECTION_OUTPUT_SCHEMA_VERSION,
                "mappings": {
                    "prompt": ConversationBehaviorProjectionKind.USER_CONVERSATION_INPUT.value,
                    "tool_call": ConversationBehaviorProjectionKind.AGENT_TOOL_CALL.value,
                    "tool_result": ConversationBehaviorProjectionKind.TOOL_EXECUTION_RESULT.value,
                    "completion": "skip",
                },
            }
        )

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
            source=envelope,
            processor_fingerprint=self.processor_fingerprint,
            projector_version=self.projector_version,
            items=items,
            recorded_at=self.clock(),
        )

    def _project_message(
        self, source_id: str, message: ConversationMessage
    ) -> ConversationBehaviorProjectionItem | None:
        # TODO(conversation-source): 修改此映射时，同步提升 CONVERSATION_BEHAVIOR_PROJECTOR_VERSION。
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
    """统一 Output 路径中的不可变 Behavior Projection Outbox。"""

    consumer = ConversationSourceConsumer.BEHAVIOR_PROJECTION

    def __init__(
        self,
        conversation_root: str | Path,
        *,
        max_files_per_source: int,
        max_file_bytes: int,
        max_items: int,
    ) -> None:
        self.root = Path(conversation_root).expanduser().resolve(strict=False)
        for name, value in (
            ("max_files_per_source", max_files_per_source),
            ("max_file_bytes", max_file_bytes),
            ("max_items", max_items),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self.max_files_per_source = max_files_per_source
        self.max_file_bytes = max_file_bytes
        self.max_items = max_items

    def expected_output_id(self, source: ConversationSourceEnvelope, processor_fingerprint: str) -> str:
        return conversation_consumer_output_id(
            source_id=source.source_id,
            source_payload_digest=source.source_payload_digest,
            consumer=self.consumer,
            processor_fingerprint=processor_fingerprint,
            output_schema_version=BEHAVIOR_PROJECTION_OUTPUT_SCHEMA_VERSION,
        )

    def put(
        self,
        source: ConversationSourceEnvelope,
        batch: ConversationBehaviorProjectionBatch,
    ) -> ConversationBehaviorProjectionBatch:
        if not isinstance(batch, ConversationBehaviorProjectionBatch):
            raise TypeError("batch must be ConversationBehaviorProjectionBatch")
        if batch.source_id != source.source_id or batch.source_payload_digest != source.source_payload_digest:
            raise ConversationSourceError("projection output belongs to another source")
        try:
            atomic_create_bytes(
                self._path(batch.source_id, batch.output_id), self._encode(batch), artifact_root=self.root
            )
        except ImmutableArtifactConflictError as exc:
            current = self.read(source, batch.output_id)
            if current is None or current.output_record_digest != batch.output_record_digest:
                raise ConversationSourceError("projection output conflicts with different content") from exc
            return current
        stored = self.read(source, batch.output_id)
        if stored is None or stored.output_record_digest != batch.output_record_digest:
            raise ConversationSourceError("projection output was not durably read back")
        return stored

    def read(
        self, source: ConversationSourceEnvelope, output_id: str
    ) -> ConversationBehaviorProjectionBatch | None:
        require_sha256(output_id, "output_id")
        try:
            encoded = read_regular_bytes(
                self._path(source.source_id, output_id), artifact_root=self.root, max_bytes=self.max_file_bytes
            )
        except FileNotFoundError:
            return None
        try:
            batch = ConversationBehaviorProjectionBatch.from_dict(json.loads(encoded))
        except (UnicodeDecodeError, json.JSONDecodeError, ConversationSourceError) as exc:
            raise ConversationSourceError("behavior projection output is corrupt") from exc
        if encoded != self._encode(batch):
            raise ConversationSourceError("behavior projection output is not canonically encoded")
        if batch.output_id != output_id:
            raise ConversationSourceError("projection output path does not match output_id")
        if batch.source_id != source.source_id or batch.source_payload_digest != source.source_payload_digest:
            raise ConversationSourceError("projection output belongs to another source")
        return batch

    def list(self, source: ConversationSourceEnvelope) -> tuple[ConversationBehaviorProjectionBatch, ...]:
        try:
            entries = list_real_directory(
                self._directory(source.source_id),
                artifact_root=self.root,
                max_entries=self.max_files_per_source,
            )
        except DurablePathIntegrityError as exc:
            raise ConversationSourceError("projection output directory is invalid or exceeds its bound") from exc
        outputs: list[ConversationBehaviorProjectionBatch] = []
        for entry in entries:
            temporary = atomic_temporary_destination(entry.name)
            if entry.is_file() and temporary is not None and _OUTPUT_FILE.fullmatch(temporary) is not None:
                continue
            match = _OUTPUT_FILE.fullmatch(entry.name)
            if not entry.is_file() or match is None:
                raise ConversationSourceError("projection output directory contains an unsupported entry")
            output = self.read(source, match.group("output_id"))
            if output is None:
                raise ConversationSourceError("projection output disappeared during enumeration")
            outputs.append(output)
        return tuple(sorted(outputs, key=lambda item: item.output_id))

    def ref(self, output: object) -> ConsumerOutputRef:
        if not isinstance(output, ConversationBehaviorProjectionBatch):
            raise ConversationSourceError("projection store received another output type")
        return ConsumerOutputRef(
            output_kind=BEHAVIOR_PROJECTION_OUTPUT_KIND,
            output_id=output.output_id,
            output_record_digest=output.output_record_digest,
            processor_fingerprint=output.processor_fingerprint,
        )

    def restore(self, output: object) -> ConversationBehaviorProjectionBatch:
        if not isinstance(output, ConversationBehaviorProjectionBatch):
            raise ConversationSourceError("projection store received another output type")
        return output

    def _directory(self, source_id: str) -> Path:
        require_sha256(source_id, "source_id")
        return self.root / "source" / "outputs" / source_id / self.consumer.value

    def _path(self, source_id: str, output_id: str) -> Path:
        require_sha256(output_id, "output_id")
        return self._directory(source_id) / f"{output_id}.json"

    def _encode(self, batch: ConversationBehaviorProjectionBatch) -> bytes:
        if len(batch.items) > self.max_items:
            raise ConversationSourceError("behavior projection exceeds its configured item bound")
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
        self.output_store = store
        self.processor_fingerprint = projector.processor_fingerprint

    async def execute(
        self,
        envelope: ConversationSourceEnvelope,
        lease: ConversationConsumerExecutionLease,
    ) -> ConversationConsumerRunResult:
        projected = self.projector.project(envelope)
        if projected is None:
            return ConversationConsumerRunResult(
                disposition=ConversationConsumerRunDisposition.SKIPPED,
                output_ref=None,
                skip_reason="NO_ELIGIBLE_MESSAGES",
                runtime_result=None,
            )
        stored = await lease.run_fenced(lambda: self.store.put(envelope, projected))
        return ConversationConsumerRunResult(
            disposition=ConversationConsumerRunDisposition.OUTPUT_WRITTEN,
            output_ref=self.store.ref(stored),
            skip_reason=None,
            runtime_result=stored,
        )


__all__ = [
    "BEHAVIOR_PROJECTION_OUTPUT_SCHEMA_VERSION",
    "CONVERSATION_BEHAVIOR_PROJECTOR_VERSION",
    "ConversationBehaviorProjectionBatch",
    "ConversationBehaviorProjectionConsumer",
    "ConversationBehaviorProjectionItem",
    "ConversationBehaviorProjectionKind",
    "ConversationBehaviorProjectionStore",
    "ConversationBehaviorProjector",
]
