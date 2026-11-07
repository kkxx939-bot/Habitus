"""Memory Conversation Consumer 首次完成结果的不可变 Output 与 Codec。"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

from habitus.conversation.source import (
    ConsumerOutputRef,
    ConversationSourceConsumer,
    ConversationSourceEnvelope,
    ConversationSourceError,
    conversation_consumer_output_id,
    discard_output_directory,
    require_sha256,
    source_timestamp,
)
from habitus.foundation.integrity import canonical_digest, canonical_json, canonicalize
from habitus.foundation.validation import is_int
from habitus.infrastructure.store.filesystem import (
    DurablePathIntegrityError,
    ImmutableArtifactConflictError,
    atomic_create_bytes,
    atomic_temporary_destination,
    durable_unlink,
    list_real_directory,
    read_regular_bytes,
)
from habitus.memory.conversation import ConversationAppendResult, ConversationAppendStatus, ConversationRetentionPlan
from habitus.memory.workflow.ingest import ConversationMemoryIngestResult
from habitus.memory.workflow.jobs import MemoryJob, MemoryJobStatus
from habitus.pre.conversation import ConversationBatch, ConversationMessage

# TODO(conversation-source): 修改 ConversationMemoryIngestResult Codec 或输出记录结构时，
# 必须同步提升本版本，避免新旧快照共享同一 output_id 身份空间。
MEMORY_CONVERSATION_OUTPUT_SCHEMA_VERSION = "memory_conversation_output_v1"
MEMORY_CONVERSATION_OUTPUT_KIND = "memory_conversation_output"
_OUTPUT_FILE = re.compile(r"^(?P<output_id>[0-9a-f]{64})\.json$")
_NORMALIZED_MESSAGE = "{} must be non-empty normalized text"


# 以下解码助手让磁盘读回的字段在构造前确认类型；报错的异常类型与文字照抄被构造值对象自身的检查
def _int_field(value: object, message: str) -> int:
    if not is_int(value):
        raise ValueError(message)
    return value


def _text_field(value: object, message: str) -> str:
    if not isinstance(value, str):
        raise ValueError(message)
    return value


def _optional_text_field(value: object, message: str) -> str | None:
    if value is not None and not isinstance(value, str):
        raise ValueError(message)
    return value


def _flag_field(value: object, message: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(message)
    return value


def _append_to_dict(value: ConversationAppendResult) -> dict[str, Any]:
    return canonicalize(
        {
            "status": value.status.value,
            "appended_count": value.appended_count,
            "live": None if value.live is None else value.live.to_dict(),
            "next_sequence": value.next_sequence,
        }
    )


def _append_from_dict(value: object) -> ConversationAppendResult:
    if not isinstance(value, Mapping) or set(value) != {
        "status",
        "appended_count",
        "live",
        "next_sequence",
    }:
        raise ConversationSourceError("memory append snapshot schema is invalid")
    live = value["live"]
    if live is not None and not isinstance(live, Mapping):
        raise ConversationSourceError("memory append live snapshot is invalid")
    try:
        status = ConversationAppendStatus(value["status"])
    except (TypeError, ValueError) as exc:
        raise ConversationSourceError("memory append status is invalid") from exc
    return ConversationAppendResult(
        status=status,
        appended_count=_int_field(value["appended_count"], "appended_count must be a non-negative integer"),
        live=None if live is None else ConversationBatch.from_dict(live),
        next_sequence=_int_field(value["next_sequence"], "next_sequence must be a non-negative integer"),
    )


def _job_to_dict(value: MemoryJob) -> dict[str, Any]:
    return canonicalize(
        {
            "memory_sequence": value.memory_sequence,
            "conversation_id": value.conversation_id,
            "started_on": value.started_on.isoformat(),
            "segment_id": value.segment_id,
            "source_segment_digest": value.source_segment_digest,
            "transaction_id": value.transaction_id,
            "status": value.status.value,
            "attempts": value.attempts,
            "claim_id": value.claim_id,
            "claim_generation": value.claim_generation,
            "worker_id": value.worker_id,
            "lease_expires_at": value.lease_expires_at,
            "next_attempt_at": value.next_attempt_at,
            "last_error": value.last_error,
            "created_at": value.created_at,
            "updated_at": value.updated_at,
        }
    )


def _job_from_dict(value: object) -> MemoryJob:
    expected = {
        "memory_sequence",
        "conversation_id",
        "started_on",
        "segment_id",
        "source_segment_digest",
        "transaction_id",
        "status",
        "attempts",
        "claim_id",
        "claim_generation",
        "worker_id",
        "lease_expires_at",
        "next_attempt_at",
        "last_error",
        "created_at",
        "updated_at",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ConversationSourceError("memory job snapshot schema is invalid")
    try:
        started_on = date.fromisoformat(value["started_on"])
        status = MemoryJobStatus(value["status"])
    except (TypeError, ValueError) as exc:
        raise ConversationSourceError("memory job snapshot identity is invalid") from exc
    optional_times = {
        name: None if value[name] is None else source_timestamp(value[name], name)
        for name in ("lease_expires_at", "next_attempt_at")
    }
    return MemoryJob(
        memory_sequence=_int_field(value["memory_sequence"], "memory_sequence must be a positive integer"),
        conversation_id=_text_field(value["conversation_id"], _NORMALIZED_MESSAGE.format("conversation_id")),
        started_on=started_on,
        segment_id=_text_field(value["segment_id"], _NORMALIZED_MESSAGE.format("segment_id")),
        source_segment_digest=_text_field(
            value["source_segment_digest"], "source_segment_digest must be lowercase SHA-256 text"
        ),
        transaction_id=_text_field(
            value["transaction_id"], "transaction_id must be 32 lowercase hexadecimal characters"
        ),
        status=status,
        attempts=_int_field(value["attempts"], "memory job attempts must be non-negative"),
        claim_id=_optional_text_field(value["claim_id"], "running memory job requires a claim_id"),
        claim_generation=_int_field(value["claim_generation"], "memory job claim_generation must be non-negative"),
        worker_id=_optional_text_field(value["worker_id"], "running memory job requires a normalized worker_id"),
        lease_expires_at=optional_times["lease_expires_at"],
        next_attempt_at=optional_times["next_attempt_at"],
        last_error=_optional_text_field(value["last_error"], "memory job last_error is invalid"),
        created_at=source_timestamp(value["created_at"], "created_at"),
        updated_at=source_timestamp(value["updated_at"], "updated_at"),
    )


def _retention_to_dict(value: ConversationRetentionPlan) -> dict[str, Any]:
    return canonicalize(
        {
            "through_sequence": value.through_sequence,
            "archive_messages": [message.to_dict() for message in value.archive_messages],
            "retained_messages": [message.to_dict() for message in value.retained_messages],
            "triggered": value.triggered,
            "flush": value.flush,
            "pending_tokens": value.pending_tokens,
            "budget_exceeded": value.budget_exceeded,
            "reason": value.reason,
            "boundary_kind": value.boundary_kind,
            "embedding_fingerprint": value.embedding_fingerprint,
            "chunker_version": value.chunker_version,
        }
    )


def _retention_from_dict(value: object) -> ConversationRetentionPlan:
    expected = {
        "through_sequence",
        "archive_messages",
        "retained_messages",
        "triggered",
        "flush",
        "pending_tokens",
        "budget_exceeded",
        "reason",
        "boundary_kind",
        "embedding_fingerprint",
        "chunker_version",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ConversationSourceError("memory retention snapshot schema is invalid")
    archive = value["archive_messages"]
    retained = value["retained_messages"]
    if not isinstance(archive, list) or not isinstance(retained, list):
        raise ConversationSourceError("memory retention message snapshots must be lists")
    if any(not isinstance(item, Mapping) for item in (*archive, *retained)):
        raise ConversationSourceError("memory retention contains an invalid message snapshot")
    through_sequence = value["through_sequence"]
    if through_sequence is not None and not is_int(through_sequence):
        raise ValueError("retention boundary must match the archived prefix")
    return ConversationRetentionPlan(
        through_sequence=through_sequence,
        archive_messages=tuple(ConversationMessage.from_dict(item) for item in archive),
        retained_messages=tuple(ConversationMessage.from_dict(item) for item in retained),
        triggered=_flag_field(value["triggered"], "retention flags must be boolean"),
        flush=_flag_field(value["flush"], "retention flags must be boolean"),
        pending_tokens=_int_field(value["pending_tokens"], "pending_tokens must be a non-negative integer"),
        budget_exceeded=_flag_field(value["budget_exceeded"], "budget_exceeded must be boolean"),
        reason=_text_field(value["reason"], "retention reason must be non-empty text"),
        boundary_kind=_optional_text_field(value["boundary_kind"], "boundary_kind must be non-empty text or None"),
        embedding_fingerprint=_optional_text_field(
            value["embedding_fingerprint"], "embedding_fingerprint must be non-empty text or None"
        ),
        chunker_version=_text_field(value["chunker_version"], "chunker_version must be non-empty text"),
    )


def conversation_memory_ingest_to_dict(value: ConversationMemoryIngestResult) -> dict[str, Any]:
    return canonicalize(
        {
            "append": _append_to_dict(value.append),
            "jobs": [_job_to_dict(job) for job in value.jobs],
            "retention": _retention_to_dict(value.retention),
        }
    )


def conversation_memory_ingest_from_dict(value: object) -> ConversationMemoryIngestResult:
    if not isinstance(value, Mapping) or set(value) != {"append", "jobs", "retention"}:
        raise ConversationSourceError("memory ingest snapshot schema is invalid")
    jobs = value["jobs"]
    if not isinstance(jobs, list):
        raise ConversationSourceError("memory ingest jobs snapshot must be a list")
    return ConversationMemoryIngestResult(
        append=_append_from_dict(value["append"]),
        jobs=tuple(_job_from_dict(job) for job in jobs),
        retention=_retention_from_dict(value["retention"]),
    )


@dataclass(frozen=True)
class MemoryConversationOutput:
    output_id: str
    source_id: str
    source_payload_digest: str
    consumer: ConversationSourceConsumer
    processor_fingerprint: str
    ingest_result_snapshot: ConversationMemoryIngestResult
    recorded_at: datetime
    output_record_digest: str

    def __post_init__(self) -> None:
        require_sha256(self.output_id, "memory output_id")
        require_sha256(self.source_id, "memory output source_id")
        require_sha256(self.source_payload_digest, "memory output source_payload_digest")
        object.__setattr__(self, "consumer", ConversationSourceConsumer(self.consumer))
        if self.consumer is not ConversationSourceConsumer.MEMORY:
            raise ConversationSourceError("memory output has the wrong consumer")
        require_sha256(self.processor_fingerprint, "memory output processor_fingerprint")
        object.__setattr__(self, "recorded_at", source_timestamp(self.recorded_at, "recorded_at"))
        require_sha256(self.output_record_digest, "memory output_record_digest")
        expected_id = conversation_consumer_output_id(
            source_id=self.source_id,
            source_payload_digest=self.source_payload_digest,
            consumer=self.consumer,
            processor_fingerprint=self.processor_fingerprint,
            output_schema_version=MEMORY_CONVERSATION_OUTPUT_SCHEMA_VERSION,
        )
        if self.output_id != expected_id:
            raise ConversationSourceError("memory output_id does not match output identity")
        if self.output_record_digest != canonical_digest(self._record_without_digest()):
            raise ConversationSourceError("memory output_record_digest does not match output record")

    @classmethod
    def create(
        cls,
        *,
        source: ConversationSourceEnvelope,
        processor_fingerprint: str,
        ingest_result: ConversationMemoryIngestResult,
        recorded_at: datetime,
    ) -> MemoryConversationOutput:
        output_id = conversation_consumer_output_id(
            source_id=source.source_id,
            source_payload_digest=source.source_payload_digest,
            consumer=ConversationSourceConsumer.MEMORY,
            processor_fingerprint=processor_fingerprint,
            output_schema_version=MEMORY_CONVERSATION_OUTPUT_SCHEMA_VERSION,
        )
        record = canonicalize(
            {
                "schema_version": MEMORY_CONVERSATION_OUTPUT_SCHEMA_VERSION,
                "output_id": output_id,
                "source_id": source.source_id,
                "source_payload_digest": source.source_payload_digest,
                "consumer": ConversationSourceConsumer.MEMORY.value,
                "processor_fingerprint": processor_fingerprint,
                "ingest_result_snapshot": conversation_memory_ingest_to_dict(ingest_result),
                "recorded_at": source_timestamp(recorded_at, "recorded_at"),
            }
        )
        return cls(
            output_id=output_id,
            source_id=source.source_id,
            source_payload_digest=source.source_payload_digest,
            consumer=ConversationSourceConsumer.MEMORY,
            processor_fingerprint=processor_fingerprint,
            ingest_result_snapshot=ingest_result,
            recorded_at=recorded_at,
            output_record_digest=canonical_digest(record),
        )

    def _record_without_digest(self) -> dict[str, Any]:
        return canonicalize(
            {
                "schema_version": MEMORY_CONVERSATION_OUTPUT_SCHEMA_VERSION,
                "output_id": self.output_id,
                "source_id": self.source_id,
                "source_payload_digest": self.source_payload_digest,
                "consumer": self.consumer.value,
                "processor_fingerprint": self.processor_fingerprint,
                "ingest_result_snapshot": conversation_memory_ingest_to_dict(self.ingest_result_snapshot),
                "recorded_at": self.recorded_at,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return canonicalize({**self._record_without_digest(), "output_record_digest": self.output_record_digest})

    @classmethod
    def from_dict(cls, value: object) -> MemoryConversationOutput:
        expected = {
            "schema_version",
            "output_id",
            "source_id",
            "source_payload_digest",
            "consumer",
            "processor_fingerprint",
            "ingest_result_snapshot",
            "recorded_at",
            "output_record_digest",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ConversationSourceError("memory output schema is invalid")
        if value.get("schema_version") != MEMORY_CONVERSATION_OUTPUT_SCHEMA_VERSION:
            raise ConversationSourceError("memory output schema version is invalid")
        try:
            consumer = ConversationSourceConsumer(value["consumer"])
        except (TypeError, ValueError) as exc:
            raise ConversationSourceError("memory output consumer is invalid") from exc
        return cls(
            output_id=require_sha256(value["output_id"], "memory output_id"),
            source_id=require_sha256(value["source_id"], "memory output source_id"),
            source_payload_digest=require_sha256(value["source_payload_digest"], "memory output source_payload_digest"),
            consumer=consumer,
            processor_fingerprint=require_sha256(value["processor_fingerprint"], "memory output processor_fingerprint"),
            ingest_result_snapshot=conversation_memory_ingest_from_dict(value["ingest_result_snapshot"]),
            recorded_at=source_timestamp(value["recorded_at"], "recorded_at"),
            output_record_digest=require_sha256(value["output_record_digest"], "memory output_record_digest"),
        )


class MemoryConversationOutputStore:
    consumer = ConversationSourceConsumer.MEMORY

    def __init__(
        self,
        conversation_root: str | Path,
        *,
        max_files_per_source: int,
        max_file_bytes: int,
    ) -> None:
        self.root = Path(conversation_root).expanduser().resolve(strict=False)
        if max_files_per_source <= 0:
            raise ValueError("max_files_per_source must be a positive integer")
        if max_file_bytes <= 0:
            raise ValueError("max_file_bytes must be a positive integer")
        self.max_files_per_source = max_files_per_source
        self.max_file_bytes = max_file_bytes

    def expected_output_id(self, source: ConversationSourceEnvelope, processor_fingerprint: str) -> str:
        return conversation_consumer_output_id(
            source_id=source.source_id,
            source_payload_digest=source.source_payload_digest,
            consumer=self.consumer,
            processor_fingerprint=processor_fingerprint,
            output_schema_version=MEMORY_CONVERSATION_OUTPUT_SCHEMA_VERSION,
        )

    def put(
        self,
        source: ConversationSourceEnvelope,
        output: MemoryConversationOutput,
    ) -> MemoryConversationOutput:
        if output.source_id != source.source_id or output.source_payload_digest != source.source_payload_digest:
            raise ConversationSourceError("memory output belongs to another source")
        try:
            atomic_create_bytes(
                self._path(output.source_id, output.output_id), self._encode(output), artifact_root=self.root
            )
        except ImmutableArtifactConflictError as exc:
            current = self.read(source, output.output_id)
            if current is None or current.output_record_digest != output.output_record_digest:
                raise ConversationSourceError("memory output conflicts with different content") from exc
            return current
        stored = self.read(source, output.output_id)
        if stored is None or stored.output_record_digest != output.output_record_digest:
            raise ConversationSourceError("memory output was not durably read back")
        return stored

    def read(self, source: ConversationSourceEnvelope, output_id: str) -> MemoryConversationOutput | None:
        require_sha256(output_id, "output_id")
        try:
            encoded = read_regular_bytes(
                self._path(source.source_id, output_id), artifact_root=self.root, max_bytes=self.max_file_bytes
            )
        except FileNotFoundError:
            return None
        except DurablePathIntegrityError as exc:
            raise ConversationSourceError("memory output exceeds its read bound or is not a regular file") from exc
        try:
            output = MemoryConversationOutput.from_dict(json.loads(encoded))
        except (UnicodeDecodeError, json.JSONDecodeError, ConversationSourceError) as exc:
            raise ConversationSourceError("memory output is corrupt") from exc
        if encoded != self._encode(output):
            raise ConversationSourceError("memory output is not canonically encoded")
        if output.source_id != source.source_id or output.source_payload_digest != source.source_payload_digest:
            raise ConversationSourceError("memory output belongs to another source")
        if output.output_id != output_id:
            raise ConversationSourceError("memory output path does not match output_id")
        return output

    def list(self, source: ConversationSourceEnvelope) -> tuple[MemoryConversationOutput, ...]:
        directory = self._directory(source.source_id)
        try:
            entries = list_real_directory(directory, artifact_root=self.root, max_entries=self.max_files_per_source)
        except DurablePathIntegrityError as exc:
            raise ConversationSourceError("memory output directory is invalid or exceeds its bound") from exc
        outputs: list[MemoryConversationOutput] = []
        for entry in entries:
            temporary = atomic_temporary_destination(entry.name)
            if entry.is_file() and temporary is not None and _OUTPUT_FILE.fullmatch(temporary) is not None:
                continue
            match = _OUTPUT_FILE.fullmatch(entry.name)
            if not entry.is_file() or match is None:
                raise ConversationSourceError("memory output directory contains an unsupported entry")
            output = self.read(source, match.group("output_id"))
            if output is None:
                raise ConversationSourceError("memory output disappeared during enumeration")
            outputs.append(output)
        return tuple(sorted(outputs, key=lambda item: item.output_id))

    def ref(self, output: object) -> ConsumerOutputRef:
        if not isinstance(output, MemoryConversationOutput):
            raise ConversationSourceError("memory output store received another output type")
        return ConsumerOutputRef(
            output_kind=MEMORY_CONVERSATION_OUTPUT_KIND,
            output_id=output.output_id,
            output_record_digest=output.output_record_digest,
            processor_fingerprint=output.processor_fingerprint,
        )

    def restore(self, output: object) -> ConversationMemoryIngestResult:
        if not isinstance(output, MemoryConversationOutput):
            raise ConversationSourceError("memory output store received another output type")
        return output.ingest_result_snapshot

    def remove(self, source: ConversationSourceEnvelope, output_id: str) -> bool:
        """删除一个陈旧输出；只供显式人工修复调用，交付路径永不使用。

        修复只在同一来源仍存在另一份、且与当前 Processor 吻合的输出时才允许，
        因此删除不会导致 Consumer 重新执行，也就不会重放已经落盘的写入副作用。
        """

        require_sha256(output_id, "output_id")
        return durable_unlink(self._path(source.source_id, output_id), artifact_root=self.root)

    def discard_all(self, source_id: str) -> int:
        """释放会话源时删这份源的全部输出。"""

        return discard_output_directory(self.root, self._directory(source_id), max_entries=self.max_files_per_source)

    def _directory(self, source_id: str) -> Path:
        require_sha256(source_id, "source_id")
        return self.root / "source" / "outputs" / source_id / self.consumer.value

    def _path(self, source_id: str, output_id: str) -> Path:
        require_sha256(output_id, "output_id")
        return self._directory(source_id) / f"{output_id}.json"

    def _encode(self, output: MemoryConversationOutput) -> bytes:
        encoded = (canonical_json(output.to_dict()) + "\n").encode("utf-8")
        if len(encoded) > self.max_file_bytes:
            raise ConversationSourceError("memory output exceeds its configured file bound")
        return encoded


__all__ = [
    "MEMORY_CONVERSATION_OUTPUT_SCHEMA_VERSION",
    "MemoryConversationOutput",
    "MemoryConversationOutputStore",
    "conversation_memory_ingest_from_dict",
    "conversation_memory_ingest_to_dict",
]
