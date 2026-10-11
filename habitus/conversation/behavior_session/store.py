"""会话 lane 回执的耐久存储：实现会话源交付契约里消费者的输出存储。"""

from __future__ import annotations

import json
import re
from pathlib import Path

from habitus.conversation.behavior_session.model import (
    BEHAVIOR_SESSION_OUTPUT_KIND,
    BEHAVIOR_SESSION_OUTPUT_SCHEMA_VERSION,
    BehaviorSessionOutput,
)
from habitus.conversation.source.model import (
    ConversationSourceEnvelope,
    ConversationSourceError,
    encode_durable_record,
    require_sha256,
)
from habitus.conversation.source.receipt import (
    ConsumerOutputRef,
    ConversationSourceConsumer,
    conversation_consumer_output_id,
)
from habitus.infrastructure.store.filesystem import (
    DurablePathIntegrityError,
    ImmutableArtifactConflictError,
    atomic_create_bytes,
    atomic_temporary_destination,
    durable_unlink,
    list_real_directory,
    read_regular_bytes,
)

_OUTPUT_FILE = re.compile(r"^(?P<output_id>[0-9a-f]{64})\.json$")


class BehaviorSessionOutputStore:
    """每份会话源一个目录、每个处理器指纹一份回执；写入即不可变。"""

    consumer = ConversationSourceConsumer.BEHAVIOR_SESSION

    def __init__(self, conversation_root: str | Path, *, max_files_per_source: int, max_file_bytes: int) -> None:
        self.root = Path(conversation_root).expanduser().resolve(strict=False)
        for name, value in (("max_files_per_source", max_files_per_source), ("max_file_bytes", max_file_bytes)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self.max_files_per_source = max_files_per_source
        self.max_file_bytes = max_file_bytes

    def expected_output_id(self, source: ConversationSourceEnvelope, processor_fingerprint: str) -> str:
        return conversation_consumer_output_id(
            source_id=source.source_id,
            source_payload_digest=source.source_payload_digest,
            consumer=self.consumer,
            processor_fingerprint=processor_fingerprint,
            output_schema_version=BEHAVIOR_SESSION_OUTPUT_SCHEMA_VERSION,
        )

    def put(self, source: ConversationSourceEnvelope, output: BehaviorSessionOutput) -> BehaviorSessionOutput:
        if not isinstance(output, BehaviorSessionOutput):
            raise TypeError("output must be BehaviorSessionOutput")
        self._require_owner(source, output)
        try:
            atomic_create_bytes(
                self._path(output.source_id, output.output_id), self._encode(output), artifact_root=self.root
            )
        except ImmutableArtifactConflictError as exc:
            current = self.read(source, output.output_id)
            if current is None or current.output_record_digest != output.output_record_digest:
                raise ConversationSourceError("behavior session receipt conflicts with different content") from exc
            return current
        stored = self.read(source, output.output_id)
        if stored is None or stored.output_record_digest != output.output_record_digest:
            raise ConversationSourceError("behavior session receipt was not durably read back")
        return stored

    def read(self, source: ConversationSourceEnvelope, output_id: str) -> BehaviorSessionOutput | None:
        require_sha256(output_id, "output_id")
        try:
            encoded = read_regular_bytes(
                self._path(source.source_id, output_id), artifact_root=self.root, max_bytes=self.max_file_bytes
            )
        except FileNotFoundError:
            return None
        try:
            output = BehaviorSessionOutput.from_dict(json.loads(encoded))
        except (UnicodeDecodeError, json.JSONDecodeError, ConversationSourceError) as exc:
            raise ConversationSourceError("behavior session receipt is corrupt") from exc
        if encoded != self._encode(output):
            raise ConversationSourceError("behavior session receipt is not canonically encoded")
        if output.output_id != output_id:
            raise ConversationSourceError("behavior session receipt path does not match output_id")
        self._require_owner(source, output)
        return output

    def list(self, source: ConversationSourceEnvelope) -> tuple[BehaviorSessionOutput, ...]:
        try:
            entries = list_real_directory(
                self._directory(source.source_id), artifact_root=self.root, max_entries=self.max_files_per_source
            )
        except DurablePathIntegrityError as exc:
            raise ConversationSourceError(
                "behavior session receipt directory is invalid or exceeds its bound"
            ) from exc
        outputs: list[BehaviorSessionOutput] = []
        for entry in entries:
            temporary = atomic_temporary_destination(entry.name)
            if entry.is_file() and temporary is not None and _OUTPUT_FILE.fullmatch(temporary) is not None:
                continue  # 写到一半的临时文件，不算输出
            match = _OUTPUT_FILE.fullmatch(entry.name)
            if not entry.is_file() or match is None:
                raise ConversationSourceError("behavior session receipt directory contains an unsupported entry")
            output = self.read(source, match.group("output_id"))
            if output is None:
                raise ConversationSourceError("behavior session receipt disappeared during enumeration")
            outputs.append(output)
        return tuple(sorted(outputs, key=lambda item: item.output_id))

    def ref(self, output: object) -> ConsumerOutputRef:
        if not isinstance(output, BehaviorSessionOutput):
            raise ConversationSourceError("behavior session store received another output type")
        return ConsumerOutputRef(
            output_kind=BEHAVIOR_SESSION_OUTPUT_KIND,
            output_id=output.output_id,
            output_record_digest=output.output_record_digest,
            processor_fingerprint=output.processor_fingerprint,
        )

    def restore(self, output: object) -> BehaviorSessionOutput:
        if not isinstance(output, BehaviorSessionOutput):
            raise ConversationSourceError("behavior session store received another output type")
        return output

    def remove(self, source: ConversationSourceEnvelope, output_id: str) -> bool:
        require_sha256(output_id, "output_id")
        return durable_unlink(self._path(source.source_id, output_id), artifact_root=self.root)

    @staticmethod
    def _require_owner(source: ConversationSourceEnvelope, output: BehaviorSessionOutput) -> None:
        if output.source_id != source.source_id or output.source_payload_digest != source.source_payload_digest:
            raise ConversationSourceError("behavior session receipt belongs to another source")

    def _directory(self, source_id: str) -> Path:
        require_sha256(source_id, "source_id")
        return self.root / "source" / "outputs" / source_id / self.consumer.value

    def _path(self, source_id: str, output_id: str) -> Path:
        require_sha256(output_id, "output_id")
        return self._directory(source_id) / f"{output_id}.json"

    def _encode(self, output: BehaviorSessionOutput) -> bytes:
        return encode_durable_record(output.to_dict(), max_bytes=self.max_file_bytes, label="behavior session receipt")


__all__ = ["BehaviorSessionOutputStore"]
