"""统一 Output 路径中的不可变 Behavior Projection Outbox。"""

from __future__ import annotations

import json
import re
from pathlib import Path

from habitus.conversation.projection.behavior.model import (
    BEHAVIOR_PROJECTION_OUTPUT_KIND,
    BEHAVIOR_PROJECTION_OUTPUT_SCHEMA_VERSION,
    ConversationBehaviorProjectionBatch,
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


class ConversationBehaviorProjectionStore:
    """按 ``output_id`` 只创建不覆盖地保存投影批次。

    ``output_id`` 是幂等键（含义见 ``model`` 模块 docstring）：同一来源在同一
    处理器版本下只会存在一个文件，重复写入按内容摘要比对后复用既有文件。
    """

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

    def remove(self, source: ConversationSourceEnvelope, output_id: str) -> bool:
        """删除一个陈旧输出；只供显式人工修复调用，交付路径永不使用。

        投影是来源的纯函数，被删除的批次随时可以从不可变 Source 重新算出，
        因此这里可以真正删除而不是留下墓碑。
        """

        require_sha256(output_id, "output_id")
        return durable_unlink(self._path(source.source_id, output_id), artifact_root=self.root)

    def _directory(self, source_id: str) -> Path:
        require_sha256(source_id, "source_id")
        return self.root / "source" / "outputs" / source_id / self.consumer.value

    def _path(self, source_id: str, output_id: str) -> Path:
        require_sha256(output_id, "output_id")
        return self._directory(source_id) / f"{output_id}.json"

    def _encode(self, batch: ConversationBehaviorProjectionBatch) -> bytes:
        if len(batch.items) > self.max_items:
            raise ConversationSourceError("behavior projection exceeds its configured item bound")
        return encode_durable_record(
            batch.to_dict(), max_bytes=self.max_file_bytes, label="behavior projection"
        )


__all__ = ["ConversationBehaviorProjectionStore"]
