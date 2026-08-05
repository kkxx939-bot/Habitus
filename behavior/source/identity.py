"""SourceRecord 的确定性身份工厂。"""

from __future__ import annotations

from behavior._validation import identifier, non_negative_int, sha256_digest
from foundation.integrity import canonical_digest


class SourceRecordIdentityFactory:
    """只从来源稳定字段生成身份，调用方不能传入任意 ID。"""

    @staticmethod
    def create(
        *,
        stream_id: object,
        source_sequence: object,
        payload_digest: object,
        source_type: object,
        schema_version: object,
    ) -> str:
        stream = identifier(stream_id, "stream_id")
        sequence = non_negative_int(source_sequence, "source_sequence")
        digest = sha256_digest(payload_digest, "payload_digest")
        source_name = getattr(source_type, "value", source_type)
        source_name = identifier(source_name, "source_type")
        version = identifier(schema_version, "schema_version")
        return "src_" + canonical_digest(
            {
                "payload_digest": digest,
                "schema_version": version,
                "source_sequence": sequence,
                "source_type": source_name,
                "stream_id": stream,
            }
        )


__all__ = ["SourceRecordIdentityFactory"]
