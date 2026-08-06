"""Owner-scoped 语义记录的确定性身份工厂。"""

from __future__ import annotations

from behavior._validation import sha256_digest
from foundation.integrity import canonical_digest


class SemanticRecordIdentityFactory:
    """只使用稳定语义输入生成身份，不包含处理时间或存储位置。"""

    @staticmethod
    def create(
        *,
        owner_identity_digest: object,
        producer_fingerprint: object,
        semantic_input: object,
    ) -> str:
        from behavior.ingress.model import SemanticRecordInput

        owner = sha256_digest(owner_identity_digest, "owner_identity_digest")
        producer = sha256_digest(producer_fingerprint, "producer_fingerprint")
        if not isinstance(semantic_input, SemanticRecordInput):
            raise TypeError("semantic_input must be SemanticRecordInput")
        return "sem_" + canonical_digest(
            {
                "owner_identity_digest": owner,
                "producer_fingerprint": producer,
                "stream_id": semantic_input.stream_id,
                "source_sequence": semantic_input.source_sequence,
                "record_kind": semantic_input.record_kind.value,
                "event_time_start": semantic_input.to_dict()["event_time_start"],
                "event_time_end": semantic_input.to_dict()["event_time_end"],
                "payload_digest": semantic_input.payload_digest,
                "evidence_digests": tuple(sorted(item.digest for item in semantic_input.evidence_refs)),
                "semantic_input_digest": canonical_digest(semantic_input.to_dict()),
                "record_schema_version": semantic_input.schema_version,
            }
        )


__all__ = ["SemanticRecordIdentityFactory"]
