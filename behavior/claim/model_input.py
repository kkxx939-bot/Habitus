"""远程 Claim Normalizer 的最小安全输入投影。"""

from __future__ import annotations

from dataclasses import dataclass, field

from behavior._validation import utc_text
from behavior.evidence.record import BehaviorEvidenceRecord
from behavior.evidence.specs import record_spec
from foundation.integrity import canonical_digest

MODEL_INPUT_POLICY_VERSION = "model_normalization_input_v1"


@dataclass(frozen=True, slots=True)
class ModelInputPolicy:
    include_context_refs: bool = False
    version: str = MODEL_INPUT_POLICY_VERSION
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        if self.version != MODEL_INPUT_POLICY_VERSION:
            raise ValueError("unsupported model input policy version")
        object.__setattr__(
            self,
            "digest",
            canonical_digest(
                {
                    "include_context_refs": self.include_context_refs,
                    "version": self.version,
                }
            ),
        )


class ModelNormalizationProjection:
    def __init__(self, policy: ModelInputPolicy | None = None) -> None:
        self.policy = policy or ModelInputPolicy()

    def project(self, record: BehaviorEvidenceRecord) -> dict[str, object]:
        content = record.semantic_content
        projected: dict[str, object] = {
            "actor_role": None if content.actor_role is None else content.actor_role.value,
            "duration_seconds": (content.event_time_end - content.event_time_start).total_seconds(),
            "event_time_end": utc_text(content.event_time_end),
            "event_time_start": utc_text(content.event_time_start),
            "integrity": content.integrity.value,
            "modality": content.modality.value,
            "payload": record_spec(content.record_kind).payload_codec.encode(content.payload),
            "record_kind": content.record_kind.value,
            "source_confidence": content.source_confidence,
            "subject_role": content.subject_role.value,
        }
        if self.policy.include_context_refs:
            projected["context_refs"] = {
                "entity_refs": content.entity_refs,
                "location_ref": content.location_ref,
                "object_refs": content.object_refs,
                "scene_ref": content.scene_ref,
            }
        return projected
