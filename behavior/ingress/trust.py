"""语义入口 Producer 来源与信任等级的系统绑定。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from behavior._validation import identifier, positive_int, sha256_digest
from behavior.errors import SemanticIngressError
from foundation.integrity import canonical_digest

PRODUCER_FINGERPRINT_SCHEMA_VERSION = "2"


class IngressTrustClass(str, Enum):
    DIRECT_SYSTEM_LOG = "DIRECT_SYSTEM_LOG"
    DIRECT_DEVICE_FACT = "DIRECT_DEVICE_FACT"
    OWNER_EXPLICIT = "OWNER_EXPLICIT"
    SENSOR_INFERRED = "SENSOR_INFERRED"
    MODEL_INFERRED = "MODEL_INFERRED"
    MULTIMODAL_MODEL_INFERRED = "MULTIMODAL_MODEL_INFERRED"


@dataclass(frozen=True, init=False)
class ProducerFingerprint:
    producer_name: str
    producer_version: str
    pipeline_version: str
    model_provider: str
    model_name: str
    prompt_version: str
    output_schema_version: str
    digest: str

    def __init__(
        self,
        producer_name: object,
        producer_version: object,
        pipeline_version: object,
        model_provider: object,
        model_name: object,
        prompt_version: object,
        output_schema_version: object,
    ) -> None:
        values = {
            "producer_name": identifier(producer_name, "producer_name"),
            "producer_version": identifier(producer_version, "producer_version"),
            "pipeline_version": identifier(pipeline_version, "pipeline_version"),
            "model_provider": identifier(model_provider, "model_provider"),
            "model_name": identifier(model_name, "model_name"),
            "prompt_version": identifier(prompt_version, "prompt_version"),
            "output_schema_version": identifier(output_schema_version, "output_schema_version"),
        }
        digest = canonical_digest({**values, "fingerprint_schema_version": PRODUCER_FINGERPRINT_SCHEMA_VERSION})
        for name, value in values.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "digest", digest)

    def to_dict(self) -> dict[str, object]:
        return {
            "producer_name": self.producer_name,
            "producer_version": self.producer_version,
            "pipeline_version": self.pipeline_version,
            "model_provider": self.model_provider,
            "model_name": self.model_name,
            "prompt_version": self.prompt_version,
            "output_schema_version": self.output_schema_version,
            "digest": self.digest,
        }

    @classmethod
    def from_dict(cls, value: object) -> ProducerFingerprint:
        from behavior._validation import require_fields, strict_fields

        fields = frozenset(
            {
                "producer_name",
                "producer_version",
                "pipeline_version",
                "model_provider",
                "model_name",
                "prompt_version",
                "output_schema_version",
                "digest",
            }
        )
        data = strict_fields(value, "producer_fingerprint", fields)
        require_fields(data, "producer_fingerprint", fields)
        result = cls(
            data["producer_name"],
            data["producer_version"],
            data["pipeline_version"],
            data["model_provider"],
            data["model_name"],
            data["prompt_version"],
            data["output_schema_version"],
        )
        try:
            stored = sha256_digest(data["digest"], "producer_fingerprint.digest")
        except ValueError as exc:
            raise SemanticIngressError(str(exc)) from exc
        if stored != result.digest:
            raise SemanticIngressError("producer fingerprint digest mismatch")
        return result


@dataclass(frozen=True)
class IngressAdapterCapability:
    trust_class: IngressTrustClass
    allowed_record_kinds: tuple[object, ...]
    maximum_batch_size: int
    owner_speaker_binding: bool = False

    def __post_init__(self) -> None:
        from behavior.ingress.model import SemanticRecordKind

        object.__setattr__(self, "trust_class", IngressTrustClass(self.trust_class))
        kinds = tuple(SemanticRecordKind(value) for value in self.allowed_record_kinds)
        if not kinds or len(set(kinds)) != len(kinds):
            raise SemanticIngressError("Adapter allowed_record_kinds must be non-empty and unique")
        object.__setattr__(self, "allowed_record_kinds", kinds)
        object.__setattr__(self, "maximum_batch_size", positive_int(self.maximum_batch_size, "maximum_batch_size"))
        if not isinstance(self.owner_speaker_binding, bool):
            raise TypeError("owner_speaker_binding must be boolean")
        if self.trust_class is IngressTrustClass.OWNER_EXPLICIT and not self.owner_speaker_binding:
            raise SemanticIngressError("OWNER_EXPLICIT capability requires confirmed Owner speaker binding")
        for kind in kinds:
            require_record_trust_compatibility(kind, self.trust_class)


_COMPATIBLE_TRUST: dict[str, frozenset[IngressTrustClass]] = {
    "OWNER_ACTIVITY_SEGMENT": frozenset(
        {
            IngressTrustClass.MODEL_INFERRED,
            IngressTrustClass.MULTIMODAL_MODEL_INFERRED,
        }
    ),
    "OWNER_UTTERANCE_SEGMENT": frozenset({IngressTrustClass.OWNER_EXPLICIT}),
    "OWNER_STATE_ASSERTION": frozenset(
        {
            IngressTrustClass.SENSOR_INFERRED,
            IngressTrustClass.MODEL_INFERRED,
            IngressTrustClass.MULTIMODAL_MODEL_INFERRED,
        }
    ),
    "OWNER_STATE_TRANSITION": frozenset(
        {
            IngressTrustClass.SENSOR_INFERRED,
            IngressTrustClass.MODEL_INFERRED,
            IngressTrustClass.MULTIMODAL_MODEL_INFERRED,
        }
    ),
    "OWNER_INTERACTION_SEGMENT": frozenset(
        {
            IngressTrustClass.MODEL_INFERRED,
            IngressTrustClass.MULTIMODAL_MODEL_INFERRED,
        }
    ),
    "ROBOT_ACTION_EVENT": frozenset({IngressTrustClass.DIRECT_SYSTEM_LOG}),
    "AGENT_ACTION_EVENT": frozenset({IngressTrustClass.DIRECT_SYSTEM_LOG}),
    "TOOL_RESULT_EVENT": frozenset({IngressTrustClass.DIRECT_SYSTEM_LOG}),
    "OWNER_SENSOR_FACT": frozenset({IngressTrustClass.DIRECT_DEVICE_FACT, IngressTrustClass.SENSOR_INFERRED}),
    "ENVIRONMENT_SENSOR_FACT": frozenset({IngressTrustClass.DIRECT_DEVICE_FACT, IngressTrustClass.SENSOR_INFERRED}),
    "DEVICE_STATE": frozenset({IngressTrustClass.DIRECT_DEVICE_FACT}),
    "ENVIRONMENT_CHANGE": frozenset(
        {
            IngressTrustClass.DIRECT_DEVICE_FACT,
            IngressTrustClass.SENSOR_INFERRED,
            IngressTrustClass.MODEL_INFERRED,
            IngressTrustClass.MULTIMODAL_MODEL_INFERRED,
        }
    ),
    "COVERAGE_INTERVAL": frozenset({IngressTrustClass.DIRECT_SYSTEM_LOG, IngressTrustClass.DIRECT_DEVICE_FACT}),
    "FREE_TEXT_SEMANTIC": frozenset(
        {
            IngressTrustClass.SENSOR_INFERRED,
            IngressTrustClass.MODEL_INFERRED,
            IngressTrustClass.MULTIMODAL_MODEL_INFERRED,
        }
    ),
}


def require_record_trust_compatibility(record_kind: object, trust_class: object) -> None:
    from behavior.ingress.model import SemanticRecordKind

    kind = SemanticRecordKind(record_kind)
    trust = IngressTrustClass(trust_class)
    if trust not in _COMPATIBLE_TRUST[kind.value]:
        raise SemanticIngressError(f"record kind {kind.value} is incompatible with ingress trust {trust.value}")


__all__ = [
    "IngressAdapterCapability",
    "IngressTrustClass",
    "PRODUCER_FINGERPRINT_SCHEMA_VERSION",
    "ProducerFingerprint",
    "require_record_trust_compatibility",
]
