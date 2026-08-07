"""Evidence 来源描述与系统绑定后的 Provenance。"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from behavior._validation import (
    external_reference,
    identifier,
    non_negative_int,
    optional_identifier,
    require_fields,
    sha256_digest,
    strict_fields,
)
from behavior.evidence.refs import CausalRef, CausalRefKind, CorrelationRef, ProjectionRef, SourceEventRef, StreamRef
from foundation.integrity import canonical_digest

PRODUCER_FINGERPRINT_SCHEMA_VERSION = "producer_fingerprint_v1"


class BehaviorOriginKind(str, Enum):
    DIRECT_PERCEPTION = "DIRECT_PERCEPTION"
    DIRECT_AMBIENT_ASR = "DIRECT_AMBIENT_ASR"
    DIRECT_RUNTIME_EVENT = "DIRECT_RUNTIME_EVENT"
    ACTION_EXECUTION_FEEDBACK = "ACTION_EXECUTION_FEEDBACK"
    CONVERSATION_PROJECTION = "CONVERSATION_PROJECTION"


class ProducerImplementationKind(str, Enum):
    ADAPTER = "ADAPTER"
    MODEL = "MODEL"
    SYSTEM = "SYSTEM"
    PROJECTOR = "PROJECTOR"


@dataclass(frozen=True)
class ProducerFingerprint:
    producer_name: str
    producer_version: str
    pipeline_version: str
    output_schema_version: str
    implementation_kind: ProducerImplementationKind
    model_provider: str | None = None
    model_name: str | None = None
    prompt_version: str | None = None
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        kind = ProducerImplementationKind(self.implementation_kind)
        values = {
            "producer_name": identifier(self.producer_name, "producer.producer_name"),
            "producer_version": identifier(self.producer_version, "producer.producer_version"),
            "pipeline_version": identifier(self.pipeline_version, "producer.pipeline_version"),
            "output_schema_version": identifier(
                self.output_schema_version,
                "producer.output_schema_version",
            ),
            "model_provider": optional_identifier(self.model_provider, "producer.model_provider"),
            "model_name": optional_identifier(self.model_name, "producer.model_name"),
            "prompt_version": optional_identifier(self.prompt_version, "producer.prompt_version"),
        }
        model_values = (values["model_provider"], values["model_name"], values["prompt_version"])
        if kind is ProducerImplementationKind.MODEL and any(value is None for value in model_values):
            raise ValueError("MODEL producer must identify provider, model, and prompt version")
        if kind is not ProducerImplementationKind.MODEL and any(value is not None for value in model_values):
            raise ValueError("non-MODEL producer cannot declare model fields")
        for name, value in values.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "implementation_kind", kind)
        object.__setattr__(
            self,
            "digest",
            canonical_digest(
                {
                    **values,
                    "implementation_kind": kind.value,
                    "schema_version": PRODUCER_FINGERPRINT_SCHEMA_VERSION,
                }
            ),
        )


@dataclass(frozen=True)
class BehaviorSourceDescriptor:
    source_event_ref: SourceEventRef
    stream_ref: StreamRef
    source_sequence: int
    source_item_index: int
    origin_kind: BehaviorOriginKind
    source_ref: str | None
    source_content_digest: str
    parent_source_event_refs: tuple[SourceEventRef, ...]
    correlation_refs: tuple[CorrelationRef, ...]
    causal_refs: tuple[CausalRef, ...]
    projection_ref: ProjectionRef | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.source_event_ref, SourceEventRef):
            raise TypeError("descriptor.source_event_ref must be SourceEventRef")
        if not isinstance(self.stream_ref, StreamRef):
            raise TypeError("descriptor.stream_ref must be StreamRef")
        origin = BehaviorOriginKind(self.origin_kind)
        if not isinstance(self.parent_source_event_refs, tuple) or any(
            not isinstance(item, SourceEventRef) for item in self.parent_source_event_refs
        ):
            raise TypeError("descriptor.parent_source_event_refs must contain SourceEventRef values")
        if not isinstance(self.correlation_refs, tuple) or any(
            not isinstance(item, CorrelationRef) for item in self.correlation_refs
        ):
            raise TypeError("descriptor.correlation_refs must contain CorrelationRef values")
        if not isinstance(self.causal_refs, tuple) or any(not isinstance(item, CausalRef) for item in self.causal_refs):
            raise TypeError("descriptor.causal_refs must contain CausalRef values")
        for name, values in (
            ("parent_source_event_refs", self.parent_source_event_refs),
            ("correlation_refs", self.correlation_refs),
            ("causal_refs", self.causal_refs),
        ):
            if len(values) > 10_000 or len(values) != len(set(values)):
                raise ValueError(f"descriptor.{name} must be bounded and unique")
        if origin is BehaviorOriginKind.CONVERSATION_PROJECTION:
            if not isinstance(self.projection_ref, ProjectionRef):
                raise ValueError("conversation projection origin requires ProjectionRef")
        elif self.projection_ref is not None:
            raise ValueError("ProjectionRef is only valid for conversation projection origin")
        object.__setattr__(self, "source_sequence", non_negative_int(self.source_sequence, "source_sequence"))
        object.__setattr__(self, "source_item_index", non_negative_int(self.source_item_index, "source_item_index"))
        object.__setattr__(self, "origin_kind", origin)
        object.__setattr__(
            self,
            "source_ref",
            None if self.source_ref is None else external_reference(self.source_ref, "source_ref", maximum=2_048),
        )
        object.__setattr__(
            self,
            "source_content_digest",
            sha256_digest(self.source_content_digest, "source_content_digest"),
        )


@dataclass(frozen=True)
class BehaviorSourceProvenance:
    descriptor: BehaviorSourceDescriptor
    adapter_name: str
    producer_fingerprint: ProducerFingerprint
    capability_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.descriptor, BehaviorSourceDescriptor):
            raise TypeError("provenance.descriptor must be BehaviorSourceDescriptor")
        if not isinstance(self.producer_fingerprint, ProducerFingerprint):
            raise TypeError("provenance.producer_fingerprint must be ProducerFingerprint")
        object.__setattr__(self, "adapter_name", identifier(self.adapter_name, "provenance.adapter_name"))
        object.__setattr__(
            self,
            "capability_digest",
            sha256_digest(self.capability_digest, "provenance.capability_digest"),
        )


def producer_to_dict(value: ProducerFingerprint) -> dict[str, Any]:
    return {
        "producer_name": value.producer_name,
        "producer_version": value.producer_version,
        "pipeline_version": value.pipeline_version,
        "output_schema_version": value.output_schema_version,
        "implementation_kind": value.implementation_kind.value,
        "model_provider": value.model_provider,
        "model_name": value.model_name,
        "prompt_version": value.prompt_version,
        "digest": value.digest,
    }


def producer_from_dict(value: object) -> ProducerFingerprint:
    fields = frozenset(
        {
            "producer_name",
            "producer_version",
            "pipeline_version",
            "output_schema_version",
            "implementation_kind",
            "model_provider",
            "model_name",
            "prompt_version",
            "digest",
        }
    )
    data = strict_fields(value, "producer_fingerprint", fields)
    require_fields(data, "producer_fingerprint", fields)
    result = ProducerFingerprint(
        producer_name=data["producer_name"],
        producer_version=data["producer_version"],
        pipeline_version=data["pipeline_version"],
        output_schema_version=data["output_schema_version"],
        implementation_kind=ProducerImplementationKind(data["implementation_kind"]),
        model_provider=data["model_provider"],
        model_name=data["model_name"],
        prompt_version=data["prompt_version"],
    )
    if result.digest != data["digest"]:
        raise ValueError("producer fingerprint digest mismatch")
    return result


def descriptor_to_dict(value: BehaviorSourceDescriptor) -> dict[str, Any]:
    return {
        "source_event_ref": _source_event_to_dict(value.source_event_ref),
        "stream_ref": {
            "namespace": value.stream_ref.namespace,
            "value": value.stream_ref.value,
            "generation": value.stream_ref.generation,
        },
        "source_sequence": value.source_sequence,
        "source_item_index": value.source_item_index,
        "origin_kind": value.origin_kind.value,
        "source_ref": value.source_ref,
        "source_content_digest": value.source_content_digest,
        "parent_source_event_refs": tuple(_source_event_to_dict(item) for item in value.parent_source_event_refs),
        "correlation_refs": tuple(
            {"namespace": item.namespace, "value": item.value, "root_value": item.root_value}
            for item in value.correlation_refs
        ),
        "causal_refs": tuple(
            {
                "kind": item.kind.value,
                "reference": item.reference,
                "reference_digest": item.reference_digest,
            }
            for item in value.causal_refs
        ),
        "projection_ref": None
        if value.projection_ref is None
        else {
            "namespace": value.projection_ref.namespace,
            "value": value.projection_ref.value,
            "source_digest": value.projection_ref.source_digest,
        },
    }


def descriptor_from_dict(value: object) -> BehaviorSourceDescriptor:
    fields = frozenset(
        {
            "source_event_ref",
            "stream_ref",
            "source_sequence",
            "source_item_index",
            "origin_kind",
            "source_ref",
            "source_content_digest",
            "parent_source_event_refs",
            "correlation_refs",
            "causal_refs",
            "projection_ref",
        }
    )
    data = strict_fields(value, "source_descriptor", fields)
    require_fields(data, "source_descriptor", fields)
    stream = strict_fields(data["stream_ref"], "stream_ref", frozenset({"namespace", "value", "generation"}))
    require_fields(stream, "stream_ref", frozenset({"namespace", "value", "generation"}))
    projection = data["projection_ref"]
    if projection is not None:
        projection_data = strict_fields(
            projection,
            "projection_ref",
            frozenset({"namespace", "value", "source_digest"}),
        )
        require_fields(projection_data, "projection_ref", frozenset({"namespace", "value", "source_digest"}))
        projection_value = ProjectionRef(**projection_data)
    else:
        projection_value = None
    return BehaviorSourceDescriptor(
        source_event_ref=_source_event_from_dict(data["source_event_ref"]),
        stream_ref=StreamRef(stream["namespace"], stream["value"], stream["generation"]),
        source_sequence=data["source_sequence"],
        source_item_index=data["source_item_index"],
        origin_kind=BehaviorOriginKind(data["origin_kind"]),
        source_ref=data["source_ref"],
        source_content_digest=data["source_content_digest"],
        parent_source_event_refs=tuple(
            _source_event_from_dict(item) for item in _array(data["parent_source_event_refs"], "parent sources")
        ),
        correlation_refs=tuple(
            _correlation_from_dict(item) for item in _array(data["correlation_refs"], "correlations")
        ),
        causal_refs=tuple(_causal_from_dict(item) for item in _array(data["causal_refs"], "causal refs")),
        projection_ref=projection_value,
    )


def provenance_to_dict(value: BehaviorSourceProvenance) -> dict[str, Any]:
    return {
        "descriptor": descriptor_to_dict(value.descriptor),
        "adapter_name": value.adapter_name,
        "producer_fingerprint": producer_to_dict(value.producer_fingerprint),
        "capability_digest": value.capability_digest,
    }


def provenance_from_dict(value: object) -> BehaviorSourceProvenance:
    fields = frozenset({"descriptor", "adapter_name", "producer_fingerprint", "capability_digest"})
    data = strict_fields(value, "provenance", fields)
    require_fields(data, "provenance", fields)
    return BehaviorSourceProvenance(
        descriptor=descriptor_from_dict(data["descriptor"]),
        adapter_name=data["adapter_name"],
        producer_fingerprint=producer_from_dict(data["producer_fingerprint"]),
        capability_digest=data["capability_digest"],
    )


def _source_event_to_dict(value: SourceEventRef) -> dict[str, str]:
    return {"namespace": value.namespace, "value": value.value, "identity_digest": value.identity_digest}


def _source_event_from_dict(value: object) -> SourceEventRef:
    fields = frozenset({"namespace", "value", "identity_digest"})
    data = strict_fields(value, "source_event_ref", fields)
    require_fields(data, "source_event_ref", fields)
    result = SourceEventRef(data["namespace"], data["value"])
    if result.identity_digest != data["identity_digest"]:
        raise ValueError("source event identity digest mismatch")
    return result


def _correlation_from_dict(value: object) -> CorrelationRef:
    fields = frozenset({"namespace", "value", "root_value"})
    data = strict_fields(value, "correlation_ref", fields)
    require_fields(data, "correlation_ref", fields)
    return CorrelationRef(data["namespace"], data["value"], data["root_value"])


def _causal_from_dict(value: object) -> CausalRef:
    fields = frozenset({"kind", "reference", "reference_digest"})
    data = strict_fields(value, "causal_ref", fields)
    require_fields(data, "causal_ref", fields)
    return CausalRef(CausalRefKind(data["kind"]), data["reference"], data["reference_digest"])


def _array(value: object, field_name: str) -> tuple[Any, ...] | list[Any]:
    if not isinstance(value, tuple | list):
        raise TypeError(f"{field_name} must be an array")
    return value


__all__ = [
    "BehaviorOriginKind",
    "BehaviorSourceDescriptor",
    "BehaviorSourceProvenance",
    "ProducerFingerprint",
    "ProducerImplementationKind",
    "descriptor_from_dict",
    "descriptor_to_dict",
    "producer_from_dict",
    "producer_to_dict",
    "provenance_from_dict",
    "provenance_to_dict",
]
