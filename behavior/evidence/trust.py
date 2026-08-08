"""Adapter 精确输出契约、时间模式与系统绑定信任。"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from behavior._validation import positive_int, typed_tuple
from behavior.evidence.content import BehaviorModality, BehaviorRecordKind, BehaviorRole
from behavior.evidence.provenance import BehaviorOriginKind
from foundation.integrity import canonical_digest

ADAPTER_CAPABILITY_SCHEMA_VERSION = "behavior_adapter_capability_v2"


class BehaviorSourceTrust(str, Enum):
    DIRECT_SYSTEM_LOG = "DIRECT_SYSTEM_LOG"
    DIRECT_DEVICE_FACT = "DIRECT_DEVICE_FACT"
    USER_EXPLICIT = "USER_EXPLICIT"
    SENSOR_INFERRED = "SENSOR_INFERRED"
    MODEL_INFERRED = "MODEL_INFERRED"
    MULTIMODAL_MODEL_INFERRED = "MULTIMODAL_MODEL_INFERRED"


class BehaviorTimeMode(str, Enum):
    LIVE = "LIVE"
    BACKFILL = "BACKFILL"


@dataclass(frozen=True, slots=True)
class AdapterOutputContract:
    origin_kind: BehaviorOriginKind
    record_kind: BehaviorRecordKind
    modality: BehaviorModality
    subject_role: BehaviorRole
    actor_role: BehaviorRole | None
    source_trust: BehaviorSourceTrust

    def __post_init__(self) -> None:
        origin = BehaviorOriginKind(self.origin_kind)
        trust = BehaviorSourceTrust(self.source_trust)
        _validate_trust_origin(trust, origin)
        object.__setattr__(self, "origin_kind", origin)
        object.__setattr__(self, "record_kind", BehaviorRecordKind(self.record_kind))
        object.__setattr__(self, "modality", BehaviorModality(self.modality))
        object.__setattr__(self, "subject_role", BehaviorRole(self.subject_role))
        object.__setattr__(
            self,
            "actor_role",
            None if self.actor_role is None else BehaviorRole(self.actor_role),
        )
        object.__setattr__(self, "source_trust", trust)

    def identity(self) -> tuple[str, str, str, str, str | None, str]:
        return (
            self.origin_kind.value,
            self.record_kind.value,
            self.modality.value,
            self.subject_role.value,
            None if self.actor_role is None else self.actor_role.value,
            self.source_trust.value,
        )


@dataclass(frozen=True, slots=True)
class BehaviorAdapterCapability:
    allowed_outputs: tuple[AdapterOutputContract, ...]
    time_mode: BehaviorTimeMode
    maximum_batch_size: int
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        outputs = typed_tuple(
            self.allowed_outputs,
            "allowed_outputs",
            AdapterOutputContract,
            maximum_items=10_000,
            allow_empty=False,
        )
        if len({item.identity() for item in outputs}) != len(outputs):
            raise ValueError("allowed_outputs must contain unique exact contracts")
        mode = BehaviorTimeMode(self.time_mode)
        maximum = positive_int(self.maximum_batch_size, "maximum_batch_size")
        object.__setattr__(self, "allowed_outputs", outputs)
        object.__setattr__(self, "time_mode", mode)
        object.__setattr__(self, "maximum_batch_size", maximum)
        object.__setattr__(
            self,
            "digest",
            canonical_digest(
                {
                    "allowed_outputs": [item.identity() for item in outputs],
                    "maximum_batch_size": maximum,
                    "schema_version": ADAPTER_CAPABILITY_SCHEMA_VERSION,
                    "time_mode": mode.value,
                }
            ),
        )

    def match(
        self,
        *,
        origin_kind: BehaviorOriginKind,
        record_kind: BehaviorRecordKind,
        modality: BehaviorModality,
        subject_role: BehaviorRole,
        actor_role: BehaviorRole | None,
    ) -> AdapterOutputContract | None:
        identity = (
            BehaviorOriginKind(origin_kind),
            BehaviorRecordKind(record_kind),
            BehaviorModality(modality),
            BehaviorRole(subject_role),
            None if actor_role is None else BehaviorRole(actor_role),
        )
        return next(
            (
                item
                for item in self.allowed_outputs
                if (
                    item.origin_kind,
                    item.record_kind,
                    item.modality,
                    item.subject_role,
                    item.actor_role,
                )
                == identity
            ),
            None,
        )


def _validate_trust_origin(
    trust: BehaviorSourceTrust,
    origin: BehaviorOriginKind,
) -> None:
    if origin is BehaviorOriginKind.DIRECT_AMBIENT_ASR and trust is not BehaviorSourceTrust.USER_EXPLICIT:
        raise ValueError("ambient ASR output requires USER_EXPLICIT trust")
    if origin is BehaviorOriginKind.DIRECT_RUNTIME_EVENT and trust is not BehaviorSourceTrust.DIRECT_SYSTEM_LOG:
        raise ValueError("runtime event output requires DIRECT_SYSTEM_LOG trust")
    if origin is BehaviorOriginKind.DIRECT_PERCEPTION and trust not in {
        BehaviorSourceTrust.MODEL_INFERRED,
        BehaviorSourceTrust.MULTIMODAL_MODEL_INFERRED,
        BehaviorSourceTrust.SENSOR_INFERRED,
        BehaviorSourceTrust.DIRECT_DEVICE_FACT,
    }:
        raise ValueError("direct perception output has an incompatible trust class")
