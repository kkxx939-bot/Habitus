"""Adapter 能力、时间模式与系统绑定的来源信任。"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from behavior._validation import enum_tuple, positive_int
from behavior.evidence.content import BehaviorModality, BehaviorRecordKind, BehaviorRole
from behavior.evidence.provenance import BehaviorOriginKind
from foundation.integrity import canonical_digest

ADAPTER_CAPABILITY_SCHEMA_VERSION = "behavior_adapter_capability_v1"


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


RolePair = tuple[BehaviorRole, BehaviorRole | None]


@dataclass(frozen=True)
class BehaviorAdapterCapability:
    source_trust: BehaviorSourceTrust
    time_mode: BehaviorTimeMode
    allowed_origin_kinds: tuple[BehaviorOriginKind, ...]
    allowed_record_kinds: tuple[BehaviorRecordKind, ...]
    allowed_modalities: tuple[BehaviorModality, ...]
    allowed_role_pairs: tuple[RolePair, ...]
    maximum_batch_size: int
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        trust = BehaviorSourceTrust(self.source_trust)
        time_mode = BehaviorTimeMode(self.time_mode)
        origins = enum_tuple(
            self.allowed_origin_kinds,
            "allowed_origin_kinds",
            BehaviorOriginKind,
            maximum_items=10_000,
        )
        record_kinds = enum_tuple(
            self.allowed_record_kinds,
            "allowed_record_kinds",
            BehaviorRecordKind,
            maximum_items=10_000,
        )
        modalities = enum_tuple(
            self.allowed_modalities,
            "allowed_modalities",
            BehaviorModality,
            maximum_items=10_000,
        )
        if not isinstance(self.allowed_role_pairs, tuple) or not self.allowed_role_pairs:
            raise ValueError("allowed_role_pairs must be a non-empty tuple")
        role_pairs: list[RolePair] = []
        for item in self.allowed_role_pairs:
            if not isinstance(item, tuple) or len(item) != 2:
                raise TypeError("allowed_role_pairs must contain two-item tuples")
            role_pairs.append(
                (
                    BehaviorRole(item[0]),
                    None if item[1] is None else BehaviorRole(item[1]),
                )
            )
        if len(set(role_pairs)) != len(role_pairs):
            raise ValueError("allowed_role_pairs must not contain duplicates")
        _validate_trust_origin(trust, origins)
        maximum = positive_int(self.maximum_batch_size, "maximum_batch_size")
        object.__setattr__(self, "source_trust", trust)
        object.__setattr__(self, "time_mode", time_mode)
        object.__setattr__(self, "allowed_origin_kinds", origins)
        object.__setattr__(self, "allowed_record_kinds", record_kinds)
        object.__setattr__(self, "allowed_modalities", modalities)
        object.__setattr__(self, "allowed_role_pairs", tuple(role_pairs))
        object.__setattr__(self, "maximum_batch_size", maximum)
        object.__setattr__(
            self,
            "digest",
            canonical_digest(
                {
                    "allowed_modalities": [item.value for item in modalities],
                    "allowed_origin_kinds": [item.value for item in origins],
                    "allowed_record_kinds": [item.value for item in record_kinds],
                    "allowed_role_pairs": [
                        [subject.value, None if actor is None else actor.value]
                        for subject, actor in role_pairs
                    ],
                    "maximum_batch_size": maximum,
                    "schema_version": ADAPTER_CAPABILITY_SCHEMA_VERSION,
                    "source_trust": trust.value,
                    "time_mode": time_mode.value,
                }
            ),
        )

    def permits(
        self,
        *,
        origin_kind: BehaviorOriginKind,
        record_kind: BehaviorRecordKind,
        modality: BehaviorModality,
        subject_role: BehaviorRole,
        actor_role: BehaviorRole | None,
    ) -> bool:
        return (
            origin_kind in self.allowed_origin_kinds
            and record_kind in self.allowed_record_kinds
            and modality in self.allowed_modalities
            and (subject_role, actor_role) in self.allowed_role_pairs
        )

def _validate_trust_origin(
    trust: BehaviorSourceTrust,
    origins: tuple[BehaviorOriginKind, ...],
) -> None:
    if BehaviorOriginKind.DIRECT_AMBIENT_ASR in origins and trust is not BehaviorSourceTrust.USER_EXPLICIT:
        raise ValueError("ambient ASR capability requires USER_EXPLICIT trust")
    if BehaviorOriginKind.DIRECT_RUNTIME_EVENT in origins and trust is not BehaviorSourceTrust.DIRECT_SYSTEM_LOG:
        raise ValueError("runtime event capability requires DIRECT_SYSTEM_LOG trust")
    if BehaviorOriginKind.DIRECT_PERCEPTION in origins and trust not in {
        BehaviorSourceTrust.MODEL_INFERRED,
        BehaviorSourceTrust.MULTIMODAL_MODEL_INFERRED,
        BehaviorSourceTrust.SENSOR_INFERRED,
        BehaviorSourceTrust.DIRECT_DEVICE_FACT,
    }:
        raise ValueError("direct perception capability has an incompatible trust class")


__all__ = [
    "BehaviorAdapterCapability",
    "BehaviorSourceTrust",
    "BehaviorTimeMode",
]
