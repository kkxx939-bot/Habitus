"""Build a complete, deterministic Core and Enhancement normalization plan."""

from __future__ import annotations

from dataclasses import dataclass

from behavior._validation import identifier, sha256_digest
from behavior.claim.normalizer import (
    ClaimNormalizer,
    ClaimNormalizerKind,
    DeterministicClaimNormalizer,
    ModelClaimNormalizer,
)
from behavior.claim.policy import ClaimNormalizerRequirement
from behavior.claim.registry import ClaimNormalizerRegistry
from behavior.config import ClaimConfig
from behavior.errors import ClaimProductionError
from behavior.evidence.manifest import EvidenceManifest
from behavior.ingress.model import OwnerScopedSemanticRecord, SemanticRecordKind
from foundation.integrity import canonical_digest


@dataclass(frozen=True)
class ClaimNormalizationRoute:
    semantic_record_id: str
    semantic_record_digest: str
    normalizer_name: str
    normalizer_fingerprint: str
    normalizer_requirement: ClaimNormalizerRequirement
    normalizer: ClaimNormalizer

    def __post_init__(self) -> None:
        object.__setattr__(self, "semantic_record_id", identifier(self.semantic_record_id, "semantic_record_id"))
        for name in ("semantic_record_digest", "normalizer_fingerprint"):
            object.__setattr__(self, name, sha256_digest(getattr(self, name), name))
        object.__setattr__(self, "normalizer_name", identifier(self.normalizer_name, "normalizer_name"))
        object.__setattr__(
            self,
            "normalizer_requirement",
            ClaimNormalizerRequirement(self.normalizer_requirement),
        )
        if not isinstance(self.normalizer, ClaimNormalizer):
            raise TypeError("normalizer must implement ClaimNormalizer")
        if self.normalizer.name != self.normalizer_name or self.normalizer.fingerprint.digest != self.normalizer_fingerprint:
            raise ClaimProductionError("route Normalizer identity mismatch")

    @property
    def route_identity(self) -> str:
        return canonical_digest(
            {
                "semantic_record_digest": self.semantic_record_digest,
                "normalizer_fingerprint": self.normalizer_fingerprint,
                "requirement": self.normalizer_requirement.value,
            }
        )


@dataclass(frozen=True)
class ClaimNormalizationPlan:
    manifest_id: str
    manifest_digest: str
    core_routes: tuple[ClaimNormalizationRoute, ...]
    enhancement_routes: tuple[ClaimNormalizationRoute, ...]
    routing_policy_digest: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "manifest_id", identifier(self.manifest_id, "manifest_id"))
        object.__setattr__(self, "manifest_digest", sha256_digest(self.manifest_digest, "manifest_digest"))
        object.__setattr__(self, "routing_policy_digest", sha256_digest(self.routing_policy_digest, "routing_policy_digest"))
        routes = (*self.core_routes, *self.enhancement_routes)
        if len({item.route_identity for item in routes}) != len(routes):
            raise ClaimProductionError("normalization plan contains a duplicate route")
        if any(
            item.normalizer_requirement is not ClaimNormalizerRequirement.REQUIRED_CORE
            for item in self.core_routes
        ):
            raise ClaimProductionError("Core plan contains an optional route")
        if any(
            item.normalizer_requirement is not ClaimNormalizerRequirement.OPTIONAL_ENHANCEMENT
            for item in self.enhancement_routes
        ):
            raise ClaimProductionError("Enhancement plan contains a required route")


class ClaimNormalizationRouter:
    def __init__(
        self,
        registry: ClaimNormalizerRegistry,
        *,
        config: ClaimConfig,
        deterministic_name: str = DeterministicClaimNormalizer.name,
        model_name: str = ModelClaimNormalizer.name,
        routing_policy_version: str = "3",
    ) -> None:
        if not isinstance(registry, ClaimNormalizerRegistry):
            raise TypeError("registry must be ClaimNormalizerRegistry")
        if not isinstance(config, ClaimConfig):
            raise TypeError("config must be ClaimConfig")
        self.registry = registry
        self.config = config
        self.routing_policy_version = identifier(
            routing_policy_version,
            "routing_policy_version",
            maximum=32,
        )
        self.deterministic_name = registry.normalize_name(deterministic_name)
        self.model_name = registry.normalize_name(model_name)
        deterministic = registry.get(self.deterministic_name)
        model = registry.get(self.model_name)
        if deterministic.kind is not ClaimNormalizerKind.DETERMINISTIC:
            raise ClaimProductionError("configured Core Normalizer is not deterministic")
        if model.kind is not ClaimNormalizerKind.MODEL:
            raise ClaimProductionError("configured Enhancement Normalizer is not model-backed")

    @property
    def routing_policy_digest(self) -> str:
        deterministic = self.registry.get(self.deterministic_name)
        model_digest = self.registry.get(self.model_name).fingerprint.digest
        return canonical_digest(
            {
                "version": self.routing_policy_version,
                "normalize_owner_utterances": self.config.normalize_owner_utterances,
                "deterministic_fingerprint": deterministic.fingerprint.digest,
                "model_fingerprint": model_digest,
                "algorithm": "core_then_independent_enhancement",
            }
        )

    def plan(
        self,
        manifest: EvidenceManifest,
        records: tuple[OwnerScopedSemanticRecord, ...],
    ) -> ClaimNormalizationPlan:
        if not isinstance(manifest, EvidenceManifest):
            raise TypeError("manifest must be EvidenceManifest")
        by_id = {item.semantic_record_id: item for item in records}
        if len(by_id) != len(records):
            raise ClaimProductionError("normalization records contain duplicate identities")
        core: list[ClaimNormalizationRoute] = []
        enhancements: list[ClaimNormalizationRoute] = []
        for snapshot in manifest.ordered_record_snapshots:
            try:
                record = by_id[snapshot.semantic_record_id]
            except KeyError as exc:
                raise ClaimProductionError("Manifest record is missing from normalization input") from exc
            if record.semantic_digest != snapshot.semantic_record_digest:
                raise ClaimProductionError("Manifest record digest conflicts with normalization input")
            kind = record.semantic_input.record_kind
            if kind is not SemanticRecordKind.FREE_TEXT_SEMANTIC:
                core.append(self._route(record, self.deterministic_name, ClaimNormalizerRequirement.REQUIRED_CORE))
            if kind is SemanticRecordKind.FREE_TEXT_SEMANTIC or (
                kind is SemanticRecordKind.OWNER_UTTERANCE_SEGMENT and self.config.normalize_owner_utterances
            ):
                enhancements.append(
                    self._route(record, self.model_name, ClaimNormalizerRequirement.OPTIONAL_ENHANCEMENT)
                )
        if len(core) + len(enhancements) > self.config.max_normalizers_per_processing:
            raise ClaimProductionError("normalization plan exceeds its configured route boundary")
        return ClaimNormalizationPlan(
            manifest_id=manifest.manifest_id,
            manifest_digest=manifest.manifest_semantic_digest,
            core_routes=tuple(core),
            enhancement_routes=tuple(enhancements),
            routing_policy_digest=self.routing_policy_digest,
        )

    def _route(
        self,
        record: OwnerScopedSemanticRecord,
        name: str,
        requirement: ClaimNormalizerRequirement,
    ) -> ClaimNormalizationRoute:
        normalizer = self.registry.get(name)
        expected_kind = (
            ClaimNormalizerKind.DETERMINISTIC
            if requirement is ClaimNormalizerRequirement.REQUIRED_CORE
            else ClaimNormalizerKind.MODEL
        )
        if normalizer.kind is not expected_kind or record.semantic_input.record_kind not in normalizer.allowed_record_kinds:
            raise ClaimProductionError("Normalizer route is incompatible with the semantic record")
        return ClaimNormalizationRoute(
            semantic_record_id=record.semantic_record_id,
            semantic_record_digest=record.semantic_digest,
            normalizer_name=normalizer.name,
            normalizer_fingerprint=normalizer.fingerprint.digest,
            normalizer_requirement=requirement,
            normalizer=normalizer,
        )


__all__ = ["ClaimNormalizationPlan", "ClaimNormalizationRoute", "ClaimNormalizationRouter"]
