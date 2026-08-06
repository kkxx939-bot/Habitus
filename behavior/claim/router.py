"""按 SemanticRecordKind 自动选择唯一规范化路径。"""

from __future__ import annotations

from behavior.claim.normalizer import ClaimNormalizer, DeterministicClaimNormalizer, ModelClaimNormalizer
from behavior.claim.registry import ClaimNormalizerRegistry
from behavior.config import ClaimConfig
from behavior.errors import ClaimProductionError
from behavior.ingress.model import OwnerScopedSemanticRecord, SemanticRecordKind


class ClaimNormalizationRouter:
    def __init__(
        self,
        registry: ClaimNormalizerRegistry,
        *,
        config: ClaimConfig,
        deterministic_name: str = DeterministicClaimNormalizer.name,
        model_name: str = ModelClaimNormalizer.name,
    ) -> None:
        if not isinstance(registry, ClaimNormalizerRegistry):
            raise TypeError("registry must be ClaimNormalizerRegistry")
        if not isinstance(config, ClaimConfig):
            raise TypeError("config must be ClaimConfig")
        self.registry = registry
        self.config = config
        self.deterministic_name = registry.normalize_name(deterministic_name)
        self.model_name = registry.normalize_name(model_name)

    def route(self, record: OwnerScopedSemanticRecord) -> tuple[ClaimNormalizer, ...]:
        if not isinstance(record, OwnerScopedSemanticRecord):
            raise TypeError("record must be OwnerScopedSemanticRecord")
        kind = record.semantic_input.record_kind
        routed: tuple[ClaimNormalizer, ...]
        if kind is SemanticRecordKind.FREE_TEXT_SEMANTIC:
            routed = (self.registry.get(self.model_name),)
        elif kind is SemanticRecordKind.OWNER_UTTERANCE_SEGMENT and self.config.normalize_owner_utterances:
            routed = (
                self.registry.get(self.deterministic_name),
                self.registry.get(self.model_name),
            )
        else:
            routed = (self.registry.get(self.deterministic_name),)
        if len(routed) > self.config.max_normalizers_per_processing:
            raise ClaimProductionError("automatic Normalizer route exceeds its configured boundary")
        if len({item.fingerprint.digest for item in routed}) != len(routed):
            raise ClaimProductionError("one semantic record cannot use equivalent Normalizers twice")
        if any(kind not in item.allowed_record_kinds for item in routed):
            raise ClaimProductionError("Normalizer route is incompatible with the semantic record kind")
        return routed


__all__ = ["ClaimNormalizationRouter"]
