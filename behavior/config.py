"""Behavior Evidence & Claim Layer 的领域内强类型配置。"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


def _bounded_int(name: str, value: object, minimum: int, maximum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer between {minimum} and {maximum}")


def _bounded_float(name: str, value: object, minimum: float, maximum: float) -> None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be numeric")
    if not math.isfinite(float(value)) or not minimum <= float(value) <= maximum:
        raise ValueError(f"{name} must be finite and between {minimum} and {maximum}")


@dataclass(frozen=True)
class IngressConfig:
    max_evidence_refs: int = 32
    max_object_refs: int = 64
    max_entity_refs: int = 64
    max_payload_chars: int = 32_768
    max_text_chars: int = 16_384
    max_batch_size: int = 256
    max_event_time_uncertainty_ms: int = 300_000
    max_future_event_skew_seconds: float = 300.0
    max_past_event_age_seconds: float = 31_536_000.0
    allowed_lateness_seconds: float = 30.0
    max_query_limit: int = 500
    max_reference_chars: int = 2_048
    max_identifier_chars: int = 256
    max_payload_items: int = 128
    max_payload_depth: int = 12

    def __post_init__(self) -> None:
        for name in (
            "max_evidence_refs",
            "max_object_refs",
            "max_entity_refs",
            "max_batch_size",
            "max_query_limit",
            "max_payload_items",
        ):
            _bounded_int(f"ingress.{name}", getattr(self, name), 1, 100_000)
        for name in ("max_payload_chars", "max_text_chars", "max_reference_chars"):
            _bounded_int(f"ingress.{name}", getattr(self, name), 32, 1_000_000)
        _bounded_int("ingress.max_identifier_chars", self.max_identifier_chars, 16, 256)
        _bounded_int("ingress.max_payload_depth", self.max_payload_depth, 1, 32)
        _bounded_int(
            "ingress.max_event_time_uncertainty_ms",
            self.max_event_time_uncertainty_ms,
            0,
            86_400_000,
        )
        for name in (
            "max_future_event_skew_seconds",
            "max_past_event_age_seconds",
            "allowed_lateness_seconds",
        ):
            _bounded_float(f"ingress.{name}", getattr(self, name), 0.0, 315_360_000.0)


@dataclass(frozen=True)
class EvidenceConfig:
    allowed_lateness_seconds: float = 30.0
    max_gap_seconds: float = 15.0
    max_bundle_duration_seconds: float = 300.0
    max_records_per_bundle: int = 256
    max_projection_chars_per_bundle: int = 65_536
    max_active_bundles: int = 1_024
    max_coverage_intervals: int = 256
    max_manifest_encoded_bytes: int = 4_194_304
    max_query_limit: int = 500

    def __post_init__(self) -> None:
        for name in (
            "allowed_lateness_seconds",
            "max_gap_seconds",
            "max_bundle_duration_seconds",
        ):
            _bounded_float(f"evidence.{name}", getattr(self, name), 0.0, 86_400.0)
        for name in (
            "max_records_per_bundle",
            "max_projection_chars_per_bundle",
            "max_active_bundles",
            "max_coverage_intervals",
            "max_query_limit",
        ):
            _bounded_int(f"evidence.{name}", getattr(self, name), 1, 1_000_000)
        _bounded_int(
            "evidence.max_manifest_encoded_bytes",
            self.max_manifest_encoded_bytes,
            1,
            1_000_000_000,
        )
        if self.max_gap_seconds > self.max_bundle_duration_seconds:
            raise ValueError("evidence.max_gap_seconds cannot exceed max_bundle_duration_seconds")


@dataclass(frozen=True)
class ClaimConfig:
    max_claims_per_record: int = 16
    max_claims_per_batch: int = 64
    max_normalizers_per_processing: int = 1_024
    max_claims_per_processing: int = 10_000
    max_model_input_chars: int = 49_152
    max_model_input_tokens: int = 16_384
    max_model_output_tokens: int = 4_096
    max_semantic_payload_chars: int = 32_768
    max_human_summary_chars: int = 2_048
    min_direct_confidence: float = 0.0
    min_sensor_confidence: float = 0.35
    min_model_confidence: float = 0.55
    repeat_state_suppression_seconds: float = 30.0
    max_alternative_group_size: int = 16
    normalize_owner_utterances: bool = False
    max_query_limit: int = 500

    def __post_init__(self) -> None:
        for name, maximum in {
            "max_claims_per_record": 100_000,
            "max_claims_per_batch": 100_000,
            "max_normalizers_per_processing": 100_000,
            "max_claims_per_processing": 100_000,
            "max_alternative_group_size": 64,
        }.items():
            _bounded_int(f"claim.{name}", getattr(self, name), 1, maximum)
        for name in (
            "max_model_input_chars",
            "max_model_input_tokens",
            "max_model_output_tokens",
            "max_semantic_payload_chars",
            "max_human_summary_chars",
            "max_query_limit",
        ):
            _bounded_int(f"claim.{name}", getattr(self, name), 1, 1_000_000)
        for name in (
            "min_direct_confidence",
            "min_sensor_confidence",
            "min_model_confidence",
        ):
            _bounded_float(f"claim.{name}", getattr(self, name), 0.0, 1.0)
        _bounded_float(
            "claim.repeat_state_suppression_seconds",
            self.repeat_state_suppression_seconds,
            0.0,
            86_400.0,
        )
        if not isinstance(self.normalize_owner_utterances, bool):
            raise TypeError("claim.normalize_owner_utterances must be boolean")
        if self.max_claims_per_record > self.max_claims_per_batch:
            raise ValueError("claim.max_claims_per_record cannot exceed max_claims_per_batch")


@dataclass(frozen=True)
class StoreConfig:
    sqlite_timeout_seconds: float = 5.0
    max_json_bytes: int = 16_777_216
    max_semantic_records: int = 1_000_000
    max_ingress_decisions: int = 1_000_000
    max_active_bundles: int = 1_024
    max_manifests: int = 250_000
    max_normalizer_attempts: int = 1_000_000
    max_claim_batches: int = 1_000_000
    max_validated_claims: int = 1_000_000
    max_accepted_claims: int = 1_000_000
    max_admission_decisions: int = 1_000_000
    max_processing_receipts: int = 250_000
    max_database_bytes: int = 8_589_934_592
    max_query_limit: int = 500

    def __post_init__(self) -> None:
        _bounded_float("store.sqlite_timeout_seconds", self.sqlite_timeout_seconds, 0.1, 300.0)
        for name in (
            "max_json_bytes",
            "max_semantic_records",
            "max_ingress_decisions",
            "max_active_bundles",
            "max_manifests",
            "max_normalizer_attempts",
            "max_claim_batches",
            "max_validated_claims",
            "max_accepted_claims",
            "max_admission_decisions",
            "max_processing_receipts",
            "max_database_bytes",
            "max_query_limit",
        ):
            _bounded_int(f"store.{name}", getattr(self, name), 1, 1_000_000_000_000)


@dataclass(frozen=True)
class BehaviorConfig:
    ingress: IngressConfig = field(default_factory=IngressConfig)
    evidence: EvidenceConfig = field(default_factory=EvidenceConfig)
    claim: ClaimConfig = field(default_factory=ClaimConfig)
    store: StoreConfig = field(default_factory=StoreConfig)

    def __post_init__(self) -> None:
        expected = (
            ("ingress", self.ingress, IngressConfig),
            ("evidence", self.evidence, EvidenceConfig),
            ("claim", self.claim, ClaimConfig),
            ("store", self.store, StoreConfig),
        )
        for name, value, value_type in expected:
            if not isinstance(value, value_type):
                raise TypeError(f"behavior.{name} must be {value_type.__name__}")
        if self.ingress.allowed_lateness_seconds != self.evidence.allowed_lateness_seconds:
            raise ValueError("ingress and evidence allowed_lateness_seconds must match")
        if self.ingress.max_payload_chars > self.evidence.max_projection_chars_per_bundle:
            raise ValueError("one maximum semantic payload cannot fit an Evidence Bundle")
        if self.evidence.max_active_bundles > self.store.max_active_bundles:
            raise ValueError("evidence active Bundle capacity cannot exceed Store capacity")
        if (
            max(
                self.ingress.max_query_limit,
                self.evidence.max_query_limit,
                self.claim.max_query_limit,
            )
            > self.store.max_query_limit
        ):
            raise ValueError("domain query limits cannot exceed the Store query boundary")
        if self.claim.max_claims_per_processing > self.store.max_validated_claims:
            raise ValueError("one Claim batch cannot exceed Store Claim capacity")
        route_factor = 2 if self.claim.normalize_owner_utterances else 1
        maximum_routes = self.evidence.max_records_per_bundle * route_factor
        if self.claim.max_normalizers_per_processing < maximum_routes:
            raise ValueError("Claim Normalizer capacity cannot process one maximum Evidence Bundle")
        maximum_claims = maximum_routes * self.claim.max_claims_per_record
        if self.claim.max_claims_per_processing < maximum_claims:
            raise ValueError("Claim capacity cannot process one maximum Evidence Bundle")
        if self.store.max_validated_claims > self.store.max_admission_decisions:
            raise ValueError("accepted Claim capacity cannot exceed AdmissionDecision capacity")
        if self.store.max_accepted_claims > self.store.max_validated_claims:
            raise ValueError("accepted Claim capacity cannot exceed validated Claim capacity")
        if self.evidence.max_manifest_encoded_bytes > self.store.max_json_bytes:
            raise ValueError("one EvidenceManifest cannot exceed the Store JSON boundary")
        if self.store.max_json_bytes > self.store.max_database_bytes:
            raise ValueError("one Store JSON value cannot exceed the database byte capacity")


__all__ = ["BehaviorConfig", "ClaimConfig", "EvidenceConfig", "IngressConfig", "StoreConfig"]
