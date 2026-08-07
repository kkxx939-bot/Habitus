"""Behavior Evidence & Claim Layer 的强类型运行边界。"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


def _bounded_int(name: str, value: object, minimum: int, maximum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer between {minimum} and {maximum}")


def _bounded_float(name: str, value: object, minimum: float, maximum: float) -> None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be numeric")
    resolved = float(value)
    if not math.isfinite(resolved) or not minimum <= resolved <= maximum:
        raise ValueError(f"{name} must be finite and between {minimum} and {maximum}")


@dataclass(frozen=True)
class BehaviorEvidenceConfig:
    max_evidence_refs: int = 32
    max_correlation_refs: int = 32
    max_causal_refs: int = 32
    max_parent_source_refs: int = 32
    max_object_refs: int = 64
    max_entity_refs: int = 64
    max_payload_chars: int = 32_768
    max_text_chars: int = 16_384
    max_batch_size: int = 256
    max_event_time_uncertainty_ms: int = 300_000
    max_future_event_skew_seconds: float = 300.0
    max_live_event_age_seconds: float = 31_536_000.0
    max_reference_chars: int = 2_048
    max_identifier_chars: int = 256
    max_payload_items: int = 128
    max_payload_depth: int = 12
    max_query_limit: int = 500

    def __post_init__(self) -> None:
        for name in (
            "max_evidence_refs",
            "max_correlation_refs",
            "max_causal_refs",
            "max_parent_source_refs",
            "max_object_refs",
            "max_entity_refs",
            "max_batch_size",
            "max_payload_items",
            "max_query_limit",
        ):
            _bounded_int(f"evidence.{name}", getattr(self, name), 1, 100_000)
        for name in ("max_payload_chars", "max_text_chars", "max_reference_chars"):
            _bounded_int(f"evidence.{name}", getattr(self, name), 32, 1_000_000)
        _bounded_int("evidence.max_identifier_chars", self.max_identifier_chars, 16, 256)
        _bounded_int("evidence.max_payload_depth", self.max_payload_depth, 1, 32)
        _bounded_int(
            "evidence.max_event_time_uncertainty_ms",
            self.max_event_time_uncertainty_ms,
            0,
            86_400_000,
        )
        _bounded_float(
            "evidence.max_future_event_skew_seconds",
            self.max_future_event_skew_seconds,
            0.0,
            315_360_000.0,
        )
        _bounded_float(
            "evidence.max_live_event_age_seconds",
            self.max_live_event_age_seconds,
            0.0,
            315_360_000.0,
        )


@dataclass(frozen=True)
class ClaimNormalizationConfig:
    max_claims_per_record: int = 16
    max_enhancement_normalizers_per_record: int = 8
    max_model_input_chars: int = 49_152
    max_model_input_tokens: int = 16_384
    max_model_output_tokens: int = 4_096
    max_semantic_payload_chars: int = 32_768
    max_human_summary_chars: int = 2_048
    max_alternative_group_size: int = 16
    normalize_user_utterances: bool = False
    max_query_limit: int = 500

    def __post_init__(self) -> None:
        for name, maximum in {
            "max_claims_per_record": 100_000,
            "max_enhancement_normalizers_per_record": 1_024,
            "max_alternative_group_size": 64,
            "max_query_limit": 100_000,
        }.items():
            _bounded_int(f"normalization.{name}", getattr(self, name), 1, maximum)
        for name in (
            "max_model_input_chars",
            "max_model_input_tokens",
            "max_model_output_tokens",
            "max_semantic_payload_chars",
            "max_human_summary_chars",
        ):
            _bounded_int(f"normalization.{name}", getattr(self, name), 1, 1_000_000)
        if not isinstance(self.normalize_user_utterances, bool):
            raise TypeError("normalization.normalize_user_utterances must be boolean")


@dataclass(frozen=True)
class BehaviorStoreConfig:
    sqlite_timeout_seconds: float = 5.0
    max_json_bytes: int = 16_777_216
    max_evidence_records: int = 1_000_000
    max_ingress_receipts: int = 1_000_000
    max_claims: int = 1_000_000
    max_normalization_attempts: int = 1_000_000
    max_normalization_receipts: int = 1_000_000
    max_database_bytes: int = 8_589_934_592
    max_query_limit: int = 500

    def __post_init__(self) -> None:
        _bounded_float("store.sqlite_timeout_seconds", self.sqlite_timeout_seconds, 0.1, 300.0)
        for name in (
            "max_json_bytes",
            "max_evidence_records",
            "max_ingress_receipts",
            "max_claims",
            "max_normalization_attempts",
            "max_normalization_receipts",
            "max_database_bytes",
            "max_query_limit",
        ):
            _bounded_int(f"store.{name}", getattr(self, name), 1, 1_000_000_000_000)
        if self.max_json_bytes > self.max_database_bytes:
            raise ValueError("store.max_json_bytes cannot exceed max_database_bytes")


@dataclass(frozen=True)
class BehaviorConfig:
    evidence: BehaviorEvidenceConfig = field(default_factory=BehaviorEvidenceConfig)
    normalization: ClaimNormalizationConfig = field(default_factory=ClaimNormalizationConfig)
    store: BehaviorStoreConfig = field(default_factory=BehaviorStoreConfig)

    def __post_init__(self) -> None:
        expected = (
            ("evidence", self.evidence, BehaviorEvidenceConfig),
            ("normalization", self.normalization, ClaimNormalizationConfig),
            ("store", self.store, BehaviorStoreConfig),
        )
        for name, value, expected_type in expected:
            if not isinstance(value, expected_type):
                raise TypeError(f"behavior.{name} must be {expected_type.__name__}")
        if max(self.evidence.max_query_limit, self.normalization.max_query_limit) > self.store.max_query_limit:
            raise ValueError("domain query limits cannot exceed the Store query boundary")
        if self.evidence.max_payload_chars > self.store.max_json_bytes:
            raise ValueError("one semantic payload cannot exceed the Store JSON boundary")
        if self.normalization.max_semantic_payload_chars > self.store.max_json_bytes:
            raise ValueError("one Claim semantic payload cannot exceed the Store JSON boundary")


__all__ = [
    "BehaviorConfig",
    "BehaviorEvidenceConfig",
    "BehaviorStoreConfig",
    "ClaimNormalizationConfig",
]
