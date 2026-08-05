"""Evidence & Claim Layer 的领域内强类型配置。"""

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
class SourceConfig:
    max_payload_ref_chars: int = 2_048
    max_semantic_text_chars: int = 16_384
    max_semantic_data_chars: int = 16_384
    max_attributes_chars: int = 8_192
    max_track_refs: int = 64
    max_entity_refs: int = 64
    max_batch_size: int = 256

    def __post_init__(self) -> None:
        for name in (
            "max_payload_ref_chars",
            "max_semantic_text_chars",
            "max_semantic_data_chars",
            "max_attributes_chars",
        ):
            _bounded_int(f"source.{name}", getattr(self, name), 64, 1_000_000)
        for name in ("max_track_refs", "max_entity_refs", "max_batch_size"):
            _bounded_int(f"source.{name}", getattr(self, name), 1, 10_000)


@dataclass(frozen=True)
class EvidenceConfig:
    allowed_lateness_seconds: float = 30.0
    max_gap_seconds: float = 15.0
    max_window_duration_seconds: float = 300.0
    max_records_per_window: int = 256
    max_projection_chars_per_window: int = 32_768
    max_active_windows: int = 1_024
    max_blind_intervals: int = 64
    max_query_limit: int = 500

    def __post_init__(self) -> None:
        for name in ("allowed_lateness_seconds", "max_gap_seconds", "max_window_duration_seconds"):
            _bounded_float(f"evidence.{name}", getattr(self, name), 0.0, 86_400.0)
        for name in (
            "max_records_per_window",
            "max_projection_chars_per_window",
            "max_active_windows",
            "max_blind_intervals",
            "max_query_limit",
        ):
            _bounded_int(f"evidence.{name}", getattr(self, name), 1, 1_000_000)
        if self.max_gap_seconds > self.max_window_duration_seconds:
            raise ValueError("evidence.max_gap_seconds cannot exceed max_window_duration_seconds")


@dataclass(frozen=True)
class ClaimConfig:
    max_producers_per_processing: int = 16
    max_claims_per_batch: int = 64
    max_model_input_chars: int = 49_152
    max_model_output_tokens: int = 4_096
    max_semantic_payload_chars: int = 32_768
    max_human_summary_chars: int = 2_048
    min_direct_score: float = 0.0
    min_model_score: float = 0.55
    repeat_state_suppression_seconds: float = 30.0
    max_alternative_group_size: int = 16
    max_query_limit: int = 500

    def __post_init__(self) -> None:
        schema_hard_limits = {
            "max_producers_per_processing": 16,
            "max_claims_per_batch": 64,
            "max_semantic_payload_chars": 32_768,
            "max_human_summary_chars": 2_048,
            "max_alternative_group_size": 16,
        }
        for name, maximum in schema_hard_limits.items():
            _bounded_int(f"claim.{name}", getattr(self, name), 1, maximum)
        for name in ("max_model_input_chars", "max_model_output_tokens", "max_query_limit"):
            _bounded_int(f"claim.{name}", getattr(self, name), 1, 1_000_000)
        for name in ("min_direct_score", "min_model_score"):
            _bounded_float(f"claim.{name}", getattr(self, name), 0.0, 1.0)
        _bounded_float(
            "claim.repeat_state_suppression_seconds",
            self.repeat_state_suppression_seconds,
            0.0,
            86_400.0,
        )


@dataclass(frozen=True)
class StoreConfig:
    sqlite_timeout_seconds: float = 5.0
    max_json_bytes: int = 16_777_216
    max_claims: int = 1_000_000
    max_receipts: int = 100_000
    max_query_limit: int = 500

    def __post_init__(self) -> None:
        _bounded_float("store.sqlite_timeout_seconds", self.sqlite_timeout_seconds, 0.1, 300.0)
        for name in ("max_json_bytes", "max_claims", "max_receipts", "max_query_limit"):
            _bounded_int(f"store.{name}", getattr(self, name), 1, 100_000_000)


@dataclass(frozen=True)
class BehaviorConfig:
    source: SourceConfig = field(default_factory=SourceConfig)
    evidence: EvidenceConfig = field(default_factory=EvidenceConfig)
    claim: ClaimConfig = field(default_factory=ClaimConfig)
    store: StoreConfig = field(default_factory=StoreConfig)

    def __post_init__(self) -> None:
        expected = (
            ("source", self.source, SourceConfig),
            ("evidence", self.evidence, EvidenceConfig),
            ("claim", self.claim, ClaimConfig),
            ("store", self.store, StoreConfig),
        )
        for name, value, value_type in expected:
            if not isinstance(value, value_type):
                raise TypeError(f"behavior.{name} must be {value_type.__name__}")
        if self.source.max_semantic_text_chars > self.claim.max_model_input_chars:
            raise ValueError("one source semantic projection cannot exceed the model input boundary")
        if (
            self.source.max_semantic_text_chars + self.source.max_semantic_data_chars
            > self.evidence.max_projection_chars_per_window
        ):
            raise ValueError("one maximum SourceRecord projection cannot fit an EvidenceWindow")
        if (
            self.evidence.max_projection_chars_per_window + 4_096
            > self.claim.max_model_input_chars
        ):
            raise ValueError("sealed EvidenceWindow projection cannot fit the model input boundary")
        manifest_upper_bound = (
            self.evidence.max_projection_chars_per_window
            + self.evidence.max_records_per_window
            * (
                self.source.max_payload_ref_chars
                + (self.source.max_track_refs + self.source.max_entity_refs) * 260
                + 2_048
            )
            + 262_144
        )
        if manifest_upper_bound > self.store.max_json_bytes:
            raise ValueError("one configured EvidenceManifest cannot fit the Store JSON boundary")
        if max(self.evidence.max_query_limit, self.claim.max_query_limit) > self.store.max_query_limit:
            raise ValueError("domain query limits cannot exceed the Store query boundary")
        if self.claim.max_query_limit > self.store.max_query_limit:
            raise ValueError("claim.max_query_limit cannot exceed store.max_query_limit")
        if self.evidence.max_query_limit > self.store.max_query_limit:
            raise ValueError("evidence.max_query_limit cannot exceed store.max_query_limit")


__all__ = ["BehaviorConfig", "ClaimConfig", "EvidenceConfig", "SourceConfig", "StoreConfig"]
