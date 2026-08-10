"""一次真实预测运行的不可变领域记录。"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from foundation.integrity import canonical_digest, canonicalize, immutable_snapshot
from prediction.context import PredictionContext
from prediction.model import PredictionKind, PredictionTargetLevel, prediction_sample_id

_RUN_CONTRACT = "prediction-run-v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class PredictionAbstentionReason(str, Enum):
    """没有输出候选时可审计的受控原因。"""

    INSUFFICIENT_CONTEXT = "insufficient_context"
    NO_MATCHING_STATE = "no_matching_state"
    LOW_CONFIDENCE = "low_confidence"
    UNSAFE = "unsafe"
    OUTSIDE_HORIZON = "outside_horizon"


@dataclass(frozen=True)
class PredictionRunSourceBinding:
    """预测时实际使用的不可变来源快照。"""

    uri: str
    revision: int
    digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.uri, str) or "://" not in self.uri or self.uri != self.uri.strip():
            raise ValueError("prediction run source URI must be an absolute URI")
        if isinstance(self.revision, bool) or not isinstance(self.revision, int) or self.revision <= 0:
            raise ValueError("prediction run source revision must be a positive integer")
        if not isinstance(self.digest, str) or _SHA256.fullmatch(self.digest) is None:
            raise ValueError("prediction run source digest must be a lowercase SHA-256 digest")

    def to_dict(self) -> dict[str, Any]:
        return {"uri": self.uri, "revision": self.revision, "digest": self.digest}

    @classmethod
    def from_dict(cls, value: object) -> PredictionRunSourceBinding:
        payload = _exact_mapping(value, {"uri", "revision", "digest"}, "prediction run source")
        return cls(payload["uri"], payload["revision"], payload["digest"])


@dataclass(frozen=True, order=True)
class PredictionRunPatternBinding:
    """预测时实际展开的一个逻辑 State 及其当时激活的那一代。

    模式图按 State 独立激活，不存在覆盖全图的单一代号，
    因此这里记录实际读过的每个 State，比全局代号更精确也更易复现。
    """

    logical_state_key: str
    generation_id: str
    manifest_digest: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "logical_state_key", prediction_sample_id(self.logical_state_key))
        object.__setattr__(self, "generation_id", prediction_sample_id(self.generation_id))
        object.__setattr__(
            self,
            "manifest_digest",
            _sha256(self.manifest_digest, "prediction run Pattern manifest digest"),
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "logical_state_key": self.logical_state_key,
            "generation_id": self.generation_id,
            "manifest_digest": self.manifest_digest,
        }

    @classmethod
    def from_dict(cls, value: object) -> PredictionRunPatternBinding:
        payload = _exact_mapping(
            value,
            {"logical_state_key", "generation_id", "manifest_digest"},
            "prediction run Pattern binding",
        )
        return cls(
            payload["logical_state_key"],
            payload["generation_id"],
            payload["manifest_digest"],
        )


@dataclass(frozen=True)
class PredictionCandidate:
    """一次预测中的一个有序候选，不承担执行授权。"""

    rank: int
    target_level: PredictionTargetLevel
    semantics: str
    probability: float
    expected_delay_seconds: float | None
    payload: Mapping[str, Any]
    evidence_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.rank, bool) or not isinstance(self.rank, int) or self.rank <= 0:
            raise ValueError("prediction candidate rank must be a positive integer")
        object.__setattr__(self, "target_level", PredictionTargetLevel(self.target_level))
        if not isinstance(self.semantics, str) or not self.semantics.strip():
            raise ValueError("prediction candidate semantics must be non-empty text")
        object.__setattr__(self, "semantics", self.semantics.strip())
        probability = _probability(self.probability, "prediction candidate probability")
        object.__setattr__(self, "probability", probability)
        if self.expected_delay_seconds is not None:
            delay = _non_negative_number(
                self.expected_delay_seconds,
                "prediction candidate expected delay",
            )
            object.__setattr__(self, "expected_delay_seconds", delay)
        if not isinstance(self.payload, Mapping):
            raise TypeError("prediction candidate payload must be a mapping")
        object.__setattr__(self, "payload", immutable_snapshot(self.payload))
        refs = tuple(self.evidence_refs)
        if any(not isinstance(ref, str) or not ref.strip() for ref in refs):
            raise ValueError("prediction candidate evidence refs must be non-empty text")
        if len(refs) != len(set(refs)):
            raise ValueError("prediction candidate evidence refs must be unique")
        object.__setattr__(self, "evidence_refs", refs)

    def to_dict(self) -> dict[str, Any]:
        value = canonicalize(
            {
                "rank": self.rank,
                "target_level": self.target_level.value,
                "semantics": self.semantics,
                "probability": self.probability,
                "expected_delay_seconds": self.expected_delay_seconds,
                "payload": self.payload,
                "evidence_refs": self.evidence_refs,
            }
        )
        assert isinstance(value, dict)
        return value

    @classmethod
    def from_dict(cls, value: object) -> PredictionCandidate:
        payload = _exact_mapping(
            value,
            {
                "rank",
                "target_level",
                "semantics",
                "probability",
                "expected_delay_seconds",
                "payload",
                "evidence_refs",
            },
            "prediction candidate",
        )
        refs = _sequence(payload["evidence_refs"], "prediction candidate evidence refs")
        return cls(
            rank=payload["rank"],
            target_level=payload["target_level"],
            semantics=payload["semantics"],
            probability=payload["probability"],
            expected_delay_seconds=payload["expected_delay_seconds"],
            payload=_mapping(payload["payload"], "prediction candidate payload"),
            evidence_refs=tuple(refs),
        )


@dataclass(frozen=True)
class PredictionRun:
    """在结果发生前冻结的预测；后续评估不得改写本记录。"""

    run_id: str
    predicted_at: datetime
    horizon_seconds: float
    context: Mapping[str, Any]
    context_digest: str
    predictor_id: str
    predictor_version: str
    predictor_digest: str
    pattern_bindings: tuple[PredictionRunPatternBinding, ...]
    source_bindings: tuple[PredictionRunSourceBinding, ...]
    candidates: tuple[PredictionCandidate, ...]
    abstention_reason: PredictionAbstentionReason | None
    signal_provenance: str | None = None

    def __post_init__(self) -> None:
        prediction_sample_id(self.run_id)
        predicted_at = _datetime(self.predicted_at, "prediction run predicted_at")
        object.__setattr__(self, "predicted_at", predicted_at)
        horizon = _non_negative_number(self.horizon_seconds, "prediction run horizon")
        if horizon == 0:
            raise ValueError("prediction run horizon must be greater than zero")
        object.__setattr__(self, "horizon_seconds", horizon)
        normalized_context = _validated_context(self.context)
        object.__setattr__(self, "context", immutable_snapshot(normalized_context))
        anchor = normalized_context["anchor"]
        decision_time = anchor["cutoff_at"] or anchor["lower_bound_at"]
        if decision_time is not None and predicted_at < _datetime(decision_time, "prediction decision time"):
            raise ValueError("prediction run cannot precede its decision point")
        expected_context_digest = canonical_digest(normalized_context)
        if self.context_digest != expected_context_digest:
            raise ValueError("prediction run context digest does not match its context")
        for name in ("predictor_id", "predictor_version"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"prediction run {name} must be non-empty text")
            object.__setattr__(self, name, value.strip())
        object.__setattr__(
            self,
            "predictor_digest",
            _sha256(self.predictor_digest, "prediction run predictor digest"),
        )
        pattern_bindings = tuple(self.pattern_bindings)
        if any(not isinstance(item, PredictionRunPatternBinding) for item in pattern_bindings):
            raise TypeError("prediction run Pattern bindings must be PredictionRunPatternBinding values")
        keys = tuple(item.logical_state_key for item in pattern_bindings)
        if len(keys) != len(set(keys)):
            raise ValueError("one logical StatePattern can bind exactly one Pattern generation")
        if pattern_bindings != tuple(sorted(pattern_bindings)):
            raise ValueError("prediction run Pattern bindings must be sorted")
        object.__setattr__(self, "pattern_bindings", pattern_bindings)
        bindings = tuple(self.source_bindings)
        if not bindings or any(not isinstance(item, PredictionRunSourceBinding) for item in bindings):
            raise ValueError("prediction run requires source bindings")
        binding_ids = tuple((item.uri, item.revision) for item in bindings)
        if len(binding_ids) != len(set(binding_ids)):
            raise ValueError("one prediction run source revision must bind exactly one digest")
        object.__setattr__(self, "source_bindings", bindings)
        candidates = tuple(self.candidates)
        if any(not isinstance(item, PredictionCandidate) for item in candidates):
            raise TypeError("prediction run candidates must contain PredictionCandidate values")
        if tuple(item.rank for item in candidates) != tuple(range(1, len(candidates) + 1)):
            raise ValueError("prediction run candidate ranks must be contiguous from one")
        expected_target = PredictionTargetLevel(
            normalized_context["prediction_scope"]["target_level"]
        )
        if any(item.target_level is not expected_target for item in candidates):
            raise ValueError("prediction candidate target level must match prediction scope")
        if sum(item.probability for item in candidates) > 1.000000001:
            raise ValueError("prediction run candidate probabilities cannot exceed one")
        if any(
            item.expected_delay_seconds is not None and item.expected_delay_seconds > horizon
            for item in candidates
        ):
            raise ValueError("prediction candidate delay cannot exceed the run horizon")
        object.__setattr__(self, "candidates", candidates)
        abstention = None if self.abstention_reason is None else PredictionAbstentionReason(self.abstention_reason)
        object.__setattr__(self, "abstention_reason", abstention)
        if bool(candidates) == (abstention is not None):
            raise ValueError("prediction run requires candidates or one abstention reason, but not both")
        if self.signal_provenance is not None:
            object.__setattr__(
                self,
                "signal_provenance",
                _sha256(self.signal_provenance, "prediction run signal provenance"),
            )
        expected_run_id = canonical_digest({"contract": _RUN_CONTRACT, **self._identity_fields()})
        if self.run_id != expected_run_id:
            raise ValueError("prediction run ID does not match its immutable content")

    @property
    def kind(self) -> PredictionKind:
        return PredictionKind(self.context["prediction_type"])

    @classmethod
    def create(
        cls,
        *,
        predicted_at: datetime,
        horizon_seconds: float,
        context: PredictionContext,
        predictor_id: str,
        predictor_version: str,
        predictor_digest: str,
        source_bindings: Sequence[PredictionRunSourceBinding],
        candidates: Sequence[PredictionCandidate] = (),
        abstention_reason: PredictionAbstentionReason | str | None = None,
        pattern_bindings: Sequence[PredictionRunPatternBinding] = (),
        signal_provenance: str | None = None,
    ) -> PredictionRun:
        if not isinstance(context, PredictionContext):
            raise TypeError("prediction run context must be PredictionContext")
        context_fields = context.to_fields()
        normalized_predicted_at = _datetime(predicted_at, "prediction run predicted_at")
        normalized_horizon = _non_negative_number(horizon_seconds, "prediction run horizon")
        if normalized_horizon == 0:
            raise ValueError("prediction run horizon must be greater than zero")
        normalized_predictor_id = _text(predictor_id, "prediction run predictor_id")
        normalized_predictor_version = _text(
            predictor_version,
            "prediction run predictor_version",
        )
        normalized_predictor_digest = _sha256(
            predictor_digest,
            "prediction run predictor digest",
        )
        normalized_pattern_bindings = tuple(sorted(pattern_bindings))
        normalized_bindings = tuple(source_bindings)
        normalized_candidates = tuple(candidates)
        normalized_abstention = (
            None if abstention_reason is None else PredictionAbstentionReason(abstention_reason)
        )
        normalized_provenance = (
            None
            if signal_provenance is None
            else _sha256(signal_provenance, "prediction run signal provenance")
        )
        context_digest = canonical_digest(context_fields)
        identity = _identity_fields(
            predicted_at=normalized_predicted_at,
            horizon_seconds=normalized_horizon,
            context=context_fields,
            context_digest=context_digest,
            predictor_id=normalized_predictor_id,
            predictor_version=normalized_predictor_version,
            predictor_digest=normalized_predictor_digest,
            pattern_bindings=normalized_pattern_bindings,
            source_bindings=normalized_bindings,
            candidates=normalized_candidates,
            abstention_reason=normalized_abstention,
            signal_provenance=normalized_provenance,
        )
        run_id = canonical_digest({"contract": _RUN_CONTRACT, **identity})
        return cls(
            run_id=run_id,
            predicted_at=normalized_predicted_at,
            horizon_seconds=normalized_horizon,
            context=context_fields,
            context_digest=context_digest,
            predictor_id=normalized_predictor_id,
            predictor_version=normalized_predictor_version,
            predictor_digest=normalized_predictor_digest,
            pattern_bindings=normalized_pattern_bindings,
            source_bindings=normalized_bindings,
            candidates=normalized_candidates,
            abstention_reason=normalized_abstention,
            signal_provenance=normalized_provenance,
        )

    def _identity_fields(self) -> dict[str, Any]:
        return _identity_fields(
            predicted_at=self.predicted_at,
            horizon_seconds=self.horizon_seconds,
            context=self.context,
            context_digest=self.context_digest,
            predictor_id=self.predictor_id,
            predictor_version=self.predictor_version,
            predictor_digest=self.predictor_digest,
            pattern_bindings=self.pattern_bindings,
            source_bindings=self.source_bindings,
            candidates=self.candidates,
            abstention_reason=self.abstention_reason,
            signal_provenance=self.signal_provenance,
        )

    def to_dict(self) -> dict[str, Any]:
        return {"schema": _RUN_CONTRACT, "run_id": self.run_id, **self._identity_fields()}

    @classmethod
    def from_dict(cls, value: object) -> PredictionRun:
        expected = {
            "schema",
            "run_id",
            "predicted_at",
            "horizon_seconds",
            "context",
            "context_digest",
            "predictor_id",
            "predictor_version",
            "predictor_digest",
            "pattern_bindings",
            "source_bindings",
            "candidates",
            "abstention_reason",
            "signal_provenance",
        }
        payload = _exact_mapping(value, expected, "prediction run")
        if payload["schema"] != _RUN_CONTRACT:
            raise ValueError("prediction run schema is unsupported")
        return cls(
            run_id=payload["run_id"],
            predicted_at=_datetime(payload["predicted_at"], "prediction run predicted_at"),
            horizon_seconds=payload["horizon_seconds"],
            context=_mapping(payload["context"], "prediction run context"),
            context_digest=payload["context_digest"],
            predictor_id=payload["predictor_id"],
            predictor_version=payload["predictor_version"],
            predictor_digest=payload["predictor_digest"],
            pattern_bindings=tuple(
                PredictionRunPatternBinding.from_dict(item)
                for item in _sequence(payload["pattern_bindings"], "prediction run Pattern bindings")
            ),
            source_bindings=tuple(
                PredictionRunSourceBinding.from_dict(item)
                for item in _sequence(payload["source_bindings"], "prediction run source bindings")
            ),
            candidates=tuple(
                PredictionCandidate.from_dict(item)
                for item in _sequence(payload["candidates"], "prediction run candidates")
            ),
            abstention_reason=payload["abstention_reason"],
            signal_provenance=payload["signal_provenance"],
        )


def _validated_context(value: object) -> dict[str, Any]:
    payload = _mapping(value, "prediction run context")
    expected = {"prediction_type", "prediction_scope", "anchor", "input"}
    if "treatment" in payload:
        expected.add("treatment")
    if set(payload) != expected:
        raise ValueError("prediction run context has an invalid shape")
    context = PredictionContext(
        kind=PredictionKind(payload["prediction_type"]),
        prediction_scope=_mapping(payload["prediction_scope"], "prediction scope"),
        anchor=_mapping(payload["anchor"], "prediction anchor"),
        input=_mapping(payload["input"], "prediction input"),
        treatment=None
        if "treatment" not in payload
        else _mapping(payload["treatment"], "prediction treatment"),
    )
    return context.to_fields()


def _identity_fields(
    *,
    predicted_at: datetime,
    horizon_seconds: float,
    context: Mapping[str, Any],
    context_digest: str,
    predictor_id: str,
    predictor_version: str,
    predictor_digest: str,
    pattern_bindings: Sequence[PredictionRunPatternBinding],
    source_bindings: Sequence[PredictionRunSourceBinding],
    candidates: Sequence[PredictionCandidate],
    abstention_reason: PredictionAbstentionReason | None,
    signal_provenance: str | None,
) -> dict[str, Any]:
    value = canonicalize(
        {
            "predicted_at": predicted_at,
            "horizon_seconds": horizon_seconds,
            "context": context,
            "context_digest": context_digest,
            "predictor_id": predictor_id,
            "predictor_version": predictor_version,
            "predictor_digest": predictor_digest,
            "pattern_bindings": [item.to_dict() for item in pattern_bindings],
            "source_bindings": [item.to_dict() for item in source_bindings],
            "candidates": [item.to_dict() for item in candidates],
            "abstention_reason": None if abstention_reason is None else abstention_reason.value,
            "signal_provenance": signal_provenance,
        }
    )
    assert isinstance(value, dict)
    return value


def _datetime(value: object, label: str) -> datetime:
    parsed = value
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(f"{label} must be an ISO timestamp") from exc
    if not isinstance(parsed, datetime) or parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{label} must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _probability(value: object, label: str) -> float:
    result = _non_negative_number(value, label)
    if result > 1:
        raise ValueError(f"{label} must be between zero and one")
    return result


def _non_negative_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{label} must be a finite non-negative number")
    return result


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be non-empty text")
    return value.strip()


def _sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{label} must be a mapping with string keys")
    return dict(value)


def _exact_mapping(value: object, expected: set[str], label: str) -> dict[str, Any]:
    payload = _mapping(value, label)
    if set(payload) != expected:
        raise ValueError(f"{label} must contain exactly {sorted(expected)}")
    return payload


def _sequence(value: object, label: str) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{label} must be a sequence")
    return tuple(value)


__all__ = [
    "PredictionAbstentionReason",
    "PredictionCandidate",
    "PredictionRun",
    "PredictionRunPatternBinding",
    "PredictionRunSourceBinding",
]
