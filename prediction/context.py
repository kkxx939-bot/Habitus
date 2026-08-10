"""只包含预测锚点时可用信息的运行时输入视图。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from foundation.integrity import canonicalize, immutable_snapshot
from prediction.context_contract import (
    PredictionContextContractError,
    normalize_prediction_anchor,
    normalize_prediction_scope,
    validate_prediction_context_contract,
)
from prediction.document import PredictionDocument
from prediction.model import PredictionKind
from prediction.visibility import validate_context_visibility, validate_treatment_boundary


@dataclass(frozen=True)
class PredictionContext:
    """从监督样本文档中隔离出来的无标签预测上下文。"""

    kind: PredictionKind
    prediction_scope: Mapping[str, Any]
    anchor: Mapping[str, Any]
    input: Mapping[str, Any]
    treatment: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        kind = PredictionKind(self.kind)
        object.__setattr__(self, "kind", kind)
        for field_name in ("prediction_scope", "anchor", "input"):
            value = getattr(self, field_name)
            if not isinstance(value, Mapping):
                raise TypeError(f"prediction context {field_name} must be a mapping")
        requires_treatment = kind is PredictionKind.CONSEQUENCE
        if requires_treatment != (self.treatment is not None):
            raise ValueError("prediction context requires treatment exactly for Consequence")
        if self.treatment is not None and not isinstance(self.treatment, Mapping):
            raise TypeError("prediction context treatment must be a mapping or None")
        try:
            scope = normalize_prediction_scope(self.prediction_scope)
            anchor = normalize_prediction_anchor(self.anchor)
            validate_prediction_context_contract(
                kind,
                scope,
                anchor,
                self.input,
                treatment=self.treatment,
            )
        except PredictionContextContractError as exc:
            raise ValueError(str(exc)) from exc
        object.__setattr__(self, "prediction_scope", immutable_snapshot(scope))
        object.__setattr__(self, "anchor", immutable_snapshot(anchor))
        object.__setattr__(self, "input", immutable_snapshot(self.input))
        validate_context_visibility(self.anchor, self.input)
        if self.treatment is not None:
            validate_treatment_boundary(self.anchor, self.treatment)
            object.__setattr__(
                self,
                "treatment",
                immutable_snapshot(_runtime_treatment(self.treatment)),
            )

    @classmethod
    def from_document(cls, document: PredictionDocument) -> PredictionContext:
        """只复制模型可见字段，不暴露标签、结果、来源和事后质量。"""

        if not isinstance(document, PredictionDocument):
            raise TypeError("document must be a PredictionDocument")
        return cls(
            kind=document.kind,
            prediction_scope=document.fields["prediction_scope"],
            anchor=document.fields["anchor"],
            input=document.fields["input"],
            treatment=document.fields.get("treatment"),
        )

    def to_fields(self) -> dict[str, Any]:
        """返回可交给预测算法的隔离副本。"""

        values: dict[str, Any] = {
            "prediction_type": self.kind.value,
            "prediction_scope": self.prediction_scope,
            "anchor": self.anchor,
            "input": self.input,
        }
        if self.treatment is not None:
            values["treatment"] = self.treatment
        fields = canonicalize(values)
        assert isinstance(fields, dict)
        return fields


def _runtime_treatment(value: Mapping[str, Any]) -> dict[str, Any]:
    """移除只用于存储闭包的来源身份，运行时只保留可观察 treatment。"""

    fields = (
        "step_kind",
        "step_ref",
        "semantics",
        "actor",
        "behavior_type",
        "target_refs",
        "status",
        "started_at",
        "available_at",
    )
    missing = set(fields) - set(value)
    if missing:
        raise ValueError(f"prediction treatment is missing runtime fields: {sorted(missing)}")
    return {field: value[field] for field in fields}


__all__ = ["PredictionContext"]
