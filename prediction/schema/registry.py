"""三类预测样本共用的唯一 Schema 注册表和规范物化入口。"""

from __future__ import annotations

from collections.abc import Mapping
from importlib import resources
from typing import Any

from foundation.integrity import canonical_json, canonicalize
from prediction.context_contract import (
    PredictionContextContractError,
    validate_prediction_context_contract,
)
from prediction.model import PredictionAddress, PredictionKind
from prediction.schema.fields import validate_field
from prediction.schema.loader import load_schema
from prediction.schema.model import (
    PredictionSchemaError,
    PredictionSchemaMaterialization,
    PredictionTypeSchema,
)
from prediction.schema.primitives import strict_mapping
from prediction.schema.renderers import render_sample
from prediction.schema.validators import validate_identity, validate_kind, validate_source_closure
from prediction.visibility import (
    PredictionVisibilityError,
    validate_context_visibility,
    validate_label_boundary,
    validate_treatment_boundary,
)

_SCHEMA_FILES = {
    PredictionKind.TRANSITION: "transitions.yaml",
    PredictionKind.TRAJECTORY: "trajectories.yaml",
    PredictionKind.CONSEQUENCE: "consequences.yaml",
}


def _prediction_kind(value: PredictionKind | str) -> PredictionKind:
    try:
        return PredictionKind(value)
    except (TypeError, ValueError) as exc:
        raise PredictionSchemaError(f"unknown prediction type: {value}") from exc


class PredictionSchemaRegistry:
    """三类预测样本共用的唯一严格校验和渲染入口。"""

    def __init__(self, schemas: tuple[PredictionTypeSchema, ...]) -> None:
        by_kind: dict[PredictionKind, PredictionTypeSchema] = {}
        for schema in schemas:
            if not isinstance(schema, PredictionTypeSchema):
                raise TypeError("schemas must contain PredictionTypeSchema values")
            if schema.kind in by_kind:
                raise PredictionSchemaError(f"duplicate prediction schema: {schema.kind.value}")
            by_kind[schema.kind] = schema
        if set(by_kind) != set(PredictionKind):
            missing = sorted(kind.value for kind in set(PredictionKind) - set(by_kind))
            raise PredictionSchemaError(f"prediction schema registry is incomplete: {missing}")
        self._schemas = tuple(by_kind[kind] for kind in PredictionKind)
        self._by_kind = by_kind

    @classmethod
    def load_default(cls) -> PredictionSchemaRegistry:
        definitions = resources.files("prediction.schema.definitions")
        return cls(
            tuple(
                load_schema(definitions.joinpath(filename).read_text(encoding="utf-8"), filename)
                for filename in _SCHEMA_FILES.values()
            )
        )

    def all(self) -> tuple[PredictionTypeSchema, ...]:
        return self._schemas

    def get(self, kind: PredictionKind | str) -> PredictionTypeSchema:
        return self._by_kind[_prediction_kind(kind)]

    def validate(self, kind: PredictionKind | str, payload: Mapping[str, Any]) -> dict[str, Any]:
        normalized_kind = _prediction_kind(kind)
        normalized = self._normalize_fields(normalized_kind, payload)
        validate_kind(normalized_kind, normalized)
        validate_identity(normalized_kind, normalized)
        validate_source_closure(normalized_kind, normalized)
        self._require_visibility(normalized_kind, normalized)
        return normalized

    def _normalize_fields(self, kind: PredictionKind, payload: Mapping[str, Any]) -> dict[str, Any]:
        """按 Schema 声明逐字段规范化；未知字段和缺失必填字段都直接拒绝。"""

        schema = self.get(kind)
        source = strict_mapping(payload, f"{kind.value} sample")
        unknown = set(source) - set(schema.field_map)
        if unknown:
            raise PredictionSchemaError(f"prediction sample contains unknown fields: {sorted(unknown)}")
        normalized: dict[str, Any] = {}
        for field in schema.fields:
            if field.name not in source:
                if field.required:
                    raise PredictionSchemaError(f"prediction sample is missing required field: {field.name}")
                continue
            normalized[field.name] = validate_field(field, source[field.name])
        return normalized

    @staticmethod
    def _require_visibility(kind: PredictionKind, normalized: Mapping[str, Any]) -> None:
        """模型可见字段不得越过切点；上下文合同与可见性异常统一收敛为 Schema 错误。"""

        try:
            validate_prediction_context_contract(
                kind,
                normalized["prediction_scope"],
                normalized["anchor"],
                normalized["input"],
                treatment=normalized.get("treatment"),
            )
            validate_context_visibility(normalized["anchor"], normalized["input"])
            validate_label_boundary(normalized["anchor"], normalized["label"])
            if kind is PredictionKind.CONSEQUENCE:
                validate_treatment_boundary(normalized["anchor"], normalized["treatment"])
        except (PredictionContextContractError, PredictionVisibilityError) as exc:
            raise PredictionSchemaError(str(exc)) from exc

    def materialize(
        self,
        kind: PredictionKind | str,
        payload: Mapping[str, Any],
    ) -> PredictionSchemaMaterialization:
        normalized_kind = _prediction_kind(kind)
        normalized = self.validate(normalized_kind, payload)
        storage_fields = canonicalize(normalized)
        try:
            canonical_json(storage_fields).encode("utf-8")
        except (TypeError, ValueError, UnicodeEncodeError) as exc:
            raise PredictionSchemaError("prediction sample must be canonical UTF-8 JSON") from exc
        return PredictionSchemaMaterialization(
            address=PredictionAddress(
                normalized_kind,
                normalized["sample_date"],
                normalized["materialization_id"],
            ),
            storage_fields=storage_fields,
            markdown_body=render_sample(normalized_kind, normalized),
        )


__all__ = ["PredictionSchemaRegistry"]
