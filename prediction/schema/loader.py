"""把声明式 YAML 加载成强类型 PredictionTypeSchema。"""

from __future__ import annotations

import yaml

from prediction.model import PredictionKind
from prediction.schema.model import (
    PredictionFieldRole,
    PredictionFieldSchema,
    PredictionFieldType,
    PredictionOperationMode,
    PredictionSchemaError,
    PredictionTypeSchema,
)
from prediction.schema.primitives import (
    boolean,
    exact_mapping,
    text,
)

_TYPE_KEYS = {"prediction_type", "description", "path_template", "operation_mode", "fields"}
_FIELD_KEYS = {"name", "type", "role", "required", "description"}

def load_schema(source: str, filename: str) -> PredictionTypeSchema:
    try:
        raw = yaml.safe_load(source)
    except yaml.YAMLError as exc:
        raise PredictionSchemaError(f"invalid YAML in {filename}") from exc
    payload = exact_mapping(raw, _TYPE_KEYS, f"schema {filename}")
    raw_fields = payload["fields"]
    if not isinstance(raw_fields, list):
        raise PredictionSchemaError(f"schema {filename} fields must be a list")
    fields: list[PredictionFieldSchema] = []
    for raw_field in raw_fields:
        field = exact_mapping(raw_field, _FIELD_KEYS, f"field in {filename}")
        fields.append(
            PredictionFieldSchema(
                name=text(field["name"], "field name"),
                field_type=PredictionFieldType(text(field["type"], "field type")),
                role=PredictionFieldRole(text(field["role"], "field role")),
                required=boolean(field["required"], "field required"),
                description=text(field["description"], "field description"),
            )
        )
    return PredictionTypeSchema(
        kind=PredictionKind(text(payload["prediction_type"], "prediction type")),
        description=text(payload["description"], "prediction description"),
        path_template=text(payload["path_template"], "prediction path template"),
        operation_mode=PredictionOperationMode(text(payload["operation_mode"], "prediction operation mode")),
        fields=tuple(fields),
    )


__all__ = ["load_schema"]
