"""把声明式 YAML 加载成强类型 BehaviorTypeSchema。"""

from __future__ import annotations

from typing import Any

import yaml

from behavior.model import BehaviorKind
from behavior.schema.fields import require_keys, strict_mapping, text
from behavior.schema.model import (
    BehaviorFieldRole,
    BehaviorFieldSchema,
    BehaviorFieldType,
    BehaviorMergeStrategy,
    BehaviorOperationMode,
    BehaviorSchemaError,
    BehaviorTypeSchema,
)

_TYPE_KEYS = {"behavior_type", "description", "path_template", "operation_mode", "fields"}
_FIELD_KEYS = {"name", "type", "role", "required", "merge", "description"}


def load_schema(source: str, filename: str) -> BehaviorTypeSchema:
    """YAML 只能声明已确认的键集；未知键和缺失键都直接拒绝。"""

    try:
        raw = yaml.safe_load(source)
    except yaml.YAMLError as exc:
        raise BehaviorSchemaError(f"invalid YAML in {filename}") from exc
    payload = strict_mapping(raw, f"schema {filename}")
    require_keys(payload, _TYPE_KEYS, f"schema {filename}")
    raw_fields = payload["fields"]
    if not isinstance(raw_fields, list):
        raise BehaviorSchemaError(f"schema {filename} fields must be a list")
    fields = tuple(_load_field(raw_field, filename) for raw_field in raw_fields)
    return BehaviorTypeSchema(
        kind=BehaviorKind(text(payload["behavior_type"], "behavior type")),
        description=text(payload["description"], "behavior description"),
        path_template=text(payload["path_template"], "behavior path template"),
        operation_mode=BehaviorOperationMode(text(payload["operation_mode"], "behavior operation mode")),
        fields=fields,
    )


def _load_field(raw_field: Any, filename: str) -> BehaviorFieldSchema:
    field = strict_mapping(raw_field, f"field in {filename}")
    require_keys(field, _FIELD_KEYS, f"field in {filename}")
    return BehaviorFieldSchema(
        name=text(field["name"], "field name"),
        field_type=BehaviorFieldType(text(field["type"], "field type")),
        role=BehaviorFieldRole(text(field["role"], "field role")),
        required=field["required"],
        merge_strategy=BehaviorMergeStrategy(text(field["merge"], "field merge")),
        description=text(field["description"], "field description"),
    )


__all__ = ["load_schema"]
