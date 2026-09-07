"""把声明式 YAML 加载成强类型 SceneTypeSchema。"""

from __future__ import annotations

from typing import Any

import yaml

from habitus.scene.schema.fields import require_keys, strict_mapping, text
from habitus.scene.schema.model import (
    SceneFieldRole,
    SceneFieldSchema,
    SceneFieldType,
    SceneSchemaError,
    SceneTypeSchema,
)

_TYPE_KEYS = {"scene_type", "description", "path_template", "fields"}
_FIELD_KEYS = {"name", "type", "role", "required", "description"}


def load_schema(source: str, filename: str) -> SceneTypeSchema:
    try:
        raw = yaml.safe_load(source)
    except yaml.YAMLError as exc:
        raise SceneSchemaError(f"invalid YAML in {filename}") from exc
    payload = strict_mapping(raw, f"schema {filename}")
    require_keys(payload, _TYPE_KEYS, f"schema {filename}")
    if text(payload["scene_type"], "scene type") != "scene":
        raise SceneSchemaError(f"schema {filename} must declare scene_type: scene")
    raw_fields = payload["fields"]
    if not isinstance(raw_fields, list):
        raise SceneSchemaError(f"schema {filename} fields must be a list")
    return SceneTypeSchema(
        description=text(payload["description"], "scene description"),
        path_template=text(payload["path_template"], "scene path template"),
        fields=tuple(_load_field(raw_field, filename) for raw_field in raw_fields),
    )


def _load_field(raw_field: Any, filename: str) -> SceneFieldSchema:
    field = strict_mapping(raw_field, f"field in {filename}")
    require_keys(field, _FIELD_KEYS, f"field in {filename}")
    return SceneFieldSchema(
        name=text(field["name"], "field name"),
        field_type=SceneFieldType(text(field["type"], "field type")),
        role=SceneFieldRole(text(field["role"], "field role")),
        required=field["required"],
        description=text(field["description"], "field description"),
    )


__all__ = ["load_schema"]
