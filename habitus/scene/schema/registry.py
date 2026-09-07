"""情景文档唯一的 Schema 注册表和规范物化入口。"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date, datetime
from importlib import resources
from typing import Any

from habitus.foundation.integrity import canonical_json, canonicalize
from habitus.scene.model import SceneAddress
from habitus.scene.schema.fields import strict_mapping, validate_field
from habitus.scene.schema.loader import load_schema
from habitus.scene.schema.model import SceneSchemaError, SceneSchemaMaterialization, SceneTypeSchema
from habitus.scene.schema.renderers import render_markdown
from habitus.scene.schema.validators import validate_payload

_SCHEMA_FILE = "scenes.yaml"


class SceneSchemaRegistry:
    def __init__(self, schema: SceneTypeSchema) -> None:
        if not isinstance(schema, SceneTypeSchema):
            raise TypeError("schema must be a SceneTypeSchema")
        self._schema = schema

    @classmethod
    def load_default(cls) -> SceneSchemaRegistry:
        definitions = resources.files("habitus.scene.schema.definitions")
        return cls(load_schema(definitions.joinpath(_SCHEMA_FILE).read_text(encoding="utf-8"), _SCHEMA_FILE))

    @property
    def schema(self) -> SceneTypeSchema:
        return self._schema

    def validate(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        source = strict_mapping(payload, "scene payload")
        unknown = set(source) - set(self._schema.field_map)
        if unknown:
            raise SceneSchemaError(f"scene payload contains unknown fields: {sorted(unknown)}")
        normalized: dict[str, Any] = {}
        for field in self._schema.fields:
            if field.name not in source:
                raise SceneSchemaError(f"scene payload is missing required field: {field.name}")
            normalized[field.name] = validate_field(field, source[field.name])
        validate_payload(normalized)
        return normalized

    def materialize(self, payload: Mapping[str, Any]) -> SceneSchemaMaterialization:
        normalized = self.validate(payload)
        return SceneSchemaMaterialization(
            address=SceneAddress(normalized["occurred_on"], normalized["label"], normalized["started_at"]),
            storage_fields=_storage_fields(normalized),
            markdown_body=render_markdown(normalized),
        )

    def address_for(self, payload: Mapping[str, Any]) -> SceneAddress:
        normalized = self.validate(payload)
        return SceneAddress(normalized["occurred_on"], normalized["label"], normalized["started_at"])


def _storage_fields(fields: Mapping[str, Any]) -> dict[str, Any]:
    stored = _storage_value(fields)
    if not isinstance(stored, dict):
        raise SceneSchemaError("scene storage fields must be a JSON object")
    try:
        canonical_json(stored).encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise SceneSchemaError("scene storage fields must be canonical UTF-8 JSON") from exc
    return stored


def _storage_value(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat(timespec="microseconds")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {key: _storage_value(item) for key, item in sorted(value.items())}
    if isinstance(value, tuple | list):
        return [_storage_value(item) for item in value]
    return canonicalize(value)


__all__ = ["SceneSchemaRegistry"]
