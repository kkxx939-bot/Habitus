"""三类 L2 文档共用的唯一 Schema 注册表和规范物化入口。"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date, datetime
from importlib import resources
from typing import Any

from behavior.model import BehaviorAddress, BehaviorKind
from behavior.schema.fields import strict_mapping, validate_field
from behavior.schema.loader import load_schema
from behavior.schema.model import (
    BehaviorSchemaError,
    BehaviorSchemaMaterialization,
    BehaviorTypeSchema,
)
from behavior.schema.renderers import render_markdown
from behavior.schema.validators import validate_payload
from behavior.uri import BehaviorURI
from foundation.integrity import canonical_json, canonicalize

_SCHEMA_FILES = {
    BehaviorKind.EVENT: "events.yaml",
    BehaviorKind.OUTCOME: "outcomes.yaml",
    BehaviorKind.EPISODE: "episodes.yaml",
}


def _behavior_kind(value: BehaviorKind | str) -> BehaviorKind:
    try:
        return BehaviorKind(value)
    except (TypeError, ValueError) as exc:
        raise BehaviorSchemaError(f"unknown behavior type: {value}") from exc


class BehaviorSchemaRegistry:
    """三类 L2 文档共用的唯一 Schema 注册表和规范渲染入口。"""

    def __init__(self, schemas: tuple[BehaviorTypeSchema, ...]) -> None:
        by_kind: dict[BehaviorKind, BehaviorTypeSchema] = {}
        for schema in schemas:
            if not isinstance(schema, BehaviorTypeSchema):
                raise TypeError("schemas must contain BehaviorTypeSchema values")
            if schema.kind in by_kind:
                raise BehaviorSchemaError(f"duplicate behavior schema: {schema.kind.value}")
            by_kind[schema.kind] = schema
        if set(by_kind) != set(BehaviorKind):
            missing = sorted(kind.value for kind in set(BehaviorKind) - set(by_kind))
            raise BehaviorSchemaError(f"behavior schema registry is incomplete: {missing}")
        self._schemas = tuple(by_kind[kind] for kind in BehaviorKind)
        self._by_kind = by_kind

    @classmethod
    def load_default(cls) -> BehaviorSchemaRegistry:
        definitions = resources.files("behavior.schema.definitions")
        return cls(
            tuple(
                load_schema(definitions.joinpath(filename).read_text(encoding="utf-8"), filename)
                for _kind, filename in _SCHEMA_FILES.items()
            )
        )

    def all(self) -> tuple[BehaviorTypeSchema, ...]:
        return self._schemas

    def get(self, kind: BehaviorKind | str) -> BehaviorTypeSchema:
        return self._by_kind[_behavior_kind(kind)]

    def validate(self, kind: BehaviorKind | str, payload: Mapping[str, Any]) -> dict[str, Any]:
        normalized_kind = _behavior_kind(kind)
        schema = self.get(normalized_kind)
        source = strict_mapping(payload, f"{normalized_kind.value} payload")
        unknown = set(source) - set(schema.field_map)
        if unknown:
            raise BehaviorSchemaError(f"behavior payload contains unknown fields: {sorted(unknown)}")
        normalized: dict[str, Any] = {}
        for field in schema.fields:
            if field.name not in source:
                if field.required:
                    raise BehaviorSchemaError(f"behavior payload is missing required field: {field.name}")
                continue
            normalized[field.name] = validate_field(field, source[field.name])
        validate_payload(normalized_kind, normalized)
        return normalized

    def materialize(
        self,
        kind: BehaviorKind | str,
        payload: Mapping[str, Any],
    ) -> BehaviorSchemaMaterialization:
        """一次校验后生成规范地址、JSON-safe 字段和人类可读正文。"""

        normalized_kind = _behavior_kind(kind)
        normalized = self.validate(normalized_kind, payload)
        return BehaviorSchemaMaterialization(
            address=self._address_for_validated(normalized_kind, normalized),
            storage_fields=_storage_fields(normalized),
            markdown_body=render_markdown(normalized_kind, normalized),
        )

    def address_for(self, kind: BehaviorKind | str, payload: Mapping[str, Any]) -> BehaviorAddress:
        normalized_kind = _behavior_kind(kind)
        normalized = self.validate(normalized_kind, payload)
        return self._address_for_validated(normalized_kind, normalized)

    def render_markdown(self, kind: BehaviorKind | str, payload: Mapping[str, Any]) -> str:
        normalized_kind = _behavior_kind(kind)
        return render_markdown(normalized_kind, self.validate(normalized_kind, payload))

    @staticmethod
    def _address_for_validated(kind: BehaviorKind, normalized: Mapping[str, Any]) -> BehaviorAddress:
        """只接受本注册表已经规范化完成的领域字段。"""

        if kind is BehaviorKind.EVENT:
            return BehaviorAddress.event(
                normalized["event_date"],
                normalized["event_name"],
                normalized["started_at"],
            )
        if kind is BehaviorKind.OUTCOME:
            event_address = BehaviorURI.parse(normalized["event_uri"]).to_address()
            return BehaviorAddress.outcome(
                event_address.occurred_on,
                event_address.name,
                event_address.started_at,
            )
        return BehaviorAddress.episode(
            normalized["episode_date"],
            normalized["episode_name"],
            normalized["started_at"],
        )


def _storage_fields(fields: Mapping[str, Any]) -> dict[str, Any]:
    """把强类型领域字段转成保留行为本地 offset 的规范 JSON 值。"""

    stored = _storage_value(fields)
    if not isinstance(stored, dict):
        raise BehaviorSchemaError("behavior storage fields must be a JSON object")
    try:
        canonical_json(stored).encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise BehaviorSchemaError("behavior storage fields must be canonical UTF-8 JSON") from exc
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


__all__ = ["BehaviorSchemaRegistry"]
