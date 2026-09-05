"""KEEP、UPDATE 和 APPEND 字段压缩协议及其确定性合并边界。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any


class SemanticFieldOperationError(ValueError):
    """字段压缩操作不能在无损保护边界内应用。"""


class SemanticFieldOperationKind(str, Enum):
    KEEP = "keep"
    UPDATE = "update"
    APPEND = "append"


@dataclass(frozen=True)
class SemanticFieldOperation:
    """模型针对一个业务字段给出的受控操作。"""

    field: str
    operation: SemanticFieldOperationKind
    content: str | None = None
    items: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.field, str) or not self.field or self.field != self.field.strip():
            raise SemanticFieldOperationError("semantic field operation requires a normalized field name")
        try:
            operation = SemanticFieldOperationKind(self.operation)
        except ValueError as exc:
            raise SemanticFieldOperationError("semantic field operation kind is unsupported") from exc
        object.__setattr__(self, "operation", operation)
        if self.content is not None and (
            not isinstance(self.content, str) or not self.content.strip()
        ):
            raise SemanticFieldOperationError("semantic field update content must be non-empty text")
        if not isinstance(self.items, tuple) or any(
            not isinstance(item, str) or not item.strip() for item in self.items
        ):
            raise SemanticFieldOperationError("semantic field append items must contain non-empty text")
        normalized_items = tuple(item.strip() for item in self.items)
        if len(normalized_items) != len(set(normalized_items)):
            raise SemanticFieldOperationError("semantic field append items must be unique")
        object.__setattr__(self, "items", normalized_items)
        if operation is SemanticFieldOperationKind.KEEP:
            if self.content is not None or normalized_items:
                raise SemanticFieldOperationError("KEEP cannot contain content or items")
        elif operation is SemanticFieldOperationKind.UPDATE:
            if self.content is None or normalized_items:
                raise SemanticFieldOperationError("UPDATE requires only complete replacement content")
            object.__setattr__(self, "content", self.content.strip())
        elif self.content is not None or not normalized_items:
            raise SemanticFieldOperationError("APPEND requires only one or more items")

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "field": self.field,
            "operation": self.operation.value,
        }
        if self.content is not None:
            result["content"] = self.content
        if self.items:
            result["items"] = list(self.items)
        return result


@dataclass(frozen=True)
class SemanticFieldOperationBatch:
    """一次模型调用返回的唯一字段操作集合。"""

    operations: tuple[SemanticFieldOperation, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.operations, tuple) or any(
            not isinstance(item, SemanticFieldOperation) for item in self.operations
        ):
            raise TypeError("semantic field operations must contain SemanticFieldOperation values")
        if not 1 <= len(self.operations) <= 128:
            raise SemanticFieldOperationError("semantic field operation batch must contain between 1 and 128 items")
        fields = tuple(item.field for item in self.operations)
        if len(fields) != len(set(fields)):
            raise SemanticFieldOperationError("semantic field operation batch cannot repeat one field")

    @classmethod
    def model_json_schema(cls) -> dict[str, object]:
        operation_variants: list[dict[str, object]] = [
            {
                "type": "object",
                "additionalProperties": False,
                "required": ["field", "operation"],
                "properties": {
                    "field": {"type": "string", "minLength": 1, "maxLength": 128},
                    "operation": {"const": SemanticFieldOperationKind.KEEP.value},
                },
            },
            {
                "type": "object",
                "additionalProperties": False,
                "required": ["field", "operation", "content"],
                "properties": {
                    "field": {"type": "string", "minLength": 1, "maxLength": 128},
                    "operation": {"const": SemanticFieldOperationKind.UPDATE.value},
                    "content": {"type": "string", "minLength": 1, "maxLength": 1_000_000},
                },
            },
            {
                "type": "object",
                "additionalProperties": False,
                "required": ["field", "operation", "items"],
                "properties": {
                    "field": {"type": "string", "minLength": 1, "maxLength": 128},
                    "operation": {"const": SemanticFieldOperationKind.APPEND.value},
                    "items": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 128,
                        "items": {"type": "string", "minLength": 1, "maxLength": 20_000},
                    },
                },
            },
        ]
        return {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": "SemanticFieldOperationBatch",
            "type": "object",
            "additionalProperties": False,
            "required": ["operations"],
            "properties": {
                "operations": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 128,
                    "items": {"oneOf": operation_variants},
                }
            },
        }

    @classmethod
    def model_validate(cls, value: object) -> SemanticFieldOperationBatch:
        if not isinstance(value, Mapping) or set(value) != {"operations"}:
            raise SemanticFieldOperationError("semantic field operation output must contain only operations")
        raw_operations = value["operations"]
        if not isinstance(raw_operations, list | tuple):
            raise SemanticFieldOperationError("semantic field operations must be an array")
        operations: list[SemanticFieldOperation] = []
        for raw in raw_operations:
            if not isinstance(raw, Mapping):
                raise SemanticFieldOperationError("semantic field operation item must be an object")
            if "operation" not in raw:
                raise SemanticFieldOperationError("semantic field operation item is missing operation")
            try:
                operation = SemanticFieldOperationKind(raw["operation"])
            except (TypeError, ValueError) as exc:
                raise SemanticFieldOperationError("semantic field operation kind is unsupported") from exc
            expected = {
                SemanticFieldOperationKind.KEEP: {"field", "operation"},
                SemanticFieldOperationKind.UPDATE: {"field", "operation", "content"},
                SemanticFieldOperationKind.APPEND: {"field", "operation", "items"},
            }[operation]
            if set(raw) != expected:
                raise SemanticFieldOperationError("semantic field operation fields do not match its kind")
            raw_field = raw.get("field")
            if not isinstance(raw_field, str):
                raise SemanticFieldOperationError("semantic field operation field must be text")
            raw_items = raw.get("items", ())
            if not isinstance(raw_items, list | tuple):
                raise SemanticFieldOperationError("semantic field append items must be an array")
            operations.append(
                SemanticFieldOperation(
                    field=raw_field,
                    operation=operation,
                    content=raw.get("content"),
                    items=tuple(raw_items),
                )
            )
        return cls(tuple(operations))


@dataclass(frozen=True)
class SemanticFieldMergePolicy:
    """返回一个业务字段可接受的确定性合并边界。"""

    field: str
    allow_update: bool = True
    allow_append: bool = True
    append_only: bool = False
    max_chars: int = 1_000_000

    def __post_init__(self) -> None:
        if not isinstance(self.field, str) or not self.field or self.field != self.field.strip():
            raise SemanticFieldOperationError("semantic field policy requires a normalized field name")
        for name in ("allow_update", "allow_append", "append_only"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"semantic field policy {name} must be boolean")
        if self.append_only and not self.allow_append:
            raise SemanticFieldOperationError("append-only semantic field must allow APPEND")
        if isinstance(self.max_chars, bool) or not isinstance(self.max_chars, int) or self.max_chars <= 0:
            raise ValueError("semantic field policy max_chars must be a positive integer")


def merge_semantic_fields(
    source: Mapping[str, Any],
    operations: SemanticFieldOperationBatch,
    policies: tuple[SemanticFieldMergePolicy, ...],
) -> dict[str, Any]:
    """由服务端应用字段操作；缺失字段等同 KEEP，未知字段明确拒绝。"""

    if not isinstance(source, Mapping) or any(not isinstance(name, str) for name in source):
        raise TypeError("semantic field source must be a mapping with string keys")
    if not isinstance(operations, SemanticFieldOperationBatch):
        raise TypeError("operations must be a SemanticFieldOperationBatch")
    if not isinstance(policies, tuple) or any(
        not isinstance(policy, SemanticFieldMergePolicy) for policy in policies
    ):
        raise TypeError("policies must contain SemanticFieldMergePolicy values")
    policy_by_field = {policy.field: policy for policy in policies}
    if len(policy_by_field) != len(policies):
        raise SemanticFieldOperationError("semantic field policies cannot repeat one field")
    unknown = sorted(item.field for item in operations.operations if item.field not in policy_by_field)
    if unknown:
        raise SemanticFieldOperationError(f"semantic field operations contain unknown fields: {unknown}")

    merged = dict(source)
    for item in operations.operations:
        policy = policy_by_field[item.field]
        current = source.get(item.field)
        if current is not None and not isinstance(current, str):
            raise SemanticFieldOperationError("semantic field merge supports text business fields only")
        current_text = "" if current is None else current
        operation = item.operation
        if operation is SemanticFieldOperationKind.KEEP:
            continue
        if operation is SemanticFieldOperationKind.UPDATE:
            assert item.content is not None
            if policy.append_only:
                candidate_items = _content_items(item.content)
                updated = _append_items(current_text, candidate_items)
            elif not policy.allow_update:
                continue
            else:
                updated = item.content
        else:
            if not policy.allow_append:
                continue
            updated = _append_items(current_text, item.items)
        if len(updated) > policy.max_chars:
            raise SemanticFieldOperationError(f"semantic field {item.field} exceeds its merge bound")
        if updated:
            merged[item.field] = updated
        elif item.field in source:
            merged[item.field] = current_text
    return merged


def _append_items(current: str, items: tuple[str, ...]) -> str:
    existing = current.rstrip()
    existing_keys = {item.casefold() for item in _content_items(existing)}
    fresh = tuple(item for item in items if item.casefold() not in existing_keys)
    if not fresh:
        return existing
    appended = "\n".join(f"- {item}" for item in fresh)
    return appended if not existing else f"{existing}\n{appended}"


def _content_items(content: str) -> tuple[str, ...]:
    result: list[str] = []
    for line in content.splitlines():
        normalized = line.strip()
        if normalized.startswith(("- ", "* ")):
            normalized = normalized[2:].strip()
        if normalized and normalized not in result:
            result.append(normalized)
    return tuple(result)


__all__ = [
    "SemanticFieldMergePolicy",
    "SemanticFieldOperation",
    "SemanticFieldOperationBatch",
    "SemanticFieldOperationError",
    "SemanticFieldOperationKind",
    "merge_semantic_fields",
]
