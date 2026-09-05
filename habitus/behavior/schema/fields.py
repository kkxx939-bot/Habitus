"""按声明的字段类型把原始载荷规范化为强类型领域值。

时间纪律（用户裁定）：occurrence 上**全部时间统一为本地时间 + 显式偏移**，``available_at``
也不例外——带偏移的本地时间就是完整瞬时，混用两种约定才是"取错时间"的事故源。UTC 归一保持在
观测/判断层的入口，不进树；时序比较一律先解析为瞬时。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import date, datetime
from typing import Any

from habitus.behavior.model import behavior_local_timestamp
from habitus.behavior.schema.model import BehaviorFieldSchema, BehaviorFieldType, BehaviorSchemaError
from habitus.behavior.schema.vocabulary import (
    GAP_KINDS,
    OCCURRENCE_STATUSES,
    SHA256,
    STATUS_BASES,
)


def strict_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise BehaviorSchemaError(f"{label} must be an object with string keys")
    return dict(value)


def require_keys(payload: Mapping[str, Any], expected: set[str], label: str) -> None:
    unknown = set(payload) - expected
    missing = expected - set(payload)
    if unknown:
        raise BehaviorSchemaError(f"{label} contains unsupported keys: {sorted(unknown)}")
    if missing:
        raise BehaviorSchemaError(f"{label} is missing keys: {sorted(missing)}")


def text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BehaviorSchemaError(f"{label} must be non-empty text")
    normalized = value.strip()
    if any(not character.isprintable() and character not in "\n\t" for character in normalized):
        raise BehaviorSchemaError(f"{label} contains control characters")
    return normalized


def optional_text(value: Any, label: str) -> str | None:
    if value is None:
        return None
    return text(value, label)


def date_value(value: Any, label: str) -> date:
    if isinstance(value, datetime):
        raise BehaviorSchemaError(f"{label} must be a date without a time")
    if isinstance(value, date):
        return value
    if not isinstance(value, str):
        raise BehaviorSchemaError(f"{label} must be an ISO date")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise BehaviorSchemaError(f"{label} must be an ISO date") from exc


def datetime_value(value: Any, label: str) -> datetime:
    """带整分钟本地偏移的时刻；序列化保留偏移，不折 UTC。"""

    parsed = value
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise BehaviorSchemaError(f"{label} must be an ISO timestamp") from exc
    try:
        return behavior_local_timestamp(parsed, label)
    except (TypeError, ValueError) as exc:
        raise BehaviorSchemaError(str(exc)) from exc


def boolean(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise BehaviorSchemaError(f"{label} must be a boolean")
    return value


def _sequence(value: Any, label: str) -> tuple[Any, ...]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise BehaviorSchemaError(f"{label} must be an array")
    return tuple(value)


def string_tuple(value: Any, label: str) -> tuple[str, ...]:
    values = tuple(text(item, f"{label} item") for item in _sequence(value, label))
    if len(values) != len(set(values)):
        raise BehaviorSchemaError(f"{label} must not contain duplicates")
    return values


def sha256_tuple(value: Any, label: str) -> tuple[str, ...]:
    values: list[str] = []
    for item in _sequence(value, label):
        if not isinstance(item, str) or not SHA256.fullmatch(item):
            raise BehaviorSchemaError(f"{label} must contain lowercase SHA-256 text")
        values.append(item)
    if len(values) != len(set(values)):
        raise BehaviorSchemaError(f"{label} must not contain duplicates")
    return tuple(values)


def sha256_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not SHA256.fullmatch(value):
        raise BehaviorSchemaError(f"{label} must be lowercase SHA-256 text")
    return value


def optional_sha256_text(value: Any, label: str) -> str | None:
    if value is None:
        return None
    return sha256_text(value, label)


def _enum(value: Any, allowed: frozenset[str], label: str) -> str:
    resolved = text(value, label)
    if resolved not in allowed:
        raise BehaviorSchemaError(f"{label} must be one of {sorted(allowed)}")
    return resolved


def occurrence_status(value: Any, label: str) -> str:
    return _enum(value, OCCURRENCE_STATUSES, label)


def status_basis(value: Any, label: str) -> str:
    return _enum(value, STATUS_BASES, label)


def gap_kind(value: Any, label: str) -> str:
    return _enum(value, GAP_KINDS, label)


def basis_steps(value: Any, label: str) -> tuple[dict[str, Any], ...]:
    """构成一件事的行为事实步骤；每步的时间在写入时从观测物化（观测之后会释放）。"""

    # 步骤不再内联 observation_ids：原料在发布后即释放，那些 id 只会是死引用（实测占文档 1/3）。
    # 步骤的起止与可知时刻在写入时已从观测物化，语义面自足。
    expected = {"semantics", "started_at", "ended_at", "available_at"}
    resolved: list[dict[str, Any]] = []
    for index, item in enumerate(_sequence(value, label), start=1):
        payload = strict_mapping(item, f"{label}[{index}]")
        require_keys(payload, expected, f"{label}[{index}]")
        started_at = datetime_value(payload["started_at"], f"{label}[{index}].started_at")
        ended_at = datetime_value(payload["ended_at"], f"{label}[{index}].ended_at")
        available_at = datetime_value(payload["available_at"], f"{label}[{index}].available_at")
        if ended_at < started_at:
            raise BehaviorSchemaError(f"{label}[{index}] ended_at cannot precede started_at")
        resolved.append(
            {
                "semantics": text(payload["semantics"], f"{label}[{index}].semantics"),
                "started_at": started_at,
                "ended_at": ended_at,
                "available_at": available_at,
            }
        )
    return tuple(resolved)


_VALIDATORS: dict[BehaviorFieldType, Callable[[Any, str], Any]] = {
    BehaviorFieldType.STRING: text,
    BehaviorFieldType.OPTIONAL_STRING: optional_text,
    BehaviorFieldType.DATE: date_value,
    BehaviorFieldType.DATETIME: datetime_value,
    BehaviorFieldType.BOOLEAN: boolean,
    BehaviorFieldType.STRING_LIST: string_tuple,
    BehaviorFieldType.SHA256_LIST: sha256_tuple,
    BehaviorFieldType.SHA256: sha256_text,
    BehaviorFieldType.OPTIONAL_SHA256: optional_sha256_text,
    BehaviorFieldType.OCCURRENCE_STATUS: occurrence_status,
    BehaviorFieldType.STATUS_BASIS: status_basis,
    BehaviorFieldType.GAP_KIND: gap_kind,
    BehaviorFieldType.BASIS_LIST: basis_steps,
}


def validate_field(field: BehaviorFieldSchema, value: Any) -> Any:
    """按声明类型规范化单个字段；非 Schema 异常统一收敛为 BehaviorSchemaError。"""

    try:
        return _VALIDATORS[field.field_type](value, field.name)
    except (TypeError, ValueError) as exc:
        if isinstance(exc, BehaviorSchemaError):
            raise
        raise BehaviorSchemaError(f"behavior field {field.name} is invalid") from exc


__all__ = [
    "require_keys",
    "strict_mapping",
    "text",
    "validate_field",
]
