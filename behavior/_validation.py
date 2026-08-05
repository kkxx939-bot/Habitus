"""Behavior 值对象共享的严格、有界、无损校验原语。"""

from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Any

from foundation.integrity import canonical_json

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:@+\-]{0,255}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REFERENCE_SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")


def strict_utc(value: object, field_name: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be a timezone-aware datetime")
    return value.astimezone(timezone.utc)


def parse_utc(value: object, field_name: str) -> datetime:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field_name} must be a normalized ISO-8601 string")
    text = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be a valid ISO-8601 datetime") from exc
    return strict_utc(parsed, field_name)


def utc_text(value: datetime) -> str:
    return strict_utc(value, "datetime").isoformat(timespec="microseconds").replace("+00:00", "Z")


def bounded_text(
    value: object,
    field_name: str,
    *,
    maximum: int,
    allow_empty: bool = False,
) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be text")
    normalized = unicodedata.normalize("NFC", value)
    if normalized != normalized.strip() or (not allow_empty and not normalized):
        raise ValueError(f"{field_name} must be normalized non-empty text")
    if len(normalized) > maximum or any(ord(character) < 32 for character in normalized):
        raise ValueError(f"{field_name} exceeds its safe text boundary")
    return normalized


def optional_bounded_text(value: object, field_name: str, *, maximum: int) -> str | None:
    if value is None:
        return None
    return bounded_text(value, field_name, maximum=maximum)


def external_reference(value: object, field_name: str, *, maximum: int) -> str:
    reference = bounded_text(value, field_name, maximum=maximum)
    folded = reference.casefold()
    if _REFERENCE_SCHEME.match(reference) is None:
        raise ValueError(f"{field_name} must be an external reference with a URI scheme")
    if folded.startswith("data:") or ";base64," in folded:
        raise ValueError(f"{field_name} cannot contain inline media or base64 data")
    return reference


def identifier(value: object, field_name: str, *, maximum: int = 256) -> str:
    normalized = bounded_text(value, field_name, maximum=maximum)
    if _IDENTIFIER.fullmatch(normalized) is None:
        raise ValueError(f"{field_name} must be a safe normalized identifier")
    return normalized


def optional_identifier(value: object, field_name: str, *, maximum: int = 256) -> str | None:
    if value is None:
        return None
    return identifier(value, field_name, maximum=maximum)


def sha256_digest(value: object, field_name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
    return value


def non_negative_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return value


def positive_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def finite_score(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{field_name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{field_name} must be finite and between zero and one")
    return result


def identifier_tuple(
    value: object,
    field_name: str,
    *,
    maximum_items: int,
    item_maximum: int = 256,
) -> tuple[str, ...]:
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise TypeError(f"{field_name} must be a sequence of identifiers")
    if len(value) > maximum_items:
        raise ValueError(f"{field_name} exceeds its item boundary")
    result = tuple(identifier(item, f"{field_name}[{index}]", maximum=item_maximum) for index, item in enumerate(value))
    if len(set(result)) != len(result):
        raise ValueError(f"{field_name} must not contain duplicates")
    return result


def json_snapshot(
    value: object,
    field_name: str,
    *,
    maximum_chars: int,
    maximum_items: int = 128,
    maximum_depth: int = 12,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be an object")
    normalized = _json_value(
        value,
        field_name,
        active=set(),
        depth=0,
        maximum_items=maximum_items,
        maximum_depth=maximum_depth,
    )
    if not isinstance(normalized, dict):
        raise TypeError(f"{field_name} must be an object")
    if len(canonical_json(normalized)) > maximum_chars:
        raise ValueError(f"{field_name} exceeds its canonical JSON boundary")
    return _freeze(normalized)


def _json_value(
    value: object,
    field_name: str,
    *,
    active: set[int],
    depth: int,
    maximum_items: int,
    maximum_depth: int,
) -> Any:
    if depth > maximum_depth:
        raise ValueError(f"{field_name} exceeds its nesting boundary")
    if value is None or isinstance(value, bool | int):
        return value
    if isinstance(value, str):
        folded = value.casefold()
        if folded.startswith("data:") and ";base64," in folded:
            raise TypeError(f"{field_name} cannot contain inline base64 media")
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{field_name} cannot contain non-finite numbers")
        return value
    if isinstance(value, bytes | bytearray | memoryview):
        raise TypeError(f"{field_name} cannot contain binary values")
    if isinstance(value, Mapping):
        marker = id(value)
        if marker in active:
            raise ValueError(f"{field_name} cannot contain recursive values")
        if len(value) > maximum_items:
            raise ValueError(f"{field_name} exceeds its mapping boundary")
        active.add(marker)
        try:
            result: dict[str, Any] = {}
            for key, item in value.items():
                if not isinstance(key, str) or not key or len(key) > 128:
                    raise TypeError(f"{field_name} must contain bounded string keys")
                if key in result:
                    raise ValueError(f"{field_name} contains duplicate keys")
                result[key] = _json_value(
                    item,
                    f"{field_name}.{key}",
                    active=active,
                    depth=depth + 1,
                    maximum_items=maximum_items,
                    maximum_depth=maximum_depth,
                )
            return {key: result[key] for key in sorted(result)}
        finally:
            active.remove(marker)
    if isinstance(value, Sequence) and not isinstance(value, str):
        marker = id(value)
        if marker in active:
            raise ValueError(f"{field_name} cannot contain recursive values")
        if len(value) > maximum_items:
            raise ValueError(f"{field_name} exceeds its sequence boundary")
        active.add(marker)
        try:
            return [
                _json_value(
                    item,
                    f"{field_name}[{index}]",
                    active=active,
                    depth=depth + 1,
                    maximum_items=maximum_items,
                    maximum_depth=maximum_depth,
                )
                for index, item in enumerate(value)
            ]
        finally:
            active.remove(marker)
    raise TypeError(f"{field_name} contains unsupported type {type(value).__name__}")


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def strict_fields(value: object, field_name: str, allowed: frozenset[str]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise TypeError(f"{field_name} must contain string keys")
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"{field_name} contains unknown fields: {unknown}")
    return dict(value)


def require_fields(value: Mapping[str, Any], field_name: str, required: frozenset[str]) -> None:
    missing = sorted(required - set(value))
    if missing:
        raise ValueError(f"{field_name} is missing required fields: {missing}")


__all__ = [
    "bounded_text",
    "external_reference",
    "finite_score",
    "identifier",
    "identifier_tuple",
    "json_snapshot",
    "non_negative_int",
    "optional_bounded_text",
    "optional_identifier",
    "parse_utc",
    "positive_int",
    "require_fields",
    "sha256_digest",
    "strict_fields",
    "strict_utc",
    "utc_text",
]
