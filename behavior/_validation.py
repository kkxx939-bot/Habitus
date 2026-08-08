"""Behavior 值对象共享的严格、有界、无损校验原语。"""

from __future__ import annotations

import base64
import json
import math
import re
import unicodedata
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from enum import Enum
from types import MappingProxyType
from typing import Any, TypeVar

from foundation.integrity import canonical_digest, canonical_json

T = TypeVar("T")
E = TypeVar("E", bound=Enum)

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:+/\-]{0,255}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REFERENCE_SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")
_MAX_SIGNED_SQLITE_INTEGER = 9_223_372_036_854_775_807
_EMAIL_LIKE = re.compile(r"(?:^|[^A-Za-z0-9._%+\-])[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}(?:$|[^A-Za-z0-9.\-])")
def strict_utc(value: object, field_name: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError(f"{field_name} must be a timezone-aware UTC datetime")
    offset = value.utcoffset()
    if offset is None:
        raise ValueError(f"{field_name} must be a timezone-aware UTC datetime")
    if offset.total_seconds() != 0:
        raise ValueError(f"{field_name} must use UTC")
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
        raise ValueError(f"{field_name} must be normalized text")
    if len(normalized) > maximum or any(unicodedata.category(character) == "Cc" for character in normalized):
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
        raise ValueError(f"{field_name} must have a URI scheme")
    if folded.startswith("data:") or ";base64," in folded or folded.startswith("base64:"):
        raise ValueError(f"{field_name} cannot contain inline or base64 data")
    return reference


def identifier(value: object, field_name: str, *, maximum: int = 256) -> str:
    normalized = bounded_text(value, field_name, maximum=maximum)
    if _IDENTIFIER.fullmatch(normalized) is None:
        raise ValueError(f"{field_name} must be a safe normalized identifier")
    return normalized


def pii_safe_identifier(value: object, field_name: str, *, maximum: int = 256) -> str:
    normalized = identifier(value, field_name, maximum=maximum)
    if "@" in normalized or _EMAIL_LIKE.search(normalized) is not None:
        raise ValueError(f"{field_name} cannot contain personal contact information")
    return normalized


def optional_identifier(value: object, field_name: str, *, maximum: int = 256) -> str | None:
    if value is None:
        return None
    return identifier(value, field_name, maximum=maximum)


def fingerprint_fields(
    *,
    name: object,
    version: object,
    pipeline_version: object,
    output_schema_version: object,
    model_provider: object,
    model_name: object,
    prompt_version: object,
    field_prefix: str,
    model_backed: bool,
) -> tuple[str, str, str, str, str | None, str | None, str | None]:
    resolved = (
        identifier(name, f"{field_prefix}.name"),
        identifier(version, f"{field_prefix}.version"),
        identifier(pipeline_version, f"{field_prefix}.pipeline_version"),
        identifier(output_schema_version, f"{field_prefix}.output_schema_version"),
        optional_identifier(model_provider, f"{field_prefix}.model_provider"),
        optional_identifier(model_name, f"{field_prefix}.model_name"),
        optional_identifier(prompt_version, f"{field_prefix}.prompt_version"),
    )
    model_values = resolved[4:]
    if model_backed and any(value is None for value in model_values):
        raise ValueError(f"{field_prefix} model fingerprint requires provider, model, and prompt version")
    if not model_backed and any(value is not None for value in model_values):
        raise ValueError(f"{field_prefix} non-model fingerprint cannot declare model fields")
    return resolved


def sha256_digest(value: object, field_name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
    return value


def non_negative_int(value: object, field_name: str) -> int:
    return _bounded_int(value, field_name, minimum=0, label="non-negative")


def positive_int(value: object, field_name: str) -> int:
    return _bounded_int(value, field_name, minimum=1, label="positive")


def _bounded_int(value: object, field_name: str, *, minimum: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= _MAX_SIGNED_SQLITE_INTEGER:
        raise ValueError(f"{field_name} must be a bounded {label} integer")
    return value


def finite_score(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{field_name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{field_name} must be finite and between zero and one")
    return result


def typed_tuple(
    value: object,
    field_name: str,
    item_type: type[T],
    *,
    maximum_items: int,
    allow_empty: bool = True,
) -> tuple[T, ...]:
    if not isinstance(value, tuple):
        raise TypeError(f"{field_name} must be a tuple")
    if (not allow_empty and not value) or len(value) > maximum_items:
        raise ValueError(f"{field_name} exceeds its item boundary")
    if any(not isinstance(item, item_type) for item in value):
        raise ValueError(f"{field_name} exceeds its typed item boundary")
    if len(set(value)) != len(value):
        raise ValueError(f"{field_name} must not contain duplicates")
    return value


def enum_tuple(
    value: object,
    field_name: str,
    enum_type: type[E],
    *,
    maximum_items: int,
    allow_empty: bool = False,
) -> tuple[E, ...]:
    if not isinstance(value, tuple):
        raise TypeError(f"{field_name} must be a tuple")
    if (not allow_empty and not value) or len(value) > maximum_items:
        raise ValueError(f"{field_name} exceeds its item boundary")
    resolved = tuple(enum_type(item) for item in value)
    if len(set(resolved)) != len(resolved):
        raise ValueError(f"{field_name} must not contain duplicates")
    return resolved


def identifier_tuple(
    value: object,
    field_name: str,
    *,
    maximum_items: int,
    item_maximum: int = 256,
) -> tuple[str, ...]:
    if not isinstance(value, tuple):
        raise TypeError(f"{field_name} must be a tuple of identifiers")
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
    forbidden_keys: frozenset[str] = frozenset(),
    reject_inline_data: bool = True,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be an object")
    normalized = _snapshot(value, field_name, maximum_chars, maximum_items, maximum_depth,
                           forbidden_keys, reject_inline_data)
    if not isinstance(normalized, dict):
        raise TypeError(f"{field_name} must be an object")
    return _freeze(normalized)


def json_value_snapshot(
    value: object,
    field_name: str,
    *,
    maximum_chars: int,
    maximum_items: int = 128,
    maximum_depth: int = 12,
    forbidden_keys: frozenset[str] = frozenset(),
    reject_inline_data: bool = True,
) -> Any:
    normalized = _snapshot(value, field_name, maximum_chars, maximum_items, maximum_depth,
                           forbidden_keys, reject_inline_data)
    return _freeze(normalized)


def _snapshot(value: object, field_name: str, maximum_chars: int, maximum_items: int,
              maximum_depth: int, forbidden_keys: frozenset[str], reject_inline_data: bool) -> Any:
    normalized = _json_value(value, field_name, active=set(), depth=0, maximum_items=maximum_items,
                             maximum_depth=maximum_depth, forbidden_keys=forbidden_keys,
                             reject_inline_data=reject_inline_data)
    if len(canonical_json(normalized)) > maximum_chars:
        raise ValueError(f"{field_name} exceeds its canonical JSON boundary")
    return normalized


def _json_value(
    value: object,
    field_name: str,
    *,
    active: set[int],
    depth: int,
    maximum_items: int,
    maximum_depth: int,
    forbidden_keys: frozenset[str],
    reject_inline_data: bool,
) -> Any:
    if depth > maximum_depth:
        raise ValueError(f"{field_name} exceeds its nesting boundary")
    if value is None or isinstance(value, bool | int):
        return value
    if isinstance(value, str):
        text = bounded_text(value, field_name, maximum=1_000_000, allow_empty=True)
        folded = text.casefold()
        if reject_inline_data and (
            folded.startswith("data:")
            or folded.startswith("base64:")
            or ";base64," in folded
        ):
            raise TypeError(f"{field_name} cannot contain inline or base64 data")
        return text
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
                folded_key = key.casefold()
                if folded_key in forbidden_keys:
                    raise ValueError(f"{field_name} contains a forbidden field")
                if key in result:
                    raise ValueError(f"{field_name} contains duplicate keys")
                result[key] = _json_value(
                    item,
                    f"{field_name}.{key}",
                    active=active,
                    depth=depth + 1,
                    maximum_items=maximum_items,
                    maximum_depth=maximum_depth,
                    forbidden_keys=forbidden_keys,
                    reject_inline_data=reject_inline_data,
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
                    forbidden_keys=forbidden_keys,
                    reject_inline_data=reject_inline_data,
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


def strict_object(value: object, field_name: str, fields: frozenset[str]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise TypeError(f"{field_name} must contain string keys")
    keys = set(value)
    unknown = sorted(keys - fields)
    if unknown:
        raise ValueError(f"{field_name} contains unknown fields: {unknown}")
    missing = sorted(fields - keys)
    if missing:
        raise ValueError(f"{field_name} is missing required fields: {missing}")
    return dict(value)


def encode_cursor(payload: Mapping[str, object]) -> str:
    return base64.urlsafe_b64encode(canonical_json(payload).encode("utf-8")).decode("ascii").rstrip("=")


def decode_cursor(value: object) -> Mapping[str, object]:
    if not isinstance(value, str) or not value or len(value) > 2_048:
        raise ValueError("cursor must be bounded text")
    try:
        padding = "=" * (-len(value) % 4)
        raw = base64.b64decode(value + padding, altchars=b"-_", validate=True)
        parsed = json.loads(raw.decode("utf-8"))
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("cursor is malformed") from exc
    if not isinstance(parsed, dict) or canonical_json(parsed).encode("utf-8") != raw:
        raise ValueError("cursor is not canonical")
    return parsed


def encode_sequence_cursor(kind: str, query: Mapping[str, object], sequence: int) -> str:
    return encode_cursor(
        {
            "kind": kind,
            "query_digest": canonical_digest(query),
            "sequence": non_negative_int(sequence, "cursor.sequence"),
        }
    )


def decode_sequence_cursor(
    value: object,
    *,
    kind: str,
    query: Mapping[str, object],
    subject: str,
) -> int:
    data = decode_cursor(value)
    if data.get("kind") != kind or data.get("query_digest") != canonical_digest(query):
        raise ValueError(f"cursor does not belong to this {subject} query")
    return non_negative_int(data.get("sequence"), "cursor.sequence")
