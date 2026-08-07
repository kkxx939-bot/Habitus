"""Behavior 值对象共享的严格、有界、无损校验原语。"""

from __future__ import annotations

import base64
import json
import math
import re
import unicodedata
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Any

from foundation.integrity import canonical_json

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:+/\-]{0,255}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REFERENCE_SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")
_MAX_SIGNED_SQLITE_INTEGER = 9_223_372_036_854_775_807
_EMAIL_LIKE = re.compile(r"(?:^|[^A-Za-z0-9._%+\-])[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}(?:$|[^A-Za-z0-9.\-])")
_RESERVED_SEMANTIC_KEYS = frozenset(
    {
        "claim_id",
        "claim_kind",
        "claim_sequence",
        "content_digest",
        "epistemic_class",
        "encoded_digest",
        "derivation_class",
        "evidence_record_id",
        "evidence_sequence",
        "ingested_at",
        "normalizer_fingerprint",
        "processing_identity",
        "producer_fingerprint",
        "capability_digest",
        "source_trust",
        "semantic_digest",
        "policy_digest",
        "compatibility_policy_digest",
        "binding_policy_digest",
        "confidence_policy_digest",
        "event_id",
        "episode_id",
        "pattern_id",
        "prediction_id",
        "storage_metadata",
        "attempt_id",
    }
)
_FORBIDDEN_IDENTITY_PREFIXES = (
    "own" + "er_",
    "us" + "er_",
    "ten" + "ant_",
    "acc" + "ount_",
)
_RESERVED_CLAIM_SYSTEM_KEYS = _RESERVED_SEMANTIC_KEYS | frozenset(
    {
        "actor_role",
        "alternative_group_key",
        "attempt_id",
        "binding_policy_digest",
        "capability_digest",
        "claim_sequence",
        "compatibility_policy_digest",
        "confidence_policy_digest",
        "content_digest",
        "created_at",
        "derivation_class",
        "effective_confidence",
        "encoded_digest",
        "evidence_record_digest",
        "evidence_record_id",
        "evidence_sequence",
        "ingested_at",
        "normalizer_fingerprint",
        "processing_identity",
        "producer_fingerprint",
        "semantic_digest",
        "semantic_fingerprint",
        "source_confidence",
        "source_epistemic_class",
        "source_trust",
        "subject_role",
        "time_end",
        "time_start",
        "time_uncertainty_ms",
    }
)


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


def sha256_digest(value: object, field_name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
    return value


def non_negative_int(value: object, field_name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 <= value <= _MAX_SIGNED_SQLITE_INTEGER
    ):
        raise ValueError(f"{field_name} must be a bounded non-negative integer")
    return value


def positive_int(value: object, field_name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 < value <= _MAX_SIGNED_SQLITE_INTEGER
    ):
        raise ValueError(f"{field_name} must be a bounded positive integer")
    return value


def finite_score(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{field_name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{field_name} must be finite and between zero and one")
    return result


def typed_tuple(value: object, field_name: str, item_type: type[Any], *, maximum_items: int) -> tuple[Any, ...]:
    if not isinstance(value, tuple):
        raise TypeError(f"{field_name} must be a tuple")
    if len(value) > maximum_items or any(not isinstance(item, item_type) for item in value):
        raise ValueError(f"{field_name} exceeds its typed item boundary")
    if len(set(value)) != len(value):
        raise ValueError(f"{field_name} must not contain duplicates")
    return value


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
    reserved_keys: frozenset[str] = _RESERVED_SEMANTIC_KEYS,
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
        reserved_keys=reserved_keys,
    )
    if not isinstance(normalized, dict):
        raise TypeError(f"{field_name} must be an object")
    if len(canonical_json(normalized)) > maximum_chars:
        raise ValueError(f"{field_name} exceeds its canonical JSON boundary")
    return _freeze(normalized)


def json_value_snapshot(
    value: object,
    field_name: str,
    *,
    maximum_chars: int,
    maximum_items: int = 128,
    maximum_depth: int = 12,
) -> Any:
    normalized = _json_value(
        value,
        field_name,
        active=set(),
        depth=0,
        maximum_items=maximum_items,
        maximum_depth=maximum_depth,
        reserved_keys=_RESERVED_SEMANTIC_KEYS,
    )
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
    reserved_keys: frozenset[str],
) -> Any:
    if depth > maximum_depth:
        raise ValueError(f"{field_name} exceeds its nesting boundary")
    if value is None or isinstance(value, bool | int):
        return value
    if isinstance(value, str):
        text = bounded_text(value, field_name, maximum=1_000_000, allow_empty=True)
        folded = text.casefold()
        if folded.startswith("data:") or folded.startswith("base64:") or ";base64," in folded:
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
                if folded_key in reserved_keys or any(
                    folded_key.startswith(prefix) for prefix in _FORBIDDEN_IDENTITY_PREFIXES
                ):
                    raise ValueError(f"{field_name} contains a reserved system semantic field")
                if key in result:
                    raise ValueError(f"{field_name} contains duplicate keys")
                result[key] = _json_value(
                    item,
                    f"{field_name}.{key}",
                    active=active,
                    depth=depth + 1,
                    maximum_items=maximum_items,
                    maximum_depth=maximum_depth,
                    reserved_keys=reserved_keys,
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
                    reserved_keys=reserved_keys,
                )
                for index, item in enumerate(value)
            ]
        finally:
            active.remove(marker)
    raise TypeError(f"{field_name} contains unsupported type {type(value).__name__}")


def claim_semantic_json_snapshot(
    value: object,
    field_name: str,
    *,
    maximum_chars: int,
    maximum_items: int = 128,
    maximum_depth: int = 12,
) -> Mapping[str, Any]:
    return json_snapshot(
        value,
        field_name,
        maximum_chars=maximum_chars,
        maximum_items=maximum_items,
        maximum_depth=maximum_depth,
        reserved_keys=_RESERVED_CLAIM_SYSTEM_KEYS,
    )


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


__all__ = [
    "bounded_text",
    "decode_cursor",
    "encode_cursor",
    "external_reference",
    "finite_score",
    "identifier",
    "identifier_tuple",
    "json_snapshot",
    "json_value_snapshot",
    "non_negative_int",
    "optional_bounded_text",
    "optional_identifier",
    "parse_utc",
    "pii_safe_identifier",
    "positive_int",
    "require_fields",
    "sha256_digest",
    "strict_fields",
    "strict_utc",
    "typed_tuple",
    "utc_text",
]
