"""预测样本 Schema 的类型原语校验器。"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from datetime import date, datetime
from typing import Any
from urllib.parse import unquote

from prediction.schema.model import PredictionSchemaError
from prediction.schema.vocabulary import RECORD_ID, SHA256, URI, URI_HEX, URI_UNRESERVED


def strict_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise PredictionSchemaError(f"{label} must be an object with string keys")
    return dict(value)


def exact_mapping(value: Any, expected: set[str], label: str) -> dict[str, Any]:
    payload = strict_mapping(value, label)
    unknown = set(payload) - expected
    missing = expected - set(payload)
    if unknown:
        raise PredictionSchemaError(f"{label} contains unsupported keys: {sorted(unknown)}")
    if missing:
        raise PredictionSchemaError(f"{label} is missing keys: {sorted(missing)}")
    return payload


def text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PredictionSchemaError(f"{label} must be non-empty text")
    normalized = value.strip()
    if any(not character.isprintable() and character not in "\n\t" for character in normalized):
        raise PredictionSchemaError(f"{label} contains control characters")
    try:
        normalized.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise PredictionSchemaError(f"{label} must be valid UTF-8 text") from exc
    return normalized


def optional_text(value: Any, label: str) -> str | None:
    return None if value is None else text(value, label)


def uri_text(value: Any, label: str) -> str:
    resolved = text(value, label)
    if URI.fullmatch(resolved) is None:
        raise PredictionSchemaError(f"{label} must be an absolute URI")
    scheme, raw_path = resolved.split("://", 1)
    raw_segments = raw_path.split("/")
    if any(not segment for segment in raw_segments):
        raise PredictionSchemaError(f"{label} contains an empty URI segment")
    canonical_segments: list[str] = []
    for segment in raw_segments:
        index = 0
        while index < len(segment):
            if segment[index] != "%":
                index += 1
                continue
            if (
                index + 2 >= len(segment)
                or segment[index + 1] not in URI_HEX
                or segment[index + 2] not in URI_HEX
            ):
                raise PredictionSchemaError(f"{label} contains malformed URI percent encoding")
            index += 3
        try:
            decoded = unquote(segment, encoding="utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise PredictionSchemaError(f"{label} contains invalid URI UTF-8") from exc
        encoded: list[str] = []
        for character in decoded:
            if character in URI_UNRESERVED or ord(character) >= 128:
                encoded.append(character)
            else:
                encoded.extend(f"%{byte:02X}" for byte in character.encode("utf-8"))
        canonical_segments.append("".join(encoded))
    return f"{scheme.lower()}://{'/'.join(canonical_segments)}"


def optional_uri(value: Any, label: str) -> str | None:
    return None if value is None else uri_text(value, label)


def record_id(value: Any, label: str) -> str:
    resolved = text(value, label)
    if RECORD_ID.fullmatch(resolved) is None:
        raise PredictionSchemaError(f"{label} must be a stable lowercase record identifier")
    return resolved


def sha256(value: Any, label: str) -> str:
    resolved = text(value, label)
    if SHA256.fullmatch(resolved) is None:
        raise PredictionSchemaError(f"{label} must be a lowercase SHA-256 digest")
    return resolved


def date_value(value: Any, label: str) -> date:
    if isinstance(value, datetime):
        raise PredictionSchemaError(f"{label} must be a date without a time")
    if isinstance(value, date):
        return value
    if not isinstance(value, str):
        raise PredictionSchemaError(f"{label} must be an ISO date")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise PredictionSchemaError(f"{label} must be an ISO date") from exc


def datetime_value(value: Any, label: str) -> datetime:
    parsed = value
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise PredictionSchemaError(f"{label} must be an ISO timestamp") from exc
    if not isinstance(parsed, datetime) or parsed.tzinfo is None or parsed.utcoffset() is None:
        raise PredictionSchemaError(f"{label} must be a timezone-aware datetime")
    return parsed


def optional_datetime(value: Any, label: str) -> datetime | None:
    return None if value is None else datetime_value(value, label)


def sequence(value: Any, label: str) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise PredictionSchemaError(f"{label} must be an array")
    return tuple(value)


def string_tuple(value: Any, label: str) -> tuple[str, ...]:
    values = tuple(text(item, f"{label} item") for item in sequence(value, label))
    if len(values) != len(set(values)):
        raise PredictionSchemaError(f"{label} must not contain duplicates")
    return values


def records(
    value: Any,
    validator: Callable[[Any, str], dict[str, Any]],
    label: str,
    *,
    identity: str | None = None,
) -> tuple[dict[str, Any], ...]:
    records = tuple(validator(item, f"{label}[{index}]") for index, item in enumerate(sequence(value, label)))
    if identity is not None:
        identities = tuple(record[identity] for record in records)
        if len(identities) != len(set(identities)):
            raise PredictionSchemaError(f"{label} contains duplicate {identity} values")
    return records


def enum(value: Any, enum_type: type[Any], label: str) -> str:
    resolved = text(value, label)
    try:
        return str(enum_type(resolved).value)
    except ValueError as exc:
        raise PredictionSchemaError(f"{label} is not an allowed value") from exc


def enum_set(value: Any, allowed: set[str] | frozenset[str], label: str) -> str:
    resolved = text(value, label)
    if resolved not in allowed:
        raise PredictionSchemaError(f"{label} must be one of {sorted(allowed)}")
    return resolved


def boolean(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise PredictionSchemaError(f"{label} must be boolean")
    return value


def positive_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise PredictionSchemaError(f"{label} must be a positive integer")
    return value


def non_negative_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PredictionSchemaError(f"{label} must be a non-negative integer")
    return value


def confidence(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PredictionSchemaError(f"{label} must be numeric")
    try:
        normalized = float(value)
    except (OverflowError, ValueError) as exc:
        raise PredictionSchemaError(f"{label} must be a finite number") from exc
    if not math.isfinite(normalized):
        raise PredictionSchemaError(f"{label} must be a finite number")
    if not 0.0 <= normalized <= 1.0:
        raise PredictionSchemaError(f"{label} must be between zero and one")
    return normalized


def optional_confidence(value: Any, label: str) -> float | None:
    return None if value is None else confidence(value, label)


def optional_non_negative_number(value: Any, label: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PredictionSchemaError(f"{label} must be a non-negative number or null")
    try:
        normalized = float(value)
    except (OverflowError, ValueError) as exc:
        raise PredictionSchemaError(f"{label} must be a finite non-negative number or null") from exc
    if not math.isfinite(normalized) or normalized < 0:
        raise PredictionSchemaError(f"{label} must be a finite non-negative number or null")
    return normalized
