"""公开数据集 Adapter 共用的严格读取与时间工具。"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

_MAX_DATASET_BYTES = 2 * 1024 * 1024 * 1024
_IDENTIFIER_CHARACTER = re.compile(r"[^A-Za-z0-9_.-]+")


def read_json_array(path: str | Path) -> tuple[Mapping[str, object], ...]:
    source = Path(path).expanduser().resolve(strict=True)
    if not source.is_file() or source.stat().st_size > _MAX_DATASET_BYTES:
        raise ValueError("benchmark dataset must be a bounded regular JSON file")
    try:
        value = json.loads(
            source.read_text(encoding="utf-8-sig"),
            object_pairs_hook=_strict_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("benchmark dataset is not valid UTF-8 JSON") from exc
    if not isinstance(value, list) or not value:
        raise ValueError("benchmark dataset root must be a non-empty array")
    if any(not isinstance(item, Mapping) for item in value):
        raise ValueError("benchmark dataset samples must be objects")
    return tuple(cast(Mapping[str, object], item) for item in value)


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"benchmark dataset contains a duplicate JSON key: {key}")
        result[key] = value
    return result


def object_value(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise ValueError(f"{label} keys must be strings")
    return cast(Mapping[str, object], value)


def list_value(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be an array")
    return value


def text_value(value: object, label: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be text")
    if not allow_empty and not value.strip():
        raise ValueError(f"{label} must be non-empty")
    return value


def safe_identifier(value: object, *, fallback: str) -> str:
    raw = str(value or "").strip()
    cleaned = _IDENTIFIER_CHARACTER.sub("-", raw).strip(".-")
    if not cleaned:
        cleaned = fallback
    if not cleaned[0].isalnum():
        cleaned = f"id-{cleaned}"
    return cleaned[:256]


def parse_time(value: object, formats: tuple[str, ...], *, fallback: datetime) -> datetime:
    if isinstance(value, str) and value.strip():
        normalized = value.strip()
        for candidate in (normalized, normalized.upper()):
            for pattern in formats:
                try:
                    parsed = datetime.strptime(candidate, pattern)
                except ValueError:
                    continue
                return parsed.replace(tzinfo=UTC)
        try:
            parsed = datetime.fromisoformat(normalized.replace("Z", "+00:00"))
        except ValueError:
            pass
        else:
            return parsed.replace(tzinfo=parsed.tzinfo or UTC).astimezone(UTC)
        raise ValueError(f"unsupported benchmark datetime: {normalized}")
    return fallback.astimezone(UTC)


def ordered_times(started_at: datetime, count: int) -> tuple[datetime, ...]:
    return tuple(started_at + timedelta(microseconds=index) for index in range(count))


__all__ = [
    "list_value",
    "object_value",
    "ordered_times",
    "parse_time",
    "read_json_array",
    "safe_identifier",
    "text_value",
]
