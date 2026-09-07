"""按声明的字段类型把原始载荷规范化为强类型领域值。

时间纪律与行为树一致：全部时间为本地时间 + 显式偏移，序列化保留偏移、不折 UTC。
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, date, datetime
from typing import Any

from habitus.behavior.model import BehaviorKind, behavior_local_timestamp
from habitus.behavior.uri import BehaviorURI, BehaviorURINodeType
from habitus.scene.model import SceneRole
from habitus.scene.schema.model import SceneFieldSchema, SceneFieldType, SceneSchemaError

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def strict_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise SceneSchemaError(f"{label} must be an object with string keys")
    return dict(value)


def require_keys(payload: Mapping[str, Any], expected: set[str], label: str) -> None:
    unknown = set(payload) - expected
    missing = expected - set(payload)
    if unknown:
        raise SceneSchemaError(f"{label} contains unsupported keys: {sorted(unknown)}")
    if missing:
        raise SceneSchemaError(f"{label} is missing keys: {sorted(missing)}")


def text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SceneSchemaError(f"{label} must be non-empty text")
    normalized = value.strip()
    if any(not character.isprintable() and character not in "\n\t" for character in normalized):
        raise SceneSchemaError(f"{label} contains control characters")
    return normalized


def line(value: Any, label: str) -> str:
    """单行文本：列表项与待用前提的一句话不得含换行——换行会在正文里伪造出新的小节。"""

    normalized = text(value, label)
    if "\n" in normalized or "\r" in normalized:
        raise SceneSchemaError(f"{label} must be a single line")
    return normalized


def date_value(value: Any, label: str) -> date:
    if isinstance(value, datetime):
        raise SceneSchemaError(f"{label} must be a date without a time")
    if isinstance(value, date):
        return value
    if not isinstance(value, str):
        raise SceneSchemaError(f"{label} must be an ISO date")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise SceneSchemaError(f"{label} must be an ISO date") from exc


def datetime_value(value: Any, label: str) -> datetime:
    parsed = value
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise SceneSchemaError(f"{label} must be an ISO timestamp") from exc
    try:
        return behavior_local_timestamp(parsed, label)
    except (TypeError, ValueError) as exc:
        raise SceneSchemaError(str(exc)) from exc


def _sequence(value: Any, label: str) -> tuple[Any, ...]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise SceneSchemaError(f"{label} must be an array")
    return tuple(value)


def string_tuple(value: Any, label: str) -> tuple[str, ...]:
    values = tuple(line(item, f"{label} item") for item in _sequence(value, label))
    if len(values) != len(set(values)):
        raise SceneSchemaError(f"{label} must not contain duplicates")
    return values


def occurrence_uri(value: Any, label: str) -> str:
    """成员与待用前提的产生方只能是行为树的 occurrence 文档 URI；规范化后以字符串存储。"""

    if not isinstance(value, str):
        raise SceneSchemaError(f"{label} must be a behavior occurrence URI")
    try:
        parsed = BehaviorURI.parse(value)
    except (TypeError, ValueError) as exc:
        raise SceneSchemaError(f"{label} must be a behavior occurrence URI") from exc
    if parsed.node_type is not BehaviorURINodeType.DOCUMENT or parsed.to_address().kind is not BehaviorKind.OCCURRENCE:
        raise SceneSchemaError(f"{label} must identify a behavior occurrence document")
    return str(parsed)


def member_list(value: Any, label: str) -> tuple[dict[str, Any], ...]:
    """成员：occurrence URI + 角色；URI 不重复。

    存储顺序按成员的开始时刻（URI 叶名里带着）再按 URI 定序——它是字段的确定性函数、不承载
    额外信息，但让正文与投影读到的就是时间序。
    """

    resolved: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, item in enumerate(_sequence(value, label), start=1):
        payload = strict_mapping(item, f"{label}[{index}]")
        require_keys(payload, {"uri", "role"}, f"{label}[{index}]")
        uri = occurrence_uri(payload["uri"], f"{label}[{index}].uri")
        if uri in seen:
            raise SceneSchemaError(f"{label} contains the same occurrence twice")
        seen.add(uri)
        try:
            role = SceneRole(text(payload["role"], f"{label}[{index}].role"))
        except ValueError as exc:
            raise SceneSchemaError(f"{label}[{index}].role must be one of {[r.value for r in SceneRole]}") from exc
        resolved.append({"uri": uri, "role": role.value})
    return tuple(
        sorted(resolved, key=lambda item: (BehaviorURI.parse(item["uri"]).to_address().started_at.astimezone(UTC), item["uri"]))
    )


def pending_effect_list(value: Any, label: str) -> tuple[dict[str, Any], ...]:
    """待用前提：一句话 + 建立它的成员 occurrence URI。"""

    resolved: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for index, item in enumerate(_sequence(value, label), start=1):
        payload = strict_mapping(item, f"{label}[{index}]")
        require_keys(payload, {"text", "producer_uri"}, f"{label}[{index}]")
        entry = {
            "text": line(payload["text"], f"{label}[{index}].text"),
            "producer_uri": occurrence_uri(payload["producer_uri"], f"{label}[{index}].producer_uri"),
        }
        key = (entry["text"], entry["producer_uri"])
        if key in seen:
            raise SceneSchemaError(f"{label} must not contain duplicates")
        seen.add(key)
        resolved.append(entry)
    return tuple(resolved)


def sha256_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise SceneSchemaError(f"{label} must be lowercase SHA-256 text")
    return value


_VALIDATORS: dict[SceneFieldType, Callable[[Any, str], Any]] = {
    SceneFieldType.STRING: text,
    SceneFieldType.DATE: date_value,
    SceneFieldType.DATETIME: datetime_value,
    SceneFieldType.STRING_LIST: string_tuple,
    SceneFieldType.MEMBER_LIST: member_list,
    SceneFieldType.PENDING_EFFECT_LIST: pending_effect_list,
    SceneFieldType.SHA256: sha256_text,
}


def validate_field(field: SceneFieldSchema, value: Any) -> Any:
    try:
        return _VALIDATORS[field.field_type](value, field.name)
    except (TypeError, ValueError) as exc:
        if isinstance(exc, SceneSchemaError):
            raise
        raise SceneSchemaError(f"scene field {field.name} is invalid") from exc


__all__ = ["line", "occurrence_uri", "require_keys", "strict_mapping", "text", "validate_field"]
