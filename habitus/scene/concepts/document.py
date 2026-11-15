"""概念定义的落盘格式：正文给人读，尾部规范 JSON 是机器读的唯一来源。

解码后重新编码必须与原文逐字节相同——正文是从字段渲染出来的，任何手改都会让回读失败，这样
"文件上写的"与"机器读到的"不可能是两回事。字段集合是严格的：多出来、少了的键都拒收（旧盘面上模型写的细分 / 汇总 /
派生情境概念带着规则、档、材料、种类这些键，读到要明确报错，不静默忽略、不自动转换）。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from habitus.scene.codec import SceneRecordError, decode_record, encode_record
from habitus.scene.concepts.model import (
    ConceptDefinition,
    ConceptError,
    ConceptOrigin,
    ConceptRole,
    ConceptSource,
)
from habitus.scene.concepts.situation import SituationRule

MARKER = "HABITUS_CONCEPT"
RECORD_TYPE = "concept"
_KEYS = frozenset(
    {
        "record_type",
        "name",
        "definition",
        "role",
        "classes",
        "lane",
        "title",
        "retired",
        "situation",
        "source",
        "created_at",
    }
)
_ROLE_LABELS = {ConceptRole.BEHAVIOR: "行为", ConceptRole.DAY_TYPE: "情境·日历"}
_ORIGIN_LABELS = {ConceptOrigin.VOCABULARY: "同步词表自动生成", ConceptOrigin.CALENDAR: "日历机械生成"}


def _method_of(definition: ConceptDefinition) -> str:
    """行为概念看记录的类编号；日历情境按日历算。"""

    if definition.situation is not None:
        return "算法按日历算——" + definition.situation.criterion()
    return "类已停用，不再命中" if definition.retired else "看记录的类编号（不判）"


def encode(definition: ConceptDefinition) -> str:
    lines = [
        f"# {definition.label}",
        "",
        f"判据：{definition.definition}",
        f"判法：{_method_of(definition)}",
        f"类别：{_ROLE_LABELS[definition.role]}" + ("" if definition.lane is None else f" · lane {definition.lane}"),
    ]
    if definition.classes:
        lines.append(f"类：{' / '.join(definition.classes)}")
    lines.append(f"来源：{_ORIGIN_LABELS[definition.source.origin]}")
    payload: dict[str, Any] = {
        "record_type": RECORD_TYPE,
        "name": definition.name,
        "definition": definition.definition,
        "role": definition.role.value,
        "classes": list(definition.classes),
        "lane": definition.lane,
        "title": definition.title,
        "retired": definition.retired,
        "situation": None if definition.situation is None else definition.situation.payload(),
        "source": {"origin": definition.source.origin.value},
        "created_at": definition.created_at,
    }
    return encode_record("\n".join(lines) + "\n", MARKER, payload)


def decode(text: str, *, expected_identity: str | None = None) -> ConceptDefinition:
    _body, payload = decode_record(text, MARKER)
    if payload.get("record_type") != RECORD_TYPE:
        raise SceneRecordError("concept record has the wrong record_type")
    if set(payload) != _KEYS:
        raise SceneRecordError(
            f"concept record has unexpected or missing fields: extra {sorted(set(payload) - _KEYS)}, "
            f"missing {sorted(_KEYS - set(payload))}"
        )
    try:
        lane, title = _optional_text(payload, "lane", "concept lane"), _optional_text(payload, "title", "concept title")
        definition = ConceptDefinition(
            name=_text(payload, "name"),
            definition=_text(payload, "definition"),
            role=ConceptRole(payload.get("role")),
            classes=tuple(str(item) for item in _list(payload, "classes")),
            lane=lane,
            title=title,
            retired=_flag(payload, "retired"),
            situation=_situation(payload.get("situation")),
            source=_source(payload.get("source")),
            created_at=datetime.fromisoformat(_text(payload, "created_at")),
        )
    except (ConceptError, TypeError, ValueError) as exc:
        raise SceneRecordError(f"concept record is not a valid definition: {exc}") from exc
    if expected_identity is not None and definition.identity != expected_identity:
        raise SceneRecordError("concept record does not carry the identity of its address")
    if encode(definition) != text:
        raise SceneRecordError("concept record body does not match its canonical rendering")
    return definition


def _optional_text(payload: dict[str, Any], key: str, label: str) -> str | None:
    value = payload.get(key)
    if value is not None and not isinstance(value, str):
        raise ConceptError(f"{label} must be text")
    return value


def _flag(payload: dict[str, Any], key: str) -> bool:
    value = payload.get(key)
    if not isinstance(value, bool):
        raise SceneRecordError(f"concept field {key!r} must be a boolean")
    return value


def _text(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        raise SceneRecordError(f"concept record field {key!r} must be text")
    return value


def _list(payload: dict[str, Any], key: str) -> list[Any]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise SceneRecordError(f"concept record field {key!r} must be a list")
    return value


def _situation(item: object) -> SituationRule | None:
    if item is None:
        return None
    if not isinstance(item, dict) or set(item) != {"value", "values"}:
        raise SceneRecordError("concept situation must be an object with value and values")
    value, values = item["value"], item["values"]
    if not isinstance(value, str) or not isinstance(values, list) or any(not isinstance(v, str) for v in values):
        raise SceneRecordError("concept situation value must be text and values a list of text")
    return SituationRule(value=value, values=tuple(values))


def _source(item: object) -> ConceptSource:
    if not isinstance(item, dict) or set(item) != {"origin"}:
        raise SceneRecordError("concept source must be an object with only an origin")
    return ConceptSource(origin=ConceptOrigin(item.get("origin")))


__all__ = ["MARKER", "RECORD_TYPE", "decode", "encode"]
