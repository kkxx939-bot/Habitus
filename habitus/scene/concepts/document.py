"""概念定义的落盘格式：正文给人读，尾部规范 JSON 是机器读的唯一来源。

解码后重新编码必须与原文逐字节相同——正文是从字段渲染出来的，任何手改都会让回读失败，这样
"文件上写的"与"机器读到的"不可能是两回事。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from habitus.scene.codec import SceneRecordError, decode_record, encode_record
from habitus.scene.concepts.model import (
    ConceptDefinition,
    ConceptError,
    ConceptGrade,
    ConceptOrigin,
    ConceptRole,
    ConceptSource,
    ContextScope,
    GradeMeasure,
    MechanicalRule,
)
from habitus.scene.concepts.situation import SituationBasis, SituationRule

MARKER = "HABITUS_CONCEPT"
RECORD_TYPE = "concept"
_ROLE_LABELS = {
    ConceptRole.BEHAVIOR: "行为",
    ConceptRole.STATE: "情境·状态",
    ConceptRole.OBJECT: "情境·对象",
    ConceptRole.DAY_TYPE: "情境·日型",
    ConceptRole.DERIVED: "情境·派生（算法从历史命中算）",
}
_ORIGIN_LABELS = {ConceptOrigin.BASELINE: "基准", ConceptOrigin.RESIDUE: "残差升级"}
_CONTEXT_LABELS = {ContextScope.OCCURRENCE: "只看这一条", ContextScope.DAY: "要看当天时间线"}


def encode(definition: ConceptDefinition) -> str:
    if not isinstance(definition, ConceptDefinition):
        raise TypeError("definition must be a ConceptDefinition")
    lines = [
        f"# {definition.name}",
        "",
        f"判据：{definition.definition}",
        f"判法：{'算法按规则判——' + definition.rule.criterion() if definition.rule is not None else '模型按判据句判'}",
        f"材料：{_CONTEXT_LABELS[definition.context]}"
        + (f"；常态 {' / '.join(definition.required_baseline_keys)}" if definition.required_baseline_keys else ""),
        f"类别：{_ROLE_LABELS[definition.role]}",
        f"上级：{definition.parent if definition.parent is not None else '（无）'}",
    ]
    if definition.situation is not None:
        lines.append(f"算法：{definition.situation.criterion()}")
    elif definition.role.is_situation:
        # 说清楚它为什么不会命中：情境概念没有说明就没有算法算得出它（事实门还没接上）。
        lines.append("算法：（还没有；这个情境要等事实门接上数据源才算得出）")
    if definition.grades:
        lines.append("档：" + " · ".join(f"{grade.name} = {grade.criterion()}" for grade in definition.grades))
    source = _ORIGIN_LABELS[definition.source.origin]
    if definition.source.kind_token is not None:
        source = f"{source}，认领 kind「{definition.source.kind_token}」"
    if definition.source.note is not None:
        source = f"{source}（{definition.source.note}）"
    lines.append(f"来源：{source}")
    payload: dict[str, Any] = {
        "record_type": RECORD_TYPE,
        "name": definition.name,
        "definition": definition.definition,
        "role": definition.role.value,
        "parent": definition.parent,
        "grades": [
            {"name": g.name, "measure": g.measure.value, "lower": g.lower, "upper": g.upper, "relative": g.relative}
            for g in definition.grades
        ],
        "rule": None
        if definition.rule is None
        else {
            "measure": definition.rule.measure.value,
            "lower": definition.rule.lower,
            "upper": definition.rule.upper,
            "relative_to": definition.rule.relative_to,
        },
        "context": definition.context.value,
        "baseline_keys": list(definition.baseline_keys),
        "situation": None if definition.situation is None else definition.situation.payload(),
        "source": {"origin": definition.source.origin.value, "note": definition.source.note, "kind_token": definition.source.kind_token},
        "created_at": definition.created_at,
    }
    return encode_record("\n".join(lines) + "\n", MARKER, payload)


def decode(text: str, *, expected_identity: str | None = None) -> ConceptDefinition:
    _body, payload = decode_record(text, MARKER)
    if payload.get("record_type") != RECORD_TYPE:
        raise SceneRecordError("concept record has the wrong record_type")
    try:
        definition = ConceptDefinition(
            name=_text(payload, "name"),
            definition=_text(payload, "definition"),
            role=ConceptRole(payload.get("role")),
            parent=payload.get("parent"),
            grades=tuple(_grade(item) for item in _list(payload, "grades")),
            rule=_rule(payload.get("rule")),
            context=ContextScope(payload.get("context")),
            baseline_keys=tuple(str(item) for item in _list(payload, "baseline_keys")),
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


def _optional_int(item: dict[str, Any], key: str) -> int | None:
    value = item.get(key)
    if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
        raise SceneRecordError(f"concept record field {key!r} must be an integer or null")
    return value


def _grade(item: object) -> ConceptGrade:
    if not isinstance(item, dict) or set(item) != {"name", "measure", "lower", "upper", "relative"}:
        raise SceneRecordError("concept grade must be an object with name, measure, lower, upper and relative")
    name, lower, upper, relative = item["name"], item["lower"], item["upper"], item["relative"]
    if not isinstance(name, str) or not isinstance(lower, int) or not isinstance(upper, int) or not isinstance(relative, bool):
        raise SceneRecordError("concept grade fields have the wrong types")
    return ConceptGrade(name=name, measure=GradeMeasure(item["measure"]), lower=lower, upper=upper, relative=relative)


def _rule(item: object) -> MechanicalRule | None:
    if item is None:
        return None
    if not isinstance(item, dict) or set(item) != {"measure", "lower", "upper", "relative_to"}:
        raise SceneRecordError("concept rule must be an object with measure, lower, upper and relative_to")
    relative_to = item["relative_to"]
    if relative_to is not None and not isinstance(relative_to, str):
        raise SceneRecordError("concept rule relative_to must be text or null")
    return MechanicalRule(
        measure=GradeMeasure(item["measure"]), lower=_optional_int(item, "lower"), upper=_optional_int(item, "upper"), relative_to=relative_to
    )


def _situation(item: object) -> SituationRule | None:
    if item is None:
        return None
    if not isinstance(item, dict) or "basis" not in item:
        raise SceneRecordError("concept situation must be an object naming its basis")
    basis = SituationBasis(item["basis"])
    weekdays = item.get("weekdays", [])
    if not isinstance(weekdays, list) or any(isinstance(day, bool) or not isinstance(day, int) for day in weekdays):
        raise SceneRecordError("concept situation weekdays must be a list of integers")
    return SituationRule(
        basis=basis,
        weekdays=tuple(weekdays),
        value=item.get("value"),
        concept=item.get("concept"),
        grade=item.get("grade"),
        days=item.get("days", 1),
    )


def _source(item: object) -> ConceptSource:
    if not isinstance(item, dict):
        raise SceneRecordError("concept source must be an object")
    return ConceptSource(origin=ConceptOrigin(item.get("origin")), note=item.get("note"), kind_token=item.get("kind_token"))


__all__ = ["MARKER", "RECORD_TYPE", "decode", "encode"]
