"""假设的落盘格式：正文给人读，尾部规范 JSON 是机器读的唯一来源；回读逐字节比对。"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from habitus.scene.codec import SceneRecordError, decode_record, encode_record
from habitus.scene.hypotheses.model import (
    Antecedent,
    Aspect,
    Direction,
    Hypothesis,
    HypothesisError,
    HypothesisOrigin,
    HypothesisSource,
    TypePrior,
)

MARKER = "HABITUS_HYPOTHESIS"
RECORD_TYPE = "hypothesis"
_DIRECTION_LABELS = {
    Aspect.PROBABILITY: {Direction.UP: "↑ 更可能发生", Direction.DOWN: "↓ 更不可能发生"},
    Aspect.TIMING: {Direction.UP: "↑ 推后", Direction.DOWN: "↓ 提前"},
    Aspect.COUNT: {Direction.UP: "↑ 次数增多", Direction.DOWN: "↓ 次数减少"},
}
_TYPE_LABELS = {TypePrior.ENABLING: "使能", TypePrior.PROMOTING: "促进", TypePrior.INHIBITING: "抑制"}
_ORIGIN_LABELS = {HypothesisOrigin.BASELINE: "基准", HypothesisOrigin.NEW_CONCEPT: "为新概念补写"}


def encode(hypothesis: Hypothesis) -> str:
    if not isinstance(hypothesis, Hypothesis):
        raise TypeError("hypothesis must be a Hypothesis")
    source = _ORIGIN_LABELS[hypothesis.source.origin]
    if hypothesis.source.note is not None:
        source = f"{source}（{hypothesis.source.note}）"
    prior = "（不标）" if hypothesis.type_prior is None else f"{_TYPE_LABELS[hypothesis.type_prior]}（基准的猜测，类型最终由账读出）"
    lines = [
        f"# {hypothesis.label()}",
        "",
        f"方向：{_DIRECTION_LABELS[hypothesis.aspect][hypothesis.direction]}",
        f"落在：{hypothesis.opportunity_label}（基准的猜测，只用来读）",
    ]
    if hypothesis.is_open_ended:
        # 无节律型才有释放条件；节律型这一行恒为空，印出来只是噪音。
        lines.append(f"释放条件：{' / '.join(hypothesis.released_by) if hypothesis.released_by else '（无，一直立着）'}")
    lines += [
        f"先验类型：{prior}",
        f"分账建议：{' / '.join(hypothesis.split_by) if hypothesis.split_by else '（无）'}",
        f"理由：{hypothesis.note}",
        f"来源：{source}",
        f"指纹：{hypothesis.fingerprint}",
    ]
    payload: dict[str, Any] = {
        "record_type": RECORD_TYPE,
        "antecedents": [{"concept": item.concept, "grade": item.grade} for item in hypothesis.antecedents],
        "consequent": hypothesis.consequent,
        "aspect": hypothesis.aspect.value,
        "direction": hypothesis.direction.value,
        "expected_at": hypothesis.expected_at,
        "horizon": hypothesis.horizon,
        "released_by": list(hypothesis.released_by),
        "type_prior": None if hypothesis.type_prior is None else hypothesis.type_prior.value,
        "split_by": list(hypothesis.split_by),
        "note": hypothesis.note,
        "source": {"origin": hypothesis.source.origin.value, "note": hypothesis.source.note},
        "created_at": hypothesis.created_at,
    }
    return encode_record("\n".join(lines) + "\n", MARKER, payload)


def decode(text: str, *, expected_identity: str | None = None) -> Hypothesis:
    _body, payload = decode_record(text, MARKER)
    if payload.get("record_type") != RECORD_TYPE:
        raise SceneRecordError("hypothesis record has the wrong record_type")
    try:
        raw_prior = payload.get("type_prior")
        hypothesis = Hypothesis(
            antecedents=tuple(_antecedent(item) for item in _list(payload, "antecedents")),
            consequent=_text(payload, "consequent"),
            aspect=Aspect(payload.get("aspect")),
            direction=Direction(payload.get("direction")),
            expected_at=_optional_int(payload, "expected_at"),
            horizon=_optional_int(payload, "horizon") or 0,
            released_by=tuple(str(item) for item in _list(payload, "released_by")),
            type_prior=None if raw_prior is None else TypePrior(raw_prior),
            split_by=tuple(str(item) for item in _list(payload, "split_by")),
            note=_text(payload, "note"),
            source=_source(payload.get("source")),
            created_at=datetime.fromisoformat(_text(payload, "created_at")),
        )
    except (HypothesisError, TypeError, ValueError) as exc:
        raise SceneRecordError(f"hypothesis record is not valid: {exc}") from exc
    if expected_identity is not None and hypothesis.identity != expected_identity:
        raise SceneRecordError("hypothesis record does not carry the identity of its address")
    if encode(hypothesis) != text:
        raise SceneRecordError("hypothesis record body does not match its canonical rendering")
    return hypothesis


def _text(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        raise SceneRecordError(f"hypothesis field {key!r} must be text")
    return value


def _optional_int(payload: dict[str, Any], key: str) -> int | None:
    value = payload.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise SceneRecordError(f"hypothesis field {key!r} must be an integer or null")
    return value


def _list(payload: dict[str, Any], key: str) -> list[Any]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise SceneRecordError(f"hypothesis field {key!r} must be a list")
    return value


def _antecedent(item: object) -> Antecedent:
    if not isinstance(item, dict) or set(item) != {"concept", "grade"} or not isinstance(item["concept"], str):
        raise SceneRecordError("an antecedent must be an object with concept and grade")
    grade = item["grade"]
    if grade is not None and not isinstance(grade, str):
        raise SceneRecordError("an antecedent grade must be text or null")
    return Antecedent(concept=item["concept"], grade=grade)


def _source(item: object) -> HypothesisSource:
    if not isinstance(item, dict):
        raise SceneRecordError("hypothesis source must be an object")
    return HypothesisSource(origin=HypothesisOrigin(item.get("origin")), note=item.get("note"))


__all__ = ["MARKER", "RECORD_TYPE", "decode", "encode"]
