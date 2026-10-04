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
    PeakWindow,
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
#: 每个来源都要有说法——缺一个就是写盘时 KeyError（2026-09-29 加闭环那两个来源时踩到）。
#: 用 dict 取值而不是 `.get(…, 默认)`：来源是受控枚举，漏了要当场炸，不该悄悄印成"未知"。
_ORIGIN_LABELS = {
    HypothesisOrigin.BASELINE: "基准",
    HypothesisOrigin.NEW_CONCEPT: "为新概念补写",
    HypothesisOrigin.MODERATION: "闭环（触点③ 读了两层不一样之后提的；从写入日起攒账、不回填）",
    HypothesisOrigin.PLACEBO: "安慰剂（前件换成无关行为，量误报率用；不是给人看的读数）",
}


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
        f"落在：{hypothesis.opportunity_label}",
    ]
    for name, items in sorted(hypothesis.windows.items()):
        lines.append(f"峰表 {name}：" + " · ".join(f"#{item.ordinal} {item.label()}" for item in items))
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
        "antecedents": [{"concept": item.concept, "grade": item.grade, "peak": item.peak} for item in hypothesis.antecedents],
        "consequent": hypothesis.consequent,
        "aspect": hypothesis.aspect.value,
        "direction": hypothesis.direction.value,
        "consequent_peak": hypothesis.consequent_peak,
        "windows": {
            name: [{"ordinal": item.ordinal, "start_minute": item.start_minute, "end_minute": item.end_minute} for item in items]
            for name, items in sorted(hypothesis.windows.items())
        },
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
            consequent_peak=_optional_int(payload, "consequent_peak"),
            windows=_windows(payload.get("windows")),
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
    if not isinstance(item, dict) or set(item) != {"concept", "grade", "peak"} or not isinstance(item["concept"], str):
        raise SceneRecordError("an antecedent must be an object with concept, grade and peak")
    grade, peak = item["grade"], item["peak"]
    if grade is not None and not isinstance(grade, str):
        raise SceneRecordError("an antecedent grade must be text or null")
    if peak is not None and (isinstance(peak, bool) or not isinstance(peak, int)):
        raise SceneRecordError("an antecedent peak must be an integer or null")
    return Antecedent(concept=item["concept"], grade=grade, peak=peak)


def _windows(value: object) -> dict[str, tuple[PeakWindow, ...]]:
    if not isinstance(value, dict):
        raise SceneRecordError("hypothesis field 'windows' must be an object")
    table: dict[str, tuple[PeakWindow, ...]] = {}
    for name, items in value.items():
        if not isinstance(name, str) or not isinstance(items, list):
            raise SceneRecordError("hypothesis windows map concept names to lists")
        windows = []
        for item in items:
            if not isinstance(item, dict) or set(item) != {"ordinal", "start_minute", "end_minute"}:
                raise SceneRecordError("a peak window carries ordinal, start_minute and end_minute")
            if any(isinstance(item[key], bool) or not isinstance(item[key], int) for key in ("ordinal", "start_minute", "end_minute")):
                raise SceneRecordError("peak window fields must be integers")
            windows.append(PeakWindow(item["ordinal"], item["start_minute"], item["end_minute"]))
        table[name] = tuple(windows)
    return table


def _source(item: object) -> HypothesisSource:
    if not isinstance(item, dict):
        raise SceneRecordError("hypothesis source must be an object")
    return HypothesisSource(origin=HypothesisOrigin(item.get("origin")), note=item.get("note"))


__all__ = ["MARKER", "RECORD_TYPE", "decode", "encode"]
