"""概念命中记录的落盘格式：正文给人读，尾部规范 JSON 是机器读的唯一来源；回读逐字节比对。"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from habitus.scene.codec import SceneRecordError, decode_record, encode_record
from habitus.scene.occurrences.model import ConceptHit, ConceptHits, ConceptHitsError

MARKER = "HABITUS_CONCEPT_HITS"
RECORD_TYPE = "concept_hits"


def encode(record: ConceptHits) -> str:
    if not isinstance(record, ConceptHits):
        raise TypeError("record must be ConceptHits")
    address = record.address
    lines = [
        f"# {address.name} · {address.started_at.isoformat(timespec='minutes')}",
        "",
        f"类别：{record.kind_token} · 最后所见 {record.last_observed_at.strftime('%H:%M')}",
        f"命中：{_render_hits(record.hits)}",
        f"情境：{_render_hits(record.situation_hits)}（判过 {'、'.join(record.situations_checked) if record.situations_checked else '无'}）",
        f"未决：{' · '.join(record.unresolved) if record.unresolved else '（无）'}",
    ]
    if record.baseline_snapshot:
        lines.append("常态：" + " · ".join(f"{key} = {value}" for key, value in record.baseline_snapshot.items()))
    lines.append(f"口径：{record.mapper}")
    if record.signals:
        lines.append("信号：" + " ｜ ".join(record.signals))
    payload: dict[str, Any] = {
        "record_type": RECORD_TYPE,
        "occurrence_uri": record.occurrence_uri,
        "kind_token": record.kind_token,
        "last_observed_at": record.last_observed_at.isoformat(timespec="microseconds"),
        "hits": [{"concept": hit.concept, "grade": hit.grade} for hit in record.hits],
        "situation_hits": [{"concept": hit.concept, "grade": hit.grade} for hit in record.situation_hits],
        "situations_checked": list(record.situations_checked),
        "unresolved": list(record.unresolved),
        "baseline_snapshot": dict(record.baseline_snapshot),
        "mapper": record.mapper,
        "mapped_at": record.mapped_at,
        "signals": list(record.signals),
    }
    return encode_record("\n".join(lines) + "\n", MARKER, payload)


def decode(text: str, *, expected_uri: str | None = None) -> ConceptHits:
    _body, payload = decode_record(text, MARKER)
    if payload.get("record_type") != RECORD_TYPE:
        raise SceneRecordError("concept hits record has the wrong record_type")
    try:
        record = ConceptHits(
            occurrence_uri=_text(payload, "occurrence_uri"),
            kind_token=_text(payload, "kind_token"),
            last_observed_at=datetime.fromisoformat(_text(payload, "last_observed_at")),
            hits=tuple(_hit(item) for item in _list(payload, "hits")),
            situation_hits=tuple(_hit(item) for item in _list(payload, "situation_hits")),
            situations_checked=tuple(str(item) for item in _list(payload, "situations_checked")),
            unresolved=tuple(str(item) for item in _list(payload, "unresolved")),
            baseline_snapshot=_mapping(payload, "baseline_snapshot"),
            mapper=_text(payload, "mapper"),
            mapped_at=datetime.fromisoformat(_text(payload, "mapped_at")),
            signals=tuple(str(item) for item in _list(payload, "signals")),
        )
    except (ConceptHitsError, TypeError, ValueError) as exc:
        raise SceneRecordError(f"concept hits record is not valid: {exc}") from exc
    if expected_uri is not None and record.occurrence_uri != expected_uri:
        raise SceneRecordError("concept hits record does not point at the occurrence of its address")
    if encode(record) != text:
        raise SceneRecordError("concept hits record body does not match its canonical rendering")
    return record


def _render_hits(hits: tuple[ConceptHit, ...]) -> str:
    if not hits:
        return "（无）"
    return " · ".join(hit.concept if hit.grade is None else f"{hit.concept}（{hit.grade}）" for hit in hits)


def _text(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        raise SceneRecordError(f"concept hits field {key!r} must be text")
    return value


def _list(payload: dict[str, Any], key: str) -> list[Any]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise SceneRecordError(f"concept hits field {key!r} must be a list")
    return value


def _mapping(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise SceneRecordError(f"concept hits field {key!r} must be an object")
    return value


def _hit(item: object) -> ConceptHit:
    if not isinstance(item, dict) or set(item) != {"concept", "grade"}:
        raise SceneRecordError("a concept hit must be an object with concept and grade")
    return ConceptHit(concept=item["concept"], grade=item["grade"])


__all__ = ["MARKER", "RECORD_TYPE", "decode", "encode"]
