"""命中记录的落盘格式：正文给人读，尾部规范 JSON 是机器读的唯一来源；回读逐字节比对。

字段集合是严格的：多出来、少了的键都拒收（旧盘面上带细分命中、未决、常态快照、材料摘要这些键的命中记录读到明确报错，
不静默忽略、不自动转换）。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from habitus.scene.codec import SceneRecordError, decode_record, encode_record
from habitus.scene.occurrences.model import ConceptHit, ConceptHits, ConceptHitsError

MARKER = "HABITUS_CONCEPT_HITS"
RECORD_TYPE = "concept_hits"
_KEYS = frozenset(
    {
        "record_type",
        "occurrence_uri",
        "kind_token",
        "classified",
        "last_observed_at",
        "situation_hits",
        "situations_checked",
        "lane",
        "hits_version",
        "sealed_at",
    }
)


def encode(record: ConceptHits) -> str:
    address = record.address
    held = " · ".join(hit.concept for hit in record.situation_hits) if record.situation_hits else "（无）"
    checked = "、".join(record.situations_checked) if record.situations_checked else "无"
    lines = [
        f"# {address.name} · {address.started_at.isoformat(timespec='minutes')}",
        "",
        # 编号不进正文（与行为树同口径，裁定 19）：类改名不该让盘上记录与规范渲染对不上。
        f"{'已归类' if record.classified else '待定'} · 最后所见 {record.last_observed_at.strftime('%H:%M')}",
        f"情境：{held}（判过 {checked}）",
        f"口径：{record.hits_version}",
    ]
    payload: dict[str, Any] = {
        "record_type": RECORD_TYPE,
        "occurrence_uri": record.occurrence_uri,
        "kind_token": record.kind_token,
        "classified": record.classified,
        "last_observed_at": record.last_observed_at.isoformat(timespec="microseconds"),
        "situation_hits": [hit.concept for hit in record.situation_hits],
        "situations_checked": list(record.situations_checked),
        "lane": record.lane,
        "hits_version": record.hits_version,
        "sealed_at": record.sealed_at,
    }
    return encode_record("\n".join(lines) + "\n", MARKER, payload)


def decode(text: str, *, expected_uri: str | None = None) -> ConceptHits:
    _body, payload = decode_record(text, MARKER)
    if payload.get("record_type") != RECORD_TYPE:
        raise SceneRecordError("concept hits record has the wrong record_type")
    if set(payload) != _KEYS:
        raise SceneRecordError(
            f"concept hits record has unexpected or missing fields: extra {sorted(set(payload) - _KEYS)}, "
            f"missing {sorted(_KEYS - set(payload))}"
        )
    try:
        record = ConceptHits(
            occurrence_uri=_text(payload, "occurrence_uri"),
            kind_token=_text(payload, "kind_token"),
            classified=_flag(payload, "classified"),
            last_observed_at=datetime.fromisoformat(_text(payload, "last_observed_at")),
            situation_hits=tuple(ConceptHit(_name(item)) for item in _list(payload, "situation_hits")),
            situations_checked=tuple(_name(item) for item in _list(payload, "situations_checked")),
            lane=_text(payload, "lane"),
            hits_version=_text(payload, "hits_version"),
            sealed_at=datetime.fromisoformat(_text(payload, "sealed_at")),
        )
    except (ConceptHitsError, TypeError, ValueError) as exc:
        raise SceneRecordError(f"concept hits record is not valid: {exc}") from exc
    if expected_uri is not None and record.occurrence_uri != expected_uri:
        raise SceneRecordError("concept hits record does not point at the occurrence of its address")
    if encode(record) != text:
        raise SceneRecordError("concept hits record body does not match its canonical rendering")
    return record


def _text(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        raise SceneRecordError(f"concept hits field {key!r} must be text")
    return value


def _name(item: object) -> str:
    if not isinstance(item, str):
        raise SceneRecordError("concept hits list items must be text")
    return item


def _flag(payload: dict[str, Any], key: str) -> bool:
    value = payload.get(key)
    if not isinstance(value, bool):
        raise SceneRecordError(f"concept hits field {key!r} must be a boolean")
    return value


def _list(payload: dict[str, Any], key: str) -> list[Any]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise SceneRecordError(f"concept hits field {key!r} must be a list")
    return value


__all__ = ["MARKER", "RECORD_TYPE", "decode", "encode"]
