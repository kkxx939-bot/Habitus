"""机会表的 JSON 形状。落盘与读回由组合根做（``runtime/foresight_opportunities.py``），这里只管编解码——
同一晚重算得到同一份字节（规范 JSON），读回时严格校验，不"尽力解析"。"""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import date, datetime
from typing import Any

from habitus.foresight.errors import ForesightError
from habitus.foresight.opportunities.model import (
    ConditionSpec,
    DayWatch,
    KindDays,
    Opportunity,
    OpportunityTable,
    RelationOpportunities,
    TreeReading,
)
from habitus.foundation.integrity import CanonicalSerializationError, canonical_json
from habitus.temporal import DayInterval, OccurrenceDay

#: v4（裁定 59）：日历换成 temporal 的复现（按天的发生 + 相邻间隔的漏看权重与截尾）、去掉 present_days、日历情境关系带键与取值。
#: v5（2026-10-18 四方评审）：表上记这一晚用的放假数据版本（裁定 59 第 4 条）。
#: v6（2026-10-18，S8）：日历关系带"见过的日型"、表上带"没见过的日型怎么办"。
#: v7（26 号清单 3.3）：表上带每天看着的比例与日型（账的久别用）。表每晚整张重写，项目没上线，旧版不迁移。
#: v8（裁定 61 第 6 条）：成立的带条件关系进表，每条关系带它挂的条件（文本 + 白天判它要的类编号 / 日历键）。
OPPORTUNITY_SCHEMA = "habitus.foresight.opportunities/v8"


def encode_table(table: OpportunityTable) -> str:
    """规范 JSON 文本（键有序、无多余空白）。"""

    payload = {
        "schema": OPPORTUNITY_SCHEMA,
        "lane": table.lane,
        "night": table.night.isoformat(),
        "relation_night": table.relation_night.isoformat() if table.relation_night is not None else None,
        "slot_minutes": table.slot_minutes,
        "window_slots": table.window_slots,
        "relations": [_relation(item) for item in table.relations],
        "kinds": [_kind(item) for item in table.kinds],
        "calendar_version": table.calendar_version,
        "unseen_day_types": table.unseen_day_types,
        "watch": [
            {"day": item.day.isoformat(), "share": item.share, "day_type": item.day_type} for item in table.watch
        ],
    }
    try:
        return canonical_json(payload)
    except CanonicalSerializationError as exc:  # pragma: no cover - 上面的值全是标量与列表
        raise ForesightError(f"opportunity table cannot be serialised: {exc}") from exc


def decode_table(text: str, *, lane: str | None = None, night: date | None = None) -> OpportunityTable:
    """读回；给了 ``lane`` / ``night`` 就核对它与文件名一致。"""

    try:
        payload = _parse(text)
        if payload["schema"] != OPPORTUNITY_SCHEMA:
            raise ForesightError(f"unknown opportunity table schema: {payload['schema']!r}")
        table = OpportunityTable(
            lane=str(payload["lane"]),
            night=_date(payload["night"]),
            relation_night=None if payload["relation_night"] is None else _date(payload["relation_night"]),
            slot_minutes=int(payload["slot_minutes"]),
            window_slots=int(payload["window_slots"]),
            relations=tuple(_relation_from(item) for item in payload["relations"]),
            kinds=tuple(_kind_from(item) for item in payload["kinds"]),
            calendar_version=None if payload["calendar_version"] is None else str(payload["calendar_version"]),
            unseen_day_types=str(payload["unseen_day_types"]),
            watch=tuple(_watch_from(item) for item in payload["watch"]),
        )
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        raise ForesightError(f"opportunity table is not valid: {exc}") from exc
    if (lane is not None and table.lane != lane) or (night is not None and table.night != night):
        raise ForesightError("opportunity table does not match its file")
    if encode_table(table) != text:
        raise ForesightError("opportunity table is not in canonical form")
    return table


def _watch_from(raw: Mapping[str, Any]) -> DayWatch:
    day_type = raw["day_type"]
    return DayWatch(
        day=_date(raw["day"]), share=float(raw["share"]), day_type=None if day_type is None else str(day_type)
    )


# ── 关系 ────────────────────────────────────────────────────────────────────


def _relation(item: RelationOpportunities) -> dict[str, Any]:
    return {
        "key": item.key,
        "antecedent": item.antecedent,
        "consequent": item.consequent,
        "antecedent_label": item.antecedent_label,
        "consequent_label": item.consequent_label,
        "antecedent_classes": list(item.antecedent_classes),
        "consequent_classes": list(item.consequent_classes),
        "segment": item.segment,
        "status": item.status,
        "upward": item.upward,
        "items": [_opportunity(chance) for chance in item.items],
        "calendar_key": item.calendar_key,
        "calendar_values": list(item.calendar_values),
        "within_day_types": list(item.within_day_types),
        "conditions": [
            {
                "text": spec.text,
                "classes": list(spec.classes),
                "calendar_key": spec.calendar_key,
                "calendar_values": list(spec.calendar_values),
            }
            for spec in item.conditions
        ],
        "reading": None
        if item.reading is None
        else {
            "treated": item.reading.treated,
            "control": item.reading.control,
            "large_enough": item.reading.large_enough,
        },
    }


def _relation_from(raw: Mapping[str, Any]) -> RelationOpportunities:
    return RelationOpportunities(
        key=str(raw["key"]),
        antecedent=str(raw["antecedent"]),
        consequent=str(raw["consequent"]),
        antecedent_label=str(raw["antecedent_label"]),
        consequent_label=str(raw["consequent_label"]),
        antecedent_classes=tuple(str(token) for token in raw["antecedent_classes"]),
        consequent_classes=tuple(str(token) for token in raw["consequent_classes"]),
        segment=str(raw["segment"]),
        status=str(raw["status"]),
        upward=_flag(raw["upward"]),
        items=tuple(_opportunity_from(chance) for chance in raw["items"]),
        reading=_reading_from(raw["reading"]),
        calendar_key=None if raw["calendar_key"] is None else str(raw["calendar_key"]),
        calendar_values=tuple(str(value) for value in raw["calendar_values"]),
        within_day_types=tuple(str(value) for value in raw["within_day_types"]),
        conditions=tuple(_condition_from(spec) for spec in raw["conditions"]),
    )


def _condition_from(raw: Mapping[str, Any]) -> ConditionSpec:
    return ConditionSpec(
        text=str(raw["text"]),
        classes=tuple(str(token) for token in raw["classes"]),
        calendar_key=None if raw["calendar_key"] is None else str(raw["calendar_key"]),
        calendar_values=tuple(str(value) for value in raw["calendar_values"]),
    )


def _reading_from(raw: object) -> TreeReading | None:
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ForesightError("relation reading must be an object or null")
    enough = raw["large_enough"]
    return TreeReading(
        treated=float(raw["treated"]),
        control=float(raw["control"]),
        large_enough=None if enough is None else _flag(enough),
    )


def _opportunity(item: Opportunity) -> dict[str, Any]:
    return {
        "antecedent_uri": item.antecedent_uri,
        "antecedent_uris": list(item.antecedent_uris),
        "day": item.day.isoformat(),
        "ended_at": item.ended_at.isoformat(),
        "came": item.came,
        "consequent_uri": item.consequent_uri,
        "consequent_at": item.consequent_at.isoformat() if item.consequent_at is not None else None,
    }


def _opportunity_from(raw: Mapping[str, Any]) -> Opportunity:
    came = raw["came"]
    if came is not None and not isinstance(came, bool):
        raise ForesightError("opportunity came must be a boolean or null")
    return Opportunity(
        antecedent_uri=str(raw["antecedent_uri"]),
        antecedent_uris=tuple(str(uri) for uri in raw["antecedent_uris"]),
        day=_date(raw["day"]),
        ended_at=_moment(raw["ended_at"]),
        came=came,
        consequent_uri=None if raw["consequent_uri"] is None else str(raw["consequent_uri"]),
        consequent_at=None if raw["consequent_at"] is None else _moment(raw["consequent_at"]),
    )


# ── 日历 ────────────────────────────────────────────────────────────────────


def _kind(item: KindDays) -> dict[str, Any]:
    return {
        "kind_token": item.kind_token,
        "label": item.label,
        "days": [
            {
                "day": day.day.isoformat(),
                "first_uri": day.first_uri,
                "first_at": day.first_at.isoformat(),
                "count": day.count,
                "natural": day.natural,
                "natural_uri": day.natural_uri,
                "natural_at": None if day.natural_at is None else day.natural_at.isoformat(),
            }
            for day in item.days
        ],
        # 第 i 段在第 i 天与第 i+1 天之间：只存漏看的权重与是否截尾，天数由两头的日子算
        "intervals": [{"seen": interval.seen, "censored": interval.censored} for interval in item.intervals],
    }


def _kind_from(raw: Mapping[str, Any]) -> KindDays:
    days = tuple(
        OccurrenceDay(
            day=_date(day["day"]),
            first_uri=str(day["first_uri"]),
            first_at=_moment(day["first_at"]),
            count=int(day["count"]),
            natural=int(day["natural"]),
            natural_uri=None if day["natural_uri"] is None else str(day["natural_uri"]),
            natural_at=None if day["natural_at"] is None else _moment(day["natural_at"]),
        )
        for day in raw["days"]
    )
    intervals = tuple(
        DayInterval(
            previous=days[index],
            following=days[index + 1],
            days=(days[index + 1].day - days[index].day).days,
            seen=float(interval["seen"]),
            censored=_flag(interval["censored"]),
        )
        for index, interval in enumerate(raw["intervals"])
    )
    return KindDays(kind_token=str(raw["kind_token"]), label=str(raw["label"]), days=days, intervals=intervals)


# ── 标量 ────────────────────────────────────────────────────────────────────


def _parse(text: str) -> dict[str, Any]:
    try:
        payload = json.loads(text)
    except ValueError as exc:
        raise ForesightError(f"opportunity table is not JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ForesightError("opportunity table must be an object")
    return payload


def _date(raw: object) -> date:
    if not isinstance(raw, str):
        raise ForesightError("a date must be text")
    return date.fromisoformat(raw)


def _moment(raw: object) -> datetime:
    if not isinstance(raw, str):
        raise ForesightError("a timestamp must be text")
    parsed = datetime.fromisoformat(raw)
    if parsed.utcoffset() is None:
        raise ForesightError("a timestamp must carry its offset")
    return parsed


def _flag(raw: object) -> bool:
    if not isinstance(raw, bool):
        raise ForesightError("a flag must be a boolean")
    return raw


__all__ = ["OPPORTUNITY_SCHEMA", "decode_table", "encode_table"]
