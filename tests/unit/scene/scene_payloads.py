"""情景树测试共用的规范载荷：一件"准备晚饭"，成员指向行为树的 occurrence URI。"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Any

from habitus.behavior.model import BehaviorAddress
from habitus.behavior.uri import BehaviorURI

CST = timezone(timedelta(hours=8))
DAY = date(2026, 8, 16)
DIGEST = "f" * 64


def local(hour: int, minute: int, second: int = 0, *, day: date = DAY) -> datetime:
    return datetime(day.year, day.month, day.day, hour, minute, second, tzinfo=CST)


def occurrence_uri(name: str, started_at: datetime) -> str:
    return str(BehaviorURI.from_address(BehaviorAddress.occurrence(started_at.date(), name, started_at)))


BUY_GROCERIES = occurrence_uri("去超市买菜", local(15, 20))
DISCUSS_DINNER = occurrence_uri("商量晚餐", local(19, 12))
LOOKUP_RECIPE = occurrence_uri("查配方", local(19, 41))
WASH_VEGETABLES = occurrence_uri("洗菜", local(19, 43))
CHECK_PHONE = occurrence_uri("看手机", local(20, 15))


def scene_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "occurred_on": DAY,
        "label": "准备晚饭",
        "started_at": local(19, 12),
        "ended_at": local(20, 30),
        "members": (
            {"uri": DISCUSS_DINNER, "role": "essential"},
            {"uri": LOOKUP_RECIPE, "role": "essential"},
            {"uri": WASH_VEGETABLES, "role": "essential"},
            {"uri": CHECK_PHONE, "role": "irrelevant"},
        ),
        "effects": ("晚饭做好了",),
        "pending_effects": ({"text": "查到了今晚做汤的配方", "producer_uri": LOOKUP_RECIPE},),
        "source_digest": DIGEST,
        "scene_version": "scene_grouping_v1+schema0000",
    }
    payload.update(overrides)
    return payload


def shopping_payload(**overrides: Any) -> dict[str, Any]:
    payload = scene_payload(
        label="去超市采购",
        started_at=local(15, 20),
        ended_at=local(16, 5),
        members=({"uri": BUY_GROCERIES, "role": "essential"},),
        effects=("买到了晚饭食材",),
        pending_effects=({"text": "家里有今晚要用的食材", "producer_uri": BUY_GROCERIES},),
    )
    payload.update(overrides)
    return payload
