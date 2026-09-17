"""钟面算术的唯一出处（scene 这一侧）：槽 = 本地时分 // ``slot_minutes``，与预测树 ``SlotKey.of`` 同一公式。

``views`` 不能 import prediction（架构边界），所以公式在这里再写一遍——但只写这一遍：投影的邻域过滤、
邻域序列的窗口起止都从这里取，两处各写一份的那天，环形距离与墙钟窗口就会静默分叉。
"""

from __future__ import annotations

from datetime import datetime

MINUTES_PER_DAY = 1440


def require_aware(at: datetime, label: str = "at") -> datetime:
    if not isinstance(at, datetime) or at.utcoffset() is None:
        raise ValueError(f"{label} must be a timezone-aware datetime")
    return at


def require_slot_minutes(slot_minutes: int) -> int:
    if isinstance(slot_minutes, bool) or not isinstance(slot_minutes, int) or slot_minutes <= 0:
        raise ValueError("slot_minutes must be a positive divisor of 1440")
    if MINUTES_PER_DAY % slot_minutes:
        raise ValueError("slot_minutes must be a positive divisor of 1440")
    return slot_minutes


def slots_per_day(slot_minutes: int) -> int:
    return MINUTES_PER_DAY // require_slot_minutes(slot_minutes)


def slot_index(at: datetime, *, slot_minutes: int) -> int:
    """``at`` 落在钟面的第几槽，按它自己的本地时分算。"""

    require_aware(at)
    return (at.hour * 60 + at.minute) // require_slot_minutes(slot_minutes)


def slot_floor(at: datetime, *, slot_minutes: int) -> datetime:
    """``at`` 所在槽的墙钟起点，保留它自己的时区。"""

    start_minute = slot_index(at, slot_minutes=slot_minutes) * slot_minutes
    return at.replace(hour=start_minute // 60, minute=start_minute % 60, second=0, microsecond=0)


__all__ = ["MINUTES_PER_DAY", "require_aware", "require_slot_minutes", "slot_floor", "slot_index", "slots_per_day"]
