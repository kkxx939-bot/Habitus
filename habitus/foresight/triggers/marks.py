"""把最近几天的行排成一条只有记号的时间线，好让链合并与语义树同一把尺（``scene.relations.chains_of`` / ``merge_for``）。

只填 ``marks``（概念 = 类编号）；人在场、对照时刻、情境这些检验用的字段不填——这里不做检验，只问"此刻前因在不在它那一段里"。
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date, datetime

from habitus.foresight.model import Seen
from habitus.scene.relations import LaneTimeline, Mark
from habitus.scene.views import slot_index

MINUTES_PER_DAY = 1440


def absolute_slot(at: datetime, *, slot_minutes: int) -> int:
    """跨天可比的绝对槽号：日序号 × 每天槽数 + 当天第几槽（语义树时间线的写法）。"""

    return at.date().toordinal() * (MINUTES_PER_DAY // slot_minutes) + slot_index(at, slot_minutes=slot_minutes)


def timeline_of(rows: Iterable[Seen], *, lane: str, slot_minutes: int, horizon: date) -> LaneTimeline:
    """``rows`` 里每一件事按类挂成记号：开始所在槽、最后所见所在槽（过了午夜截在当天最后一槽）。"""

    per_day = MINUTES_PER_DAY // slot_minutes
    line = LaneTimeline(lane=lane, slots_per_day=per_day, horizon=horizon.toordinal() * per_day)
    for row in rows:
        start = absolute_slot(row.started_at, slot_minutes=slot_minutes)
        if row.ended_at.date() == row.day:
            end = absolute_slot(row.ended_at, slot_minutes=slot_minutes)
        else:
            end = row.day.toordinal() * per_day + per_day - 1
        line.marks.setdefault(row.kind_token, []).append(
            Mark(slot=start, started_at=row.started_at, uri=row.source, last_slot=max(start, end))
        )
    for items in line.marks.values():
        items.sort(key=lambda item: (item.slot, item.started_at, item.uri))
    return line


__all__ = ["MINUTES_PER_DAY", "absolute_slot", "timeline_of"]
