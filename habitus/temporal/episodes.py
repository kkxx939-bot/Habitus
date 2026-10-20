"""一类的"一次一次"与按天的发生（裁定 59 第 6 条第 1 步）。

``episodes_of``：同一类的记录按开始时刻扎堆合并，一堆就是一次；这一次是不是自然发生，看它的第一条开始之前有没有被提醒
（裁定 54：被提醒才做的不算他自己的规律；同一次里接着做的几条跟着第一条走）。
``occurrence_days``：按本地日期归天——这一天第一次的记录与时刻、当天几次、其中几次自然发生、第一次自然发生的记录与时刻。
"离上次多久"按全部发生算（他确实做了），命中只数自然发生的那几天，时窗只取自然发生的时刻（提醒的时刻不许漏回"往常几点做"）。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime

from habitus.series.model import SeriesRecord
from habitus.temporal.merge import merge_starts


@dataclass(frozen=True)
class Episode:
    """合并后的一次：开始（第一条的开始）、最后所见（几条里最晚的）、几条记录的 URI、是不是自然发生。"""

    started_at: datetime
    last_seen: datetime
    uris: tuple[str, ...]
    natural: bool

    def __post_init__(self) -> None:
        if not self.uris:
            raise ValueError("an episode holds at least one record")
        if self.last_seen < self.started_at:
            raise ValueError("an episode cannot end before it starts")

    @property
    def day(self) -> date:
        return self.started_at.date()

    @property
    def first_uri(self) -> str:
        return self.uris[0]


@dataclass(frozen=True)
class OccurrenceDay:
    """一类在某一天：第一次的记录与时刻、当天几次、其中几次自然发生、第一次自然发生的记录与时刻（没有自然的是 None）。"""

    day: date
    first_uri: str
    first_at: datetime
    count: int
    natural: int
    natural_uri: str | None = None
    natural_at: datetime | None = None

    def __post_init__(self) -> None:
        if self.count < 1 or not 0 <= self.natural <= self.count:
            raise ValueError("an occurrence day has at least one occurrence and no more natural ones than that")
        if (self.natural == 0) != (self.natural_uri is None) or (self.natural_uri is None) != (self.natural_at is None):
            raise ValueError("natural_uri / natural_at are given exactly when the day has a natural occurrence")


def episodes_of(records: Sequence[SeriesRecord], *, window_seconds: int) -> tuple[Episode, ...]:
    """一组发生（一个类，或汇总概念的几个成员类合在一起）、按开始时刻升序的记录 → 一次一次。没排好序是调用方的错，硬拒。"""

    ordered = tuple(records)
    groups = merge_starts([record.started_at for record in ordered], window_seconds=window_seconds)
    found = []
    for group in groups:
        members = [ordered[index] for index in group]
        found.append(
            Episode(
                started_at=members[0].started_at,
                last_seen=max(member.last_observed_at for member in members),
                uris=tuple(member.uri for member in members),
                natural=not members[0].reminded,
            )
        )
    return tuple(found)


def occurrence_days(episodes: Sequence[Episode]) -> tuple[OccurrenceDay, ...]:
    by_day: dict[date, list[Episode]] = {}
    for item in episodes:
        by_day.setdefault(item.day, []).append(item)
    found = []
    for day in sorted(by_day):
        items = by_day[day]
        natural = [item for item in items if item.natural]
        found.append(
            OccurrenceDay(
                day=day,
                first_uri=items[0].first_uri,
                first_at=items[0].started_at,
                count=len(items),
                natural=len(natural),
                natural_uri=natural[0].first_uri if natural else None,
                natural_at=natural[0].started_at if natural else None,
            )
        )
    return tuple(found)


__all__ = ["Episode", "OccurrenceDay", "episodes_of", "occurrence_days"]
