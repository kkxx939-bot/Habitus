"""temporal 测试共用的造数：一条 lane 上某一类的记录，时刻都带 +08:00。"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

from habitus.series.model import EventSeries, GapKind, SeriesGap, SeriesRecord

CST = timezone(timedelta(hours=8))
LANE = "session"


def at(day: date, hour: int, minute: int = 0) -> datetime:
    return datetime(day.year, day.month, day.day, hour, minute, tzinfo=CST)


def record(
    moment: datetime, kind: str = "s-k0001", *, reminded: bool = False, minutes: int = 10, number: int = 0
) -> SeriesRecord:
    return SeriesRecord(
        uri=f"behavior://{kind}/{moment.isoformat()}/{number}",
        name=kind,
        day=moment.date(),
        started_at=moment,
        last_observed_at=moment + timedelta(minutes=minutes),
        kind_token=kind,
        lane=LANE,
        classified=True,
        reminded=reminded,
        summary=kind,
    )


def gap(start: datetime, end: datetime, kind: GapKind = GapKind.UNOBSERVED) -> SeriesGap:
    return SeriesGap(uri=f"behavior://gap/{start.isoformat()}", started_at=start, ended_at=end, kind=kind)


def series(records: list[SeriesRecord], *, cutoff: date, gaps: list[SeriesGap] | None = None) -> EventSeries:
    ordered = sorted(records, key=lambda item: (item.started_at, item.uri))
    return EventSeries(
        cutoff=cutoff, records=tuple(ordered), gaps=tuple(sorted(gaps or [], key=lambda g: (g.started_at, g.uri)))
    )
