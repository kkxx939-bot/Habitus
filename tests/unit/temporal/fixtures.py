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


def with_sessions(records: list[SeriesRecord], *, cutoff: date) -> list[SeriesRecord]:
    """``records`` 加上从最早那天起到截止日前每天 06:00 一条叫不出名的「待定」：每天都开过会话。会话 lane 一整天没有会话算
    没在看（裁定 89 第 2 条），要"在看、只是没做"的日子就得有它；「待定」不是类，不进任何一类的发生与间隔。"""

    if not records:
        return list(records)
    day, found = min(item.day for item in records), list(records)
    while day < cutoff:
        moment = at(day, 6)
        found.append(
            SeriesRecord(
                uri=f"behavior://s-待定/{moment.isoformat()}/session",
                name="会话",
                day=day,
                started_at=moment,
                last_observed_at=moment + timedelta(minutes=5),
                kind_token="s-待定",
                lane=LANE,
                classified=False,
                reminded=False,
                summary="会话",
            )
        )
        day += timedelta(days=1)
    return found


def gap(start: datetime, end: datetime, kind: GapKind = GapKind.UNOBSERVED) -> SeriesGap:
    return SeriesGap(uri=f"behavior://gap/{start.isoformat()}", started_at=start, ended_at=end, kind=kind)


def series(records: list[SeriesRecord], *, cutoff: date, gaps: list[SeriesGap] | None = None) -> EventSeries:
    ordered = sorted(records, key=lambda item: (item.started_at, item.uri))
    return EventSeries(
        cutoff=cutoff, records=tuple(ordered), gaps=tuple(sorted(gaps or [], key=lambda g: (g.started_at, g.uri)))
    )
