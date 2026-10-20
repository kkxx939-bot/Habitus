"""「离上次够久」：从机会表里这个类的复现读（语义树登记、``temporal`` 算，裁定 59），不按周几、不按几号。

离上次 e 天：历史上恰好隔过 e 天、且不短于 e 的在险间隔 ≥2 个，才有读数（``temporal.cluster_at``）；从 e 起相差 1 天连着的
那一簇间隔是这次"该来的那几天"，时窗落在这几天里（取历史上落在这一簇里那几次自然发生的时刻范围）。命中只数自然发生的那几天
（被提醒才做的不是他自己的规律，裁定 54）；以被提醒那次结尾的间隔是截尾，不算命中、也从机会里拿掉。触发实例是
``<上次那条的 URI>@<这一簇的第一天>``，第一天按含 e 的整簇算，同一簇哪天重算都是同一个身份、全账本只建一条。

每周的事跳一周照样接得住：历史上隔过 14 天，离上次 14 天时那一簇就有读数。锚在周三、每月 20 号这类日历上的规律不归这里，
归日历情境关系（``triggers.situations``）。**今天已做的照建**（裁定 49）；此刻已经过了时窗而今天还没做的不建（退场）。
时窗只取这一簇里最近 M 次命中的时刻（证据的窗口，裁定 57）：在险间隔照旧数全历史，"几点做"跟着最近的习惯走。
"""

from __future__ import annotations

from datetime import date, timedelta

from habitus.foresight.model import Moment
from habitus.foresight.opportunities.model import KindDays, OpportunityTable
from habitus.foresight.triggers.draft import Draft, at_time_of, clamp, midnight_after, midnight_of
from habitus.temporal import cluster_at

INTERVAL = "interval"


def day_drafts(
    table: OpportunityTable, moment: Moment, *, done_today: frozenset[str], recent: int
) -> tuple[Draft, ...]:
    found: list[Draft] = []
    for kind in table.kinds:
        draft = _interval(table, kind, moment, done_now=kind.kind_token in done_today, recent=recent)
        if draft is not None:
            found.append(draft)
    return tuple(found)


def _interval(table: OpportunityTable, kind: KindDays, moment: Moment, *, done_now: bool, recent: int) -> Draft | None:
    days = [item for item in kind.days if item.day < moment.day]
    if len(days) < 2:
        return None
    intervals = [item for item in kind.intervals if item.following.day < moment.day]
    last = days[-1]
    reading = cluster_at(intervals, (moment.day - last.day).days)
    if reading is None:
        return None
    first_day = last.day + timedelta(days=reading.block[0])
    last_day = last.day + timedelta(days=reading.cluster[-1])
    basis = sorted((item.following for item in reading.hits), key=lambda day: day.day)
    now = moment.at
    # 时窗只看最近 M 次命中（裁定 57）：一年前从 19:00 改到 07:00 的跑步，时窗末不再拖在 19:00
    times = [item.natural_at for item in basis if item.natural_at is not None][-recent:]
    if not times:
        return None
    start = min(at_time_of(first_day, moment_, now) for moment_ in times)
    end = max(at_time_of(last_day, moment_, now) for moment_ in times)
    clamped = clamp(
        start,
        end,
        midnight_after(last_day, now),
        now=now,
        slot_minutes=table.slot_minutes,
        keep=done_now,
        caught_from=moment.caught_from,
    )
    if clamped is None:
        return None
    window, deadline = clamped
    return Draft(
        kind_token=kind.kind_token,
        source="日子",
        trigger=INTERVAL,
        trigger_instance=f"{last.first_uri}@{first_day.isoformat()}",
        window=window,
        deadline=deadline,
        basis=tuple(item.natural_uri for item in basis if item.natural_uri is not None),
        opportunities=reading.opportunities,
        hits=len(basis),
        observed_from=midnight_of(moment.day, now),
    )


def first_window_day(trigger_instance: str) -> date | None:
    """「离上次够久」实例里这一簇的第一天；别的触发实例是 None。"""

    _uri, separator, tail = trigger_instance.rpartition("@")
    if not separator:
        return None
    try:
        return date.fromisoformat(tail)
    except ValueError:
        return None


__all__ = ["INTERVAL", "day_drafts", "first_window_day"]
