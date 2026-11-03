"""「离上次够久」：从机会表里这个类的复现读（语义树登记、``temporal`` 算，裁定 59），不按周几、不按几号。

离上次 e 天：历史上恰好隔过 e 天、且不短于 e 的在险间隔 ≥2 个，才有读数（``temporal.cluster_at``）；从 e 起相差 1 天连着的
那一簇间隔是这次"该来的那几天"，时窗落在这几天里（取历史上落在这一簇里那几次自然发生的时刻范围）。命中只数自然发生的那几天
（被提醒才做的不是他自己的规律，裁定 54）；以被提醒那次结尾的间隔是截尾，不算命中、也从机会里拿掉。触发实例是
``<上次那条的 URI>@<这段时窗的第一天>``，第一天是盖住 e 的那段时窗的起点（``opened``），同一段时窗哪天重算都是同一个身份、
全账本只建一条。

每周的事跳一周照样接得住：历史上隔过 14 天，离上次 14 天时那一簇就有读数。锚在周三、每月 20 号这类日历上的规律不归这里，
归日历情境关系（``triggers.situations``）。**今天已做的照建**（裁定 49）；上一次看的时候（``Moment.caught_from``）已经过了
时窗而今天还没做的不建（退场）——夜里那条线从零点起看（裁定 94 第 2 条），所以当天的日子预期不会退场。
时窗只取这一簇里最近 M 次命中的时刻（证据的窗口，裁定 57）：在险间隔照旧数全历史，"几点做"跟着最近的习惯走。
簇往后接几天看账的线 θ（``line``，裁定 89 第 1 条）：到离上次 e 天为止他自己来了的在险间隔占比已超过 θ，时窗就只在这一天——
天天做的事偶尔漏过一天，以后单独漏的那天照常到时窗末提醒，不会因为历史上隔过一次 2 天就把时窗撑到次日。簇被线截成先后几段时，
身份与时窗起点取盖住今天的那一段的第一天（``ClusterReading.opened``）：漏过几次的天天做的事连漏第 2 天是新的一条。

**连漏、超过了历史**（裁定 90，``temporal.overdue_at``）：离上次已比历史上（够得上读数的）间隔都久、常规读法读不出时，按常见间隔
继续——天天做的每天一条，每周一次的每 7 天一条，每条只管那一天、各是新的身份，照常递送与结算；什么时候不再开口交给账的
"规律断了"与提醒反馈，这里不定第几天停。

**一天只算一次、只用含昨天的那张表**（裁定 94 第 1 条）：日子预期是夜里那条线的最后一步（前一天定稿 → 重试 → 语义树 →
机会表 → 单元 → 当天的日子预期，``runtime.foresight_days``），用的就是这条线刚写的、含昨天的那张表——间隔、簇、命中、时窗都
从这一张读，只有这一个来源；白天那一拍不建日子预期。每天都建、从当天零点起算（``observed_from``，裁定 94 第 2 条）：零点到这条线
跑完之间做过的照建、照算；时窗整个落在这段里而没做的也照建，结过期待决（裁定 49 第 1 条：不许只建做了的一侧）。
"""

from __future__ import annotations

from datetime import date, timedelta

from habitus.foresight.model import Moment
from habitus.foresight.opportunities.model import KindDays, OpportunityTable
from habitus.foresight.triggers.draft import Draft, at_time_of, clamp, midnight_after, midnight_of
from habitus.temporal import cluster_at, overdue_at

INTERVAL = "interval"


def day_drafts(
    table: OpportunityTable,
    moment: Moment,
    *,
    done_today: frozenset[str],
    recent: int,
    line: float,
) -> tuple[Draft, ...]:
    found: list[Draft] = []
    for kind in table.kinds:
        draft = _interval(
            table,
            kind,
            moment,
            done_now=kind.kind_token in done_today,
            recent=recent,
            line=line,
        )
        if draft is not None:
            found.append(draft)
    return tuple(found)


def _interval(
    table: OpportunityTable,
    kind: KindDays,
    moment: Moment,
    *,
    done_now: bool,
    recent: int,
    line: float,
) -> Draft | None:
    days = [item for item in kind.days if item.day < moment.day]
    if len(days) < 2:
        return None
    intervals = [item for item in kind.intervals if item.following.day < moment.day]
    last_day, last_uri = days[-1].day, days[-1].first_uri
    elapsed = (moment.day - last_day).days
    reading = cluster_at(intervals, elapsed, line=line) or overdue_at(intervals, elapsed, line=line)
    if reading is None:
        return None
    first_day = last_day + timedelta(days=reading.opened)
    end_day = last_day + timedelta(days=reading.cluster[-1])
    basis = sorted((item.following for item in reading.hits), key=lambda day: day.day)
    now = moment.at
    # 时窗只看最近 M 次命中（裁定 57）：一年前从 19:00 改到 07:00 的跑步，时窗末不再拖在 19:00
    times = [item.natural_at for item in basis if item.natural_at is not None][-recent:]
    if not times:
        return None
    start = min(at_time_of(first_day, moment_, now) for moment_ in times)
    end = max(at_time_of(end_day, moment_, now) for moment_ in times)
    clamped = clamp(
        start,
        end,
        midnight_after(end_day, now),
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
        trigger_instance=f"{last_uri}@{first_day.isoformat()}",
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
