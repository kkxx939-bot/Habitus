"""「到了日子 / 离上次够久」：从机会表的日历（每个类发生过的日子）读，不用预测树的复发间隔（裁定 47 改过窗与形状）。

- **到了日子**（``day_of_month``）：往月同一个几号做过（还信用卡 10 号）。一类一本账（触发身份不带几号，几号在触发实例里）；
  机会 = 人在场的往月里有这一天的月份数，命中 = 那几个月里这一天做过的；只在一个月做过的不建（样本少就没有读数）。
  按月形状：命中只算那个月**只落在这一天、前一天后一天都没做**的（天天看日程的类 15 号也做，但 14、16 号也做，15 号对它
  不是日子；9 号 10 号连着还信用卡的那种是一对、归离上次够久管）。时窗是那几次第一次的时刻范围（今天）。
- **离上次够久**（``interval``）：按逝去天数算离散危险率。历史相邻两次的间隔 g₁…gₙ 里，到了逝去 e 天还没做的那些
  （gᵢ ≥ e）是此刻在险的机会，恰在第 e 天做的（gᵢ = e）是命中。今天只有在"历史上有一次恰好隔 e 天"时才算到了日子；
  时窗落在"该来的那几天"——从今天起、到与 e 连着（相邻天数）的那一簇间隔的最后一天，不是"上次 + 最大间隔"。
  在险的机会不到两次不建。触发实例是 ``<上次那条的 URI>@<这一簇的第一天>``，第一天按含 e 的整簇算（往前也连着），
  簇里哪一天重算都是同一个身份、全账本只建一条；同一个上次能分几簇（不连着的）到期，各建一条。

**今天已做的照建**（裁定 49：从当天零点起的发生都算这条说中的）；此刻已经过了时窗而今天还没做的不建（退场）。
时窗至少一槽宽（单样本的时刻只是一个点，拍点落不上）。
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, datetime, timedelta

from habitus.foresight.model import Moment
from habitus.foresight.opportunities.model import KindDay, KindDays, OpportunityTable
from habitus.foresight.triggers.draft import Draft, at_time_of, clamp, midnight_after, midnight_of

DAY_OF_MONTH = "day_of_month"
INTERVAL = "interval"


def day_drafts(table: OpportunityTable, moment: Moment, *, done_today: frozenset[str]) -> tuple[Draft, ...]:
    found: list[Draft] = []
    for kind in table.kinds:
        days = tuple(item for item in kind.days if item.day < moment.day)
        if not days:
            continue
        today = kind.kind_token in done_today
        found.extend(_same_day_of_month(table, kind, days, moment, done_now=today))
        found.extend(_interval(table, kind, days, moment, done_now=today))
    return tuple(found)


def _same_day_of_month(
    table: OpportunityTable, kind: KindDays, days: Sequence[KindDay], moment: Moment, *, done_now: bool
) -> tuple[Draft, ...]:
    done = {item.day for item in days}
    same = [
        item
        for item in days
        if item.day.day == moment.day.day
        and item.natural > 0
        and item.day - timedelta(days=1) not in done
        and item.day + timedelta(days=1) not in done
    ]
    # 那个月这一天只有被提醒才做的：不提醒做不做看不到，那个月不是机会、也不是命中（裁定 54：分子分母都不计）
    unknown = {
        (item.day.year, item.day.month) for item in days if item.day.day == moment.day.day and item.natural == 0
    }
    months = {
        (day.year, day.month) for day in table.present_days if day.day == moment.day.day and day < moment.day
    } - unknown
    hit_months = {(item.day.year, item.day.month) for item in same}
    if len(hit_months) < 2:
        return ()  # 只在一个月落在这一天：样本少就没有读数
    opportunities = max(len(months | hit_months), len(hit_months))
    now = moment.at
    start = min(at_time_of(moment.day, _natural_at(item), now) for item in same)
    end = max(at_time_of(moment.day, _natural_at(item), now) for item in same)
    clamped = clamp(
        start,
        end,
        midnight_after(moment.day, now),
        now=now,
        slot_minutes=table.slot_minutes,
        keep=done_now,
        caught_from=moment.caught_from,
    )
    if clamped is None:
        return ()
    window, deadline = clamped
    return (
        Draft(
            kind_token=kind.kind_token,
            source="日子",
            trigger=DAY_OF_MONTH,
            trigger_instance=moment.day.isoformat(),
            window=window,
            deadline=deadline,
            basis=tuple(_natural_uri(item) for item in same),
            opportunities=opportunities,
            hits=len(hit_months),
            observed_from=midnight_of(moment.day, now),
        ),
    )


def _interval(
    table: OpportunityTable, kind: KindDays, days: Sequence[KindDay], moment: Moment, *, done_now: bool
) -> tuple[Draft, ...]:
    if len(days) < 2:
        return ()
    gaps = [(later.day - earlier.day).days for earlier, later in zip(days, days[1:], strict=False)]
    last = days[-1]
    elapsed = (moment.day - last.day).days
    at_risk = [gap for gap in gaps if gap >= elapsed]
    if elapsed not in gaps or len(at_risk) < 2:
        return ()
    cluster = _cluster(sorted(set(at_risk)), elapsed)
    block = _block(sorted(set(gaps)), elapsed)
    first_day, last_day = last.day + timedelta(days=block[0]), last.day + timedelta(days=cluster[-1])
    # 命中只数自然发生的那几天（被提醒才做的不是他自己的规律，裁定 54）；间隔照全部发生算。落在这一簇里、却是被提醒才做的
    # 那几次：不提醒那天做不做看不到，分子分母都不计；比这一簇更长才被提醒的，到这一簇时他确实还没做，照常是在险的机会
    pairs = list(zip(days, days[1:], strict=False))
    basis_days = [later for earlier, later in pairs if (later.day - earlier.day).days in cluster and later.natural > 0]
    unknown = sum(1 for earlier, later in pairs if (later.day - earlier.day).days in cluster and later.natural == 0)
    if not basis_days:
        return ()  # 恰好隔这么久的那几次全是被提醒才做的：没有自然命中
    opportunities = len(at_risk) - unknown
    now = moment.at
    start = min(at_time_of(first_day, _natural_at(item), now) for item in basis_days)
    end = max(at_time_of(last_day, _natural_at(item), now) for item in basis_days)
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
        return ()
    window, deadline = clamped
    return (
        Draft(
            kind_token=kind.kind_token,
            source="日子",
            trigger=INTERVAL,
            trigger_instance=f"{last.first_uri}@{first_day.isoformat()}",
            window=window,
            deadline=deadline,
            basis=tuple(_natural_uri(item) for item in basis_days),
            opportunities=opportunities,
            hits=len(basis_days),
            observed_from=midnight_of(moment.day, now),
        ),
    )


def _natural_at(item: KindDay) -> datetime:
    """命中那天第一次自然发生的时刻（调用方只对 ``natural > 0`` 的那天问）。"""

    assert item.natural_at is not None
    return item.natural_at


def _natural_uri(item: KindDay) -> str:
    assert item.natural_uri is not None
    return item.natural_uri


def _cluster(distinct_gaps: Sequence[int], elapsed: int) -> tuple[int, ...]:
    """从 ``elapsed`` 起、相邻天数连着的那一簇间隔（29、30 是一簇；2、3、5 里从 2 起是 2、3）。"""

    found = [elapsed]
    for gap in distinct_gaps:
        if gap > found[-1] and gap - found[-1] <= 1:
            found.append(gap)
    return tuple(found)


def _block(distinct_gaps: Sequence[int], elapsed: int) -> tuple[int, ...]:
    """含 ``elapsed`` 的那一整簇（往前往后都按相邻天数连着）：触发实例取它的第一天，同一簇哪天重算都是同一个身份。

    只从 ``elapsed`` 往后数会让间隔 2、3、4 的类第 2 天建一条（簇 2–4）、第 3 天再建一条（簇 3–4），两条时窗重叠，
    第 3 天那一次发生被同一本账结成两次命中。"""

    found = [elapsed]
    for gap in sorted(distinct_gaps, reverse=True):
        if gap < found[0] and found[0] - gap <= 1:
            found.insert(0, gap)
    return tuple(found)


def first_window_day(trigger_instance: str) -> date | None:
    """「离上次够久」实例里这一簇的第一天；别的触发实例是 None。"""

    _uri, separator, tail = trigger_instance.rpartition("@")
    if not separator:
        return None
    try:
        return date.fromisoformat(tail)
    except ValueError:
        return None


__all__ = ["DAY_OF_MONTH", "INTERVAL", "day_drafts", "first_window_day"]
