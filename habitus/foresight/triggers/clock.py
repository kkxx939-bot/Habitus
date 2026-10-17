"""「到了钟点」：预测树上这个周几这个行为起过头的那些日子（格子的出处日），时窗从那几天的实际时刻取（裁定 47）。

预测树只回答"这个周几这个行为在哪些日子起过头"（``nodes`` 的 ``days``）；几点从机会表的日历配（那几天第一次的时刻），
窗是这些时刻的 p10–p90（样本少时就是最早到最晚），不从池化收缩的曲线取——曲线借了别的周几的质量，窗会比实际宽。
出处日不到三天不建（样本少就没有读数）；机会的分母是机会表里人在场的、同一个周几的日子；到了钟点（窗开了）才建；
**今天已做的照建**（裁定 49：从当天零点起的发生都算这条说中的，结算照常给准时 / 早到），此刻过了窗而今天还没做的不建；
最晚时限是今天过完。
"""

from __future__ import annotations

from datetime import date, datetime

from habitus.foresight.model import Moment
from habitus.foresight.opportunities.model import OpportunityTable
from habitus.foresight.triggers.draft import Draft, at_time_of, clamp, midnight_after, midnight_of
from habitus.prediction.model import PredictionTree

#: 出处日少于这么多天不建：两天的时刻只是两个点，说不出"往常几点"。
MIN_PROVENANCE_DAYS = 3


def clock_drafts(
    tree: PredictionTree, table: OpportunityTable | None, moment: Moment, *, done_today: frozenset[str]
) -> tuple[Draft, ...]:
    """``done_today`` 是今天到此刻已经开始过的类（照建，过了窗也建：那是一次命中）。没有机会表就没有实际时刻，不建。"""

    if table is None:
        return ()
    weekday = moment.weekday
    provenance: dict[str, set[date]] = {}
    for (key, action), node in tree.nodes.items():
        if key.weekday == weekday:
            provenance.setdefault(action, set()).update(day for day in node.days if day < moment.day)
    present = tuple(day for day in table.present_days if day.weekday() == weekday and day < moment.day)
    found: list[Draft] = []
    for action in sorted(provenance):
        days = sorted(provenance[action])
        if len(days) < MIN_PROVENANCE_DAYS:
            continue
        calendar = table.kind(action)
        if calendar is None:
            continue
        by_day = {item.day: item for item in calendar.days}
        hits = [by_day[day] for day in days if day in by_day and by_day[day].natural > 0]
        if len(hits) < MIN_PROVENANCE_DAYS:
            continue
        # 这个周几只有被提醒才做的那几天：不提醒做不做看不到，不是机会（裁定 54：分子分母都不计，与预测树同口径）
        unknown = {item.day for item in calendar.days if item.natural == 0}
        chances = [day for day in present if day not in unknown]
        times = sorted(at_time_of(moment.day, item.natural_at, moment.at) for item in hits if item.natural_at is not None)
        start, end = _quantiles(times)
        if moment.at < start:
            continue  # 还没到钟点
        clamped = clamp(
            start,
            end,
            midnight_after(moment.day, moment.at),
            now=moment.at,
            slot_minutes=tree.slot_minutes,
            keep=action in done_today,
            caught_from=moment.caught_from,
        )
        if clamped is None:
            continue
        window, deadline = clamped
        found.append(
            Draft(
                kind_token=action,
                source="钟面",
                trigger=f"weekday={weekday}",
                trigger_instance=moment.day.isoformat(),
                window=window,
                deadline=deadline,
                basis=tuple(item.natural_uri for item in hits if item.natural_uri is not None),
                opportunities=max(len(chances), len(hits)),
                hits=len(hits),
                observed_from=midnight_of(moment.day, moment.at),
            )
        )
    return tuple(found)


def _quantiles(times: list[datetime]) -> tuple[datetime, datetime]:
    """排好序的时刻的 p10 与 p90（按位置取，不插值）：十个以内取最早到最晚，第 11 个起才开始去掉两头。"""

    last = len(times) - 1
    return times[int(0.1 * last)], times[-1 - int(0.1 * last)]


__all__ = ["MIN_PROVENANCE_DAYS", "clock_drafts"]
