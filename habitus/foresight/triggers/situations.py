"""「处在情境里」：语义树上"日历条件 → 后果"的关系（裁定 59；19 号方案第三节原来推后的那一种触发）。

今天这个日历条件成立（周三、20 号、工作日……，外部条件包按天给，没有放假数据时日型按名义周末），而且到了时窗，就建一条预期：
时窗是历史上条件成立那几天里后果自然发生的时刻的 p10–p90（十个以内取最早到最晚），最晚时限是今天过完。只命中过一次的不建
（样本少就没有读数）。**今天已做的照建**（裁定 49：从当天零点起的发生都算这条说中的），此刻过了窗而今天还没做的不建。

一本账 = 后果 + "情境" + 关系键：锚在周三的例会材料是一本，锚在工作日的叫车是一本。节假日不用另写否决：锚在工作日的，
放假那天前因不成立，自然不建。
"""

from __future__ import annotations

from collections.abc import Iterable

from habitus.foresight.model import Moment
from habitus.foresight.opportunities.model import OpportunityTable
from habitus.foresight.triggers.draft import Draft, at_time_of, clamp, midnight_after, midnight_of, spread

#: 只命中过一次的不建：一次的时刻只是一个点，说不出"往常几点"。
MIN_HITS = 2


def situation_drafts(
    table: OpportunityTable, moment: Moment, *, done_today: frozenset[str], statuses: Iterable[str]
) -> tuple[Draft, ...]:
    wanted = set(statuses)
    found: list[Draft] = []
    for relation in table.relations:
        if relation.segment != "situation" or relation.status not in wanted or not relation.upward:
            continue
        if not relation.calendar_holds(moment.calendar, official=table.unseen_day_types == "official"):
            continue  # 今天不在这个日历情境里，或今天的日型是它没见过的（只在工作日的周三做过，国庆的周三不建）
        hits = [
            item for item in relation.items if item.came and item.consequent_at is not None and item.day < moment.day
        ]
        if len(hits) < MIN_HITS:
            continue
        now = moment.at
        times = sorted(
            at_time_of(moment.day, item.consequent_at, now) for item in hits if item.consequent_at is not None
        )
        start, end = spread(times)
        if now < start:
            continue  # 还没到时窗
        kind = relation.consequent_classes[0]
        clamped = clamp(
            start,
            end,
            midnight_after(moment.day, now),
            now=now,
            slot_minutes=table.slot_minutes,
            keep=kind in done_today,
            caught_from=moment.caught_from,
        )
        if clamped is None:
            continue
        window, deadline = clamped
        known = [item for item in relation.items if item.known and item.day < moment.day]
        found.append(
            Draft(
                kind_token=kind,
                source="情境",
                trigger=relation.key,
                trigger_instance=moment.day.isoformat(),
                window=window,
                deadline=deadline,
                basis=tuple(item.consequent_uri for item in hits if item.consequent_uri is not None),
                opportunities=max(len(known), len(hits)),
                hits=len(hits),
                observed_from=midnight_of(moment.day, now),
            )
        )
    return tuple(found)


__all__ = ["MIN_HITS", "situation_drafts"]
