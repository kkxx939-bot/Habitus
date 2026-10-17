"""「有件事刚发生」：前因此刻在它那一段跨度里，后果就是候选；以及反向——一个候选的全部前因此刻在不在场。

前因"在不在它那一段里"的判法与机会表、关系检验同一把尺：先按 ``chains_of`` 把相邻的同类合成一条链、再按 ``merge_for`` 合并
窗口重叠的，然后看链所在的段位——

- 同一条链：链今天做完的那一槽到此刻不超过转移窗口的槽数；
- 当天剩下的：今天第一条链做完、已经过了转移窗口；
- 次日：链在昨天；第 2–7 天：链在 2–7 天前（合并后的锚）。

前两段里前因做完之后后果**已经开始过**的、后两段里后果在窗口里已经来过的**照建**（裁定 49：那是一次命中，不建账上就只剩
没来的那几次），而且不退场；从前因做完那一刻（按天的两段：窗口第一天零点）起的发生都算这条说中的，结算照常给准时 / 早到。
时窗从机会表里这条关系历史上命中的那几次取：按槽的两段取"做完之后多久"，按天的两段取"第几天、几点"；只命中过一次的不建
（样本少就没有读数，裁定 47）。**前因还在进行中不建**：同一条链这一段只认最后所见已经过去至少一槽的链（未封口的判断最后所见
就是此刻，开始那一拍就落在链段、窗锚在开始槽，是评审指出的毛病）。
**退场**：此刻已经过了历史上最晚那一次（方案第三节），不再建——按槽的两段多宽限一槽（未封口的判断最多晚一拍才到场，
紧挨着的链不然永远赶不上）。触发实例是前因那条链的锚的**开始时刻**，不是它的 URI：今天的前因还没封口时只有 ``unsealed:<开始时刻>``
这个临时出处，封口后换成 URI，身份不能跟着换（前因清单里的 ``source`` 仍记出处，给人看）。
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date, datetime, time, timedelta

from habitus.foresight.ledger.model import Antecedent
from habitus.foresight.model import Moment
from habitus.foresight.opportunities.model import OpportunityTable, RelationOpportunities
from habitus.foresight.triggers.draft import Draft, at_time_of, clamp, instance_at, midnight_after, midnight_of
from habitus.scene.relations import Chain, LaneTimeline, Segment, chains_of, merge_for

_MERGED = "*"


def chains_in_segment(
    line: LaneTimeline, classes: Iterable[str], segment: Segment, *, today: date, now_slot: int, window: int
) -> tuple[Chain, ...]:
    """这些类的链里，此刻正处在 ``segment`` 这一段的那几条（通常一条；没有就是不在场）。``now_slot`` 是此刻的绝对槽号。"""

    marks = sorted(
        (mark for token in set(classes) for mark in line.marks.get(token, [])),
        key=lambda item: (item.slot, item.started_at, item.uri),
    )
    merged = LaneTimeline(
        lane=line.lane, slots_per_day=line.slots_per_day, horizon=line.horizon, marks={_MERGED: marks}
    )
    kept = merge_for(segment, chains_of(merged, _MERGED, window))
    if segment is Segment.CHAIN:
        return tuple(chain for chain in kept if chain.day == today and 1 <= now_slot - chain.end <= window)
    if segment is Segment.REST_OF_DAY:
        return tuple(chain for chain in kept if chain.day == today and now_slot - chain.end > window)
    if segment is Segment.NEXT_DAY:
        return tuple(chain for chain in kept if (today - chain.day).days == 1)
    return tuple(chain for chain in kept if 2 <= (today - chain.day).days <= 7)


def consequent_came(line: LaneTimeline, classes: Iterable[str], segment: Segment, chain: Chain, *, today: date) -> bool:
    """这条链之后、这一段窗口里后果已经来过没有（链里自己的记录不算）。"""

    marks = [mark for token in set(classes) for mark in line.marks.get(token, []) if mark.uri not in chain.uris]
    if segment in (Segment.CHAIN, Segment.REST_OF_DAY):
        return any(mark.day == today and mark.slot >= chain.end for mark in marks)
    if segment is Segment.NEXT_DAY:
        return any(mark.day == today for mark in marks)
    return any(chain.day + timedelta(days=2) <= mark.day <= today for mark in marks)


def event_drafts(
    table: OpportunityTable, line: LaneTimeline, moment: Moment, *, now_slot: int, statuses: Iterable[str]
) -> tuple[Draft, ...]:
    """此刻在场的、抬高后果的关系，每条一份草稿。"""

    wanted = set(statuses)
    found: list[Draft] = []
    for relation in table.relations:
        if relation.status not in wanted or not relation.upward:
            continue
        segment = Segment(relation.segment)
        for chain in chains_in_segment(
            line, relation.antecedent_classes, segment, today=moment.day, now_slot=now_slot, window=table.window_slots
        ):
            came = consequent_came(line, relation.consequent_classes, segment, chain, today=moment.day)
            draft = _draft(relation, segment, chain, line, moment, table, came=came)
            if draft is not None:
                found.append(draft)
    return tuple(found)


def antecedents_of(
    table: OpportunityTable,
    kind_token: str,
    line: LaneTimeline,
    moment: Moment,
    *,
    now_slot: int,
    statuses: Iterable[str],
) -> tuple[Antecedent, ...]:
    """反向：以这个类为后果的全部关系，逐条标前因此刻在不在它那一段里、是抬高还是压低；在场的带前因这一次的出处。"""

    wanted = set(statuses)
    found: list[Antecedent] = []
    for relation in table.by_consequent(kind_token):
        if relation.status not in wanted:
            continue
        chains = chains_in_segment(
            line,
            relation.antecedent_classes,
            Segment(relation.segment),
            today=moment.day,
            now_slot=now_slot,
            window=table.window_slots,
        )
        found.append(
            Antecedent(
                relation=relation.key,
                present=bool(chains),
                upward=relation.upward,
                source=chains[0].anchor.uri if chains else "",
            )
        )
    return tuple(found)


def _draft(
    relation: RelationOpportunities,
    segment: Segment,
    chain: Chain,
    line: LaneTimeline,
    moment: Moment,
    table: OpportunityTable,
    *,
    came: bool = False,
) -> Draft | None:
    hits = [item for item in relation.items if item.came and item.consequent_at is not None]
    if len(hits) < 2:
        return None
    now = moment.at
    day_start = line.day_start(chain.day)
    ended = datetime.combine(chain.day, time(0), tzinfo=now.tzinfo) + timedelta(
        minutes=(chain.end - day_start) * table.slot_minutes
    )
    if segment in (Segment.CHAIN, Segment.REST_OF_DAY):
        offsets = [item.consequent_at - item.ended_at for item in hits if item.consequent_at is not None]
        start, end = ended + min(offsets), ended + max(offsets)
        if segment is Segment.CHAIN:
            boundary = ended + timedelta(minutes=(table.window_slots + 1) * table.slot_minutes)
        else:
            boundary = midnight_after(chain.day, now)
    else:
        day_offsets = [(item.consequent_at.date() - item.day).days for item in hits if item.consequent_at is not None]
        times = [item.consequent_at for item in hits if item.consequent_at is not None]
        first_day, last_day = chain.day + timedelta(days=min(day_offsets)), chain.day + timedelta(days=max(day_offsets))
        start = min(at_time_of(first_day, item, now) for item in times)
        end = max(at_time_of(last_day, item, now) for item in times)
        boundary = midnight_after(chain.day + timedelta(days=1 if segment is Segment.NEXT_DAY else 7), now)
    grace = timedelta(minutes=table.slot_minutes) if segment in (Segment.CHAIN, Segment.REST_OF_DAY) else timedelta(0)
    if segment in (Segment.CHAIN, Segment.REST_OF_DAY):
        observed_from = min(ended, now)
    else:
        observed_from = midnight_of(chain.day + timedelta(days=1 if segment is Segment.NEXT_DAY else 2), now)
    clamped = clamp(
        start,
        end,
        boundary,
        now=now,
        slot_minutes=table.slot_minutes,
        open_until=ended + grace,
        keep=came,
        caught_from=moment.caught_from,
    )
    if clamped is None:
        return None  # 退场：历史上最晚的那一次也过了
    window, deadline = clamped
    return Draft(
        kind_token=relation.consequent_classes[0],
        source="事件",
        trigger=relation.key,
        trigger_instance=instance_at(chain.anchor.started_at),
        window=window,
        deadline=deadline,
        basis=tuple(item.consequent_uri for item in hits if item.consequent_uri is not None),
        opportunities=relation.opportunities,
        hits=relation.hits,
        observed_from=observed_from,
    )


__all__ = ["antecedents_of", "chains_in_segment", "consequent_came", "event_drafts"]
