"""夜里算机会表（N1）：在语义树同一条时间线上、用同一把尺（``scene.relations`` 的链合并与在场口径），对每条成立 / 候选关系
列出历史每次前因之后后果来没来，再给每个行为类列发生过的日子。零模型、零统计：这里不排序、不给读数，只摆事实；
读数怎么出、门怎么设是白天那一拍的事（第九节第 3 步）。

哪些关系进表：短跨度四段、不带条件、后果是基础概念、前因是基础或
汇总概念（长跨度要重算标志、带条件的要对着此刻那条记录判、细分概念今天的记录还没映射——都还不进预测层）。
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date, datetime, time, timedelta

from habitus.foresight.opportunities.model import (
    KindDay,
    KindDays,
    Opportunity,
    OpportunityTable,
    RelationOpportunities,
    TreeReading,
)
from habitus.scene.concepts.model import ConceptKind, ConceptSet
from habitus.scene.relations import Chain, Controls, LaneTimeline, Relation, Segment, Status, chains_of, merge_for

#: 进表的四段（短跨度）。
_SEGMENTS = (Segment.CHAIN, Segment.REST_OF_DAY, Segment.NEXT_DAY, Segment.DAYS_2_7)
_STATUSES = {Status.ESTABLISHED: "established", Status.CANDIDATE: "candidate"}


def build_table(
    line: LaneTimeline,
    relations: Iterable[Relation],
    concepts: ConceptSet,
    *,
    night: date,
    relation_night: date | None,
    slot_minutes: int,
    window_slots: int,
) -> OpportunityTable:
    """这条 lane 第 ``night`` 晚的机会表。``window_slots`` 是转移窗口的槽数（与预测树、关系检验同一个数）。"""

    controls = Controls(line, window_slots)
    found: list[RelationOpportunities] = []
    for relation in sorted(relations, key=lambda item: item.key.text):
        key = relation.key
        status = _STATUSES.get(relation.status)
        # TODO(FORESIGHT-EVENT-FIRST-001)：带条件的关系整条跳过，语义树量出的调节（周末 / 日历 / 地点）到不了预测层；
        # 规格见 triggers/assemble.py。
        if status is None or key.condition or key.segment not in _SEGMENTS or key.lane != line.lane:
            continue
        consequent = concepts.get(key.consequent)
        antecedent = concepts.get(key.antecedent)
        if consequent is None or antecedent is None or consequent.kind is not ConceptKind.BASE:
            continue
        if antecedent.kind not in (ConceptKind.BASE, ConceptKind.GROUP):
            continue
        chains = merge_for(key.segment, chains_of(line, key.antecedent, window_slots))
        items = tuple(
            _opportunity(controls, key.segment, chain, key.consequent, line, slot_minutes) for chain in chains
        )
        found.append(
            RelationOpportunities(
                key=key.text,
                antecedent=key.antecedent,
                consequent=key.consequent,
                antecedent_label=concepts.label_of(key.antecedent),
                consequent_label=concepts.label_of(key.consequent),
                antecedent_classes=tuple(antecedent.classes),
                consequent_classes=tuple(consequent.classes),
                segment=key.segment.value,
                status=status,
                upward=relation.upward,
                items=items,
                reading=_reading(relation),
            )
        )
    return OpportunityTable(
        lane=line.lane,
        night=night,
        relation_night=relation_night,
        slot_minutes=slot_minutes,
        window_slots=window_slots,
        present_days=line.days(),
        relations=tuple(found),
        kinds=_kinds(line, concepts),
    )


def _reading(relation: Relation) -> TreeReading | None:
    test = relation.tonight
    if test is None:
        return None
    effect = test.effect
    return TreeReading(
        treated=min(1.0, max(0.0, effect.treated_rate)),
        control=min(1.0, max(0.0, effect.control_rate)),
        large_enough=test.large_enough,
    )


def _opportunity(
    controls: Controls, segment: Segment, chain: Chain, consequent: str, line: LaneTimeline, slot_minutes: int
) -> Opportunity:
    arrival = controls.arrival(segment, chain, consequent)
    return Opportunity(
        antecedent_uri=chain.anchor.uri,
        antecedent_uris=tuple(sorted(chain.uris)),
        day=chain.day,
        ended_at=_slot_start(chain.day, chain.end - line.day_start(chain.day), slot_minutes, chain.anchor.started_at),
        came=arrival.came,
        consequent_uri=arrival.uri if arrival.came else None,
        consequent_at=arrival.started_at if arrival.came else None,
    )


def _slot_start(day: date, slot: int, slot_minutes: int, like: datetime) -> datetime:
    """当天第 ``slot`` 槽的开头，用 ``like`` 的时区写（行为树上的时刻是本地时刻加显式偏移）。"""

    return datetime.combine(day, time(0), tzinfo=like.tzinfo) + timedelta(minutes=slot * slot_minutes)


def _kinds(line: LaneTimeline, concepts: ConceptSet) -> tuple[KindDays, ...]:
    """每个基础概念（= 行为类）发生过的日子；没发生过的不列。"""

    found: list[KindDays] = []
    for identity in sorted(concepts.behaviors()):
        item = concepts[identity]
        if item.kind is not ConceptKind.BASE or concepts.lane_of(identity) != line.lane:
            continue
        marks = line.marks.get(identity, [])
        if not marks:
            continue
        days: list[KindDay] = []
        for mark in marks:  # 时间线上的记号已按（槽, 开始时刻, uri）排好
            natural = 0 if mark.reminded else 1
            if days and days[-1].day == mark.day:
                last = days[-1]
                first_natural = natural == 1 and last.natural_uri is None
                days[-1] = KindDay(
                    day=last.day,
                    first_uri=last.first_uri,
                    first_at=last.first_at,
                    count=last.count + 1,
                    natural=last.natural + natural,
                    natural_uri=mark.uri if first_natural else last.natural_uri,
                    natural_at=mark.started_at if first_natural else last.natural_at,
                )
            else:
                days.append(
                    KindDay(
                        day=mark.day,
                        first_uri=mark.uri,
                        first_at=mark.started_at,
                        count=1,
                        natural=natural,
                        natural_uri=mark.uri if natural else None,
                        natural_at=mark.started_at if natural else None,
                    )
                )
        found.append(KindDays(kind_token=item.classes[0], label=item.label, days=tuple(days)))
    return tuple(found)


__all__ = ["build_table"]
