"""夜里算机会表（N1）：在语义树同一条时间线上、用同一把尺（``scene.relations`` 的链合并与在场口径），对每条成立 / 候选关系
列出历史每次前因之后后果来没来，再把每个行为类的复现（语义树登记的）抄进来。零模型、零统计：这里不排序、不给读数，只摆事实；
读数怎么出、门怎么设是白天那一拍的事（第九节第 3 步）。

哪些关系进表：后果是基础概念，并且是
- 短跨度四段、前因是基础或汇总概念的；带条件的只收**成立的**（裁定 61 第 6 条：成立的组合当单元的一条边，没成立的照旧由
  "同一组"计数），它的机会只列条件全成立的链；
- 日历情境段（日历条件 → 后果，裁定 59）：一个后果的几条日历关系只留最具体的那一族（周几集合在就不留被它包住的单个周几）。
长跨度的其余几段、细分概念当前因的还不进预测层（要重算标志、要对着此刻那条记录判、今天的记录还没映射）。
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import date, datetime, time, timedelta

from habitus.conditions import DAY_TYPE_KEY, DayType, DayTypeCalendar
from habitus.foresight.opportunities.model import (
    ConditionSpec,
    DayWatch,
    KindDays,
    Opportunity,
    OpportunityTable,
    RelationOpportunities,
    TreeReading,
)
from habitus.scene.concepts.calendar_situations import calendar_key, most_specific
from habitus.scene.concepts.model import ConceptKind, ConceptSet
from habitus.scene.recurrence import Recurrence
from habitus.scene.relations import (
    Chain,
    Condition,
    Controls,
    LaneTimeline,
    Relation,
    Segment,
    Status,
    Template,
    chains_of,
    evaluator_all,
    merge_for,
)
from habitus.temporal import Coverage

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
    recurrences: Mapping[str, Recurrence] | None = None,
    calendar_version: str | None = None,
    unseen_day_types: str = "skip",
    coverage: Coverage | None = None,
    calendar: DayTypeCalendar | None = None,
) -> OpportunityTable:
    """这条 lane 第 ``night`` 晚的机会表。``window_slots`` 是转移窗口的槽数（与关系检验同一个数）；``recurrences`` 是语义树
    这一晚登记的每个类的复现（日历部分从这里抄）；
    ``calendar_version`` 是这一晚日历情境用的放假数据版本，记在表上（裁定 59 第 4 条：不存每天的结果，只记所用数据的版本）；
    ``coverage`` 是这条 lane 的观测覆盖，抄成每天看着的比例，日型从 ``calendar`` 取（账的久别用；时间线上的日历条件只有
    有记录的日子，空白日也要日型）。"""

    controls = Controls(line, window_slots)
    found: list[RelationOpportunities] = []
    wanted = [
        relation
        for relation in sorted(relations, key=lambda item: item.key.text)
        if _admitted(relation, line, concepts)
    ]
    calendar_kept = _most_specific_calendar(wanted, concepts)
    for relation in wanted:
        key = relation.key
        status = _STATUSES[relation.status]
        consequent = concepts[key.consequent]
        if key.segment is Segment.SITUATION:
            if key.text not in calendar_kept:
                continue
            rule = calendar_key(concepts, key.antecedent)
            assert rule is not None
            items, within = _calendar_opportunities(line, key.antecedent, key.consequent, rule[0])
            found.append(
                RelationOpportunities(
                    key=key.text,
                    antecedent=key.antecedent,
                    consequent=key.consequent,
                    antecedent_label=concepts.label_of(key.antecedent),
                    consequent_label=concepts.label_of(key.consequent),
                    antecedent_classes=(key.antecedent,),
                    consequent_classes=tuple(consequent.classes),
                    segment=key.segment.value,
                    status=status,
                    upward=relation.upward,
                    items=items,
                    reading=_reading(relation),
                    calendar_key=rule[0],
                    calendar_values=rule[1],
                    within_day_types=within,
                )
            )
            continue
        antecedent = concepts[key.antecedent]
        chains = merge_for(key.segment, chains_of(line, key.antecedent, window_slots))
        if key.condition:
            # 带条件的关系：机会只列条件全成立的链（判不了的不算机会）
            judge = evaluator_all(line, key.conditions, window=window_slots)
            chains = tuple(chain for chain in chains if judge(chain) is True)
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
                conditions=tuple(_condition_spec(concepts, condition) for condition in key.conditions),
            )
        )
    return OpportunityTable(
        lane=line.lane,
        night=night,
        relation_night=relation_night,
        slot_minutes=slot_minutes,
        window_slots=window_slots,
        relations=tuple(found),
        kinds=_kinds(line.lane, concepts, recurrences or {}),
        calendar_version=calendar_version,
        unseen_day_types=unseen_day_types,
        watch=_watch(coverage, calendar),
    )


def _admitted(relation: Relation, line: LaneTimeline, concepts: ConceptSet) -> bool:
    key = relation.key
    if relation.status not in _STATUSES or key.lane != line.lane:
        return False
    if key.condition and (relation.status is not Status.ESTABLISHED or key.segment is Segment.SITUATION):
        return False  # 带条件的只收成立的；日历情境关系不挂条件
    consequent = concepts.get(key.consequent)
    if consequent is None or consequent.kind is not ConceptKind.BASE:
        return False
    if key.segment is Segment.SITUATION:
        # 只收抬高的日历关系：一个取值高了，同一维的其余取值就显得低（"周一 → 例会材料 ↓"只是"周三 ↑"的反面），
        # 进单元会当压低的输入把历史切碎；语义树照样记着它们（2026-10-18，S8）
        return relation.upward and calendar_key(concepts, key.antecedent) is not None
    antecedent = concepts.get(key.antecedent)
    return (
        key.segment in _SEGMENTS and antecedent is not None and antecedent.kind in (ConceptKind.BASE, ConceptKind.GROUP)
    )


def _most_specific_calendar(relations: Iterable[Relation], concepts: ConceptSet) -> frozenset[str]:
    """每个后果的日历关系里，被同状态、同方向的更粗的日历关系解释了的不留（``most_specific``：集合包住的取值、月末包住的
    31 号）。候选不挤掉成立的，压低的不挤掉抬高的。"""

    groups: dict[tuple[str, Status, bool], list[Relation]] = {}
    for relation in relations:
        if relation.key.segment is Segment.SITUATION:
            groups.setdefault((relation.key.consequent, relation.status, relation.upward), []).append(relation)
    kept: set[str] = set()
    for items in groups.values():
        chosen = set(most_specific(concepts, [item.key.antecedent for item in items]))
        kept.update(item.key.text for item in items if item.key.antecedent in chosen)
    return frozenset(kept)


def _calendar_opportunities(
    line: LaneTimeline, antecedent: str, consequent: str, key: str
) -> tuple[tuple[Opportunity, ...], tuple[str, ...]]:
    """日历条件成立的每一个人在场的日子是一次机会：后果那天有没有自然发生（只有被提醒才做的那天不知道）。
    机会的"前因"就是那一天本身（``calendar:<日子>``），起算是那天零点。

    周几、几号、月末这类关系还交出它在哪几种日型上见过他自然地做（检验时日型是固定住的，裁定 59 第 7 条）；机会只列这几种日型的
    日子与日型答不出的日子——建预期时也只在这几种日型上建，账上的机会与建预期的口径一致。"""

    held = line.calendar_days(antecedent) or {}
    marks = line.marks.get(consequent, [])
    unknown = set(line.unknown.get(consequent, []))
    rows = []
    seen: set[str] = set()
    for day in line.days():
        if not held.get(day):
            continue
        today = [mark for mark in marks if mark.day == day]
        natural = [mark for mark in today if not mark.reminded and mark.slot not in unknown]
        day_type = line.calendar_conditions.get(day, {}).get(DAY_TYPE_KEY)
        if natural and day_type is not None:
            seen.add(day_type)
        rows.append((day, today, natural, day_type))
    within = () if key == DAY_TYPE_KEY else tuple(kind.value for kind in DayType if kind.value in seen)
    found = []
    for day, today, natural, day_type in rows:
        if within and day_type is not None and day_type not in within:
            continue
        like = (today[0].started_at if today else _any_moment(line, day)).tzinfo
        anchor = f"calendar:{day.isoformat()}"
        if natural:
            first = min(natural, key=lambda mark: (mark.started_at, mark.uri))
            came: bool | None = True
            uri, started = first.uri, first.started_at
        else:
            came, uri, started = (None if today else False), None, None
        found.append(
            Opportunity(
                antecedent_uri=anchor,
                antecedent_uris=(anchor,),
                day=day,
                ended_at=datetime.combine(day, time(0), tzinfo=like),
                came=came,
                consequent_uri=uri,
                consequent_at=started,
            )
        )
    return tuple(found), within


def _watch(coverage: Coverage | None, calendar: DayTypeCalendar | None) -> tuple[DayWatch, ...]:
    if coverage is None:
        return ()
    found = []
    for day in coverage.days:
        kind = calendar.day_type(day) if calendar is not None else None
        found.append(DayWatch(day, coverage.day_share(day), None if kind is None else kind.value))
    return tuple(found)


def _any_moment(line: LaneTimeline, day: date) -> datetime:
    """这一天任何一条记录的开始时刻（借它的时区写零点）。"""

    return next(record.started_at for record in line.records.values() if record.day == day)


def _kinds(lane: str, concepts: ConceptSet, recurrences: Mapping[str, Recurrence]) -> tuple[KindDays, ...]:
    """这条 lane 上每个行为类的复现（语义树登记的），名字取类名。"""

    found = []
    for kind in sorted(recurrences):
        recurrence = recurrences[kind]
        if recurrence.lane != lane or not recurrence.days:
            continue
        found.append(
            KindDays(
                kind_token=kind,
                label=concepts.class_label(kind),
                days=recurrence.days,
                intervals=recurrence.intervals,
            )
        )
    return tuple(found)


def _condition_spec(concepts: ConceptSet, condition: Condition) -> ConditionSpec:
    """条件 → 白天判它要的材料（见 ``ConditionSpec``）。"""

    if condition.template is Template.TIMELINE:
        item = concepts.get(condition.value)
        classes = () if item is None or item.kind is ConceptKind.REFINEMENT else tuple(item.classes)
        return ConditionSpec(condition.text, classes=classes)
    if condition.template is Template.CONCEPT:
        rule = calendar_key(concepts, condition.value)
        if rule is not None:
            return ConditionSpec(condition.text, calendar_key=rule[0], calendar_values=tuple(rule[1]))
    return ConditionSpec(condition.text)


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


__all__ = ["build_table"]
