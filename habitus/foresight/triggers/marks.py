"""把最近几天的行排成一条只有记号的时间线，好让链合并与语义树同一把尺（``scene.relations.chains_of`` / ``merge_for``）。

填 ``marks``（概念 = 类编号）；人在场、对照时刻这些检验用的字段不填——这里不做检验，只问"此刻前因在不在它那一段里"。
带条件的关系要在白天判"此刻条件成不成立"（裁定 61 第 6 条），所以按机会表上各关系挂的条件再补三样、让语义树的
``evaluator`` 在这条时间线上也能判：每一行的记录（地点、同在的人）；时间线条件指向汇总概念时，成员类的记号也挂在那个
概念身份下；日历情境条件按每一天的日历条件机械填。映射器打的情境标签白天没有，那种条件判不了（``conditions.judge_for``）。
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from datetime import date

from habitus.foresight.model import Seen
from habitus.foresight.opportunities.model import ConditionSpec
from habitus.scene.relations import LaneTimeline, Mark, Template
from habitus.scene.relations import parse_condition as parse
from habitus.series import SeriesRecord
from habitus.temporal.slots import absolute_slot

MINUTES_PER_DAY = 1440


def timeline_of(
    rows: Iterable[Seen],
    *,
    lane: str,
    slot_minutes: int,
    horizon: date,
    conditions: Iterable[ConditionSpec] = (),
    calendar_of: Callable[[date], Mapping[str, str]] | None = None,
) -> LaneTimeline:
    """``rows`` 里每一件事按类挂成记号：开始所在槽、最后所见所在槽（过了午夜截在当天最后一槽）。
    ``conditions`` 是机会表上各关系挂的条件；``calendar_of`` 给某一天的日历条件（日历情境条件要它；不给就判不了）。"""

    per_day = MINUTES_PER_DAY // slot_minutes
    line = LaneTimeline(lane=lane, slots_per_day=per_day, horizon=horizon.toordinal() * per_day)
    specs = tuple(conditions)
    aliases: dict[str, set[str]] = {}
    for spec in specs:
        condition = parse(spec.text)
        if condition.template is Template.TIMELINE:
            for token in spec.classes:
                aliases.setdefault(token, set()).add(condition.value)
    days: set[date] = set()
    for row in rows:
        start = absolute_slot(row.started_at, slot_minutes=slot_minutes)
        if row.ended_at.date() == row.day:
            end = absolute_slot(row.ended_at, slot_minutes=slot_minutes)
        else:
            end = row.day.toordinal() * per_day + per_day - 1
        mark = Mark(slot=start, started_at=row.started_at, uri=row.source, last_slot=max(start, end))
        for identity in {row.kind_token, *aliases.get(row.kind_token, ())}:
            line.marks.setdefault(identity, []).append(mark)
        line.records[row.source] = _record(row, lane)
        days.add(row.day)
    for items in line.marks.values():
        items.sort(key=lambda item: (item.slot, item.started_at, item.uri))
    if calendar_of is not None:
        _calendar(line, specs, sorted(days | {horizon}), calendar_of)
    return line


def _calendar(
    line: LaneTimeline,
    specs: tuple[ConditionSpec, ...],
    days: list[date],
    calendar_of: Callable[[date], Mapping[str, str]],
) -> None:
    """日历情境条件按天机械填（与语义树建时间线同一个判法：那天日历答不出的不填，不知道 ≠ 否）。"""

    keyed = [
        (parse(spec.text).value, spec.calendar_key, set(spec.calendar_values)) for spec in specs if spec.calendar_key
    ]
    if not keyed:
        return
    for day in days:
        conditions = dict(calendar_of(day))
        line.calendar_conditions[day] = conditions
        for identity, key, values in keyed:
            line.calendar.add(identity)
            known = conditions.get(key)
            if known is not None:
                line.situations.setdefault(identity, {})[day] = known in values


def _record(row: Seen, lane: str) -> SeriesRecord:
    """一行的结构字段，给"地点 / 同在的人"条件读（其余字段只是形状要求）。"""

    return SeriesRecord(
        uri=row.source,
        name=row.name or row.kind_token,
        day=row.day,
        started_at=row.started_at,
        last_observed_at=row.ended_at,
        kind_token=row.kind_token,
        lane=lane,
        classified=True,
        reminded=False,
        summary=row.summary,
        place=row.place,
        subjects=row.subjects,
    )


__all__ = ["MINUTES_PER_DAY", "timeline_of"]
