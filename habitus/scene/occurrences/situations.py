"""情境命中：算法按概念上的情境说明算出"那一刻外面是什么样"。

映射器不算这个——它只判"这一条是不是那个行为"（``situation_hits`` 由调用方填，七d-7 那条边界）。
这里是纯计算：给定一天的日历、这条 occurrence 的字段、以及它之前一段时间的命中事件，
把概念集里**带情境说明**的那些逐条判一遍。

**材料由调用方备好**：历史命中以"事件"传进来（时刻 · 概念身份 · 档，含祖先），本支不读别的存储。

**派生情境按事件数，不按日历日**（B14"窗与机会按事件维度锚定"；2026-09-30 按评审 A-8 / B-11 / C-8 改）：
02:10 的晚睡落在"今天"的目录里，按日历日取"昨天"会漏掉刚熬完夜的这个早上、却让第二天早上成立。
所以 ``YESTERDAY`` 问的是"这条 occurrence 开始之前 24 小时内命中过「X」没有"，``STREAK`` 问的是
"往前每 24 小时一格、连续 N 格里格格都命中过「X」"。**只看这条 occurrence 之前**：把它自己算进去
就是拿今天解释今天，而这些情境正是要给它分层用的。

**"判过"与"成立"分开记**（2026-09-30 裁定八）：``situation_hits`` 返回成立的那些和**判过的全部**。
没判过（材料不全、情境概念是后来才加的）与判过不成立在盘上必须长得不一样，分层时才分得开。
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

from habitus.scene.concepts.model import ConceptSet, concept_identity
from habitus.scene.concepts.situation import SituationBasis, SituationRule
from habitus.scene.occurrences.model import ConceptHit, ConceptHits

#: 派生情境的"一天"：以这条 occurrence 的开始时刻为锚往前数的 24 小时。
DAY_HOURS = 24


@dataclass(frozen=True)
class HitEvent:
    """历史里的一次命中：什么时候、哪个概念（身份）、哪一档。祖先以 ``grade=None`` 记。"""

    at: datetime
    identity: str
    grade: str | None = None


@dataclass(frozen=True)
class SituationInputs:
    """算一条 occurrence 的情境要的全部材料。``moment`` 是这条 occurrence 的开始时刻。"""

    day: date
    moment: datetime | None = None
    calendar_note: str | None = None
    subjects: tuple[str, ...] = ()
    place: str | None = None
    history: tuple[HitEvent, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not isinstance(self.day, date):
            raise TypeError("day must be a date")
        if self.moment is not None and (not isinstance(self.moment, datetime) or self.moment.utcoffset() is None):
            raise TypeError("moment must be a timezone-aware datetime")


@dataclass(frozen=True)
class SituationOutcome:
    """成立的情境（``hits``）与判过的全部情境概念名（``checked``，含成立的）。"""

    hits: tuple[ConceptHit, ...]
    checked: tuple[str, ...]


def hit_events(records: Iterable[ConceptHits], concepts: ConceptSet) -> tuple[HitEvent, ...]:
    """一批命中记录 → 命中事件，**含祖先**（派生情境多半盯着粗的那一层：「赶工中」盯的是「写代码」，命中的是「修改代码」）。"""

    found: list[HitEvent] = []
    for record in records:
        names = [hit.concept for hit in record.hits if hit.concept in concepts]
        for hit in record.hits:
            if hit.concept in concepts:
                found.append(HitEvent(record.started_at, hit.identity, hit.grade))
        for ancestor in concepts.with_ancestors(names) - {concepts[name].identity for name in names}:
            found.append(HitEvent(record.started_at, ancestor, None))
    return tuple(sorted(found, key=lambda item: (item.at.timestamp(), item.identity)))


def situation_hits(concepts: ConceptSet, inputs: SituationInputs) -> SituationOutcome:
    """概念集里带情境说明的情境概念逐条判：成立的进 ``hits``，判过的进 ``checked``，都按身份排序。

    没有说明的情境概念一个都不判（「出差中」要等事实门）；带说明但材料给不全的（要日历却没有日历、
    要历史却没给 ``moment``）**不判**——不把"不知道"算成"成立"，也不把它算成"判过不成立"。
    """

    if not isinstance(concepts, ConceptSet):
        raise TypeError("concepts must be a ConceptSet")
    if not isinstance(inputs, SituationInputs):
        raise TypeError("inputs must be SituationInputs")
    hits: list[ConceptHit] = []
    checked: list[str] = []
    for identity in sorted(concepts.situations()):
        rule = concepts[identity].situation
        if rule is None:
            continue
        verdict = _holds(rule, inputs)
        if verdict is None:
            continue
        checked.append(concepts[identity].name)
        if verdict:
            hits.append(ConceptHit(concepts[identity].name))
    return SituationOutcome(hits=tuple(hits), checked=tuple(checked))


def _holds(rule: SituationRule, inputs: SituationInputs) -> bool | None:
    """成立 / 不成立 / ``None`` = 材料不全、判不了。"""

    if rule.basis is SituationBasis.WEEKDAYS:
        return inputs.day.weekday() in rule.weekdays
    if rule.basis is SituationBasis.CALENDAR_NOTE:
        if inputs.calendar_note is None:
            return None
        return rule.value is not None and rule.value in inputs.calendar_note
    if rule.basis is SituationBasis.SUBJECT:
        return rule.value in inputs.subjects
    if rule.basis is SituationBasis.PLACE:
        return inputs.place is not None and inputs.place == rule.value
    if inputs.moment is None or rule.concept is None:
        return None
    watched = concept_identity(rule.concept)
    if rule.basis is SituationBasis.YESTERDAY:
        return _hit_within(inputs, watched, rule.grade, since=inputs.moment - timedelta(hours=DAY_HOURS), until=inputs.moment)
    return all(
        _hit_within(
            inputs,
            watched,
            rule.grade,
            since=inputs.moment - timedelta(hours=DAY_HOURS * step),
            until=inputs.moment - timedelta(hours=DAY_HOURS * (step - 1)),
        )
        for step in range(1, rule.days + 1)
    )


def _hit_within(inputs: SituationInputs, watched: str, grade: str | None, *, since: datetime, until: datetime) -> bool:
    """``[since, until)`` 里命中过「watched」（要档的话档也要对）没有。"""

    return any(
        event.identity == watched and (grade is None or event.grade == grade) and since <= event.at < until for event in inputs.history
    )


def history_hours(concepts: ConceptSet) -> int:
    """要备多长的命中历史：概念集里最长的连续天数 × 24 小时。一条派生说明都没有 → 0。"""

    spans = [
        item.situation.days
        for identity in concepts.situations()
        for item in (concepts[identity],)
        if item.situation is not None and item.situation.basis.reads_history
    ]
    return DAY_HOURS * (max(spans) if spans else 0)


def history_days(concepts: ConceptSet, day: date) -> tuple[date, ...]:
    """要读哪几天的记录才凑得出 ``history_hours``：``day`` 之前的那几天（含 ``day`` 自己——凌晨那条晚睡就在今天的目录里）。"""

    hours = history_hours(concepts)
    if hours == 0:
        return ()
    depth = hours // DAY_HOURS
    return tuple(day - timedelta(days=step) for step in range(0, depth + 1))


__all__ = [
    "DAY_HOURS",
    "HitEvent",
    "SituationInputs",
    "SituationOutcome",
    "history_days",
    "history_hours",
    "hit_events",
    "situation_hits",
]
