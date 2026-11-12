"""关系行退休的"久没来"（方案 ``19`` §8，用户 11-10「2 同意」、11-12「全部按建议」）：纯函数，取材料由组合根做。

- **尺**：账的久别同一把（``temporal.absence.long_absence``）——这次空白比历史上没截尾的间隔都长，且不短于 H 个典型间隔；
  样本不够不判。H 与最少间隔数在 ``RelationConfig``（组合根从预测层抄）。
- **来过**：行为概念是时间线上有记号的日子，那天的记号全是被提醒之后才做的记成 ``Visit(prompted=True)``（以它结尾的间隔截尾）；
  非日历情境当前因是它成立的日子。没来过的一端不判。
- **在看**：来过的日子按 1，其余按观测覆盖的日占比（``temporal.coverage``；不是时间线的"人在场"）。
- **只数日型**：只数这个概念来过的日型的日子（账的久别同一个口径，``foresight.absence._rulers``）；日型答不出的归 None。
- **两端**：前因、后果各判一次，任一端久没来就退；不看条件。前因是日历情境时只判后果。

与语义树的"久别重来"（``exceeds_longest``、按链日、要 ⌈1/α⌉−1 个样本）不是一回事。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date

from habitus.scene.concepts.calendar_situations import calendar_key
from habitus.scene.concepts.model import ConceptSet
from habitus.scene.relations.config import RelationConfig
from habitus.scene.relations.engine import RelationKey
from habitus.scene.relations.timeline import LaneTimeline
from habitus.temporal.absence import AbsenceVerdict, GapSummary, Visit, effective_gaps, long_absence, open_gap
from habitus.temporal.coverage import Coverage


@dataclass(frozen=True)
class Watching:
    """一条 lane 这一代的观测材料：每天在看的比例、某天是什么日型（答不出是 None）、空白算到哪天（最后一个已封口的日子）。"""

    coverage: Coverage
    day_type: Callable[[date], str | None]
    through: date


@dataclass(frozen=True)
class Absence:
    """一端久没来：这次空白的有效长度、历来没截尾的间隔里最长的、典型间隔（中位数），单位都是天。"""

    current: float
    longest: float
    typical: float


def visits(line: LaneTimeline, identity: str) -> tuple[Visit, ...]:
    """这个概念来过的日子：有记号的是行为概念（那天全是被提醒之后才做的记 ``prompted``）；否则是情境成立的日子。"""

    marks = line.marks.get(identity)
    if marks:
        prompted: dict[date, bool] = {}
        for mark in marks:
            prompted[mark.day] = prompted.get(mark.day, True) and mark.reminded
        return tuple(Visit(day, prompted=flag) for day, flag in sorted(prompted.items()))
    held = line.situations.get(identity, {})
    return tuple(Visit(day) for day, value in sorted(held.items()) if value)


def absence(came: tuple[Visit, ...], watching: Watching, config: RelationConfig) -> Absence | None:
    """来过的这些日子之后到 ``watching.through`` 是不是久没来；是就给出读数，不是、不判（没来过、样本不够）都是 None。"""

    if not came:
        return None
    seen = {visit.day for visit in came}
    types = {watching.day_type(day) for day in seen}

    def watched(day: date) -> float:
        return 1.0 if day in seen else watching.coverage.day_share(day)

    def counts(day: date) -> bool:
        return day in seen or watching.day_type(day) in types

    prior = effective_gaps(came, watched=watched, counts=counts)
    current = open_gap(came[-1].day, watching.through, watched=watched, counts=counts)
    if current is None:
        return None
    verdict = long_absence(
        prior,
        current,
        min_samples=config.absence_min_gaps,
        typical_multiple=config.absence_typical_multiple,
    )
    if verdict is not AbsenceVerdict.EXCEEDS:
        return None
    history = GapSummary().including(prior)
    return Absence(current=current.length, longest=history.longest, typical=history.typical)


def absences(line: LaneTimeline, watching: Watching, config: RelationConfig) -> Callable[[str], Absence | None]:
    """一条 lane 这一代每个概念久没来的读数，按概念缓存（同一代里一个概念的结论与它出现在哪条关系上无关）。"""

    found: dict[str, Absence | None] = {}

    def away(identity: str) -> Absence | None:
        if identity not in found:
            found[identity] = absence(visits(line, identity), watching, config)
        return found[identity]

    return away


def retirement_reason(key: RelationKey, concepts: ConceptSet, away: Callable[[str], Absence | None]) -> str | None:
    """这条关系该不该退休：前因（日历情境不判）、后果任一端久没来就退，返回迁移原因；不退是 None。
    ``away`` 是一个概念久没来的读数（``absences``）。"""

    ends: tuple[tuple[str, str], ...] = (("后果", key.consequent),)
    if calendar_key(concepts, key.antecedent) is None:
        ends = (("前因", key.antecedent), *ends)
    for label, identity in ends:
        found = away(identity)
        if found is not None:
            return (
                f"{label}久没来（这次空白有效长度 {found.current:.4g} 天，历来最长 {found.longest:.4g} 天，"
                f"典型间隔 {found.typical:.4g} 天），关系退休"
            )
    return None


__all__ = ["Absence", "Watching", "absence", "absences", "retirement_reason", "visits"]
