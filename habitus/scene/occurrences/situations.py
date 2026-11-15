"""情境命中：算法按日历情境上的说明算出"那一天日历上是什么样"（裁定 59 第 7 条）。

纯计算：给定一天的日历条件，把概念集里的日历情境逐条判一遍。

**"判过"与"成立"分开记**（2026-09-30 裁定八）：``situation_hits`` 返回成立的那些和**判过的全部**。
没判过（那天日历答不出这个键）与判过不成立在盘上必须长得不一样，分层时才分得开。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date

from habitus.scene.concepts.model import ConceptSet
from habitus.scene.occurrences.model import ConceptHit


@dataclass(frozen=True)
class SituationInputs:
    """算一天的日历情境要的材料。"""

    day: date
    #: 这一天的日历条件（外部条件包给的键值，裁定 59）；``None`` = 没有日历可读。答不出的键不出现（不知道 ≠ 否）。
    calendar: Mapping[str, str] | None = None


@dataclass(frozen=True)
class SituationOutcome:
    """成立的情境（``hits``）与判过的全部情境概念名（``checked``，含成立的）。"""

    hits: tuple[ConceptHit, ...]
    checked: tuple[str, ...]


def situation_hits(concepts: ConceptSet, inputs: SituationInputs) -> SituationOutcome:
    """概念集里的日历情境逐条判：成立的进 ``hits``，判过的进 ``checked``，都按身份排序。那天日历答不出这个键的**不判**——
    不把"不知道"算成"成立"，也不把它算成"判过不成立"。"""

    hits: list[ConceptHit] = []
    checked: list[str] = []
    for identity in sorted(concepts.situations()):
        rule = concepts[identity].situation
        known = None if rule is None or inputs.calendar is None else inputs.calendar.get(rule.value)
        if rule is None or known is None:
            continue
        checked.append(concepts[identity].name)
        if known in rule.values:
            hits.append(ConceptHit(concepts[identity].name))
    return SituationOutcome(hits=tuple(hits), checked=tuple(checked))


__all__ = ["SituationInputs", "SituationOutcome", "situation_hits"]
