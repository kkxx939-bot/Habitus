"""情境命中：只有日历一族（裁定 59 第 7 条）。判过的全部记下，成立的是其子集（裁定八）；那天日历答不出的键不判。"""

from __future__ import annotations

from datetime import timedelta

import pytest

from habitus.scene.concepts import ConceptSet
from habitus.scene.concepts.situation import SituationError, SituationRule
from habitus.scene.occurrences import ConceptHit
from habitus.scene.occurrences.situations import SituationInputs, situation_hits
from tests.unit.scene.concept_fixtures import calendar
from tests.unit.scene.fixtures import DAY1
from tests.unit.scene.hit_fixtures import CONCEPTS

SATURDAY = DAY1  # 2026-08-15 是周六
FRIDAY = SATURDAY - timedelta(days=1)
SWAP_DAY = calendar("调休日", "日历.日型", "调休上班")
SITUATIONS = ConceptSet((*CONCEPTS.values(), SWAP_DAY))


def names(hits: tuple[ConceptHit, ...]) -> set[str]:
    return {hit.concept for hit in hits}


def test_calendar_situations_compute_and_checked_is_a_superset_of_holding() -> None:
    inputs = SituationInputs(day=SATURDAY, calendar={"日历.周几": "周六", "日历.日型": "工作日"})
    outcome = situation_hits(SITUATIONS, inputs)
    assert names(outcome.hits) == {"周末"}
    assert set(outcome.checked) == {"周末", "调休日"}  # 调休日判过、不成立
    # 那天日历答不出日型：调休日不判（不当成立、也不当"判过不成立"）
    partial = situation_hits(SITUATIONS, SituationInputs(day=SATURDAY, calendar={"日历.周几": "周五"}))
    assert partial.hits == () and partial.checked == ("周末",)
    # 没有日历可读：一个都不判
    bare = situation_hits(SITUATIONS, SituationInputs(day=FRIDAY))
    assert bare.hits == () and bare.checked == ()


def test_a_calendar_rule_needs_a_key_and_distinct_values() -> None:
    with pytest.raises(SituationError, match="calendar situation names one or more values"):
        SituationRule(value="日历.周几", values=())
    with pytest.raises(SituationError, match="calendar key"):
        SituationRule(value="", values=("周六",))
    with pytest.raises(SituationError, match="distinct"):
        SituationRule(value="日历.周几", values=("周六", "周六"))
