"""情境命中：三族各算一遍、派生按事件锚定（不按日历日）、判过与成立分开、没有说明的情境概念永远不命中。"""

from __future__ import annotations

from datetime import timedelta

import pytest

from habitus.scene.concepts import ConceptRole, ConceptSet
from habitus.scene.concepts.situation import SituationBasis as B
from habitus.scene.concepts.situation import SituationError, SituationRule
from habitus.scene.occurrences import ConceptHit
from habitus.scene.occurrences.situations import (
    HitEvent,
    SituationInputs,
    history_days,
    history_hours,
    hit_events,
    situation_hits,
)
from tests.unit.scene.concept_fixtures import concept
from tests.unit.scene.fixtures import DAY1, at
from tests.unit.scene.ledger_fixtures import CONCEPTS, record

SATURDAY = DAY1  # 2026-08-15 是周六
FRIDAY = SATURDAY - timedelta(days=1)
THURSDAY = SATURDAY - timedelta(days=2)

SWAP_DAY = concept("调休日", "当地日历说这天要补班", role=ConceptRole.DAY_TYPE, situation=SituationRule(B.CALENDAR_NOTE, value="补班"))
WITH_FAMILY = concept("和家人一起", "这件事是和家人一起做的", role=ConceptRole.OBJECT, situation=SituationRule(B.SUBJECT, value="家人B"))
AT_OFFICE = concept("在公司", "地点是公司", role=ConceptRole.OBJECT, situation=SituationRule(B.PLACE, value="公司"))
CRUNCH = concept("赶工中", "往前连着三个 24 小时都在运动", role=ConceptRole.DERIVED, situation=SituationRule(B.STREAK, concept="运动", days=3))
LAST_NIGHT = concept("昨晚晚睡", "这条行为之前 24 小时内的就寝命中了晚睡重档", role=ConceptRole.DERIVED, situation=SituationRule(B.YESTERDAY, concept="晚睡", grade="重"))
SITUATIONS = ConceptSet((*CONCEPTS.values(), SWAP_DAY, WITH_FAMILY, AT_OFFICE, CRUNCH, LAST_NIGHT))
BREAKFAST = at(SATURDAY, 7, 30)


def names(hits: tuple[ConceptHit, ...]) -> set[str]:
    return {hit.concept for hit in hits}


def test_the_three_families_each_compute_and_checked_is_a_superset_of_holding() -> None:
    """日型（周几 / 日历那句话）· 对象（同在 / 地点）· 派生（之前 24 小时）。判过的全部记下，成立的是其子集（裁定八）。"""

    inputs = SituationInputs(
        day=SATURDAY,
        moment=BREAKFAST,
        calendar_note="补班日，按周一上班",
        subjects=("家人B",),
        place="公司",
        history=(HitEvent(at(SATURDAY, 2, 10), "晚睡", "重"),),
    )
    outcome = situation_hits(SITUATIONS, inputs)
    assert names(outcome.hits) == {"周末", "调休日", "和家人一起", "在公司", "昨晚晚睡"}
    assert set(outcome.checked) == {"周末", "调休日", "和家人一起", "在公司", "昨晚晚睡", "赶工中"}  # 赶工中判过、不成立
    # 材料给不全 → 那一族**不判**（不当成立、也不当"判过不成立"）：没有日历那句话、没有 moment
    bare = situation_hits(SITUATIONS, SituationInputs(day=FRIDAY))
    assert bare.hits == () and set(bare.checked) == {"周末", "和家人一起", "在公司"}


def test_yesterday_is_anchored_on_the_occurrence_not_the_calendar_day() -> None:
    """02:10 的晚睡落在**今天**的目录里。刚熬完夜的这个早上要算「昨晚晚睡」，第二天早上不算（评审 A-8 / B-11 / C-8）。"""

    late = (HitEvent(at(SATURDAY, 2, 10), "晚睡", "重"),)
    assert "昨晚晚睡" in names(situation_hits(SITUATIONS, SituationInputs(day=SATURDAY, moment=BREAKFAST, history=late)).hits)
    next_morning = SituationInputs(day=SATURDAY + timedelta(days=1), moment=at(SATURDAY + timedelta(days=1), 7, 30), history=late)
    assert "昨晚晚睡" not in names(situation_hits(SITUATIONS, next_morning).hits)
    # 只是轻档 → 不算（档是剂量，不同档是不同的前件）
    light = (HitEvent(at(SATURDAY, 2, 10), "晚睡", "轻"),)
    assert "昨晚晚睡" not in names(situation_hits(SITUATIONS, SituationInputs(day=SATURDAY, moment=BREAKFAST, history=light)).hits)


def test_a_streak_counts_24_hour_slices_backwards_and_needs_every_slice() -> None:
    """连着三个 24 小时 = 往前每一格都命中；这条 occurrence 自己不算（拿今天解释今天）。"""

    def history(*hours_before: int) -> tuple[HitEvent, ...]:
        return tuple(HitEvent(BREAKFAST - timedelta(hours=hours), "运动", None) for hours in hours_before)

    assert "赶工中" in names(situation_hits(SITUATIONS, SituationInputs(day=SATURDAY, moment=BREAKFAST, history=history(12, 36, 60))).hits)
    # 中间断一格 → 不算连续
    assert "赶工中" not in names(situation_hits(SITUATIONS, SituationInputs(day=SATURDAY, moment=BREAKFAST, history=history(12, 60, 84))).hits)
    # 这条 occurrence 之后的不算
    assert "赶工中" not in names(situation_hits(SITUATIONS, SituationInputs(day=SATURDAY, moment=BREAKFAST, history=history(-1, 36, 60))).hits)
    # 只有两格 → 不算
    assert "赶工中" not in names(situation_hits(SITUATIONS, SituationInputs(day=SATURDAY, moment=BREAKFAST, history=history(12, 36))).hits)


def test_a_situation_concept_without_a_rule_never_hits() -> None:
    """「出差中」要等事实门接上数据源；在那之前它进得了概念集、也能被假设引用，只是分不出层。"""

    inputs = SituationInputs(day=SATURDAY, moment=BREAKFAST, subjects=("出差中",), place="出差中")
    outcome = situation_hits(SITUATIONS, inputs)
    assert "出差中" not in names(outcome.hits) and "出差中" not in outcome.checked


def test_hit_events_carry_grades_and_ancestors() -> None:
    """派生情境多半盯粗的那一层：命中「打球」要能满足盯着「运动」的那条说明。"""

    records = [record(FRIDAY, "打球", 18, 0, ConceptHit("打球")), record(FRIDAY, "就寝", 2, 10, ConceptHit("晚睡", "重"))]
    events = hit_events(records, SITUATIONS)
    assert {(e.identity, e.grade) for e in events} == {("打球", None), ("运动", None), ("晚睡", "重")}
    assert all(e.at.tzinfo is not None for e in events)


def test_how_much_history_to_prepare_comes_from_the_concepts() -> None:
    """要读多少由概念集里最长那条连续决定——不写死一个"读七天"；含今天自己的目录（凌晨那条就在里面）。"""

    assert history_hours(SITUATIONS) == 72
    assert history_days(SITUATIONS, SATURDAY) == (SATURDAY, FRIDAY, THURSDAY, SATURDAY - timedelta(days=3))
    assert history_days(CONCEPTS, SATURDAY) == ()  # 一条派生说明都没有 → 一天都不用读


def test_a_rule_that_the_algorithm_cannot_compute_is_refused() -> None:
    with pytest.raises(SituationError, match="takes no weekdays"):
        SituationRule(B.PLACE, weekdays=(5,), value="公司")
    with pytest.raises(SituationError, match="watches no concept"):
        SituationRule(B.SUBJECT, value="家人B", concept="打球")
    with pytest.raises(SituationError, match="names the concept it watches"):
        SituationRule(B.YESTERDAY)
    with pytest.raises(ValueError):
        B("open_claim")  # 「约了球还没打」那一族 2026-09-30 取消
