"""情境命中：四族各算一遍、派生只看过去的日子、没有说明的情境概念永远不命中。"""

from __future__ import annotations

from datetime import timedelta

import pytest

from habitus.scene.concepts import ConceptRole, ConceptSet
from habitus.scene.concepts.situation import SituationBasis as B
from habitus.scene.concepts.situation import SituationError, SituationRule
from habitus.scene.occurrences import ConceptHit
from habitus.scene.occurrences.situations import (
    SituationInputs,
    day_signature,
    history_days,
    situation_hits,
)
from tests.unit.scene.concept_fixtures import concept
from tests.unit.scene.fixtures import DAY1
from tests.unit.scene.ledger_fixtures import CONCEPTS, record

SATURDAY = DAY1  # 2026-08-15 是周六
FRIDAY = SATURDAY - timedelta(days=1)

SWAP_DAY = concept("调休日", "当地日历说这天要补班", role=ConceptRole.DAY_TYPE, situation=SituationRule(B.CALENDAR_NOTE, value="补班"))
WITH_FAMILY = concept("和家人一起", "这件事是和家人一起做的", role=ConceptRole.OBJECT, situation=SituationRule(B.SUBJECT, value="家人B"))
AT_OFFICE = concept("在公司", "地点是公司", role=ConceptRole.OBJECT, situation=SituationRule(B.PLACE, value="公司"))
BOOKED = concept("约了球还没打", "约球的承诺还没结", role=ConceptRole.STATE, situation=SituationRule(B.OPEN_CLAIM, concept="打球"))
CRUNCH = concept("赶工中", "连着三天都在写代码", role=ConceptRole.DERIVED, situation=SituationRule(B.STREAK, concept="运动", days=3))
LAST_NIGHT = concept("昨晚晚睡", "昨天的就寝命中了晚睡", role=ConceptRole.DERIVED, situation=SituationRule(B.YESTERDAY, concept="晚睡", grade="重"))
SITUATIONS = ConceptSet((*CONCEPTS.values(), SWAP_DAY, WITH_FAMILY, AT_OFFICE, BOOKED, CRUNCH, LAST_NIGHT))


def names(hits: tuple[ConceptHit, ...]) -> set[str]:
    return {hit.concept for hit in hits}


def test_the_four_families_each_compute() -> None:
    """日型（周几 / 日历那句话）· 对象（同在 / 地点）· 开放承诺 · 派生（昨天）。"""

    inputs = SituationInputs(
        day=SATURDAY,
        calendar_note="补班日，按周一上班",
        subjects=("家人B",),
        place="公司",
        open_consequents={"打球"},
        history={FRIDAY: frozenset({("晚睡", "重")})},
    )
    assert names(situation_hits(SITUATIONS, inputs)) == {"周末", "调休日", "和家人一起", "在公司", "约了球还没打", "昨晚晚睡"}
    # 材料全换掉 → 一个都不成立（没有日历那句话时不拿"不知道"当"成立"）
    assert situation_hits(SITUATIONS, SituationInputs(day=FRIDAY)) == ()


def test_a_graded_derived_situation_needs_that_grade() -> None:
    """「昨晚晚睡」盯的是重档：昨天只是轻档就不算（档是剂量，不同档是不同的前件）。"""

    light = SituationInputs(day=SATURDAY, history={FRIDAY: frozenset({("晚睡", "轻")})})
    assert "昨晚晚睡" not in names(situation_hits(SITUATIONS, light))
    heavy = SituationInputs(day=SATURDAY, history={FRIDAY: frozenset({("晚睡", "重")})})
    assert "昨晚晚睡" in names(situation_hits(SITUATIONS, heavy))


def test_a_streak_counts_backwards_from_yesterday_and_needs_every_day() -> None:
    """连续三天 = 昨天、前天、前前天都命中；今天不算进去（拿今天解释今天）。"""

    def history(*offsets: int):
        return {SATURDAY - timedelta(days=offset): frozenset({("运动", None)}) for offset in offsets}

    assert "赶工中" in names(situation_hits(SITUATIONS, SituationInputs(day=SATURDAY, history=history(1, 2, 3))))
    # 中间断一天 → 不算连续
    assert "赶工中" not in names(situation_hits(SITUATIONS, SituationInputs(day=SATURDAY, history=history(1, 3, 4))))
    # 只有今天命中 → 不算（今天不进连续）
    assert "赶工中" not in names(situation_hits(SITUATIONS, SituationInputs(day=SATURDAY, history=history(0))))
    # 那几天还没映射过（没有签名）→ 算没命中，不算成立
    assert "赶工中" not in names(situation_hits(SITUATIONS, SituationInputs(day=SATURDAY, history=history(1, 2))))


def test_a_situation_concept_without_a_rule_never_hits() -> None:
    """「出差中」要等事实门接上数据源；在那之前它进得了概念集、也能被假设引用，只是分不出层。"""

    inputs = SituationInputs(day=SATURDAY, subjects=("出差中",), place="出差中", open_consequents={"出差中"})
    assert "出差中" not in names(situation_hits(SITUATIONS, inputs))


def test_the_day_signature_carries_grades_and_ancestors() -> None:
    """派生情境多半盯粗的那一层：命中「打球」要能满足盯着「运动」的那条说明。"""

    records = [record(FRIDAY, "打球", 18, 0, ConceptHit("打球")), record(FRIDAY, "就寝", 2, 10, ConceptHit("晚睡", "重"))]
    signature = day_signature(records, SITUATIONS)
    assert ("打球", None) in signature and ("运动", None) in signature and ("晚睡", "重") in signature
    # 概念集里没有的名字不进签名
    assert all(identity in SITUATIONS for identity, _grade in signature)


def test_how_many_days_of_history_to_prepare_comes_from_the_concepts() -> None:
    """要读几天由概念集里最长那条连续决定——不写死一个"读七天"。"""

    assert history_days(SITUATIONS, SATURDAY) == tuple(SATURDAY - timedelta(days=step) for step in (1, 2, 3))
    assert history_days(CONCEPTS, SATURDAY) == ()  # 一条派生说明都没有 → 一天都不用读


def test_a_rule_that_the_algorithm_cannot_compute_is_refused() -> None:
    with pytest.raises(SituationError, match="takes no weekdays"):
        SituationRule(B.PLACE, weekdays=(5,), value="公司")
    with pytest.raises(SituationError, match="watches no concept"):
        SituationRule(B.SUBJECT, value="家人B", concept="打球")
    with pytest.raises(SituationError, match="names the concept it watches"):
        SituationRule(B.OPEN_CLAIM)
