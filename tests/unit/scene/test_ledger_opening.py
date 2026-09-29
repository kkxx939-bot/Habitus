"""④ 开承诺：前件集合全部命中才开、锚在最后开始的行为概念、档要对上、祖先聚合、同一次机会去重、幂等、机会快照。"""

from __future__ import annotations

import pytest

from habitus.scene.hypotheses import Antecedent
from habitus.scene.ledger import LedgerConfig, LedgerStore, open_claims_for_day
from habitus.scene.occurrences import ConceptHit, ConceptHitStore
from tests.unit.scene.fixtures import DAY1, DAY2, at
from tests.unit.scene.ledger_fixtures import (
    BALL_LATE_TO_BREAKFAST,
    CONCEPTS,
    EXERCISE_TO_COFFEE,
    LATE_TO_BREAKFAST,
    LATE_TRAVEL_TO_BREAKFAST,
    NOW,
    TableOpportunities,
    hypothesis,
    record,
)


def stores(tmp_path) -> tuple[ConceptHitStore, LedgerStore]:
    return ConceptHitStore(tmp_path / "scene"), LedgerStore(tmp_path / "scene")


def test_a_single_antecedent_opens_one_claim_anchored_at_its_start_with_the_opportunity_snapshot(tmp_path) -> None:
    hits, ledger = stores(tmp_path)
    late = record(DAY1, "就寝", 2, 10, ConceptHit("晚睡", "轻"), situations=("周末",), lasts_minutes=7 * 60)  # 被认到 09:10：锚不受影响
    hits.write(late)
    hits.write(record(DAY1, "早餐", 7, 30, "早餐"))
    provider = TableOpportunities()

    report = open_claims_for_day(DAY1, hypotheses=(LATE_TO_BREAKFAST,), concepts=CONCEPTS, hits=hits, ledger=ledger, opportunities=provider, now=NOW)

    assert (report.opened, report.already_open, report.overlapping, report.without_control) == (1, 0, 0, 0)
    (claim,) = ledger.claims_for(LATE_TO_BREAKFAST.identity)
    assert claim.anchor == at(DAY1, 2, 10)
    assert claim.antecedent_hits == (ConceptHit("晚睡", "轻"),) and claim.antecedent_uris == (late.occurrence_uri,)
    assert claim.situation_snapshot == ("周末",) and claim.hypothesis_fingerprint == LATE_TO_BREAKFAST.fingerprint
    assert claim.control is not None and len(claim.control.opportunities) == 16 and claim.control.generation == "gen-1"
    assert claim.control.at(1).at == at(DAY1, 7, 45) and claim.control.at(1).probability == 0.88  # type: ignore[union-attr]  # 七h 第 2 步的 88%
    assert provider.requests[0].consequent == "早餐" and provider.requests[0].anchor == claim.anchor and provider.requests[0].count == 16

    # 幂等：重跑同一天不重开，也不比对内容（快照事后不改）。
    report = open_claims_for_day(DAY1, hypotheses=(LATE_TO_BREAKFAST,), concepts=CONCEPTS, hits=hits, ledger=ledger, opportunities=TableOpportunities({}), now=NOW)
    assert (report.opened, report.already_open) == (0, 1) and len(ledger.claims_for(LATE_TO_BREAKFAST.identity)) == 1


def test_grades_and_situations_must_match_the_antecedent_set(tmp_path) -> None:
    """「晚睡@重 + 出差中」：轻档夜晚不开；重档但不在出差不开；重档且出差 → 开。"""

    hits, ledger = stores(tmp_path)
    hits.write(record(DAY1, "就寝", 2, 10, ConceptHit("晚睡", "轻"), situations=("出差中",)))
    hits.write(record(DAY1, "午睡", 14, 0, ConceptHit("晚睡", "重")))
    hits.write(record(DAY1, "再睡", 23, 50, ConceptHit("晚睡", "重"), situations=("出差中", "周末")))
    report = open_claims_for_day(DAY1, hypotheses=(LATE_TRAVEL_TO_BREAKFAST,), concepts=CONCEPTS, hits=hits, ledger=ledger, opportunities=TableOpportunities(), now=NOW)
    assert report.opened == 1
    (claim,) = ledger.claims_for(LATE_TRAVEL_TO_BREAKFAST.identity)
    assert claim.anchor == at(DAY1, 23, 50)
    assert claim.antecedent_hits == (ConceptHit("出差中"), ConceptHit("晚睡", "重"))
    assert claim.control is not None and claim.control.at(1).at == at(DAY2, 7, 45)  # type: ignore[union-attr]  # 23:50 的锚：下一个早餐机会是次日早上


def test_a_multi_behaviour_set_is_anchored_at_the_last_element_and_gathers_within_the_horizon(tmp_path) -> None:
    hits, ledger = stores(tmp_path)
    ball = record(DAY1, "打球", 19, 0, "打球")
    late = record(DAY2, "就寝", 2, 10, ConceptHit("晚睡", "轻"))  # 7 小时后
    hits.write(ball)
    hits.write(late)
    # 打球那天：晚睡还没到，凑不齐。
    report = open_claims_for_day(DAY1, hypotheses=(BALL_LATE_TO_BREAKFAST,), concepts=CONCEPTS, hits=hits, ledger=ledger, opportunities=TableOpportunities(), now=NOW)
    assert report.opened == 0
    # 就寝那天：打球在 24 小时内 → 凑齐，锚在最后开始的「晚睡」，出力的两条都记下。
    report = open_claims_for_day(DAY2, hypotheses=(BALL_LATE_TO_BREAKFAST,), concepts=CONCEPTS, hits=hits, ledger=ledger, opportunities=TableOpportunities(), now=NOW)
    assert report.opened == 1
    (claim,) = ledger.claims_for(BALL_LATE_TO_BREAKFAST.identity)
    assert claim.anchor == at(DAY2, 2, 10) and set(claim.antecedent_uris) == {ball.occurrence_uri, late.occurrence_uri}
    assert claim.antecedent_hits == (ConceptHit("打球"), ConceptHit("晚睡", "轻"))
    # 凑齐时长是保护闸：缩到 5 小时就凑不齐。
    ledger2 = LedgerStore(tmp_path / "scene2")
    report = open_claims_for_day(
        DAY2, hypotheses=(BALL_LATE_TO_BREAKFAST,), concepts=CONCEPTS, hits=hits, ledger=ledger2, opportunities=TableOpportunities(), now=NOW, config=LedgerConfig(gathering_hours=5)
    )
    assert report.opened == 0


def test_ancestor_concepts_match_through_read_time_aggregation(tmp_path) -> None:
    """「运动 → 咖啡」：记录上只有叶子「打球」，祖先「运动」由 parent 链聚合出来；祖先不带档。咖啡一天三个机会。"""

    hits, ledger = stores(tmp_path)
    hits.write(record(DAY1, "打球", 19, 0, "打球"))
    report = open_claims_for_day(DAY1, hypotheses=(EXERCISE_TO_COFFEE,), concepts=CONCEPTS, hits=hits, ledger=ledger, opportunities=TableOpportunities(), now=NOW)
    assert report.opened == 1
    (claim,) = ledger.claims_for(EXERCISE_TO_COFFEE.identity)
    assert claim.antecedent_hits == (ConceptHit("运动"),) and claim.control is not None
    assert [item.at for item in claim.control.opportunities[:4]] == [at(DAY1, 20, 0), at(DAY2, 8, 30), at(DAY2, 14, 0), at(DAY2, 20, 0)]
    assert claim.control.expected_count(1, 3) == pytest.approx(0.3 + 0.5 + 0.4)


def test_two_triggers_awaiting_the_same_opportunity_collapse_to_one_claim(tmp_path) -> None:
    """同一晚两条「晚睡」记录（02:10 与 03:00）等的都是同一个早餐机会 → 只开一条；隔天的另开。"""

    hits, ledger = stores(tmp_path)
    hits.write(record(DAY1, "就寝", 2, 10, ConceptHit("晚睡", "轻")))
    hits.write(record(DAY1, "又醒了再睡", 3, 0, ConceptHit("晚睡", "轻")))
    hits.write(record(DAY2, "就寝", 1, 0, ConceptHit("晚睡", "轻")))  # 隔天：下一个早餐机会是另一个
    report = open_claims_for_day(DAY1, hypotheses=(LATE_TO_BREAKFAST,), concepts=CONCEPTS, hits=hits, ledger=ledger, opportunities=TableOpportunities(), now=NOW)
    assert (report.opened, report.overlapping) == (1, 1)
    report = open_claims_for_day(DAY2, hypotheses=(LATE_TO_BREAKFAST,), concepts=CONCEPTS, hits=hits, ledger=ledger, opportunities=TableOpportunities(), now=NOW)
    assert (report.opened, report.overlapping) == (1, 0)
    assert [claim.anchor for claim in ledger.claims_for(LATE_TO_BREAKFAST.identity)] == [at(DAY1, 2, 10), at(DAY2, 1, 0)]


def test_a_claim_opens_without_control_when_the_tree_has_nothing_and_skips_unknown_concepts(tmp_path) -> None:
    hits, ledger = stores(tmp_path)
    hits.write(record(DAY1, "就寝", 2, 10, ConceptHit("晚睡", "轻")))
    ghost = hypothesis("晚睡", consequent="咖啡", aspect=EXERCISE_TO_COFFEE.aspect, direction=EXERCISE_TO_COFFEE.direction, type_prior=None, horizon=3, note="x")
    report = open_claims_for_day(DAY1, hypotheses=(LATE_TO_BREAKFAST, ghost), concepts=CONCEPTS, hits=hits, ledger=ledger, opportunities=TableOpportunities({}), now=NOW)
    assert (report.opened, report.without_control) == (2, 2)
    assert all(claim.control is None for claim in ledger.claims_for(LATE_TO_BREAKFAST.identity))
    # 没有快照的承诺不参与"同一次机会"去重：只按触发去重。
    hits.write(record(DAY1, "又睡", 3, 0, ConceptHit("晚睡", "轻")))
    report = open_claims_for_day(DAY1, hypotheses=(LATE_TO_BREAKFAST,), concepts=CONCEPTS, hits=hits, ledger=ledger, opportunities=TableOpportunities({}), now=NOW)
    assert (report.opened, report.already_open, report.overlapping) == (1, 1, 0)


def test_hypotheses_whose_concepts_are_gone_are_reported_not_silently_skipped(tmp_path) -> None:
    from habitus.scene.concepts import ConceptSet

    hits, ledger = stores(tmp_path)
    hits.write(record(DAY1, "就寝", 2, 10, ConceptHit("晚睡", "轻")))
    without = ConceptSet([CONCEPTS[identity] for identity in CONCEPTS if identity != "早餐"])
    report = open_claims_for_day(DAY1, hypotheses=(LATE_TO_BREAKFAST,), concepts=without, hits=hits, ledger=ledger, opportunities=TableOpportunities(), now=NOW)
    assert (report.opened, report.unmappable) == (0, (LATE_TO_BREAKFAST.identity,))


def test_ledger_config_guards_its_numbers() -> None:
    with pytest.raises(ValueError, match="censor_after cannot exceed"):
        LedgerConfig(snapshot_opportunities=4, censor_after=10)
    with pytest.raises(ValueError, match="opportunity_coverage"):
        LedgerConfig(opportunity_coverage=0.0)
    assert LedgerConfig().snapshot_opportunities == 16 and LedgerConfig().censor_after == 10
    assert Antecedent("晚睡").identity == "晚睡"


def test_one_unusable_snapshot_does_not_abort_the_whole_night(tmp_path) -> None:
    """机会口给了一份账本收不下的快照（第一个机会在锚之前就结束了）：那条承诺按"要不到对照"开，并报进
    ``unusable_snapshots``；不能让一条坏快照把整晚所有假设的承诺都掀掉（十 ④"对照要不到 → control=null 照样开"）。"""

    from datetime import timedelta

    from habitus.scene.ledger import Opportunity, OpportunityRequest, OpportunitySnapshot, WindowSpan

    class StaleOpportunities:
        def opportunities(self, request: OpportunityRequest) -> OpportunitySnapshot:
            centre = request.anchor - timedelta(hours=3)
            return OpportunitySnapshot("gen-1", (Opportunity(centre, WindowSpan(centre - timedelta(minutes=30), centre + timedelta(minutes=30)), 0.5),))

    hits, ledger = stores(tmp_path)
    hits.write(record(DAY1, "就寝", 2, 10, ConceptHit("晚睡", "轻")))
    report = open_claims_for_day(DAY1, hypotheses=(LATE_TO_BREAKFAST,), concepts=CONCEPTS, hits=hits, ledger=ledger, opportunities=StaleOpportunities(), now=NOW)
    assert (report.opened, report.without_control) == (1, 1)
    assert report.unusable_snapshots and "has not ended by the anchor" in report.unusable_snapshots[0]
    (claim,) = ledger.claims_for(LATE_TO_BREAKFAST.identity)
    assert claim.control is None  # 机会是事实，照样开；强度读时算不出而已


def test_the_snapshot_is_long_enough_for_what_the_hypothesis_measures(tmp_path) -> None:
    """时刻/次数量第 k 次机会时，要的机会数不能少于 k（少了那条账永远结不了，评审 B5）。"""

    hits, ledger = stores(tmp_path)
    hits.write(record(DAY1, "就寝", 2, 10, ConceptHit("晚睡", "轻")))
    far = hypothesis("晚睡", consequent="咖啡", aspect=EXERCISE_TO_COFFEE.aspect, direction=EXERCISE_TO_COFFEE.direction, type_prior=None, expected_at=12, horizon=5, note="x")
    provider = TableOpportunities()
    open_claims_for_day(DAY1, hypotheses=(far,), concepts=CONCEPTS, hits=hits, ledger=ledger, opportunities=provider, now=NOW, config=LedgerConfig(snapshot_opportunities=8, censor_after=8))
    assert provider.requests[-1].count == 16  # max(8, 12 + 5 − 1)
    (claim,) = ledger.claims_for(far.identity)
    assert claim.control is not None and claim.control.at(16) is not None
