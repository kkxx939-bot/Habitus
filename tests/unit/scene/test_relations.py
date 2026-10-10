"""关系检验（语义树新方案 ``13`` ①③，第 4 步）：全部两两都量、病例交叉对照、四段跨度、剔除、加权 BH、按块看稳、区间够大。

数据是合成的：一个人 40 个工作日 9:00–18:00 在用电脑，按固定种子随机做几类事；其中埋一条真关系——
「讨论方案」之后 30 分钟内几乎总接着「修改代码」——看它能不能被找出来，而随机配的对找不出来。
"""

from __future__ import annotations

import math
from dataclasses import replace
from datetime import datetime, timedelta
from itertools import product

import pytest

from habitus.scene.concepts import ConceptSet
from habitus.scene.relations import (
    LaneTests,
    RelationConfig,
    RelationConfigError,
    RelationKey,
    RelationTest,
    RelationThresholds,
    Segment,
    Verdict,
    build_timelines,
    comparable,
    examine_night,
    stats,
)
from habitus.scene.relations.spans import Chain, Stratum
from habitus.scene.relations.timeline import Mark
from habitus.series import EventSeries, GapKind, SeriesGap
from tests.unit.kind_ids import kind_id
from tests.unit.scene.concept_fixtures import EXERCISE, REWORK, base, cid
from tests.unit.scene.relation_fixtures import (
    CODE,
    CONCEPTS,
    CONFIG,
    CST,
    DISCUSS,
    DOCS,
    PUSH,
    RESEARCH,
    START,
    at,
    key,
    planted,
    record,
    workdays,
)


def tests_by_key(series: EventSeries) -> dict[RelationKey, RelationTest]:
    (night,) = examine_night(series, CONCEPTS, {}, CONFIG).values()
    return {test.key: test for test in night.tests}


def test_the_planted_relation_is_found_and_passes_all_three_gates() -> None:
    found = tests_by_key(planted())
    chain = found[key(DISCUSS, CODE)]
    assert chain.verdict is Verdict.SIGNIFICANT and chain.upward
    assert chain.effect.treated_rate > 0.85 and chain.effect.control_rate < 0.4
    assert chain.stable and chain.large_enough
    assert chain.interval is not None and chain.interval[0] > CONFIG.thresholds.min_absolute_lift
    assert chain.excess_share is not None and 0.0 < chain.excess_share < 1.0
    # 反过来（修改代码 → 讨论方案）没有埋：不显著
    assert found[key(CODE, DISCUSS)].verdict is not Verdict.SIGNIFICANT


@pytest.mark.parametrize("seed", [7, 11, 23])
def test_random_pairs_and_fixed_routines_stay_quiet(seed: int) -> None:
    """没埋关系的数据上一条都不该过第 1 道；9:00 开工、17:50 收工这两件固定作息也不该被读成"被别的事带出来 / 压下去"
    （对照只量人在场的时间、当天剩下的按同一钟点比）。"""

    found = tests_by_key(planted(seed=seed, follow=0.0))
    noisy = [
        test.key
        for test in found.values()
        if test.verdict is Verdict.SIGNIFICANT and (test.key.antecedent, test.key.consequent) != (DISCUSS, CODE)
    ]
    assert noisy == []


def test_without_the_planted_link_nothing_is_found_in_the_chain_segment() -> None:
    found = tests_by_key(planted(follow=0.0))
    assert found[key(DISCUSS, CODE)].verdict is not Verdict.SIGNIFICANT


def test_a_rare_antecedent_is_sparse_with_no_interval_and_says_how_far_it_is() -> None:
    series = planted(days=6)
    (night,) = examine_night(series, CONCEPTS, {}, CONFIG).values()
    push = [test for test in night.tests if test.key.antecedent == PUSH]
    assert push and all(test.verdict is Verdict.SPARSE for test in push)
    assert all(test.interval is None for test in push)
    assert any(test.more_needed for test in push)
    assert night.family < len(night.tests)  # 剔除真的发生了


def test_the_night_is_reproducible() -> None:
    series = planted()
    assert examine_night(series, CONCEPTS, {}, CONFIG) == examine_night(series, CONCEPTS, {}, CONFIG)


def test_forward_validation_reads_only_the_new_data() -> None:
    series = planted()
    line = build_timelines(series, CONCEPTS, {}, CONFIG)["session"]
    tests = LaneTests(line, CONCEPTS, CONFIG, series.cutoff)
    since = workdays(40)[30]
    later = tests.evaluate(key(DISCUSS, CODE), upward=True, since=since, purpose="forward")
    everything = tests.evaluate(key(DISCUSS, CODE), upward=True, purpose="forward")
    assert 0 < later.antecedents < everything.antecedents
    assert later.antecedents >= CONFIG.thresholds.forward_min_antecedents and later.p_value < CONFIG.thresholds.forward_alpha
    recent = tests.evaluate(key(DISCUSS, CODE), upward=True, last_blocks=CONFIG.thresholds.maintenance_blocks, purpose="maintain")
    assert recent.antecedents < everything.antecedents and recent.interval is not None


def test_only_present_time_counts_gaps_and_the_cutoff_make_the_outcome_unknown() -> None:
    """一天：9:00 调研、10:00 讨论方案、10:15–11:00 观测空白、14:00 提交推送、17:00 写文档；截止日是第二天。

    - 空白盖住的槽不算人在场；
    - 讨论方案之后 45 分钟碰到空白：这条链没整段在场，这一次结果未知（不是"没来"）；
    - 次日一段伸过截止日：未知；
    - 写文档是当天最后一件：之后没有在场的槽，"当天剩下的"未知（不是"它让一切都不发生"）；
    - 调研那一次的对照是同一天别的事做完的时刻：讨论方案之后碰到空白、写文档之后没有在场的槽，都不能比，只剩提交推送那一个。
    """

    day = START
    records = (
        record(day, 9, 0, "调研"),
        record(day, 10, 0, "讨论方案"),
        record(day, 14, 0, "提交推送"),
        record(day, 17, 0, "写文档"),
    )
    gap = SeriesGap(
        uri="behavior://gaps/x", started_at=at(day, 10, 15), ended_at=at(day, 11, 0), kind=GapKind.UNOBSERVED
    )
    series = EventSeries(cutoff=day + timedelta(days=1), records=records, gaps=(gap,))
    line = build_timelines(series, CONCEPTS, {}, CONFIG)["session"]
    assert {41, 42, 43}.isdisjoint(line.present[day]) and {40, 44, 68}.issubset(line.present[day])
    tests = LaneTests(line, CONCEPTS, CONFIG, series.cutoff)
    assert tests.strata(key(DISCUSS, CODE)) == ()
    assert tests.strata(key(DISCUSS, CODE, Segment.NEXT_DAY)) == ()
    assert tests.strata(key(DOCS, CODE, Segment.REST_OF_DAY)) == ()
    morning = tests.strata(key(RESEARCH, CODE))
    assert len(morning) == 1 and morning[0].treated == 0 and morning[0].controls == 1


def test_pairs_that_read_the_same_records_are_never_compared() -> None:
    concepts = ConceptSet([base("修改代码"), REWORK, base("打球"), base("跑步"), EXERCISE])
    assert not comparable(concepts, cid("修改代码"), cid("修改代码"))
    assert not comparable(concepts, "返工", cid("修改代码")) and not comparable(concepts, cid("修改代码"), "返工")
    assert not comparable(concepts, "运动", cid("打球"))
    assert comparable(concepts, cid("打球"), cid("跑步")) and comparable(concepts, "返工", "运动")


def test_the_configuration_refuses_contradictions() -> None:
    with pytest.raises(RelationConfigError):
        RelationConfig(slot_minutes=7, transition_window_slots=3, recurrence_window_days=90)
    with pytest.raises(RelationConfigError):
        RelationConfig(slot_minutes=15, transition_window_slots=96, recurrence_window_days=90)
    with pytest.raises(RelationConfigError):
        RelationThresholds(fdr=1.0)



# --- 统计 ----------------------------------------------------------------------------------------


def stratum(treated: int, positives: int, controls: int, block: int = 0) -> Stratum:
    mark = Mark(slot=0, started_at=at(START, 9, 0), uri=f"u{block}{treated}{positives}{controls}")
    return Stratum(
        chain=Chain(anchor=mark, uris=frozenset({mark.uri}), goals=frozenset(), end=mark.slot),
        treated=treated,
        positives=positives,
        controls=controls,
        block=block,
    )


def test_the_exact_tails_match_brute_force() -> None:
    strata = [stratum(1, 2, 5), stratum(0, 1, 3), stratum(1, 0, 4), stratum(1, 3, 3)]
    upper, lower = stats.tails(strata, exact_limit=100)
    probabilities = [item.null_probability for item in strata]
    observed = sum(item.treated for item in strata)
    brute_upper = brute_lower = 0.0
    for outcome in product((0, 1), repeat=len(strata)):
        mass = math.prod(p if y else 1.0 - p for p, y in zip(probabilities, outcome, strict=True))
        brute_upper += mass if sum(outcome) >= observed else 0.0
        brute_lower += mass if sum(outcome) <= observed else 0.0
    assert upper == pytest.approx(brute_upper) and lower == pytest.approx(brute_lower)
    # 正态近似与精确值同一量级
    approx_upper, _ = stats.tails(strata * 30, exact_limit=10)
    exact_upper, _ = stats.tails(strata * 30, exact_limit=1000)
    assert approx_upper == pytest.approx(exact_upper, abs=0.02)


def test_the_minimum_attainable_p_is_the_all_one_way_probability() -> None:
    strata = [stratum(1, 1, 3), stratum(0, 1, 1)]  # q = 0.5, 0.5
    assert stats.minimum_p(strata) == pytest.approx(2 * 0.25)
    assert stats.minimum_p([stratum(0, 0, 4)]) == 1.0  # 对照里一个阳性都没有、自己也没有：没信息


def test_tarone_prunes_hopeless_tests_and_bh_controls_the_rest() -> None:
    p_values = {"a": 0.001, "b": 0.02, "c": 0.5, "hopeless": 0.9}
    minimum = {"a": 1e-6, "b": 1e-6, "c": 1e-6, "hopeless": 0.5}
    family, passed = stats.tarone_bh(p_values, minimum, {}, q=0.1)
    assert family == {"a", "b", "c"} and passed == {"a", "b"}
    # 先验权重：给 b 低权重，它就过不了
    _, weighted = stats.tarone_bh(p_values, minimum, {"b": 0.1, "a": 2.0, "c": 2.0}, q=0.1)
    assert weighted == {"a"}


def test_heterogeneity_and_leave_one_block_out() -> None:
    steady = [stratum(1, 1, 9, block) for block in range(4) for _ in range(5)]
    assert stats.heterogeneity(steady) > 0.5 and stats.leave_one_block_out(steady)
    flipping = [stratum(1, 1, 9, 0) for _ in range(10)] + [stratum(0, 9, 9, 1) for _ in range(10)]
    assert stats.heterogeneity(flipping) < 0.01 and not stats.leave_one_block_out(flipping)
    assert not stats.leave_one_block_out(steady[:5])  # 只有一块：谈不上稳


def test_the_chi_square_tail_matches_known_values() -> None:
    assert stats._chi2_sf(3.841458820694124, 1) == pytest.approx(0.05, abs=1e-6)
    assert stats._chi2_sf(11.070497693516351, 5) == pytest.approx(0.05, abs=1e-6)
    assert stats._chi2_sf(0.0, 3) == 1.0


def test_a_pending_record_in_the_window_makes_the_outcome_unknown() -> None:
    """讨论方案 10:00 做完、10:20 一件「待定」（叫不出名）、之后没有修改代码：这一次结果未知，不算"后果没来"（E13，裁定 21-3）。"""

    day = START
    pending = replace(record(day, 10, 20, "调研", number=9), kind_token="s-待定", classified=False)
    records = (record(day, 9, 0, "调研"), record(day, 10, 0, "讨论方案"), pending, record(day, 14, 0, "提交推送"))
    series = EventSeries(cutoff=day + timedelta(days=1), records=records)
    line = build_timelines(series, CONCEPTS, {}, CONFIG)["session"]
    tests = LaneTests(line, CONCEPTS, CONFIG, series.cutoff)
    assert tests.strata(key(DISCUSS, CODE)) == ()
    assert any(slot == line.day_start(day) + 41 for slot in line.unknown[CODE])


def test_a_hit_record_mapped_under_another_class_counts_as_unmapped() -> None:
    """映射时这条还是「调研」，之后词表拆改把它重打成「修改代码」：那份命中不是现在这个类的，细分概念「返工」当没映射过（判不了），
    不把旧类的细分命中算到新类头上（E3）。"""

    from habitus.series import SeriesRecord
    from tests.unit.scene.concept_fixtures import CODING
    from tests.unit.scene.hit_fixtures import record as hit_record
    from tests.unit.scene.hit_fixtures import uri_for

    day = START
    concepts = ConceptSet([CODING, REWORK, base("调研")])
    started = at(day, 10, 0)
    series_record = SeriesRecord(
        uri=uri_for(day, "改一处", 10, 0),
        name="改一处",
        day=day,
        started_at=started,
        last_observed_at=started + timedelta(minutes=10),
        kind_token=kind_id("修改代码"),
        lane="session",
        classified=True,
        reminded=False,
        summary="改一处",
    )
    stale = hit_record(day, "改一处", 10, 0, "返工", kind="调研")
    series = EventSeries(cutoff=day + timedelta(days=1), records=(series_record,))
    line = build_timelines(series, concepts, {series_record.uri: stale}, CONFIG)["session"]
    assert not line.marks.get("返工")
    assert line.unknown.get("返工")


def _hits_for(series: EventSeries, blind_every: int) -> dict:
    """修改代码的每条记录都映射过：每 ``blind_every`` 条里有一条「返工」判不了，其余判了"不是返工"。"""

    from habitus.scene.occurrences import ConceptHits
    from habitus.scene.occurrences.model import UnresolvedReason
    from tests.unit.scene.hit_fixtures import uri_for

    found = {}
    coding = [item for item in series.records if item.kind_token == CODE]
    for index, item in enumerate(coding):
        blind = index % blind_every == 0
        found[item.uri] = ConceptHits(
            occurrence_uri=uri_for(item.day, item.name, item.started_at.hour, item.started_at.minute),
            kind_token=item.kind_token,
            last_observed_at=item.last_observed_at,
            hits=(),
            situation_hits=(),
            situations_checked=(),
            unresolved={"返工": UnresolvedReason.MODEL_UNSEEN} if blind else {},
            baseline_snapshot={},
            mapper="test",
            mapped_at=datetime(2026, 10, 1, tzinfo=CST),  # 晚于「返工」的创建：映射时已经有这个细分概念
            lane="session",
            classified=True,
        )
    return found


def test_a_refinement_consequent_that_is_mostly_blind_stays_out_of_the_family() -> None:
    """「返工」当后果，修改代码的记录八成判不了：上下界宽到什么都说明不了，不进检验族（裁定 27 第 6 条）。"""

    concepts = ConceptSet([*CONCEPTS.values(), REWORK])
    series = planted(40)
    line = build_timelines(series, concepts, _hits_for(series, blind_every=1), CONFIG)["session"]
    tests = LaneTests(line, concepts, CONFIG, series.cutoff)
    night = tests.night()
    rework = [test for test in night.tests if test.key == key(DISCUSS, "返工")]
    assert rework and rework[0].verdict is Verdict.SPARSE


def test_a_refinement_consequent_is_judged_on_both_bounds() -> None:
    """一成判不了：照常检验，判不了的窗口按"当没出现 / 当出现了"各算一遍。埋的是「讨论方案 → 修改代码」、没有一条是返工：
    当没出现那一头读不出"更多"，这条不显著——不再把"窗口里有修改代码"读成"返工更多"。"""

    concepts = ConceptSet([*CONCEPTS.values(), REWORK])
    series = planted(40)
    line = build_timelines(series, concepts, _hits_for(series, blind_every=10), CONFIG)["session"]
    tests = LaneTests(line, concepts, CONFIG, series.cutoff)
    (rework,) = [test for test in tests.night().tests if test.key == key(DISCUSS, "返工")]
    assert rework.verdict is not Verdict.SIGNIFICANT
