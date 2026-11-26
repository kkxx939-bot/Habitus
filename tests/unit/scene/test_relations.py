"""关系检验（语义树新方案 ``13`` ①③，第 4 步）：全部两两都量、病例交叉对照、四段跨度、剔除、加权 BH、按块看稳、区间够大。

数据是合成的：一个人 40 个工作日 9:00–18:00 在用电脑，按固定种子随机做几类事；其中埋一条真关系——
「讨论方案」之后 30 分钟内几乎总接着「修改代码」——看它能不能被找出来，而随机配的对找不出来。
"""

from __future__ import annotations

import math
from dataclasses import replace
from datetime import timedelta
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
from tests.unit.scene.concept_fixtures import base, cid
from tests.unit.scene.relation_fixtures import (
    CODE,
    CONCEPTS,
    CONFIG,
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
    assert later.p_value < CONFIG.thresholds.forward_alpha
    recent = tests.evaluate(key(DISCUSS, CODE), upward=True, last_blocks=8, purpose="maintain")
    assert recent.antecedents < everything.antecedents and recent.interval is not None


def test_only_present_time_counts_gaps_and_the_cutoff_make_the_outcome_unknown() -> None:
    """一天：9:00 调研、10:00 讨论方案、10:15–11:00 观测空白、14:00 提交推送、17:00 写文档；截止日是第二天。

    - 空白盖住的槽不算人在场；
    - 讨论方案之后 45 分钟碰到空白：这条链没整段在场，这一次结果未知（不是"没来"）；
    - 次日一段伸过截止日：未知；
    - 写文档是当天最后一件：之后没有在场的槽，"当天剩下的"未知（不是"它让一切都不发生"）；
    - 调研那一次的对照是同一天别的事做完的时刻：讨论方案之后碰到空白、写文档之后没有在场的槽，都不能比，只剩提交推送那一个。

    修改代码在这一天之前自然发生过一次（上周五），这一天对它才是"结果已知"（第一次之前一律不知道，裁定 104 第 1 条）；
    上周五只有这一条记录，没有前因的链，也进不了这一天的对照。
    """

    day = START
    records = (
        record(day - timedelta(days=3), 9, 0, "修改代码"),
        record(day, 9, 0, "调研"),
        record(day, 10, 0, "讨论方案"),
        record(day, 14, 0, "提交推送"),
        record(day, 17, 0, "写文档"),
    )
    gap = SeriesGap(
        uri="habitus://gaps/session/x", started_at=at(day, 10, 15), ended_at=at(day, 11, 0), kind=GapKind.UNOBSERVED
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


def test_a_concept_is_never_compared_with_itself_nor_with_its_parent_or_child() -> None:
    """自己与自己不比；上下级也不比（上下级之间不量关系，裁定 105 第 5 条）；兄弟、不相干的照常比。"""

    from dataclasses import replace

    from habitus.scene.concepts import ConceptSet
    from tests.unit.scene.concept_fixtures import BALL, RUNNING

    nested = ConceptSet([replace(BALL, classes=(BALL.name, RUNNING.name)), RUNNING])
    assert not comparable(CONCEPTS, cid("修改代码"), cid("修改代码"))
    assert comparable(CONCEPTS, cid("打球"), cid("跑步")) and comparable(CONCEPTS, "周末", cid("打球"))
    assert not comparable(nested, cid("打球"), cid("跑步")) and not comparable(nested, cid("跑步"), cid("打球"))


def test_the_configuration_refuses_contradictions() -> None:
    with pytest.raises(RelationConfigError):
        RelationConfig(
            slot_minutes=7, transition_window_slots=3, absence_typical_multiple=8.0, absence_min_gaps=2
        )
    with pytest.raises(RelationConfigError):
        RelationConfig(
            slot_minutes=15, transition_window_slots=96, absence_typical_multiple=8.0, absence_min_gaps=2
        )
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


def test_a_one_way_test_only_reaches_as_low_as_its_own_side() -> None:
    """只朝一侧检时最小可达 p 取这一侧：6 层、每层对照率 0.75，朝上全中也只有 0.75⁶ ≈ 0.178；朝下全阴是 0.25⁶。"""

    strata = [stratum(1, 2, 3, block=index) for index in range(6)]  # q = 0.75
    assert stats.minimum_one_sided(strata, upward=True) == pytest.approx(0.75**6)
    assert stats.minimum_one_sided(strata, upward=False) == pytest.approx(0.25**6)
    assert stats.minimum_one_sided([stratum(0, 0, 4)], upward=True) == 1.0


def test_a_chain_that_cannot_reach_alpha_upward_stays_out_of_the_family(monkeypatch) -> None:
    """同一条链只检"更多"：入族看朝上那一侧的最小可达 p。每层对照率 0.75 的 6 层朝上全中也只有 0.178，不进检验族
    （按两侧小的那侧读成 2.4e-4 会进族，把族撑大、压住别的真检验）。"""

    from habitus.scene.relations.measures import ChainMeasure

    series = planted()
    line = build_timelines(series, CONCEPTS, {}, CONFIG)["session"]
    tests = LaneTests(line, CONCEPTS, CONFIG, series.cutoff)
    chain = next(item for item in tests.keys() if item.segment is Segment.CHAIN and not item.condition)
    strata = tuple(stratum(1, 2, 3, block=index) for index in range(6))
    monkeypatch.setattr(tests, "keys", lambda: (chain,))
    monkeypatch.setattr(tests, "measure", lambda key, since=None: ChainMeasure(strata))
    (test,) = tests.night().tests
    assert test.minimum_p == pytest.approx(0.75**6) and test.verdict is Verdict.SPARSE


def _tarone_family_by_definition(minimum, weights, q):  # type: ignore[no-untyped-def]
    """Tarone 修剪照定义一个 K 一个 K 数（与实现对照用）。"""

    keys = sorted(minimum)
    scale = len(keys) / sum(max(weights.get(key, 1.0), 1e-12) for key in keys)
    ratio = {key: minimum[key] / (max(weights.get(key, 1.0), 1e-12) * scale) for key in keys}
    for limit in range(1, len(keys) + 1):
        members = [key for key in keys if ratio[key] <= q / limit]
        if len(members) <= limit:
            return frozenset(members)
    return frozenset()


def test_tarone_family_matches_its_definition_and_stays_fast_on_a_large_night() -> None:
    """检验族与"一个 K 一个 K 数"的定义逐项相同；四十万个键、四百个够得着的，一次不到五秒（原来每个 K 扫一遍全部键；本机约 0.2 秒，门槛给 CI 慢机器和覆盖率插桩留余量）。"""

    import random
    import time

    generator = random.Random(3)
    for _ in range(200):
        keys = [f"k{index}" for index in range(generator.randint(1, 60))]
        minimum = {key: generator.choice([1e-6, 1e-3, 0.01, 0.05, 0.2, 1.0, generator.random()]) for key in keys}
        weights = {key: generator.choice([0.5, 2.0, generator.random() + 0.1]) for key in keys if generator.random() < 0.5}
        p_values = {key: generator.random() for key in keys}
        family, _passed = stats.tarone_bh(p_values, minimum, weights, q=0.1)
        assert family == _tarone_family_by_definition(minimum, weights, 0.1)
    keys = [f"k{index}" for index in range(400_000)]
    minimum = {key: 1e-6 if index < 400 else 1.0 for index, key in enumerate(keys)}
    p_values = {key: generator.random() if index < 400 else 1.0 for index, key in enumerate(keys)}
    started = time.perf_counter()
    family, _passed = stats.tarone_bh(p_values, minimum, {}, q=0.05)
    elapsed = time.perf_counter() - started
    assert len(family) == 400
    assert elapsed < 5.0, elapsed


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


def test_a_pending_record_in_the_window_is_a_thing_that_is_not_the_consequent() -> None:
    """讨论方案 10:00 做完、10:20 一件「待定」（叫不出名）、之后没有修改代码：这一次是"修改代码没来"。「待定」是发生过的一件事，
    只是词表还没把它稳定成类（用户 2026-10-12）：它算人在场、算那天一件事做完的时刻（对照时刻），不让任何概念判不了。
    修改代码上周五自然发生过一次，这一天对它才是"结果已知"（第一次之前一律不知道，裁定 104 第 1 条）。"""

    day = START
    pending = replace(record(day, 10, 20, "调研", number=9), kind_token="s-待定", classified=False)
    records = (
        record(day - timedelta(days=3), 9, 0, "修改代码"),
        record(day, 9, 0, "调研"),
        record(day, 10, 0, "讨论方案"),
        pending,
        record(day, 14, 0, "提交推送"),
    )
    series = EventSeries(cutoff=day + timedelta(days=1), records=records)
    line = build_timelines(series, CONCEPTS, {}, CONFIG)["session"]
    tests = LaneTests(line, CONCEPTS, CONFIG, series.cutoff)
    assert not line.unknown.get(CODE)
    (stratum,) = tests.strata(key(DISCUSS, CODE))
    assert stratum.treated == 0
    # 对照时刻：同一天别的事做完的时刻——调研 9:00、待定 10:20（提交推送 14:00 是当天最后一件，之后没有在场的槽，不能比）
    assert stratum.controls == 2
    assert any(uri == pending.uri for _slot, uri in line.ends[day])


def test_a_daily_antecedent_at_random_times_still_has_same_slot_controls() -> None:
    """讨论方案每个工作日都做、时刻随机：别的周同周几同一钟点多半没在讨论，第二道对照有得比；
    每天 9:00 固定的调研在别的周同一钟点总在做，第二道对照为空 → 这条链样本不够（每天都来的是基线，不是关系）。"""

    series = planted()
    line = build_timelines(series, CONCEPTS, {}, CONFIG)["session"]
    tests = LaneTests(line, CONCEPTS, CONFIG, series.cutoff)
    from habitus.scene.relations.spans import Pairing

    second = tests.strata(key(DISCUSS, CODE), pairing=Pairing.SAME_SLOT)
    assert second and all(item.controls >= 1 for item in second)
    # 9:00 那条自己成链的（没和随机时刻的调研并成一条）：别的周同周几 9:00 总在调研，没有对照。
    # 合并按开始时刻的秒数（裁定 59）：9:52 起的随机调研离 9:00 超过 45 分钟，自成一条，不是这里要看的 9:00 那条。
    opening = [
        chain
        for chain in tests.chains(RESEARCH)
        if (chain.anchor.started_at.hour, chain.anchor.started_at.minute) == (9, 0) and len(chain.uris) == 1
    ]
    assert opening
    assert all(
        tests.controls.stratum(Segment.CHAIN, chain, CODE, antecedent=RESEARCH, pairing=Pairing.SAME_SLOT) is None
        for chain in opening
    )


def test_a_record_counts_only_toward_its_current_class() -> None:
    """一条记录只算它现在那个类的行为概念：词表迁移把它从「调研」重打成「修改代码」之后，盘上那份按旧编号写的命中记录
    不影响它算哪个概念（命中记录只带日历情境）。"""

    from habitus.series import SeriesRecord
    from tests.unit.scene.concept_fixtures import CODING
    from tests.unit.scene.hit_fixtures import record as hit_record
    from tests.unit.scene.hit_fixtures import uri_for

    day = START
    concepts = ConceptSet([CODING, base("调研")])
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
    stale = hit_record(day, "改一处", 10, 0, kind="调研")
    series = EventSeries(cutoff=day + timedelta(days=1), records=(series_record,))
    line = build_timelines(series, concepts, {series_record.uri: stale}, CONFIG)["session"]
    assert [mark.uri for mark in line.marks[cid("修改代码")]] == [series_record.uri]
    # 调研从没自然发生过，人在场的槽对它本来就是"不知道"（裁定 104 第 1 条），这里只看记录现在那个类：
    # 陈旧的命中记录既没给调研添记号，也没让修改代码变成看不到
    assert not line.marks.get(cid("调研")) and not line.unknown.get(cid("修改代码"))


def test_a_record_past_midnight_ends_in_the_last_slot_of_its_day() -> None:
    """23:00 开始、次日 00:20 最后所见：人在场按日子算，做完的那一槽截在当天最后一槽（第 95 槽）。"""

    day = START
    late = replace(record(day, 23, 0, "调研"), last_observed_at=at(day + timedelta(days=1), 0, 20))
    line = build_timelines(EventSeries(cutoff=day + timedelta(days=2), records=(late,)), CONCEPTS, {}, CONFIG)["session"]
    ((ended, _uri),) = line.ends[day]
    assert ended == day.toordinal() * 96 + 95


def test_only_the_last_start_of_a_merged_run_ends_its_chain() -> None:
    """同一类 17:00、17:20 两次（合并窗口内连着）：那一天"做完"的时刻只记最后那条，前一条后面还接着。"""

    day = START
    first, second = record(day, 17, 0, "调研", number=1), record(day, 17, 20, "调研", number=2)
    line = build_timelines(EventSeries(cutoff=day + timedelta(days=1), records=(first, second)), CONCEPTS, {}, CONFIG)[
        "session"
    ]
    assert [uri for _ended, uri in line.ends[day]] == [second.uri]


def test_two_days_2_7_windows_six_days_apart_do_not_merge() -> None:
    """后几段窗口重叠的再合一次：第 2–7 天的窗口，隔 5 天的两条链窗口还挨着第 7 天、合成一条；隔 6 天的不重叠、各算各的。"""

    from habitus.scene.relations import chains_of, merge_for

    def chains(gap: int):  # type: ignore[no-untyped-def]
        days = (START, START + timedelta(days=gap))
        series = EventSeries(cutoff=days[-1] + timedelta(days=10), records=tuple(record(d, 9, 0, "调研") for d in days))
        line = build_timelines(series, CONCEPTS, {}, CONFIG)["session"]
        return merge_for(Segment.DAYS_2_7, chains_of(line, RESEARCH, CONFIG.transition_window_slots))

    assert len(chains(5)) == 1 and len(chains(6)) == 2
