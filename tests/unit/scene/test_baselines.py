"""常态：只算被声明过的键、两个窗、环形中位数、不含当天、样本不够就不给、沿 parent 聚合、漂移是信号。"""

from __future__ import annotations

from datetime import timedelta

import pytest

from habitus.scene.concepts import BaselineStatistic, ConceptRole, ConceptSet
from habitus.scene.occurrences.baselines import baseline_table, declared_keys
from tests.unit.scene.concept_fixtures import BEDTIME_KEY, BEDTIME_KEY_ALL, concept
from tests.unit.scene.fixtures import DAY1
from tests.unit.scene.ledger_fixtures import CONCEPTS, record

#: 「就寝」的两个窗都声明一遍：近期由 LATE_RULE 声明（判据），历来由一条判据句声明（读漂移）。
DRIFT_READER = concept("作息漂移", "这个人的就寝时刻在往后移", role=ConceptRole.STATE, baseline_keys=(BEDTIME_KEY_ALL,))
WITH_BOTH_WINDOWS = ConceptSet((*CONCEPTS.values(), DRIFT_READER))
TODAY = DAY1 + timedelta(days=30)


def bedtimes(*offsets_and_times: tuple[int, int, int]) -> list:
    """(几天前, 时, 分) → 一条命中「就寝」的记录。"""

    return [record(TODAY - timedelta(days=days), "睡觉", hour, minute, "就寝") for days, hour, minute in offsets_and_times]


def test_only_declared_keys_are_computed() -> None:
    """概念声明了什么就算什么：不声明的概念不出现，也不去算（32 条上限、以及没人读的数字不该算）。"""

    assert [key.text for key in declared_keys(CONCEPTS)] == [BEDTIME_KEY]
    assert [key.text for key in declared_keys(WITH_BOTH_WINDOWS)] == [BEDTIME_KEY_ALL, BEDTIME_KEY]
    # 一条命中都没有 → 键在 missing 里，values 里没有；映射器据此把「晚睡」记成未决
    empty = baseline_table((), CONCEPTS, day=TODAY)
    assert empty.values == {} and empty.missing == (BEDTIME_KEY,)


def test_the_clock_median_wraps_midnight_and_the_two_windows_differ() -> None:
    """23:40 与 00:20 的常态是午夜前后，不是中午；近期窗只看最近 14 天，历来窗看全部。"""

    records = bedtimes(
        (1, 23, 40),
        (2, 0, 20),
        (3, 23, 50),  # 近期三条：午夜前后
        (20, 21, 0),
        (21, 21, 30),
        (22, 22, 0),  # 两周以前三条：九点多
    )
    snapshot = baseline_table(records, WITH_BOTH_WINDOWS, day=TODAY)
    assert snapshot.values[BEDTIME_KEY] == "23:50"  # 近期三条的环形中位数
    assert snapshot.samples[BEDTIME_KEY] == 3
    assert snapshot.values[BEDTIME_KEY_ALL] == "22:50"  # 六条一起：环上排序后中间两条 22:00 与 23:40 的中点
    assert snapshot.samples[BEDTIME_KEY_ALL] == 6


def test_today_is_not_part_of_its_own_baseline() -> None:
    """今天这一条要和"今天以前"比；把它算进去就是拿它和自己比，晚睡那条规则会被自己稀释。"""

    history = bedtimes((1, 23, 0), (2, 23, 10), (3, 23, 20))
    today = record(TODAY, "睡觉", 3, 0, "就寝")  # 今天凌晨三点才睡
    assert baseline_table([*history, today], CONCEPTS, day=TODAY).values[BEDTIME_KEY] == "23:10"


def test_a_key_with_too_few_samples_is_missing_rather_than_guessed() -> None:
    """两次谈不上"常态"：不给这个键（映射器记未决），不编一个值——编出来的值会让一条错判定看上去很确定。"""

    thin = baseline_table(bedtimes((1, 23, 0), (2, 23, 30)), CONCEPTS, day=TODAY)
    assert BEDTIME_KEY not in thin.values and thin.missing == (BEDTIME_KEY,)
    # 门槛可调：按 2 算就给
    loose = baseline_table(bedtimes((1, 23, 0), (2, 23, 30)), CONCEPTS, day=TODAY, min_samples=2)
    assert loose.values[BEDTIME_KEY] == "23:15"


def test_hits_on_a_child_concept_count_for_its_parent() -> None:
    """命中「打球」也算「运动」一次——否则上级概念永远攒不出常态（读侧沿 parent 链聚合的同一条规矩）。"""

    exercise_duration = concept("长时运动", "一次超过常态时长的运动", baseline_keys=("运动:usual_duration:recent",))
    concepts = ConceptSet((*CONCEPTS.values(), exercise_duration))
    records = [
        record(TODAY - timedelta(days=days), "打球", 18, 0, "打球", lasts_minutes=minutes)
        for days, minutes in ((1, 60), (2, 90), (3, 120))
    ]
    snapshot = baseline_table(records, concepts, day=TODAY)
    assert snapshot.values["运动:usual_duration:recent"] == "90"


def test_the_drift_between_the_windows_is_a_signal_not_a_criterion() -> None:
    """近期比历来晚多少 = "他在往后漂"，进 profile 的作息骨架、也是闭环的第二个触发源。"""

    records = bedtimes((1, 1, 0), (2, 1, 30), (3, 0, 30), (30, 22, 0), (31, 22, 30), (32, 23, 0))
    snapshot = baseline_table(records, WITH_BOTH_WINDOWS, day=TODAY)
    (drift,) = snapshot.drifts
    assert drift.concept == "就寝" and drift.statistic is BaselineStatistic.USUAL_START
    assert drift.minutes == pytest.approx(75.0)  # 近期 01:00 vs 历来 23:45
    assert drift.drifting and "近期比历来晚 75 分钟" in drift.render()
    assert snapshot.drifting == (drift,)
    # 只有一个窗算得出来时没有漂移可说（不拿"没有对照"当"没漂"）
    one_window = baseline_table(bedtimes((1, 23, 0), (2, 23, 10), (3, 23, 20)), CONCEPTS, day=TODAY)
    assert one_window.drifts == ()
