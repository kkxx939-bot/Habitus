"""带条件关系的开测门（裁定 60 第 9 条、L10.17、裁定 94 第 5 条）：最小可达 p 条件下按实测层、条件外按假设估，不看条件外实测的
对照率；条件外当天从不出现 B、两边对照都没有 B，都不再被判成"永远不显著"；每边至少 3 次；前向验证用的单侧版同一口径。
层的数字取自 v0.11 E5（看作业 → 打印｜之前发过卷）。"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable
from types import SimpleNamespace

import pytest

from habitus.scene.relations import stats
from habitus.scene.relations.conditions import MIN_PER_SIDE, ModeratedMeasure, ready_to_test, split
from habitus.scene.relations.measures import ChainMeasure
from habitus.scene.relations.spans import Stratum
from tests.unit.scene.relation_fixtures import CONFIG

ALPHA = CONFIG.thresholds.forward_alpha
EXACT = CONFIG.thresholds.exact_strata_limit


def grouped(inside: list[tuple[int, int, int]], outside: list[tuple[int, int, int]]) -> ModeratedMeasure:
    """（A 那一刻来没来, 对照里来了几个, 对照几个）→ 按条件分好的两组，每层一个块。"""

    strata = [
        Stratum(chain=SimpleNamespace(held=held), treated=treated, positives=positives, controls=controls, block=index)  # type: ignore[arg-type]
        for index, (held, (treated, positives, controls)) in enumerate(
            [(True, item) for item in inside] + [(False, item) for item in outside]
        )
    ]
    return split(ChainMeasure(tuple(strata)), [lambda chain: chain.held])  # type: ignore[attr-defined]


def observed_p(measure: ModeratedMeasure) -> float:
    return stats.two_sided(*measure.tails(exact_limit=EXACT))


E5_INSIDE = [(1, 1, 22), (1, 1, 11), (1, 1, 11), (1, 2, 22)]
E5_OUTSIDE = [(0, 0, 18), (0, 0, 14), (0, 0, 23), (0, 0, 19)]


def test_an_outside_that_never_sees_the_consequent_opens_once_the_cleanest_split_can_reach_alpha() -> None:
    """E5：条件外那几天当天从不打印。两边各 4 次，分得最干净（就是实测）p = 0.0093 → 开测；两边各 3 次分得最干净 0.024 → 开测；
    条件下 3、条件外 2 次（分得最干净本有 0.040）不开——每边至少 ``MIN_PER_SIDE`` = 3 次。"""

    night_0423 = grouped(E5_INSIDE, E5_OUTSIDE)
    assert ready_to_test(night_0423, alpha=ALPHA)
    assert night_0423.minimum_p() == pytest.approx(observed_p(night_0423)) and observed_p(night_0423) < 0.01

    three_each = grouped(E5_INSIDE[:3], E5_OUTSIDE[:3])
    assert ready_to_test(three_each, alpha=ALPHA) and three_each.minimum_p() == pytest.approx(0.0239, abs=1e-4)

    assert not ready_to_test(grouped(E5_INSIDE[:3], E5_OUTSIDE[:2]), alpha=ALPHA)
    assert not ready_to_test(grouped(E5_INSIDE, E5_OUTSIDE[:1]), alpha=ALPHA)


def test_no_consequent_in_any_control_still_has_a_cleanest_split() -> None:
    """两边对照里都没有 B：两边都"不中"的那一端交互方差为 0、量不出，跳过它，不把整条关系判成永远不显著。条件下 3 次全中、
    条件外 3 次全不中时交互 z = √(2n) = √6，双侧 0.0143 → 开测；单侧（前向验证）0.0072。"""

    measure = grouped([(1, 0, 20)] * 3, [(0, 0, 20)] * 3)
    assert measure.minimum_p() == pytest.approx(math.erfc(math.sqrt(6.0) / math.sqrt(2.0)))
    assert ready_to_test(measure, alpha=ALPHA)
    assert measure.minimum_toward(upward=True) == pytest.approx(0.5 * math.erfc(math.sqrt(6.0) / math.sqrt(2.0)))


def _exact_size(measure_of: Callable[[tuple[int, ...]], ModeratedMeasure], sides: int, reported: float) -> float:
    """零假设（两组同一个来的概率 π、对照里都没有 B）下，检验报出的双侧 p ≤ ``reported`` 的概率，在 π 上取最大。"""

    sizes = []
    for step in range(1, 100):
        rate = step / 100
        size = 0.0
        for outcome in itertools.product((0, 1), repeat=2 * sides):
            if observed_p(measure_of(outcome)) <= reported + 1e-12:
                size += math.prod(rate if hit else 1.0 - rate for hit in outcome)
        sizes.append(size)
    return max(sizes)


@pytest.mark.parametrize("sides", [2, 3])
def test_two_on_each_side_is_too_few_for_the_interaction_z(sides: int) -> None:
    """裁定 94 第 5 条：每边 2 次时交互 z 的正态近似偏乐观。两边对照都没有 B、条件下全中、条件外全不中：2 次报双侧 0.0455，
    零假设下真能得到它的概率最大 1/16 = 0.0625（π = ½），门开了也判不出——每边至少 3 次挡住；3 次报 0.0143、真实最大
    1/64 ≈ 0.0156，在 α 以内，开测。"""

    def measure_of(outcome: tuple[int, ...]) -> ModeratedMeasure:
        # 直接给 enough=True：看的是检验本身报出的 p，不经过 split 的次数判断
        split_up = grouped([(hit, 0, 20) for hit in outcome[:sides]], [(hit, 0, 20) for hit in outcome[sides:]])
        return ModeratedMeasure(split_up.inside, split_up.outsides, enough=True)

    cleanest = measure_of((1,) * sides + (0,) * sides)
    reported = observed_p(cleanest)
    assert reported == pytest.approx(math.erfc(math.sqrt(sides)))  # z = √(2n)
    true_size = _exact_size(measure_of, sides, reported)
    assert true_size == pytest.approx(0.5 ** (2 * sides), abs=1e-9)
    assert (true_size <= ALPHA) is (sides >= MIN_PER_SIDE)
    assert ready_to_test(cleanest, alpha=ALPHA) is (sides >= MIN_PER_SIDE)


def test_the_one_sided_minimum_carries_the_same_outside_gate_as_the_opening_gate() -> None:
    """裁定 94 第 5 条：前向验证用的 ``minimum_toward`` 与开测门同一口径。条件外每层只有 2 个对照、其中 1 个有 B：交互 z 分得最
    干净时单侧只有 1.2e-4，但条件外自己全中 / 全不中都只有 (2/3)³ ≈ 0.30，开测门不开，单侧也读 0.30（不能读成够得着）。
    发现之后的新数据（``subset``）里条件外只剩 2 次时，全量的 ``enough`` 还在，单侧照样按每边至少 3 次读成到不了。"""

    few_controls = grouped([(1, 0, 20)] * 3, [(0, 1, 2)] * 3)
    assert not ready_to_test(few_controls, alpha=ALPHA)
    assert few_controls.minimum_toward(upward=True) == pytest.approx((2 / 3) ** 3)

    full = grouped([(1, 0, 20)] * 4, [(0, 0, 20)] * 4)
    newer = ModeratedMeasure(full.inside, (full.outsides[0][2:],), full.enough)
    assert newer.enough and len(newer.outsides[0]) == 2
    assert newer.minimum_toward(upward=True) == 1.0


def test_a_suppressing_condition_opens_by_the_same_rule_and_only_on_a_split_the_test_can_reach() -> None:
    """压低型（条件下 A 之后 B 不来、平时三成；条件外 A 之后都来、对照里没有 B）：按同一条规则开——次数到了分得最干净能显著才开。
    3 次时朝哪个方向都到不了：朝上交互 z 只到双侧 0.079；朝下条件下全不中的精确概率 (8/11)³ ≈ 0.38，交互再大也过不了交并检验。
    （两个方向不能拼：拿朝下的交互配朝上的精确检验是 3 次就"开"的旧算法，开了也判不出。）4 次开。"""

    inside, outside = (0, 3, 10), (1, 0, 10)
    assert not ready_to_test(grouped([inside] * 3, [outside] * 3), alpha=ALPHA)
    opened = grouped([inside] * 4, [outside] * 4)
    assert ready_to_test(opened, alpha=ALPHA) and opened.minimum_p() == pytest.approx(0.0424, abs=1e-4)


def test_a_relation_without_information_stays_sparse_while_the_same_counts_with_information_open() -> None:
    """真没信息的照旧不开：条件下自己没来、对照里也没有 B（检验里条件下的层内精确检验怎么都没信息），或某一边太少。
    同样的次数，条件下来了的那一组就开（条件外照旧从不出现 B）。"""

    assert not ready_to_test(grouped([(0, 0, 20)] * 6, [(0, 0, 20)] * 6), alpha=ALPHA)
    assert not ready_to_test(grouped([(1, 0, 20)] * 6, [(0, 0, 20)]), alpha=ALPHA)
    assert ready_to_test(grouped([(1, 0, 20)] * 6, [(0, 0, 20)] * 6), alpha=ALPHA)
