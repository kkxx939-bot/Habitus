"""覆盖口：两种空白都算、重叠的空白并集相减、跨午夜的空白挂在开始那天、按天缓存。"""

from __future__ import annotations

import pytest

from habitus.runtime.scene_coverage import TreeCoverage
from habitus.scene.ledger import WindowSpan
from tests.unit.scene.fixtures import DAY1, DAY2, Site, at, publish_gap

NOW = at(DAY1, 23, 0)


def site(tmp_path) -> Site:
    return Site(tmp_path, now=NOW)


def test_two_overlapping_gaps_are_subtracted_once_and_both_kinds_count(tmp_path) -> None:
    """未观测 07:00–08:00 与没读懂 07:30–08:30 重叠：按次相减会把 07:30–08:00 算两遍，算出负覆盖。"""

    tree = site(tmp_path).behavior_tree
    publish_gap(tree, DAY1, (7, 0), (8, 0))
    publish_gap(tree, DAY1, (7, 30), (8, 30), kind="没读懂")
    provider = TreeCoverage(tree)
    # 07:00–09:00 共 120 分钟，空白并集是 07:00–08:30 共 90 分钟 → 看清了 25%
    covered = provider.coverage(WindowSpan(at(DAY1, 7, 0), at(DAY1, 9, 0)))
    assert covered.observed_fraction == pytest.approx(0.25)
    assert {gap.kind for gap in covered.gaps} == {"未观测", "没读懂"}
    # 完全在空白之外 → 全看清，不列空白
    clear = provider.coverage(WindowSpan(at(DAY1, 10, 0), at(DAY1, 11, 0)))
    assert clear.observed_fraction == 1.0 and clear.gaps == ()


def test_a_gap_across_midnight_is_found_from_the_previous_day(tmp_path) -> None:
    """02:00 睡到 09:00 的空白挂在开始那天；只读次日会把次日早上算成"看清了"，早餐就成了反面证据。"""

    tree = site(tmp_path).behavior_tree
    publish_gap(tree, DAY1, (23, 0), (9, 0), end_day=DAY2)
    provider = TreeCoverage(tree)
    breakfast = provider.coverage(WindowSpan(at(DAY2, 7, 0), at(DAY2, 8, 30)))
    assert breakfast.observed_fraction == 0.0 and len(breakfast.gaps) == 1
    # 往前不读的话就漏掉它（这正是 lookbehind_days 存在的理由）
    blind = TreeCoverage(tree, lookbehind_days=0)
    assert blind.coverage(WindowSpan(at(DAY2, 7, 0), at(DAY2, 8, 30))).observed_fraction == 1.0


def test_an_instant_gap_is_legal_and_counted_but_does_not_reduce_coverage(tmp_path) -> None:
    """起止同刻的空白是**合法**的（归约那边明说"起止同刻是合法的单观测段"），而且是多数：
    真实数据里 7 月 46 个空白段有 27 个是同刻的「没读懂」。

    它说的是"那一瞬间读不出是什么行为"，不声称周围那段时间没观测到，所以零宽度减不掉任何时间；
    但要数出来——不然"覆盖 1.0"看上去像"全看清了"。真实数据上这条让整夜塌过一次（2026-09-29 探针）。
    """

    tree = site(tmp_path).behavior_tree
    publish_gap(tree, DAY1, (19, 58), (19, 58), kind="没读懂")
    publish_gap(tree, DAY1, (7, 0), (8, 0))
    provider = TreeCoverage(tree, lookbehind_days=0)
    clear = provider.coverage(WindowSpan(at(DAY1, 19, 0), at(DAY1, 21, 0)))
    assert clear.observed_fraction == 1.0 and clear.gaps == ()  # 同刻的减不掉时间
    assert provider.instant_gaps[DAY1] == 1  # 但数出来了
    dark = provider.coverage(WindowSpan(at(DAY1, 7, 0), at(DAY1, 9, 0)))
    assert dark.observed_fraction == pytest.approx(0.5)  # 真空白照减


def test_a_day_is_read_once_and_forget_drops_it(tmp_path) -> None:
    """夜批里一天的空白会被问几十次（每条承诺每个机会一次），而已封口的那天不会再变。"""

    tree = site(tmp_path).behavior_tree
    publish_gap(tree, DAY1, (7, 0), (8, 0))
    reads: list[object] = []
    inner = tree.read_day

    def counting(kind, occurred_on):
        reads.append((kind, occurred_on))
        return inner(kind, occurred_on)

    tree.read_day = counting  # type: ignore[method-assign]
    provider = TreeCoverage(tree, lookbehind_days=0)
    for _ in range(5):
        provider.coverage(WindowSpan(at(DAY1, 7, 0), at(DAY1, 9, 0)))
    assert len(reads) == 1
    provider.forget(DAY1)
    provider.coverage(WindowSpan(at(DAY1, 7, 0), at(DAY1, 9, 0)))
    assert len(reads) == 2
    provider.forget()
    provider.coverage(WindowSpan(at(DAY1, 7, 0), at(DAY1, 9, 0)))
    assert len(reads) == 3
