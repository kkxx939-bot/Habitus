"""丙一"在用"那一段（预期表 C10–C15；共享说明 3.5）：候选转在用之前，它的记录在数"两个不同日子"时也算给它最近的不是候选的
上级；转在用以后只算自己。"最后被选中"与 20 天退出仍然只看直接记录（靠下级的日子转了在用的上级，从转在用那天起算 20 天）。

走组合根装好的整套 Runtime（``tests.unit.word_grow.site.GrowSite``：会话 lane 选词 → 行为树 → 换代数在用 → 语义树概念）；
"在用" = 语义树里没停用的行为概念。改动前（``780e298``）在用只数直接选中的日子：C10、C11、C12、C13、C14 在旧代码上应当失败；
C15 是"初始词的记录不往上算"的守门，旧代码上本来就成立。
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import date
from pathlib import Path

import pytest

from tests.unit.word_grow.site import QUERY, WEEKLY, WRITE_DOC, Grow, GrowSite, Named, at, grown, node
from tests.unit.word_pick.site import Word

Y, Z = "查销售数据", "查月度销量"


def d(month: int, day: int) -> date:
    return date(2026, month, day)


@pytest.fixture
def site(tmp_path: Path) -> Iterator[GrowSite]:
    built = GrowSite(tmp_path)
    yield built
    built.close()


def _y_on_two_days(site: GrowSite) -> None:
    """候选 Y 挂在查数据下面：04-13 写出它的那一轮记在 Y 上，04-15 又选中 Y；查数据没有直接记录。"""

    site.record("conv-y", at(d(4, 13), 9), "帮我查一下这个月的销售数据", Grow("查数据", node(Y, "查销售相关的数据")))
    site.record("conv-y", at(d(4, 15), 9), "再查一下销售数据", Named(Y))


def test_c10_a_candidates_record_counts_one_day_for_its_parent(site: GrowSite) -> None:
    site.record("conv-y", at(d(4, 13), 9), "帮我查一下这个月的销售数据", Grow("查数据", node(Y, "查销售相关的数据")))
    site.record("conv-q", at(d(4, 15), 9), "帮我查一下库存数据", Named("查数据"))
    site.generate(d(4, 16))
    in_use = site.in_use()
    assert (QUERY in in_use, grown(1) in in_use) == (True, False)


def test_c11_a_candidate_on_two_days_brings_itself_and_its_parent_into_use(site: GrowSite) -> None:
    _y_on_two_days(site)
    site.generate(d(4, 16))
    in_use = site.in_use()
    assert (grown(1) in in_use, QUERY in in_use) == (True, True)


def _then_only_y(site: GrowSite, *days: date) -> None:
    for day in days:
        site.record("conv-y", at(day, 9), "再查一下销售数据", Named(Y))


def test_c12_a_parent_in_use_only_through_its_childs_days_leaves_twenty_days_after_entering(site: GrowSite) -> None:
    _y_on_two_days(site)
    site.generate(d(4, 16))
    assert QUERY in site.in_use()
    _then_only_y(site, d(4, 20), d(4, 25), d(4, 30), d(5, 5))
    site.generate(d(5, 5))
    assert {QUERY, grown(1)} <= site.in_use()
    site.generate(d(5, 6))
    in_use = site.in_use()
    assert (QUERY in in_use, grown(1) in in_use) == (False, True)


def test_c13_once_the_candidate_is_in_use_its_records_no_longer_count_for_the_parent(site: GrowSite) -> None:
    _y_on_two_days(site)
    site.generate(d(4, 16))
    assert QUERY in site.in_use()  # C11：查数据靠 Y 转在用之前的两个日子在用
    _then_only_y(site, d(4, 20), d(4, 25), d(4, 30), d(5, 5))
    site.generate(d(5, 6))
    assert QUERY not in site.in_use() and grown(1) in site.in_use()
    _then_only_y(site, d(5, 7), d(5, 8))
    site.generate(d(5, 9))
    assert QUERY not in site.in_use()


def test_c14_a_candidate_under_a_candidate_counts_for_the_nearest_non_candidate_parent_and_itself(
    site: GrowSite,
) -> None:
    site.record(
        "conv-z",
        at(d(4, 13), 9),
        "帮我查一下这个月每周的销量",
        Grow("查数据", node(Y, "查销售相关的数据"), node(Z, "按月查销量")),
    )
    site.record("conv-z", at(d(4, 15), 9), "再查一下月度销量", Named(Z))
    assert site.token_at(at(d(4, 13), 9)) == grown(2)
    site.generate(d(4, 16))
    in_use = site.in_use()
    assert (QUERY in in_use, grown(2) in in_use, grown(1) in in_use) == (True, True, False)


def test_c15_records_on_an_initial_child_do_not_count_for_its_parent(site: GrowSite) -> None:
    """守门：写周报是初始词、下面没有候选，记录直接挂在它上面，不算给写文档的日子（旧代码上本来就成立）。"""

    site.record("conv-w", at(d(4, 13), 9), "帮我写一份本周周报", Word(WEEKLY))
    site.record("conv-w", at(d(4, 15), 9), "再写一份周报", Word(WEEKLY))
    site.generate(d(4, 16))
    assert WRITE_DOC not in site.in_use()
