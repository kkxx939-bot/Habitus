"""机会口：从预测树的 marginal 曲线取峰。叶子概念一条曲线、聚合概念相加；第 1 次机会含锚。"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest

from habitus.runtime.scene_opportunities import (
    TreeOpportunities,
    generation_of,
    merged_marginal,
    peaks_of,
)
from habitus.scene.ledger import OpportunityRequest
from tests.unit.runtime.prediction_tree_fixtures import curve, tree

CST = timezone(timedelta(hours=8))


def test_a_peak_is_a_run_above_the_days_average_and_its_centre_is_the_highest_slot() -> None:
    """峰 = 连续一段高于当日均值的槽；``at`` 取峰内最高的那一槽（最可能的那一刻），不是几何中点。"""

    # 槽 14/15/16 = 07:00–08:30，其中 15 最高
    found = peaks_of(curve({14: 0.2, 15: 0.5, 16: 0.1}).marginal)
    assert len(found) == 1
    peak = found[0]
    assert (peak.first_slot, peak.last_slot, peak.centre_slot) == (14, 16, 15)
    assert peak.mass == pytest.approx(0.8)
    # 整条曲线全 0 → 没有峰（均值 0，谈不上"高于均值"）
    assert peaks_of(curve({}).marginal) == ()
    # 两段分开的高值 = 两个峰
    assert len(peaks_of(curve({14: 0.5, 15: 0.4, 30: 0.3, 31: 0.2}).marginal)) == 2


def test_a_run_above_the_mean_is_not_a_peak_unless_it_carries_enough_mass() -> None:
    """**没有绝对下限，"高于均值"在稀疏曲线上会退化**（2026-09-29 探针实测）。

    一个 45 天里 97 次的行为在 96 槽钟面上均值只有 0.0107，于是 6 个质量 0.03–0.05 的**单槽**跟着两个
    真峰（各 0.31）一起被当成机会——一天报出 8 个"机会"。后果是一条链：伪峰 → 逐峰各一条假设 →
    一个后件打满 24 本账 → 账按"第 7 次机会"分层、样本被切碎 → 永远攒不够 3 次。
    """

    sparse = curve({20: 0.30, 21: 0.28, 30: 0.04, 35: 0.05, 40: 0.03}).marginal  # 一个真峰 + 三个单槽噪声
    assert [round(peak.mass, 2) for peak in peaks_of(sparse, floor=0.0)] == [0.58, 0.04, 0.05, 0.03]
    assert [round(peak.mass, 2) for peak in peaks_of(sparse)] == [0.58]  # 默认下限滤掉三个单槽
    # 下限是"这一次机会值得叫机会"，与"高于均值"各管一件事：整条平坦的曲线仍然没有峰
    assert peaks_of(curve(dict.fromkeys(range(48), 0.5)).marginal) == ()


def test_a_leaf_concept_uses_its_one_curve_and_an_aggregate_concept_adds_them_up() -> None:
    """叶子概念一个 kind → 相加退化成"就用那条"；聚合概念跨几个 kind → 逐槽相加、截到 1（B13：同槽近似互斥）。"""

    built = tree(
        {
            (4, "排查CI失败原因"): curve({28: 0.3}),
            (4, "排查性能问题"): curve({28: 0.2, 29: 0.4}),
        }
    )
    # 叶子：只有一个 kind
    leaf = merged_marginal(built, 4, ("排查CI失败原因",))
    assert leaf is not None and leaf[28] == pytest.approx(0.3) and leaf[29] == 0.0
    # 聚合：两个 kind 逐槽相加
    aggregate = merged_marginal(built, 4, ("排查CI失败原因", "排查性能问题"))
    assert aggregate is not None and aggregate[28] == pytest.approx(0.5) and aggregate[29] == pytest.approx(0.4)
    # 截到 1
    heavy = tree({(4, "a"): curve({10: 0.7}), (4, "b"): curve({10: 0.8})})
    capped = merged_marginal(heavy, 4, ("a", "b"))
    assert capped is not None and capped[10] == 1.0
    # 这个周几一条曲线都没有 → None，调用方跳过这一天
    assert merged_marginal(built, 0, ("排查CI失败原因",)) is None


def test_the_first_opportunity_is_the_peak_the_anchor_sits_in() -> None:
    """评审 A-2：锚正在其中的峰也算第 1 次机会。按"峰的开始晚于锚"铺会把正在进行的峰丢掉。"""

    # 树上的键是 kind，映射靠 kinds；这里让「早餐」对应 kind「吃饭」
    built = tree({(weekday, "吃饭"): curve({14: 0.2, 15: 0.5, 16: 0.1}) for weekday in range(7)})
    provider = TreeOpportunities(built, {"早餐": ("吃饭",)}, generation="gen-1")

    # 峰是 07:00–08:30。锚 07:30 正落在峰里 → 它就是第 1 次机会
    inside = provider.opportunities(OpportunityRequest(consequent="早餐", anchor=datetime(2026, 8, 14, 7, 30, tzinfo=CST), count=3))
    assert inside is not None
    first = inside.opportunities[0]
    assert first.span.start == datetime(2026, 8, 14, 7, 0, tzinfo=CST)
    assert first.span.end == datetime(2026, 8, 14, 8, 30, tzinfo=CST)
    assert first.at == datetime(2026, 8, 14, 7, 45, tzinfo=CST)  # 槽 15 的中心
    assert first.probability == pytest.approx(0.8)
    # 锚在峰之后 → 第 1 次机会是次日那个峰
    after = provider.opportunities(OpportunityRequest(consequent="早餐", anchor=datetime(2026, 8, 14, 9, 0, tzinfo=CST), count=2))
    assert after is not None and after.opportunities[0].span.start == datetime(2026, 8, 15, 7, 0, tzinfo=CST)
    assert [item.at.date() for item in after.opportunities] == [date(2026, 8, 15), date(2026, 8, 16)]


def test_a_consequent_with_no_hits_or_no_curve_gets_no_snapshot() -> None:
    """要不到对照就明说要不到：承诺仍然开（机会本身是事实），读时算不出强度而已。"""

    built = tree({(4, "吃饭"): curve({15: 0.5})})
    provider = TreeOpportunities(built, {"早餐": ("吃饭",)}, generation="gen-1", max_lookahead_days=3)
    # 这个概念从没命中过任何 kind
    assert provider.opportunities(OpportunityRequest(consequent="打球", anchor=datetime(2026, 8, 14, 2, 0, tzinfo=CST), count=2)) is None
    # 只有周五有曲线，从周六起翻三天都翻不到 → None
    assert provider.opportunities(OpportunityRequest(consequent="早餐", anchor=datetime(2026, 8, 15, 2, 0, tzinfo=CST), count=2)) is None
    # 周五那天够得到，但只铺出一个峰（要 5 个也只给 1 个）
    friday = provider.opportunities(OpportunityRequest(consequent="早餐", anchor=datetime(2026, 8, 14, 2, 0, tzinfo=CST), count=5))
    assert friday is not None and len(friday.opportunities) == 1


def test_the_generation_name_is_time_then_digest() -> None:
    """代名要字典序即时间序：构建时刻在前、配置摘要在后（与 prediction/store 的命名同向）。"""

    name = generation_of(tree({}))
    assert name.startswith("20260818T") and name.endswith("-" + "d" * 12)


def test_the_snapshot_is_capped_and_slots_map_to_local_time() -> None:
    """要的个数超过账本收得下的上限时按上限给；槽位换算跟着锚的时区走。"""

    built = tree({(weekday, "吃饭"): curve({15: 0.5, 30: 0.3}) for weekday in range(7)})
    provider = TreeOpportunities(built, {"早餐": ("吃饭",)}, generation="gen-1")
    snapshot = provider.opportunities(OpportunityRequest(consequent="早餐", anchor=datetime(2026, 8, 14, 2, 0, tzinfo=CST), count=4))
    assert snapshot is not None and len(snapshot.opportunities) == 4
    # 一天两个峰（07:30–08:00 与 15:00–15:30），所以四个机会跨两天
    assert [item.at.strftime("%m-%d %H:%M") for item in snapshot.opportunities] == [
        "08-14 07:45",
        "08-14 15:15",
        "08-15 07:45",
        "08-15 15:15",
    ]
    assert all(item.at.tzinfo == CST for item in snapshot.opportunities)
