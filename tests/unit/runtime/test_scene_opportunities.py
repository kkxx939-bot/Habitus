"""机会口：把假设上的钟面窗口落到各天、量那天曲线在窗口里的质量；峰的发现（节律口用）也在这里。叶子概念一条曲线、聚合概念相加；第 1 个落点含锚。"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest

from habitus.runtime.scene_opportunities import (
    TreeOpportunities,
    generation_of,
    merged_marginal,
    peaks_of,
)
from habitus.scene.hypotheses import PeakWindow
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


BREAKFAST = PeakWindow(1, 7 * 60, 8 * 60 + 30)  # 07:00–08:30，槽 14–16


def test_the_first_landing_is_the_window_the_anchor_sits_in() -> None:
    """评审 A-2：锚正在其中的窗口也算第 1 个落点。按"窗口开始晚于锚"铺会把正在进行的那个丢掉。

    窗口是假设上写死的钟面时段（``PeakWindow``），机会口只负责把它落到各天、量那天曲线在窗口里的质量。
    """

    # 树上的键是 kind，映射靠 kinds；这里让「早餐」对应 kind「吃饭」
    built = tree({(weekday, "吃饭"): curve({14: 0.2, 15: 0.5, 16: 0.1}) for weekday in range(7)})
    provider = TreeOpportunities(built, {"早餐": ("吃饭",)}, generation="gen-1")

    # 锚 07:30 正落在窗口里 → 它就是第 1 个落点
    inside = provider.opportunities(OpportunityRequest(consequent="早餐", anchor=datetime(2026, 8, 14, 7, 30, tzinfo=CST), count=1, window=BREAKFAST))
    assert inside is not None and len(inside.opportunities) == 1
    first = inside.opportunities[0]
    assert first.span.start == datetime(2026, 8, 14, 7, 0, tzinfo=CST)
    assert first.span.end == datetime(2026, 8, 14, 8, 30, tzinfo=CST)
    assert first.at == datetime(2026, 8, 14, 7, 45, tzinfo=CST)  # 窗口中心
    assert first.probability == pytest.approx(0.8)
    # 锚在窗口之后 → 第 1 个落点是次日那个窗口
    after = provider.opportunities(OpportunityRequest(consequent="早餐", anchor=datetime(2026, 8, 14, 9, 0, tzinfo=CST), count=2, window=BREAKFAST))
    assert after is not None and after.opportunities[0].span.start == datetime(2026, 8, 15, 7, 0, tzinfo=CST)
    assert [item.at.date() for item in after.opportunities] == [date(2026, 8, 15), date(2026, 8, 16)]
    # 容差（2026-10-01 用户定复用预测树的 pool_half_width）：两边各展 slack_slots 槽，质量也按展宽后的段算
    slack = TreeOpportunities(built, {"早餐": ("吃饭",)}, generation="gen-1", slack_slots=1)
    assert slack.slack_minutes == 30
    widened = slack.opportunities(OpportunityRequest(consequent="早餐", anchor=datetime(2026, 8, 14, 2, 0, tzinfo=CST), count=1, window=BREAKFAST))
    assert widened is not None and widened.opportunities[0].span.start == datetime(2026, 8, 14, 6, 30, tzinfo=CST)
    assert widened.opportunities[0].span.end == datetime(2026, 8, 14, 9, 0, tzinfo=CST) and widened.opportunities[0].probability == pytest.approx(0.8)


def test_a_consequent_with_no_curve_gets_a_window_without_a_control() -> None:
    """要不到对照就明说要不到：窗口照样落地（它是假设上写死的事实）、承诺照样开，只是 ``probability=None``——不猜（二-4）。"""

    built = tree({(4, "吃饭"): curve({15: 0.5})})
    provider = TreeOpportunities(built, {"早餐": ("吃饭",)}, generation="gen-1", max_lookahead_days=3)
    # 这个概念从没命中过任何 kind：窗口在、对照 None
    blind = provider.opportunities(OpportunityRequest(consequent="打球", anchor=datetime(2026, 8, 14, 2, 0, tzinfo=CST), count=2, window=BREAKFAST))
    assert blind is not None and [item.probability for item in blind.opportunities] == [None, None]
    # 只有周五有曲线：周五那天的落点有数，周六起的没有
    friday = provider.opportunities(OpportunityRequest(consequent="早餐", anchor=datetime(2026, 8, 14, 2, 0, tzinfo=CST), count=3, window=BREAKFAST))
    assert friday is not None and [item.probability for item in friday.opportunities] == [pytest.approx(0.5), None, None]


def test_the_generation_name_is_time_then_digest() -> None:
    """代名要字典序即时间序：构建时刻在前、配置摘要在后（与 prediction/store 的命名同向）。"""

    name = generation_of(tree({}))
    assert name.startswith("20260818T") and name.endswith("-" + "d" * 12)


def test_the_snapshot_cycles_the_consequents_window_table_from_the_requested_one() -> None:
    """次数方面要接下来几个窗口：从请求的那个窗口起、按后件的整张峰表轮，跨天继续；槽位换算跟着锚的时区走。"""

    built = tree({(weekday, "吃饭"): curve({15: 0.5, 30: 0.3}) for weekday in range(7)})
    provider = TreeOpportunities(built, {"早餐": ("吃饭",)}, generation="gen-1")
    table = (PeakWindow(1, 7 * 60 + 30, 8 * 60), PeakWindow(2, 15 * 60, 15 * 60 + 30))
    snapshot = provider.opportunities(OpportunityRequest(consequent="早餐", anchor=datetime(2026, 8, 14, 2, 0, tzinfo=CST), count=4, window=table[0], windows=table))
    assert snapshot is not None and len(snapshot.opportunities) == 4
    assert [item.at.strftime("%m-%d %H:%M") for item in snapshot.opportunities] == ["08-14 07:45", "08-14 15:15", "08-15 07:45", "08-15 15:15"]
    assert [item.probability for item in snapshot.opportunities] == [pytest.approx(0.5), pytest.approx(0.3), pytest.approx(0.5), pytest.approx(0.3)]
    assert all(item.at.tzinfo == CST for item in snapshot.opportunities)
    # 从第 2 个窗口起：锚 02:00 之后第一次落地是当天 15:00，再轮到次日 07:30
    second = provider.opportunities(OpportunityRequest(consequent="早餐", anchor=datetime(2026, 8, 14, 2, 0, tzinfo=CST), count=2, window=table[1], windows=table))
    assert second is not None and [item.at.strftime("%m-%d %H:%M") for item in second.opportunities] == ["08-14 15:15", "08-15 07:45"]


def test_r3_08_a_peak_across_midnight_is_one_opportunity_owned_by_the_day_it_starts() -> None:
    """就寝常态 23:00–01:00：零点前后的质量在同一条周几曲线的两头，按环接成**一个**峰，归给它开始的那一天，尾巴落在次日
    （评审 A-7 / B-8 / C-7：按日历日各取各的会切成 0.55 与 0.35 两个机会，00:20 入睡被记成"当晚没睡"）。"""

    # 槽 46、47 = 23:00–24:00，槽 0、1 = 00:00–01:00；每个周几都是这条曲线，所以"次日"的开头就是它自己的开头
    ring = curve({46: 0.3, 47: 0.25, 0: 0.2, 1: 0.15})
    (peak,) = peaks_of(ring.marginal, following=ring.marginal, preceding=ring.marginal)
    assert (peak.first_slot, peak.last_slot, peak.centre_slot) == (46, 49, 46) and peak.mass == pytest.approx(0.9)
    # 次日没数据 → 不猜：峰截在 24:00（二-4，"没看过就当没看到"）；前一天没数据 → 00:00 那段算当天自己的峰
    truncated = peaks_of(ring.marginal)
    assert [(p.first_slot, p.last_slot) for p in truncated] == [(0, 1), (46, 47)] and truncated[1].mass == pytest.approx(0.55)
    # 有前一天、它的尾巴高于均值 → 当天 00:00 那段是它的尾巴，不再单独成峰
    assert [(p.first_slot, p.last_slot) for p in peaks_of(ring.marginal, preceding=ring.marginal)] == [(46, 47)]
    built = tree({(weekday, "睡觉"): ring for weekday in range(7)})
    provider = TreeOpportunities(built, {"就寝": ("睡觉",)}, generation="gen-1")
    bedtime = PeakWindow(1, peak.first_slot * 30, (peak.last_slot + 1) * 30)  # 23:00–25:00：跨午夜的窗口终点过 24:00
    snapshot = provider.opportunities(OpportunityRequest(consequent="就寝", anchor=datetime(2026, 8, 14, 2, 10, tzinfo=CST), count=2, window=bedtime))
    assert snapshot is not None
    first, second = snapshot.opportunities
    assert first.span.start == datetime(2026, 8, 14, 23, 0, tzinfo=CST) and first.span.end == datetime(2026, 8, 15, 1, 0, tzinfo=CST)
    assert first.probability == pytest.approx(0.9)  # 两头的质量各取那天周几的曲线，加在一起
    assert second.span.start == datetime(2026, 8, 15, 23, 0, tzinfo=CST)
    # 00:20 入睡落在第 1 个窗口里
    assert snapshot.index_of(datetime(2026, 8, 15, 0, 20, tzinfo=CST)) == 1
    # 每半都够不上 MIN_PEAK_MASS、合起来够：切开就整个消失，接起来是一个机会
    thin = curve({46: 0.06, 47: 0.05, 0: 0.04, 1: 0.03})
    assert len(peaks_of(thin.marginal, following=thin.marginal, preceding=thin.marginal)) == 1
    # 真正的次日曲线不同（周五晚特别晚）：接的是次日的开头，不是当天的
    friday = curve({46: 0.3, 47: 0.25, 0: 0.1, 1: 0.1})
    saturday = curve({0: 0.3, 1: 0.3, 2: 0.2, 46: 0.2, 47: 0.2})
    (late,) = peaks_of(friday.marginal, following=saturday.marginal, preceding=friday.marginal)
    assert (late.first_slot, late.last_slot) == (46, 50) and late.mass == pytest.approx(0.3 + 0.25 + 0.3 + 0.3 + 0.2)
