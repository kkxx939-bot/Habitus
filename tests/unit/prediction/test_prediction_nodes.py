"""钟面节点：计数、曝光、三种率、两个 lift、收缩与趋势。

数字都能手算——这是纯函数层的正确性主战场（见 TODO(PRED-TREE-001)）。
"""

from __future__ import annotations

import random
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from prediction import nodes
from prediction.errors import PredictionTreeError
from prediction.model import ObservedGap, SlotKey
from prediction.nodes import (
    accumulate,
    completion_curves,
    day_coverage,
    decay_weight,
    derive_all,
    period_scale,
    pool_indexes,
    pooled_trends,
)
from prediction.recurrence import derive as derive_recurrence
from tests.unit.prediction.prediction_fixtures import (
    action,
    at,
    config,
    daily,
    gap,
    production_config,
    publish,
    reference,
    weekly,
)

MORNING = SlotKey(weekday=0, slot=28)  # 周一 07:00–07:15


def build(actions, gaps=(), *, days: int, **overrides):
    """账本 + **发布形态**的读取视图；率与 lift 一律按已发布的曲线与基线取。"""

    cfg = config(**overrides)
    at = reference(days - 1)
    ledger = accumulate(list(actions), list(gaps), config=cfg, reference=at)
    return cfg, ledger, publish(actions, gaps, config=cfg, reference_day=at)


def build_curves(actions, gaps=(), *, days: int, cfg=None, **overrides):
    """走夜批的真实顺序：先复发（给出每个行为自己的周期）→ 计数 → 趋势 → 格子与曲线。"""

    resolved = cfg if cfg is not None else config(**overrides)
    at = reference(days - 1)
    actions = list(actions)
    recurrences = derive_recurrence(actions, config=resolved, reference=at)
    periods = {
        name: item.intervals.p50 / 86_400.0 for name, item in recurrences.items()
    }
    ledger = accumulate(actions, list(gaps), config=resolved, reference=at)
    trends = pooled_trends(
        actions, list(gaps), config=resolved, reference=at, periods=periods
    )
    completion = completion_curves(actions, list(gaps), config=resolved, reference=at)
    derived = derive_all(
        ledger, config=resolved, trends=trends, completion=completion
    )
    return dict(derived.cells), dict(derived.curves)


# --- 计数与曝光 ---------------------------------------------------------------------------


def test_exposure_counts_every_calendar_day_not_only_days_with_records() -> None:
    """只数"有记录的天"会按"这天发生了点什么"条件化，把所有概率整体抬高。"""

    # 28 天里只有 4 个周一做过这件事，但 28 天全部是机会。
    _cfg, ledger, _stats = build(weekly("吃药", 4, weekday=0, hour=7), days=28)
    assert ledger.observed_days == 28
    assert ledger.exposure[MORNING].observed_days == pytest.approx(4.0, rel=0.01)


def test_same_slot_twice_in_one_day_is_capped_but_raw_count_survives() -> None:
    """封顶 1 让 P 保持是概率；原始次数另存，强度信息不丢。"""

    actions = [action("洗手", 0, 7, 2), action("洗手", 0, 7, 9)]  # 同一个 15 分钟槽
    _cfg, ledger, _stats = build(actions, days=1)
    counts = ledger.counts[(MORNING, "洗手")]
    assert counts.occurred_days == pytest.approx(1.0)
    assert counts.raw_occurrences == pytest.approx(2.0)


def test_gaps_reduce_exposure_and_lift_the_rate() -> None:
    """没在看的时段不算机会——否则观测越差、算出的概率越低，方向就反了。"""

    actions = daily("吃药", 6, hour=7)
    # 另外 4 天该时段没有观测：分母从 10 降到 6。
    gaps = [gap(offset, 7, 8) for offset in range(6, 10)]
    _cfg, ledger, stats = build(actions, gaps, days=10, slot_minutes=15)
    key = (SlotKey(weekday=0, slot=28), "吃药")
    monday_exposure = ledger.exposure[SlotKey(weekday=0, slot=28)].observed_days
    # 10 天里周一出现两次（第 0 天与第 7 天），第 7 天被空白扣掉。
    assert monday_exposure == pytest.approx(1.0, rel=0.01)
    assert stats[key].marginal == pytest.approx(1.0, abs=1e-3)


def test_a_gap_spanning_midnight_is_split_across_both_days() -> None:
    """跨日空白必须按天切开，否则次日那半段被整段丢掉、曝光虚高。"""

    from prediction.model import ObservedGap
    from tests.unit.prediction.prediction_fixtures import at

    crossing = ObservedGap(started_at=at(0, 23), ended_at=at(1, 1), watched=True)
    _cfg, ledger, _stats = build([action("吃药", 3, 7)], [crossing], days=4)
    # 两端都被空白完全盖住 → 曝光为零 → 稀疏存储里根本没有这两个格子。
    assert SlotKey(weekday=0, slot=92) not in ledger.exposure  # 周一 23:00
    assert SlotKey(weekday=1, slot=0) not in ledger.exposure  # 周二 00:00
    # 而没被盖住的槽照常有曝光，证明只扣掉了重叠的那部分。
    assert ledger.exposure[SlotKey(weekday=0, slot=88)].observed_days > 0.9  # 周一 22:00


def test_a_partly_covered_day_weighs_less_on_both_sides() -> None:
    """一天对一个格子的证据量 ``m = 衰减权重 × 覆盖比例``，分子分母**同一个 m**。

    早先分子按整份权重记、分母按覆盖比例记，覆盖 0.33 的槽里发生一次会算出 3 倍的"概率"；
    它没暴露，是因为另有一条"这一槽产出过 occurrence 就把覆盖记满"的特判把分母也顶满——
    两个特判互相撑着，一起掩盖了口径不一致。这条测试同时钉住两边：分子必须缩到 1/3，
    分母也必须是 1/3（而不是被谁补成 1）。
    """

    hit = action("吃药", 0, 7)  # 07:00，落在槽 28（07:00–07:15）
    # 空白盖住这一槽的后 2/3，且**不含**那条行为的起点 → 不被证伪，照常扣减。
    partial = ObservedGap(started_at=at(0, 7, 5), ended_at=at(0, 7, 15), watched=True)
    kept = nodes.reconcile_gaps([hit], [partial])
    assert kept == (partial,)

    cfg, ledger, _stats = build([hit], kept, days=1)
    key = SlotKey(weekday=0, slot=28)
    assert ledger.exposure[key].observed_days == pytest.approx(1 / 3, abs=1e-6)
    counts = ledger.counts[(key, "吃药")]
    assert counts.occurred_days == pytest.approx(1 / 3, abs=1e-6)
    assert counts.first_days == pytest.approx(1 / 3, abs=1e-6)
    # 同口径的直接后果：裸比值落在 [0,1] 内，夹紧退化成防御性护栏。
    assert counts.occurred_days / ledger.exposure[key].observed_days == pytest.approx(1.0)


def test_a_behaviour_read_inside_an_unobserved_gap_is_refused() -> None:
    """"没在看"与"看见了"同时成立是上游的矛盾；本层报出来，不替它圆场。"""

    hit = action("吃药", 0, 7)
    blind = ObservedGap(started_at=at(0, 7), ended_at=at(0, 8), watched=False)
    assert nodes.reconcile_gaps([hit], [blind]) == (blind,)  # 未观测不被证伪
    with pytest.raises(PredictionTreeError, match="unobserved gap"):
        build([hit], [blind], days=1)


def test_one_observation_does_not_saturate_the_completion_line() -> None:
    """整整四周只做过一次的行为，"该完成线"不得满信心触发。

    旧实现是裸比值 ``(之前累积+首次)/曝光``：那个周几只有那一天做过，分子分母逐字相等，
    末槽精确 1.0——一次观测就让缺失检测确信"这个点该做完了"。两级分解把它拉回基线：π 的样本
    单位是天，一次命中兑四天曝光再经收缩，只剩一成上下。留出回测的完整对照见 PRED-RATES-002。
    """

    cfg = production_config()
    _cells, curves = build_curves([action("体检", 0, 9)], days=28, cfg=cfg)
    curve = curves[(0, "体检")]  # 第 0 天是周一
    assert curve.cumulative[-1] < 0.3
    assert curve.cumulative[-1] > 0.0  # 也不能压到零：他**确实**做过一次


def test_a_steady_daily_habit_still_completes() -> None:
    """反过来也要成立：天天做的事，到晚上该完成线必须接近 1，收缩不能把它一起压平。"""

    cfg = production_config()
    _cells, curves = build_curves(daily("吃药", 28, 7, 30), days=28, cfg=cfg)
    curve = curves[(0, "吃药")]
    assert curve.cumulative[-1] > 0.9
    # 分布函数的形状：早于最早那次之前没有质量，之后单调爬升到 π。
    assert curve.cumulative[0] == 0.0
    assert all(
        later >= earlier
        for earlier, later in zip(curve.cumulative[:-1], curve.cumulative[1:], strict=True)
    )


# --- 三种率 -------------------------------------------------------------------------------


def test_marginal_hazard_and_cumulative_differ_for_a_twice_daily_action() -> None:
    """一天两次的行为最能看出三种率的分工。"""

    actions = []
    for offset in range(10):
        actions.append(action("吃药", offset, 7))  # 每天早上都吃
        if offset < 5:
            actions.append(action("吃药", offset, 20))  # 前 5 天晚上也吃
    _cfg, _ledger, stats = build(actions, days=10)

    evening_mondays = stats[(SlotKey(weekday=0, slot=80), "吃药")]  # 周一 20:00
    # 边际率：这个槽发生过的比例（周一共 2 天，第 0 天晚上吃了）
    assert evening_mondays.marginal == pytest.approx(0.5, abs=1e-3)
    # 危险率：两个周一早上都已经吃过，风险集为空 → 无信息 → 退回先验（很小），
    # 不是"精确 0"，因为 0/0 没有定义；退回先验是标准的贝叶斯答案。
    assert evening_mondays.hazard < 0.05
    # 累积率：到 20:00 为止，两个周一都已经吃过了
    assert evening_mondays.cumulative == pytest.approx(1.0, abs=1e-3)


def test_hazard_conditions_on_not_having_happened_yet() -> None:
    """危险率的分母是风险集：到这个槽为止还没发生的天数。"""

    actions = []
    for offset in range(10):
        actions.append(action("起床", offset, 7 if offset % 2 == 0 else 8))
    _cfg, _ledger, stats = build(actions, days=14)
    eight = stats[(SlotKey(weekday=0, slot=32), "起床")]  # 周一 08:00
    # 周一共 2 天：第 0 天 07:00 起床、第 7 天 08:00 起床。
    # 边际率 = 1/2；危险率 = 1/1（在"还没起床"的那 1 天里，08:00 起了）
    assert eight.marginal == pytest.approx(0.5, abs=1e-3)
    assert eight.hazard == pytest.approx(1.0, abs=1e-3)


# --- lift ---------------------------------------------------------------------------------


def test_lift_separates_a_true_peak_from_background_noise() -> None:
    """同样的 P 可能是真峰也可能是底噪——lift 的分母是该动作自己的全天平均。"""

    actions = list(daily("吃药", 14, hour=7))
    for offset in range(14):  # 看电视：每天 12:00–22:00 连续铺开（40 个槽）
        for slot_index in range(40):
            actions.append(action("看电视", offset, 12 + slot_index // 4, 15 * (slot_index % 4)))
    _cfg, _ledger, stats = build(actions, days=14)

    medicine = stats[(SlotKey(weekday=0, slot=28), "吃药")]
    television = stats[(SlotKey(weekday=0, slot=76), "看电视")]  # 周一 19:00
    # 两者在各自的格子里概率都是 1，光看 P 完全分不出
    assert medicine.marginal == pytest.approx(1.0, abs=1e-3)
    assert television.marginal == pytest.approx(1.0, abs=1e-3)
    # lift 把它们分开：吃药只占 1/96 个槽（真峰），看电视占 40/96（底噪）
    assert medicine.lift_all_day == pytest.approx(96.0, rel=0.05)
    assert television.lift_all_day == pytest.approx(2.4, rel=0.1)


def test_weekday_lift_surfaces_a_weekly_habit_that_the_clock_alone_would_bury() -> None:
    """每周二打球在日钟面上只有 1/7 的概率，在周维度上是强规律。"""

    actions = weekly("打球", 8, weekday=1, hour=19)  # 每周二 19:00
    _cfg, _ledger, stats = build(actions, days=56)
    tuesday = stats[(SlotKey(weekday=1, slot=76), "打球")]
    assert tuesday.marginal == pytest.approx(1.0, abs=1e-2)
    # 同一时刻跨周几的平均只有 1/7，所以周几 lift 应当接近 7
    assert tuesday.lift_weekday == pytest.approx(7.0, rel=0.2)


# --- 收缩 ---------------------------------------------------------------------------------


def test_shrinkage_pulls_sparse_cells_and_lets_go_when_data_grows() -> None:
    """收缩正确工作的判据：样本少时拉回去，样本多时几乎不动。"""

    sparse = build(daily("吃药", 3, hour=7), days=8, shrink_slot_to_pool=5.0)
    dense = build(daily("吃药", 300, hour=7), days=300, shrink_slot_to_pool=5.0)
    key = (SlotKey(weekday=0, slot=28), "吃药")
    sparse_rate = sparse[2][key].marginal
    dense_rate = dense[2][key].marginal
    assert sparse_rate < 0.95  # 被明显拉低
    assert dense_rate > 0.99  # 几乎不动


def test_a_never_seen_cell_is_unlikely_but_not_impossible() -> None:
    """30 天没见过不等于永不发生；发布出去的必须是极小值而不是 0，也不是"没有答案"。

    格子是稀疏发布的，但曲线是密集的：这一格有没有被记过账不该决定它有没有答案。
    """

    actions = list(daily("吃药", 30, hour=7)) + [action("做饭", 0, 18)]
    _cfg, _ledger, published = build(actions, days=30, shrink_slot_to_pool=5.0)
    key = (SlotKey(weekday=0, slot=28), "做饭")
    assert key not in published  # 这一格没有计数，格子确实不发布
    cooking_morning = published[key]  # 但曲线照样答得出来
    assert 0.0 < cooking_morning.marginal < 0.05
    assert cooking_morning.count == 0.0  # 伴随值如实说"这一格一次都没见过"


def test_pooling_wraps_around_midnight() -> None:
    """钟面是环形的：00:00 的邻域必须含前一天最后那几个槽，而不是在边界上截断。"""

    assert pool_indexes(0, 2, 96) == (94, 95, 0, 1, 2)
    assert pool_indexes(95, 2, 96) == (93, 94, 95, 0, 1)
    assert pool_indexes(40, 0, 96) == (40,)  # 不池化时只有自己


def test_a_neighbouring_slot_borrows_evidence_across_midnight() -> None:
    """环形池化的**后果**：23:45 从来没发生过，但 00:00 的证据要能把它抬起来。

    用生产档参数跑——夹具那组收缩几乎是关的，借不借得到看不出来。
    """

    actions = [action("起夜", offset, 0, 5) for offset in range(28)]
    cfg = production_config()
    _cells, curves = build_curves(actions, days=28, cfg=cfg)
    curve = curves[(0, "起夜")]
    late = curve.marginal[95]  # 23:45，自身零计数，但邻域里有 00:05 的证据
    early = curve.marginal[48]  # 12:00，邻域里同样没有证据
    assert late > 3 * early


# --- 衰减与趋势 ---------------------------------------------------------------------------


def test_decay_lifts_a_habit_that_formed_recently() -> None:
    """衰减让新习惯抬头：同样 10/30 天，最近 10 天的比 30 天前的强得多。"""

    # 短窗必须真的比长窗短（config 的自洽校验），所以这里一并调小近期半衰期。
    settings = dict(decay_half_life_days=14.0, recent_half_life_days=5.0)
    recent = build(daily("晨跑", 10, hour=6, start=20), days=30, **settings)
    old = build(daily("晨跑", 10, hour=6, start=0), days=30, **settings)
    key = (SlotKey(weekday=0, slot=24), "晨跑")  # 周一 06:00
    assert recent[2][key].marginal > 3 * old[2][key].marginal


def test_trend_signals_a_habit_that_is_fading() -> None:
    """长期率还高但近期率已掉——没有这个数，换药停服后会连续误报两个月。"""

    _cells, stopped = build_curves(
        daily("吃药", 40, hour=7), days=60, decay_half_life_days=60.0
    )
    _cells2, continuing = build_curves(
        daily("吃药", 60, hour=7), days=60, decay_half_life_days=60.0
    )

    faded = stopped[(0, "吃药")].trend
    steady = continuing[(0, "吃药")].trend
    assert faded is not None and steady is not None
    assert steady == pytest.approx(1.0, rel=0.1)  # 一直在做 → 近期与长期一致
    assert faded < 0.6 * steady  # 停了 20 天 → 近期率明显低于长期率


def test_decay_weight_halves_at_the_half_life() -> None:
    assert decay_weight(0.0, 60.0) == pytest.approx(1.0)
    assert decay_weight(60.0, 60.0) == pytest.approx(0.5)
    assert decay_weight(120.0, 60.0) == pytest.approx(0.25)


# --- 边界 ---------------------------------------------------------------------------------


def test_reference_before_the_earliest_observation_is_rejected() -> None:
    with pytest.raises(PredictionTreeError, match="must not precede"):
        build(daily("吃药", 3, hour=7), days=0)


def test_slot_mapping_is_local_wall_clock() -> None:
    """槽位按本地时分映射；同一瞬时在不同偏移下落在不同槽是**正确**的。"""

    from datetime import timedelta as _td
    from datetime import timezone as _tz

    from prediction.model import SlotKey as _SlotKey
    from tests.unit.prediction.prediction_fixtures import at

    east = at(0, 7, 30)  # 07:30+08:00 → 第 30 槽
    west = east.astimezone(_tz(_td(hours=-5)))  # 同一瞬时，本地是前一天 18:30
    assert _SlotKey.of(east, slot_minutes=15).slot == 30
    assert _SlotKey.of(west, slot_minutes=15).slot == 74  # 18:30 → 74，不是"随便一个别的值"
    assert _SlotKey.of(west, slot_minutes=15).weekday == 6  # 连周几都跟着退了一天


# --- 夏令时 -------------------------------------------------------------------------------


def test_daylight_saving_day_maps_by_wall_clock_not_by_elapsed_time() -> None:
    """春季跳表那天本地时钟只有 92 个槽：02:00–03:00 根本不存在。

    槽位由**本地时分**决定，所以跳表之后的 03:05 落在 12 号槽（03:00），不会因为"距离
    午夜过了两小时零五分"被算到 8 号槽去。曝光那边仍按 96 槽记这一天——那 4 个不存在的槽
    因此被算作"在看但没发生"，一年一次、影响四个槽的分母。这是已知并接受的近似：
    真按 92/100 记账要给曝光引入日历时区依赖，代价远大于收益。
    """

    from prediction.model import ObservedAction

    new_york = ZoneInfo("America/New_York")
    spring_forward = datetime(2026, 3, 8, 3, 5, tzinfo=new_york)  # 跳表后的第一个小时
    assert spring_forward.utcoffset() == timedelta(hours=-4)

    ledger = accumulate(
        [ObservedAction(action="起床", started_at=spring_forward, day=spring_forward.date())],
        [],
        config=config(),
        reference=spring_forward.date(),
    )
    occurred = [key for key, counts in ledger.counts.items() if counts.occurred_days > 0.0]
    assert [key[0].slot for key in occurred] == [12]  # 03:00，不是"距午夜两小时零五分"的第 8 槽


def test_the_repeated_autumn_hour_still_counts_as_one_day() -> None:
    """秋季回拨那天 01:00–02:00 走两遍；同槽同日封顶 1 让 P 依然是概率。"""

    from prediction.model import ObservedAction

    new_york = ZoneInfo("America/New_York")
    day = date(2026, 11, 1)
    first_pass = datetime(2026, 11, 1, 1, 30, tzinfo=new_york, fold=0)
    second_pass = datetime(2026, 11, 1, 1, 30, tzinfo=new_york, fold=1)
    assert first_pass.utcoffset() != second_pass.utcoffset()  # 同一面钟，两个瞬时

    ledger = accumulate(
        [
            ObservedAction(action="喂猫", started_at=first_pass, day=day),
            ObservedAction(action="喂猫", started_at=second_pass, day=day),
        ],
        [],
        config=config(),
        reference=day,
    )
    counts = ledger.counts[(SlotKey(weekday=day.weekday(), slot=6), "喂猫")]  # 01:30
    assert counts.occurred_days == pytest.approx(1.0)  # 概率不会因此超过 1
    assert counts.raw_occurrences == pytest.approx(2.0)  # 强度信息仍然留着


# --- 收缩链（生产档参数；夹具那组收缩几乎是关的，测不出这里的任何东西）-------------------


def test_borrowing_starts_from_the_weakest_assumption() -> None:
    """收缩顺序：``(周几,槽) ← (周几,池化邻域) ← (跨周几,池化邻域) ← 全天平均``。

    时间邻域借用只假设"相邻时刻的率相近"（弱，通常成立）；跨周几借用假设"各周几的率相近"
    （强，对周规律行为**直接错误**）。顺序颠倒会系统性把周规律压平——而那正是加周维度要
    捕捉的东西。规格把这条列为"本轮修正的技术错误"，所以它必须有测试守着：变异测试证明
    过，把两层调换之后全部单测照样全绿。

    判据是**结果**而不是调用顺序：同一批数据下，先借时间邻域必须比先借跨周几给出更接近
    真值的估计。
    """

    # 12 周，每周二 19:00 打球；其余周几同一时刻从不打球。
    actions = weekly("打球", 12, weekday=1, hour=19)
    cfg = production_config()
    ledger = accumulate(actions, [], config=cfg, reference=reference(12 * 7 - 1))
    key = (SlotKey(weekday=1, slot=76), "打球")

    slot_key, action_name = key
    completion = completion_curves(actions, [], config=cfg, reference=reference(12 * 7 - 1))
    curves = derive_all(ledger, config=cfg, trends={}, completion=completion).curves
    correct = curves[(slot_key.weekday, action_name)].marginal[slot_key.slot]
    reversed_order = _marginal_with_reversed_shrinkage(ledger, cfg, key)
    assert correct > reversed_order
    # 真值是 1.0（12 个周二打了 12 次）；先跨周几会被其余 6 个周几的零证据拖下去。
    assert correct - reversed_order > 0.03


def _marginal_with_reversed_shrinkage(ledger, cfg, key) -> float:
    """把收缩链的前两层调换之后重算同一个格子——只在本测试里存在的对照组。"""

    from prediction import nodes

    slot_key, action = key
    pool = pool_indexes(slot_key.slot, cfg.pool_half_width, cfg.slots_per_day)
    baselines = nodes._all_day_rates(ledger, config=cfg)

    def totals(weekday, slots):
        top = bottom = 0.0
        weekdays = range(7) if weekday is None else (weekday,)
        for day_index in weekdays:
            for slot_index in slots:
                cell = SlotKey(weekday=day_index, slot=slot_index)
                exposure = ledger.exposure.get(cell)
                if exposure is None:
                    continue
                bottom += exposure.observed_days
                counts = ledger.counts.get((cell, action))
                if counts is not None:
                    top += counts.occurred_days
        return top, bottom

    # 颠倒：先同周几池化 ← 全天，再跨周几池化 ← 它，最后本格
    pooled = nodes._shrink(*totals(slot_key.weekday, pool), baselines.marginal[action], cfg.shrink_pool_to_weekday)
    cross = nodes._shrink(*totals(None, pool), pooled, cfg.shrink_weekday_to_all_day)
    return nodes._shrink(*totals(slot_key.weekday, (slot_key.slot,)), cross, cfg.shrink_slot_to_pool)


def test_production_shrinkage_keeps_the_ranking_even_where_it_flattens_the_level() -> None:
    """生产启动档下收缩很重：30 天 100% 的习惯只发布成 0.57 左右（实测）。

    这个**绝对值偏低是参数问题不是实现问题**——十三个参数全是没有数据依据的启动档
    （见 TODO(PRED-TUNING-001)），而 30 天里同一周几只有 4–5 天，样本量与收缩强度 5 同量级，
    算出 0.57 在算术上是对的。这里因此**不去断言一个我自己拍的下界**，只钉两件真正必须成立
    的事：一是排序不能塌（扎实的习惯必须远高于只见过一次的动作），二是 lift 必须把它认出来。

    绝对值本身留给留出回测去定档；那正是 evaluation 存在的理由。
    """

    cfg = production_config()
    actions = list(daily("吃药", 30, hour=7)) + [action("剪指甲", 0, 7)]
    published = publish(actions, config=cfg, reference_day=reference(29))
    habit = published[(SlotKey(weekday=0, slot=28), "吃药")]
    once = published[(SlotKey(weekday=0, slot=28), "剪指甲")]

    assert habit.marginal > 5 * once.marginal  # 排序没塌
    assert habit.lift_all_day > 10  # 这一格远高于它自己的全天平均
    assert habit.n_eff > 4  # 伴随值如实说明证据只有四个多周一


# --- 密集曲线与自适应趋势（本轮修复） ------------------------------------------------------


def test_a_slot_before_the_first_occurrence_is_estimated_the_same_way_as_one_after() -> None:
    """同一个"从没在这一格发生过"的事实，落在首次之前和之后必须走同一个机制。

    早先"树上有没有这一格"由累积率的记账规则顺手决定（当天首次之后的每个槽都留一笔账、
    之前的一个都没有），于是首次之前退到全天基线、之后拿到池化收缩估计，真实数据实测
    1,915 例、中位差 19.8 倍、最大 50 倍。
    """

    cfg = production_config()
    _cells, curves = build_curves(daily("午睡", 28, hour=13), days=28, cfg=cfg)
    curve = curves[(0, "午睡")]
    first_slot = 13 * 4  # 13:00
    before, after = curve.marginal[first_slot - 8], curve.marginal[first_slot + 8]
    # 两侧对称、量级相同——不再相差一个数量级
    assert 0.5 < before / after < 2.0
    # 而且都不是零：从没在这个点见过不等于永不发生
    assert before > 0.0


def test_the_hazard_curve_carries_mass_before_the_earliest_time_ever_seen() -> None:
    """"他今天比以往任何一天都早"不能是概率 0。

    早先危险率只在有格子的槽上有值，首次之前的质量恒为 0，于是时刻分布只能向右胖：真实
    数据实测 827 个 (动作,天) 里 821 个预计时刻偏晚、0 个偏早，p10 永远正好等于历史见过的
    最早那次。
    """

    cfg = production_config()
    _cells, curves = build_curves(daily("晨跑", 28, hour=7), days=28, cfg=cfg)
    curve = curves[(0, "晨跑")]
    assert all(value > 0.0 for value in curve.hazard)
    assert sum(curve.hazard[: 7 * 4]) > 0.0  # 07:00 之前也有质量


def test_a_weekly_habit_that_died_reads_as_gone() -> None:
    """停了十七个周一的习惯必须读到接近 0——这个字段存在的全部理由就是接住它。

    规格："没有这个数，换药停服后会连续误报两个月。"证据窗按周期缩放时若把单位弄错（按天
    分档而不是按"周几通道里的一次机会"），同一份数据会读出 0.635，"→0 = 习惯正在消失"的
    判据根本不会触发。
    """

    weeks = 57
    workouts = [item for item in weekly("健身", weeks, weekday=0, hour=20)][:40]
    trend, evidence = _trend_of(workouts, days=7 * weeks, cfg=production_config())
    assert trend is not None and trend < 0.05
    assert evidence > 0.0  # 长窗里还看得见旧证据，所以这个 0 是"消失"不是"没数据"


def test_the_evidence_window_scales_with_how_long_you_wait_between_occurrences() -> None:
    """双周行为跳过一次，不该和周频行为跳过一次读成同样严重。

    趋势是在**周几通道**里算的：一个窗口里有几次机会由通道决定（每周一次），与行为周期无关。
    随周期变的是命中率——双周行为隔一个周一才命中一次，要攒同样多的命中就得等两倍长的窗。
    """

    haircuts = [action("理发", 14 * turn, 10) for turn in range(19)]  # 每两周一次，最后一次跳过
    cfg = production_config()
    adaptive, _evidence = _trend_of(haircuts, days=14 * 20, cfg=cfg)
    fixed, _ = _trend_of(haircuts, days=14 * 20, cfg=cfg, periods={"理发": 1.0})
    assert adaptive is not None and fixed is not None
    assert fixed < 0.6 < adaptive < 0.85  # 不缩放 0.511 / 缩放 0.730


def test_a_behaviour_without_recurrence_evidence_gets_no_trend() -> None:
    """量不出节奏就定不了窗——给 None，不要偷偷按最短的窗算。

    ``recurrence.derive`` 丢弃超过 ``recurrence_window_days`` 的间隔，于是季频行为在
    ``periods`` 里根本不出现。把它当成日频（最短的窗）实测读出 2.744（"新习惯正在建立"），
    而它一点没变；换个基准日相位一变就翻到接近 0——纯噪声。
    """

    checkups = [action("体检", 112 * turn, 9) for turn in range(6)]  # 间隔 112 天 > 90 天窗
    cfg = production_config()
    at = reference(112 * 6 - 1)
    periods = {
        name: item.intervals.p50 / 86_400.0
        for name, item in derive_recurrence(checkups, config=cfg, reference=at).items()
    }
    assert "体检" not in periods
    trends = pooled_trends(checkups, [], config=cfg, reference=at, periods=periods)
    assert set(trends.values()) == {(None, 0.0)}


def test_the_trend_follows_the_behaviour_across_its_pooled_neighbourhood() -> None:
    """时刻在邻域内漂移仍算同一件事发生了——早饭从 7:15 挪到 9:00 不是"习惯消失"。

    必须用生产档跑：夹具那组 ``pool_half_width=0``，邻域退化成单槽，这条主张一次都执行不到。
    """

    drifting = [action("健身", 7 * week, 20, 0 if week < 10 else 30) for week in range(20)]
    trend, evidence = _trend_of(drifting, days=7 * 20, cfg=production_config())
    assert trend == pytest.approx(1.0, abs=0.05)
    assert evidence > 5.0


def test_period_scale_buckets_instead_of_following_a_noisy_median() -> None:
    """档位的单位是"周几通道里的一次机会"（= 一周），不是天。

    分档而不是用连续值：中位间隔本身会在 7 和 14 之间跳，窗口跟着抖会自己造出趋势变化。
    """

    assert period_scale(1.0) == 1.0  # 日频：每个周一都命中
    assert period_scale(7.0) == 1.0  # 周频：同样每个周一都命中，窗口不该拉长
    assert period_scale(8.5) == 2.0  # 偶尔跳一周 → 已经要等两个周一
    assert period_scale(14.0) == 2.0  # 双周
    assert period_scale(30.0) == 4.0  # 月频
    assert period_scale(112.0) == 4.0  # 再稀也不再拉长：档位到顶


def _trend_of(actions, *, days: int, cfg, periods=None):
    """走夜批的真实顺序取某个 (周一, 动作) 的趋势。"""

    at = reference(days - 1)
    resolved = periods
    if resolved is None:
        resolved = {
            name: item.intervals.p50 / 86_400.0
            for name, item in derive_recurrence(list(actions), config=cfg, reference=at).items()
        }
    trends = pooled_trends(list(actions), [], config=cfg, reference=at, periods=resolved)
    return next(value for key, value in trends.items() if key[0] == 0)


def test_a_cell_carries_this_slots_own_evidence_not_the_whole_actions() -> None:
    """伴随值必须与概率同源：格子上的 ``count``/``n_eff`` 说的是**这一格**的命中数与机会数。

    率现在只有曲线一份（格子上没有率可以跟它对不上），所以这里守的是另一半：把槽内概率配上
    全时段的伴随值，会让"这一格只见过一次"读成"n_eff≈19 的实测结论"，下游的支持度门直接放行。
    """

    actions = list(daily("吃药", 30, hour=7, minute=30)) + [action("吃药", 3, 19)]
    _cfg, _ledger, published = build(actions, days=30)
    morning = published[(SlotKey(weekday=0, slot=30), "吃药")]
    evening = published[(SlotKey(weekday=3, slot=76), "吃药")]
    assert morning.count > 3.0  # 五个周一都吃了
    assert evening.count == pytest.approx(1.0, abs=0.05)  # 那一格只有一次
    # 机会数是**这一格**的（30 天里该周几出现四五次），不是这个动作的全部 31 次。
    assert morning.n_eff < 6.0 and evening.n_eff < 6.0


def test_the_circular_window_matches_the_naive_neighbourhood_sum() -> None:
    """滑动窗必须与朴素的环形邻域求和逐位一致——这是它唯一的契约。

    滑窗最容易犯的错是"窗宽从 2h+1 塌成 2h 并逐槽累积漂移"：变异实测下全部定性测试照样绿
    （收缩链的跨午夜借证据、排序不塌都还成立），而池化和的相对误差中位已经到 215%。
    """

    generator = random.Random(20260901)
    for total in (3, 8, 96):
        values = [generator.random() * 5.0 for _ in range(total)]
        for half in range(0, (total - 1) // 2 + 1):
            naive = [
                sum(values[(index + offset) % total] for offset in range(-half, half + 1))
                for index in range(total)
            ]
            assert nodes._circular_window_sums(values, half) == pytest.approx(naive, abs=1e-12)


def test_a_pooling_window_wider_than_the_clock_face_is_refused() -> None:
    """窗口宽过整圈时朴素写法会重复计数、滑窗不会——两者会静默分叉，所以硬拒。"""

    with pytest.raises(PredictionTreeError, match="wider than the clock face"):
        nodes._circular_window_sums([1.0, 2.0, 3.0, 4.0], 2)


def test_the_cross_weekday_layer_borrows_from_the_pooled_neighbourhood() -> None:
    """收缩链倒数第二层借的是**跨周几的池化邻域**，不是跨周几的同一个槽。

    只测"顺序没颠倒"守不住这一条：把那一层的证据换成该槽本身（等于把时间池化从这一层摘掉），
    顺序判据照样过。构造上让证据全部落在邻域里、该槽本身跨全部周几一次都没有。
    """

    actions = [action("遛狗", offset, 10, 0) for offset in range(0, 28, 2)]  # 偶数天 10:00
    actions += [action("遛狗", offset, 10, 30) for offset in range(1, 28, 2)]  # 奇数天 10:30
    cfg = production_config()  # pool_half_width=2 → ±30 分钟，邻域盖住 10:00 与 10:30
    _cells, curves = build_curves(actions, days=28, cfg=cfg)
    middle = 10 * 4 + 1  # 10:15，跨全部周几一次都没发生过
    borrowed = curves[(0, "遛狗")].marginal[middle]
    faraway = curves[(0, "遛狗")].marginal[10 * 4 + 20]  # 15:15，邻域里什么也没有
    assert borrowed > 20 * faraway


def test_day_coverage_ignores_a_zero_width_gap() -> None:
    """曝光这一边对零宽度空白不扣任何东西——``edges`` 那边"不算洞"的立论就建立在这上面。

    真实 WP4 数据里 21 段 gap 有 17 段是零宽度，不是边角情形。
    """

    moment = at(0, 9, 30)
    zero_width = ObservedGap(started_at=moment, ended_at=moment, watched=True)
    covered = day_coverage(reference(0), [zero_width], 15)
    assert covered == [1.0] * 96


def test_the_neighbourhood_is_centred_on_where_the_behaviour_actually_happens() -> None:
    """邻域中心取该 (周几, 动作) 发生**最多**的那个槽，不是随便一个发生过的槽。

    中心选错，邻域就罩不住这个行为的常规时刻：十六个周一里十五个在 20:00 做、一个在 08:00 做，
    中心若落到 08:00，那十五次全部算成"没发生"，一个稳定的习惯会被读成正在消失。
    """

    workouts = [action("健身", 7 * week, 20) for week in range(16)]
    workouts.append(action("健身", 7 * 3, 8))  # 某个周一还额外早上做过一次
    trend, evidence = _trend_of(workouts, days=7 * 16, cfg=production_config())
    assert trend == pytest.approx(1.0, abs=0.05)
    assert evidence > 5.0
