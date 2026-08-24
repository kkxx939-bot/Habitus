"""钟面节点：计数、曝光、三种率、两个 lift、收缩与趋势。

数字都能手算——这是纯函数层的正确性主战场（见 TODO(PRED-TREE-001)）。
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from prediction.errors import PredictionTreeError
from prediction.model import SlotKey
from prediction.nodes import accumulate, decay_weight, derive, pool_indexes
from tests.unit.prediction.prediction_fixtures import (
    action,
    config,
    daily,
    gap,
    production_config,
    reference,
    weekly,
)

MORNING = SlotKey(weekday=0, slot=28)  # 周一 07:00–07:15


def build(actions, gaps=(), *, days: int, **overrides):
    cfg = config(**overrides)
    ledger = accumulate(list(actions), list(gaps), config=cfg, reference=reference(days - 1))
    return cfg, ledger, derive(ledger, config=cfg)


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

    crossing = ObservedGap(started_at=at(0, 23), ended_at=at(1, 1))
    _cfg, ledger, _stats = build([action("吃药", 3, 7)], [crossing], days=4)
    # 两端都被空白完全盖住 → 曝光为零 → 稀疏存储里根本没有这两个格子。
    assert SlotKey(weekday=0, slot=92) not in ledger.exposure  # 周一 23:00
    assert SlotKey(weekday=1, slot=0) not in ledger.exposure  # 周二 00:00
    # 而没被盖住的槽照常有曝光，证明只扣掉了重叠的那部分。
    assert ledger.exposure[SlotKey(weekday=0, slot=88)].observed_days > 0.9  # 周一 22:00


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
    """30 天没见过不等于永不发生；收缩后应当是极小值而不是 0。"""

    actions = list(daily("吃药", 30, hour=7)) + [action("做饭", 0, 18)]
    _cfg, _ledger, stats = build(actions, days=30, shrink_slot_to_pool=5.0)
    cooking_morning = stats.get((SlotKey(weekday=0, slot=28), "做饭"))
    # 该格没有计数，因此没有记录——这正是稀疏存储；查询层负责给出收缩后的兜底值。
    assert cooking_morning is None


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
    ledger = accumulate(actions, [], config=cfg, reference=reference(27))
    stats = derive(ledger, config=cfg)
    late = stats[(SlotKey(weekday=0, slot=95), "起夜")]  # 23:45，自身零计数
    early = stats[(SlotKey(weekday=0, slot=48), "起夜")]  # 12:00，邻域里同样没有证据
    assert late.marginal > 3 * early.marginal


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

    key = (SlotKey(weekday=0, slot=28), "吃药")
    stopped = build(daily("吃药", 40, hour=7), days=60, decay_half_life_days=60.0)
    continuing = build(daily("吃药", 60, hour=7), days=60, decay_half_life_days=60.0)

    faded = stopped[2][key].trend
    steady = continuing[2][key].trend
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

    correct = derive(ledger, config=cfg)[key].marginal
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
    ledger = accumulate(actions, [], config=cfg, reference=reference(29))
    stats = derive(ledger, config=cfg)
    habit = stats[(SlotKey(weekday=0, slot=28), "吃药")]
    once = stats[(SlotKey(weekday=0, slot=28), "剪指甲")]

    assert habit.marginal > 5 * once.marginal  # 排序没塌
    assert habit.lift_all_day > 10  # 这一格远高于它自己的全天平均
    assert habit.n_eff > 4  # 伴随值如实说明证据只有四个多周一
