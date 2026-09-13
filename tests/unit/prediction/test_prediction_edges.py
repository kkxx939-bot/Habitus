"""转移与并行边、删失、联合查询与复发间隔。"""

from __future__ import annotations

from datetime import UTC

import pytest

from habitus.prediction import query, recurrence
from habitus.prediction.edges import (
    NO_SUCCESSOR,
    derive,
    derive_parallels,
    pair,
    parallel_totals,
    quantiles,
    require_ordered,
)
from habitus.prediction.errors import PredictionTreeError
from habitus.prediction.model import ObservedAction, ObservedGap, SlotKey
from tests.unit.prediction.prediction_fixtures import action, at, config, gap, reference


def build(actions, gaps=(), concurrent=(), *, days: int, **overrides):
    cfg = config(**overrides)
    ledger = pair(
        list(actions), list(gaps), list(concurrent), config=cfg, reference=reference(days - 1)
    )
    return cfg, ledger


# --- 转移与 ∅ -----------------------------------------------------------------------------


def test_no_successor_is_a_first_class_outcome() -> None:
    """"洗完手之后什么都没做"必须进分母——它正是提醒逻辑最依赖的那个数。"""

    actions = []
    for offset in range(10):
        actions.append(action("洗手", offset, 12))
        if offset < 6:  # 6 天接着吃饭，4 天什么都没做
            actions.append(action("吃饭", offset, 12, 5))
    cfg, ledger = build(actions, days=10)
    stats = derive(ledger, config=cfg)

    assert stats[("洗手", "吃饭")].probability == pytest.approx(0.6, abs=0.01)
    assert stats[("洗手", NO_SUCCESSOR)].probability == pytest.approx(0.4, abs=0.01)
    total = sum(
        item.probability for (source, _target), item in stats.items() if source == "洗手"
    )
    assert total == pytest.approx(1.0, abs=0.02)  # 归一：含弃权的多分类


def test_censoring_keeps_a_broken_window_out_of_the_statistics() -> None:
    """窗口内有观测空洞时那一对**扔掉**——否则"什么都没做"会被虚增。"""

    actions = [action("洗手", offset, 12) for offset in range(10)]
    # 后 4 天洗完手之后画面就断了：这 4 次既不是转移，也不是"没做下一件事"。
    gaps = [gap(offset, 12, 15) for offset in range(6, 10)]
    cfg, ledger = build(actions, gaps, days=10)
    stats = derive(ledger, config=cfg)

    assert ledger.censored == pytest.approx(4.0, rel=0.05)
    assert stats[("洗手", NO_SUCCESSOR)].n_eff == pytest.approx(6.0, rel=0.05)
    # 若把删失算成"没做"，分母会是 10、P(∅) 仍是 1.0 但 n_eff 虚高——这里守住了 6。


def test_parallel_actions_do_not_become_fake_transitions() -> None:
    """边吃饭边看手机是**同时**不是先后；不分开会造出假因果并挤掉真正的下一步。"""

    actions = []
    concurrent = []
    for offset in range(8):
        base = len(actions)
        actions.append(action("吃饭", offset, 18))
        actions.append(action("看手机", offset, 18, 5))  # 并行
        actions.append(action("洗碗", offset, 18, 40))  # 真正的下一步
        concurrent.append((base, base + 1))
    cfg, ledger = build(actions, concurrent=concurrent, days=8)
    transitions = derive(ledger, config=cfg)
    parallels = derive_parallels(ledger)

    # 并行对没有污染转移：吃饭之后的下一步是洗碗，不是看手机
    assert ("吃饭", "看手机") not in transitions
    assert transitions[("吃饭", "洗碗")].probability > 0.9
    # 并行本身有预测价值，单独记账而不是丢掉。键按动作身份序规范化（"吃饭" < "看手机"），
    # 与这一对谁先开始无关；条件概率由查询层拿参与口径的分母现算，这里只断言对称的事实。
    assert parallels[("吃饭", "看手机")].count > 0.0
    assert parallel_totals(ledger)["吃饭"] == pytest.approx(parallels[("吃饭", "看手机")].count)


def test_participation_totals_cover_every_pair_the_action_takes_part_in() -> None:
    """分母是"A 参与的**全部**并行"——只有一条边时任何写法都恒真，得有第三个动作才测得到。"""

    actions = []
    concurrent = []
    for offset in range(6):
        base = len(actions)
        actions.append(action("吃饭", offset, 12))
        actions.append(action("看手机", offset, 12, 1))
        actions.append(action("听歌", offset, 12, 2))
        concurrent.append((base, base + 1))  # 吃饭 ∥ 看手机
        concurrent.append((base, base + 2))  # 吃饭 ∥ 听歌
        concurrent.append((base + 1, base + 2))  # 看手机 ∥ 听歌
    cfg, ledger = build(actions, concurrent=concurrent, days=6)
    parallels = derive_parallels(ledger)
    totals = parallel_totals(ledger)
    for name in ("吃饭", "看手机", "听歌"):
        taken_part_in = sum(
            item.count for key, item in parallels.items() if name in key
        )
        assert totals[name] == pytest.approx(taken_part_in)
    # 同一个动作自己与自己并行只计一次，否则 Σ P 不等于 1
    self_paired = [action("交谈", 0, 9), action("交谈", 0, 9, 1)]
    _cfg, self_ledger = build(self_paired, concurrent=[(0, 1)], days=1)
    assert parallel_totals(self_ledger)["交谈"] == pytest.approx(
        derive_parallels(self_ledger)[("交谈", "交谈")].count
    )


def test_transition_window_bounds_what_counts_as_next() -> None:
    """超出窗口的后继不算"接下来"，那一次记为"什么都没做"。"""

    actions = [action("洗手", 0, 12), action("吃饭", 0, 15)]  # 隔了 3 小时
    cfg, ledger = build(actions, days=1, transition_window_seconds=3_600.0)
    stats = derive(ledger, config=cfg)
    assert ("洗手", "吃饭") not in stats
    assert stats[("洗手", NO_SUCCESSOR)].count == pytest.approx(1.0, rel=0.01)


def test_edge_lift_uses_the_same_denominator_convention_on_both_sides() -> None:
    """lift 的分子分母口径必须一致（都含 ∅），否则被 ∅ 系统性拉偏。"""

    actions = []
    for offset in range(20):
        actions.append(action("洗手", offset, 12))
        actions.append(action("吃饭", offset, 12, 3))
        actions.append(action("看书", offset, 20))  # 与洗手无关的高频动作
    cfg, ledger = build(actions, days=20)
    stats = derive(ledger, config=cfg)
    # 洗手之后必然吃饭（P≈1），而吃饭在所有转移目标里只占一部分 → lift 明显大于 1
    assert stats[("洗手", "吃饭")].probability > 0.95
    assert stats[("洗手", "吃饭")].lift > 1.5


# --- 联合查询 -----------------------------------------------------------------------------


def test_joint_query_answers_without_any_independence_assumption() -> None:
    """周二 19:00 且刚吃完饭——两个边缘概率不能相乘，联合格子可以直接数。"""

    actions = []
    for week in range(10):
        tuesday = 1 + 7 * week
        actions.append(action("吃饭", tuesday, 19))
        actions.append(action("打球", tuesday, 19, 30))
        thursday = 3 + 7 * week
        actions.append(action("吃饭", thursday, 19))  # 周四吃饭但不打球
    cfg, ledger = build(actions, days=70)
    edges = derive(ledger, config=cfg)
    tree = _tree(edges, cfg)

    tuesday = _by_target(query.successors(tree, "吃饭", slot=SlotKey(weekday=1, slot=76)))
    thursday = _by_target(query.successors(tree, "吃饭", slot=SlotKey(weekday=3, slot=76)))
    assert tuesday["打球"].probability == pytest.approx(1.0, abs=0.01)  # 周二必打
    assert tuesday["打球"].approximate is False
    assert thursday["打球"].probability == pytest.approx(0.0, abs=0.01)  # 周四不打

    # 而边缘概率把两天混在一起，只有 0.5——这正是不能拿它当联合用的原因
    assert edges[("吃饭", "打球")].probability == pytest.approx(0.5, abs=0.05)


def test_joint_query_flags_an_approximation_when_the_cell_has_no_evidence() -> None:
    """该槽没有源动作的记录时退回近似并**标注**——不把近似值冒充成实测值。"""

    cfg, ledger = build([action("洗手", 0, 12), action("吃饭", 0, 12, 2)], days=1)
    tree = _tree(derive(ledger, config=cfg), cfg)
    empty = _by_target(query.successors(tree, "洗手", slot=SlotKey(weekday=0, slot=80)))
    assert empty["吃饭"].approximate is True


def test_a_real_action_named_like_the_sentinel_is_refused() -> None:
    """∅ 是本层的哨兵；真有个动作叫这个名字，无后继的账会静默覆盖掉真转移边。"""

    moment = action("洗手", 0, 12).started_at
    with pytest.raises(PredictionTreeError, match="reserved"):
        pair(
            [ObservedAction(action=NO_SUCCESSOR, started_at=moment, day=moment.date())],
            [],
            [],
            config=config(),
            reference=reference(0),
        )


def _by_target(candidates):
    return {item.target: item for item in candidates}


def _tree(edges, cfg):
    """把一批边包成一棵最小的树——联合查询的生产路径在 query 上，测试就该走那条。"""

    from datetime import datetime

    from habitus.prediction.model import PredictionTree

    return PredictionTree(
        built_at=datetime(2026, 8, 16, tzinfo=UTC),
        reference_day=reference(0),
        config_digest="test",
        slot_minutes=cfg.slot_minutes,
        nodes={},
        curves={},
        weekday_baselines={},
        edges=edges,
        parallels={},
        parallel_totals={},
        recurrences={},
        exposure={},
        baselines={},
        actions=tuple(sorted({source for source, _target in edges})),
        observed_days=1,
        censored_transitions=0.0,
    )


# --- 间隔 ---------------------------------------------------------------------------------


def test_interval_quantiles_answer_when_to_speak() -> None:
    actions = []
    for offset, delay_minutes in enumerate([2, 3, 3, 4, 5, 6, 20]):
        actions.append(action("洗手", offset, 12))
        actions.append(action("吃饭", offset, 12, delay_minutes))
    cfg, ledger = build(actions, days=7)
    stats = derive(ledger, config=cfg)
    intervals = stats[("洗手", "吃饭")].intervals
    assert intervals is not None
    assert intervals.p50 == pytest.approx(4 * 60, abs=90)
    assert intervals.p90 >= intervals.p50 >= intervals.p10


def test_quantiles_of_an_empty_sample_is_none_not_a_made_up_range() -> None:
    assert quantiles(()) is None


# --- 复发间隔 -----------------------------------------------------------------------------


def test_recurrence_covers_monthly_behaviour_without_bucketing() -> None:
    """一年剪 12 次头发 = 11 个间隔样本全部用上；按"几号"分桶则每桶 0.4 个，什么也算不出。"""

    actions = []
    day = 0
    for interval in (28, 33, 31, 26, 35, 30, 29, 32, 30, 31, 27):
        day += interval
        actions.append(action("剪头发", day, 14))
    cfg = config()
    stats = recurrence.derive(actions, config=cfg, reference=reference(day))
    intervals = stats["剪头发"].intervals
    assert intervals.p50 == pytest.approx(30 * 86_400, rel=0.15)
    assert recurrence.overdue_ratio(stats["剪头发"], 34 * 86_400) > 1.0


def test_recurrence_ignores_intervals_beyond_the_window() -> None:
    """中断半年之后重新开始不是"复发"，算进中位数只会污染估计。"""

    actions = [action("大扫除", 0, 10), action("大扫除", 14, 10), action("大扫除", 300, 10)]
    cfg = config(recurrence_window_days=90.0)
    stats = recurrence.derive(actions, config=cfg, reference=reference(300))
    intervals = stats["大扫除"].intervals
    # 样本数是**衰减加权**的，所以略小于 1；关键是"只有一个间隔进来"。
    assert intervals.sample_count == pytest.approx(1.0, rel=0.1)
    assert intervals.p50 == pytest.approx(14 * 86_400, rel=0.01)


# --- 输入纪律 -----------------------------------------------------------------------------


def test_unordered_actions_fail_loudly() -> None:
    later = action("洗手", 3, 12)
    earlier = action("吃饭", 1, 12)
    with pytest.raises(PredictionTreeError, match="ordered"):
        require_ordered([later, earlier])


# --- 删失与空白（本轮修复） ------------------------------------------------------------


def test_a_transition_across_an_observation_hole_is_censored_too() -> None:
    """找到了后继也要查空洞：中间断过档，我们不知道洞里是不是还发生过别的。

    只在"没找到后继"的分支查空洞，等于在观测最差的地方记下最确凿的因果。真实数据实测：
    新粒度的 DAY1 上 345 对配到后继的里有 23 对是跨着洞记的。
    """

    actions = [action("出门", 0, 9), action("回家", 0, 9, 40)]
    holed = pair(
        actions,
        [gap(0, 9, 10)],  # 09:00–10:00 没观测，正好盖住这一对之间
        [],
        config=config(),
        reference=reference(0),
    )
    assert ("出门", "回家") not in holed.transitions
    assert holed.censored > 0.0

    clean = pair(actions, [], [], config=config(), reference=reference(0))
    assert clean.transitions[("出门", "回家")] > 0.0
    assert clean.censored == 0.0


def test_a_zero_width_gap_is_not_a_hole() -> None:
    """单观测的"没读懂"段起止同刻：曝光那边扣不掉任何东西，删失这边也不能算洞。

    观测模型明文不携带时段，所以零宽度是**忠实记录**而不是坏数据；两个消费者对同一条记录
    必须给出同一个解读。真实数据实测：DAY1 的 21 段 gap 里 17 段是零宽度。
    """

    moment = at(0, 9, 30)
    zero_width = ObservedGap(started_at=moment, ended_at=moment, watched=True)
    actions = [action("出门", 0, 9), action("回家", 0, 9, 40)]
    ledger = pair(actions, [zero_width], [], config=config(), reference=reference(0))
    assert ledger.transitions[("出门", "回家")] > 0.0
    assert ledger.censored == 0.0


def test_parallel_keys_do_not_depend_on_which_one_started_first() -> None:
    """并行是对称关系，"谁先开始"是每次发生各不相同的偶然，不该决定证据落进哪个键。

    按时间序建键会把同一对行为劈成 ``(A,B)`` 与 ``(B,A)`` 两个键，读侧取对称闭包时后者
    覆盖前者：真实数据上 425 条并行边里 51 个无序对同时存在正反两键，同一个事实被读成
    count 0.989 与 6.897、概率 0.855 与 0.005。
    """

    actions = [
        action("吃饭", 0, 12),
        action("看手机", 0, 12, 5),  # 第一天吃饭先开始
        action("看手机", 1, 12),
        action("吃饭", 1, 12, 5),  # 第二天看手机先开始
    ]
    ledger = pair(
        actions, [], [(0, 1), (2, 3)], config=config(), reference=reference(1)
    )
    assert set(ledger.parallels) == {("吃饭", "看手机")}  # 一个键，不是两个
    totals = parallel_totals(ledger)
    assert totals["吃饭"] == pytest.approx(totals["看手机"])
    assert totals["吃饭"] == pytest.approx(ledger.parallels[("吃饭", "看手机")])


# --- 出处：这条边是哪几天看见的 ---------------------------------------------------------


def test_an_edge_names_the_days_it_was_seen() -> None:
    """转移边与 ``→ ∅`` 都要带上源那条行为发生的日子，语义层据此反查那天接着做了什么。"""

    actions = []
    for offset in range(5):
        actions.append(action("洗手", offset, 12))
        if offset < 3:
            actions.append(action("吃饭", offset, 12, 5))
    cfg, ledger = build(actions, days=5)
    stats = derive(ledger, config=cfg)
    assert stats[("洗手", "吃饭")].days == tuple(reference(offset) for offset in range(3))
    assert stats[("洗手", NO_SUCCESSOR)].days == tuple(reference(offset) for offset in (3, 4))


def test_a_censored_pair_leaves_no_provenance() -> None:
    """删失的那一对既不算转移也不算"什么都没做"，出处里同样不能留下这一天。

    留下了，语义侧就会去取一段我们**承认自己没看清**的历史，还以为它是一条实打实的证据。
    """

    actions = [action("洗手", offset, 12) for offset in range(4)]
    cfg, ledger = build(actions, [gap(2, 12, 14)], days=4)
    stats = derive(ledger, config=cfg)
    assert ledger.censored > 0.0
    assert reference(2) not in stats[("洗手", NO_SUCCESSOR)].days


def test_twice_in_one_day_is_two_counts_but_one_day() -> None:
    """边按次记、出处按天去重，所以 ``count`` 可以大于 ``len(days)``——两者不同量纲。"""

    actions = [action("吃药", 0, 8), action("喝水", 0, 8, 5), action("吃药", 0, 20), action("喝水", 0, 20, 5)]
    cfg, ledger = build(actions, days=1)
    edge = derive(ledger, config=cfg)[("吃药", "喝水")]
    assert edge.count == pytest.approx(2.0)
    assert edge.days == (reference(0),)


# --- 邻域联合格子 -----------------------------------------------------------------------


def _pooled_tree(cfg, ledger):
    """把一批边包成树，再补上曲线要用的最小骨架（联合查询只读 edges 与 slot_minutes）。"""

    return _tree(derive(ledger, config=cfg), cfg)


def test_the_joint_cell_can_be_spread_over_the_neighbourhood(tmp_path) -> None:
    """15 分钟一格上单格的联合样本很容易是空的，于是"上一步→下一步"频繁掉进 lift 近似。

    摊到 ±k 槽之后，分子与分母必须摊在**同一批**格子上、并且都含 ∅——少了 ∅ 那一半，
    "这个槽做完 A 通常就收工"会被算成"必然接着做 B"。
    """

    actions = []
    for week in range(6):
        # 直方图的键是 (周几, 槽)，所以现场按周递增落在同一个周几上；分钟数让它在 12:00 前后
        # 飘 20 分钟：单格只罩得住四次，邻域才罩得住六次。
        minute = 10 * (week % 3)
        actions.append(action("洗手", 7 * week, 12, minute))
        if week < 4:
            actions.append(action("吃饭", 7 * week, 12, minute + 5))
    cfg, ledger = build(actions, days=7 * 6)
    tree = _pooled_tree(cfg, ledger)
    centre = SlotKey(weekday=reference(0).weekday(), slot=48)  # 12:00–12:15

    single = _by_target(query.successors(tree, "洗手", slot=centre))
    pooled = _by_target(query.successors(tree, "洗手", slot=centre, half_width=2))
    assert pooled["吃饭"].count > single["吃饭"].count  # 邻域罩进更多次
    assert pooled["吃饭"].n_eff > single["吃饭"].n_eff
    # 分母含 ∅：一个源的全部目标（含 ∅）在同一批格子上求和，所以概率加起来正好是 1。
    assert sum(item.probability for item in pooled.values()) == pytest.approx(1.0)
    assert NO_SUCCESSOR in pooled


def test_half_width_zero_is_exactly_the_old_single_cell_behaviour(tmp_path) -> None:
    """默认值必须与改动前逐位相同，否则这个参数就不是"加了一条路"，而是改了所有既有调用。"""

    actions = []
    for week in range(5):
        actions.append(action("洗手", 7 * week, 12))
        if week < 3:
            actions.append(action("吃饭", 7 * week, 12, 5))
    cfg, ledger = build(actions, days=7 * 5)
    tree = _pooled_tree(cfg, ledger)
    centre = SlotKey(weekday=reference(0).weekday(), slot=48)
    assert query.successors(tree, "洗手", slot=centre) == query.successors(
        tree, "洗手", slot=centre, half_width=0
    )
    assert query.successors(tree, "洗手") == query.successors(tree, "洗手", half_width=0)


def test_a_neighbourhood_without_a_slot_is_refused(tmp_path) -> None:
    """没有槽就没有"邻域"可言。静默忽略会让调用方以为自己按邻域查过了。"""

    cfg, ledger = build([action("洗手", 0, 12)], days=1)
    tree = _pooled_tree(cfg, ledger)
    with pytest.raises(PredictionTreeError, match="needs a slot"):
        query.successors(tree, "洗手", half_width=2)
    for bad in (-1, True, 1.0):
        with pytest.raises(PredictionTreeError, match="half_width"):
            query.successors(tree, "洗手", slot=SlotKey(weekday=0, slot=48), half_width=bad)


def test_a_window_wider_than_the_clock_face_is_refused(tmp_path) -> None:
    """宽过整圈的环形窗口会把同一个格子数两遍；这条硬拒要与节点池化那边一字不差地一致。"""

    cfg, ledger = build([action("洗手", 0, 12)], days=1)
    tree = _pooled_tree(cfg, ledger)
    with pytest.raises(PredictionTreeError, match="count slots twice"):
        query.neighbourhood(tree, SlotKey(weekday=0, slot=48), 48)
    assert len(query.neighbourhood(tree, SlotKey(weekday=0, slot=48), 47)) == 95


def test_the_neighbourhood_wraps_inside_one_weekday_only(tmp_path) -> None:
    """周一 23:50 的邻域含周一 00:05，不含周二 00:05——与节点池化逐行做窗口和同一个语义。"""

    cfg, ledger = build([action("洗手", 0, 12)], days=1)
    tree = _pooled_tree(cfg, ledger)
    keys = query.neighbourhood(tree, SlotKey(weekday=0, slot=95), 2)
    assert {key.weekday for key in keys} == {0}
    assert sorted(key.slot for key in keys) == [0, 1, 93, 94, 95]
