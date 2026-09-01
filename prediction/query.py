"""对一代已发布的树提问。

五类问题：这个时刻会开始做什么（``slot_outlook``）、这个行为通常几点、范围多宽
（``day_outlook``）、做完这件接着做什么（``successors``）、跟这件同时会做什么
（``parallels``）、这件事该做了没有（``recurrence_status``）。另有按单点取值的
``marginal_at`` / ``hazard_at`` / ``cumulative_at`` 与整格取值的 ``node_at``。

累积率不单独成一类问题，它作为字段挂在节点候选上——"到这个点为止今天通常做了没有"和
"这个槽会不会开始做"来自同一批计数。

三条纪律，全部来自 ``TODO(PRED-TREE-001)`` 的组合契约：

- **一次查询钉住一代**：每个函数都接一棵完整的 ``PredictionTree``，本层不碰存储、
  不会中途换代。
- **全量返回，不 top-N**：每周二打球在按概率的排行榜上永远排在洗手后面，截断等于在树这一层
  就废掉了预测范围里的一半。返回序按名字排，**排序不是排名**——不给"取前 N 个"留下顺手的接口。
- **禁止朴素相乘**：两个边缘概率相乘等于把基线乘两遍。先查联合格子，格子空了才退回 lift 相乘，
  并把结果标成近似。

本层不持有当日状态：风险集（"今天还没发生的"）由上层拿 ``cumulative`` 对比当日实况自行构造。
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime

from prediction.edges import NO_SUCCESSOR
from prediction.errors import PredictionTreeError
from prediction.model import (
    DayCurve,
    EdgeStatistics,
    IntervalQuantiles,
    NodeStatistics,
    PredictionTree,
    SlotKey,
)
from prediction.recurrence import overdue_ratio


@dataclass(frozen=True)
class NodeCandidate:
    """一个格子里的一个动作。``count`` 单列，因为单次巧合只能靠计数本身识别。"""

    action: str
    marginal: float
    hazard: float
    cumulative: float
    lift_all_day: float
    lift_weekday: float
    n_eff: float
    trend: float | None
    count: float


@dataclass(frozen=True)
class SlotOutlook:
    """一个格子的完整答案：全部候选 + 累积率 + 两条留白曲线。

    ``irregularity`` 是候选分布的归一化熵——"认识这些动作，但此刻没有赢家"；
    ``escape`` 是 Good-Turing 式的"可能发生没见过的事"。两条都高就该闭嘴。

    ``candidates`` 覆盖这个周几**全部有曲线的动作**，不只是历史上恰好落在这一格的那些——
    率一律取曲线在这个槽的取值，``count``/``n_eff`` 是这一格自己的命中数与机会数（没发生过
    就是 0 与该槽的曝光）。**"树上有没有这一格"不得决定谁能当候选**：一个每周二 20:00/20:15
    交替打球的人，20:30 那一格的边际率是全天基线的 46 倍，而它历史上从没恰好落在那一格；
    按格子筛会让这个信号整格消失，同时 ``escape`` 报 1.0（读侧含义是"太杂，闭嘴"）。这也是
    组合契约里"查询必须全量返回，不得筛掉重要低频行为"的同一条——筛选条件换成"历史上恰好
    落在这一格"，对有时间抖动的周频行为效果一样。

    两条留白曲线**只看真的发生过的格子**（``_seen`` 按 ``occurred_days > 0`` 过滤）：把从没
    发生过的候选算进去会凭空拉低熵、也会稀释逃逸质量。
    """

    slot: SlotKey
    candidates: tuple[NodeCandidate, ...]
    irregularity: float
    escape: float


@dataclass(frozen=True)
class EdgeCandidate:
    """一条边的答案（``target`` 可能是 ``∅``，即"接下来什么都没做"）。

    **伴随值必须与概率同源**：给了槽位、且该槽有联合样本时，``count`` 与 ``n_eff`` 说的就是
    那一格的命中数与机会数，不是整条边的。旧写法把槽内概率配上全时段的伴随值，于是"这一格
    只见过一次"会被读成"n_eff≈19 的实测结论"，下游的支持度门会直接放行——正是规格里
    "每个数字必带伴随值"要防的那个失真。

    ``approximate`` 为真表示该槽没有联合样本、概率是由 lift 近似出来的，上层据此降低信任
    或干脆闭嘴——这个标记不能丢，丢了近似值就冒充了实测值。
    """

    target: str
    probability: float
    # 并行边没有 lift：转移边的 lift 有明确口径（该边份额 ÷ 目标整体份额，两边都含 ∅），
    # 并行没有对应的分母。填一个恒为 0 的占位会被读成"比基线低到不可能"，所以给 None。
    lift: float | None
    n_eff: float
    count: float
    intervals: IntervalQuantiles | None
    approximate: bool


@dataclass(frozen=True)
class DayOutlook:
    """一个行为在某个周几的时间画像：通常几点、范围多宽、在升还是在降。

    ``earliest``/``median``/``latest`` 是危险率时刻分布的 p10/p50/p90；``mass`` 是归一化之前
    的总质量（"当天会发生一次"的整体概率），质量低时那三个分位说的是"万一发生的话几点"。
    ``trend`` 与它的伴随值 ``trend_n_eff`` 见 ``DayCurve``。
    """

    weekday: int
    action: str
    earliest: SlotKey
    median: SlotKey
    latest: SlotKey
    mass: float
    trend: float | None
    trend_n_eff: float


@dataclass(frozen=True)
class RecurrenceStatus:
    """复发查询的答案：间隔分位数与"已经拖了多久"的倍数。"""

    action: str
    intervals: IntervalQuantiles
    overdue: float


def slot_at(tree: PredictionTree, moment: datetime) -> SlotKey:
    """把一个**本地**时刻映射到这棵树的钟面位置；槽宽随树发布，不用另行传入。"""

    return SlotKey.of(moment, slot_minutes=tree.slot_minutes)


def marginal_at(tree: PredictionTree, slot: SlotKey, action: str) -> float:
    """这个格子里 ``action`` 开始的概率。

    取值只有**一条路径**：这个 (周几, 动作) 的曲线在这个槽的取值。曲线是密集的，所以"这一格
    从没发生过"与"这一格发生过"用的是同一个机制（三层收缩），不再分叉。只有这个周几**从来
    没做过**这个动作时才退到全天平均——那是真的没有任何时刻证据可借。

    退回**不是** 0：三十天没在这个点见过不等于永不发生，给 0 会让任何依赖它的比值直接爆掉。
    """

    curve = tree.curves.get((slot.weekday, action))
    if curve is not None:
        return curve.marginal[slot.slot]
    return tree.baselines.get(action, 0.0)


def hazard_at(tree: PredictionTree, slot: SlotKey, action: str) -> float:
    """"如果到现在还没做，这个槽会做"的概率；时刻分布逐槽累乘用的就是它。

    与 ``marginal_at`` 同一条纪律：整条钟面上都有值。曲线缺失（这个周几从没做过）时给 0——
    没有任何"首次落在这个周几"的证据，编一个数出来没有依据。
    """

    curve = tree.curves.get((slot.weekday, action))
    return curve.hazard[slot.slot] if curve is not None else 0.0


def cumulative_at(tree: PredictionTree, slot: SlotKey, action: str) -> float:
    """到这个槽为止今天通常做了没有；缺失检测的落点。

    值来自 ``nodes.completion_curves`` 的两级分解 ``π · F(t)``：单调不减、落在 [0,1]，且在
    小样本下被收缩往基线拉——旧的裸比值在样本少时只会给 0 或 1，一次观测就能让"该完成线"
    满信心触发。

    曲线缺失（这个周几从没做过这个动作）时给 0——"没有该完成的期望"，这与 ``hazard_at``
    同向、与 ``marginal_at`` 退到全天基线**不同向**：那两个回答的是"会不会做"，可以借全天的
    证据；这一个回答的是"今天到现在该做完几成"，没有该周几的证据就没有期望可言，编一个数
    出来会让缺失检测凭空报警。曲线**存在**但当天最早的发生还没到的槽同样是 0，那是分布函数
    在质量到来之前的正常取值，不是"没有证据"。
    """

    curve = tree.curves.get((slot.weekday, action))
    return curve.cumulative[slot.slot] if curve is not None else 0.0


def node_at(tree: PredictionTree, slot: SlotKey, action: str) -> NodeCandidate | None:
    """这一格关于某个动作的完整答案：三种率 + 两个 lift + 伴随值 + 趋势。

    这个周几从没做过这个动作时返回 None——那时连曲线都没有，只有全天基线可退，而"退到基线"
    是 ``marginal_at`` 的语义，不是一个候选。
    """

    _require_tree(tree)
    curve = tree.curves.get((slot.weekday, action))
    if curve is None:
        return None
    exposure = tree.exposure.get(slot)
    return _candidate(
        tree,
        action,
        slot,
        curve,
        tree.nodes.get((slot, action)),
        exposure.observed_days if exposure is not None else 0.0,
    )


def slot_outlook(tree: PredictionTree, slot: SlotKey) -> SlotOutlook:
    """这个格子里会开始做什么。候选全量返回，按名字排序。"""

    _require_tree(tree)
    if not isinstance(slot, SlotKey):
        raise PredictionTreeError("slot must be a SlotKey")
    cells = {
        action: statistics
        for (key, action), statistics in tree.nodes.items()
        if key == slot
    }
    curves = {
        action: curve
        for (weekday, action), curve in tree.curves.items()
        if weekday == slot.weekday
    }
    exposure = tree.exposure.get(slot)
    opportunities = exposure.observed_days if exposure is not None else 0.0
    return SlotOutlook(
        slot=slot,
        candidates=tuple(
            _candidate(tree, action, slot, curves[action], cells.get(action), opportunities)
            for action in sorted(curves)
        ),
        irregularity=_normalized_entropy(cells),
        escape=_escape(cells),
    )


def day_outlook(tree: PredictionTree, weekday: int, action: str) -> DayOutlook | None:
    """这个行为在这个周几**通常几点、范围多宽**，以及它在升还是在降。

    时窗由危险率逐槽累乘得到：``P(第 t 槽才第一次发生) = h(t)·Π_{s<t}(1−h(s))``，归一化后取
    p10/p50/p90。这条链此前只存在于 ``evaluation.first_occurrence_timing``（离线定档工具），
    对外没有接口——"这个行为通常几点"是预测层最基本的问题，不该只有回测能回答。

    ``mass`` 是归一化之前的总质量，即"当天会发生一次"的整体概率。它必须跟着窗口一起给：
    质量很低时那三个分位数说的是"万一发生的话大概几点"，不是"今天会在几点"。
    """

    _require_tree(tree)
    curve = tree.curves.get((weekday, action))
    if curve is None:
        return None
    survive = 1.0
    distribution: list[float] = []
    mass = 0.0
    for hazard in curve.hazard:
        probability = survive * hazard
        distribution.append(probability)
        mass += probability
        survive *= 1.0 - hazard
    if mass <= 0.0:
        return None
    slots = [_quantile_slot(distribution, mass, fraction) for fraction in (0.10, 0.50, 0.90)]
    return DayOutlook(
        weekday=weekday,
        action=action,
        earliest=SlotKey(weekday=weekday, slot=slots[0]),
        median=SlotKey(weekday=weekday, slot=slots[1]),
        latest=SlotKey(weekday=weekday, slot=slots[2]),
        mass=mass,
        trend=curve.trend,
        trend_n_eff=curve.trend_n_eff,
    )


def _quantile_slot(distribution: Sequence[float], mass: float, fraction: float) -> int:
    accumulated = 0.0
    for slot, probability in enumerate(distribution):
        accumulated += probability / mass
        if accumulated >= fraction:
            return slot
    return len(distribution) - 1


def successors(
    tree: PredictionTree, source: str, *, slot: SlotKey | None = None
) -> tuple[EdgeCandidate, ...]:
    """做完 ``source`` 接着做什么。

    给了 ``slot`` 就按组合契约走：先查该槽的联合格子（无独立性假设），格子空了才退回
    ``P(b│a) × lift_全天(b @ 槽)`` 的近似并截断到 1。这里刻意**不**再乘 ``lift_周几``——
    格子键已含周几，``lift_全天`` 里已经包着这份提升，再乘一遍就是重复计入。
    """

    _require_tree(tree)
    outgoing = _outgoing(tree.edges, source)
    # 联合查询的分母对该源的每个目标都是同一个，算一次就够。
    opportunities = (
        0.0
        if slot is None
        else sum(item.slot_histogram.get(slot, 0.0) for item in outgoing.values())
    )
    return tuple(
        _successor(tree, target, outgoing[target], slot, opportunities)
        for target in sorted(outgoing)
    )


def parallels(tree: PredictionTree, action: str) -> tuple[EdgeCandidate, ...]:
    """跟 ``action`` 同时会做什么。并行不是转移，它有自己的分母，永不参与转移的归一。

    存储按**动作身份序**规范化成一个键（并行是对称关系，谁先开始是每次发生各不相同的偶然），
    方向在这里给出：``P(与 A 同时做 B) = count(A,B) ÷ A 参与的全部并行``。分母来自
    ``tree.parallel_totals``，所以两个方向问同一对得到的是同一份证据的两个条件概率，而不是
    两半被劈开的证据——旧写法按时间序建键、又按"存储源"各自归一，真实数据上同一个事实被读成
    count 0.989 与 6.897、概率 0.855 与 0.005。
    """

    _require_tree(tree)
    total = tree.parallel_totals.get(action, 0.0)
    counts: dict[str, float] = {}
    for (left, right), statistics in tree.parallels.items():
        if action == left:
            counts[right] = counts.get(right, 0.0) + statistics.count
        elif action == right:
            counts[left] = counts.get(left, 0.0) + statistics.count
    return tuple(
        EdgeCandidate(
            target=target,
            probability=counts[target] / total if total > 0.0 else 0.0,
            lift=None,
            n_eff=total,
            count=counts[target],
            intervals=None,
            approximate=False,
        )
        for target in sorted(counts)
    )


def recurrence_status(
    tree: PredictionTree, action: str, *, elapsed_seconds: float
) -> RecurrenceStatus | None:
    """距上次 ``action`` 已经过了这么久，该做了没有。没有间隔样本时返回 None。"""

    _require_tree(tree)
    statistics = tree.recurrences.get(action)
    if statistics is None:
        return None
    return RecurrenceStatus(
        action=action,
        intervals=statistics.intervals,
        overdue=overdue_ratio(statistics, elapsed_seconds),
    )


# --- 内部 -------------------------------------------------------------------------------


def _successor(
    tree: PredictionTree,
    target: str,
    statistics: EdgeStatistics,
    slot: SlotKey | None,
    opportunities: float,
) -> EdgeCandidate:
    """一条边在给定槽位下的答案；伴随值随概率一起换口径。"""

    if slot is None or opportunities <= 0.0:
        if slot is not None and target != NO_SUCCESSOR:
            # 该槽没有任何联合样本：退回 lift 近似并如实标注。∅ 没有自己的时刻分布可借
            # （它是"没有下一件事"，不是一个动作），保持边缘值。
            return EdgeCandidate(
                target=target,
                probability=min(1.0, statistics.probability * _node_lift(tree, target, slot)),
                lift=statistics.lift,
                n_eff=statistics.n_eff,
                count=statistics.count,
                intervals=statistics.intervals,
                approximate=True,
            )
        return EdgeCandidate(
            target=target,
            probability=statistics.probability,
            lift=statistics.lift,
            n_eff=statistics.n_eff,
            count=statistics.count,
            intervals=statistics.intervals,
            approximate=False,
        )

    hits = statistics.slot_histogram.get(slot, 0.0)
    probability = hits / opportunities
    return EdgeCandidate(
        target=target,
        probability=probability,
        # 这一格相对该边全时段份额的提升；两边口径一致（都含 ∅），所以可以直接比。
        lift=probability / statistics.probability if statistics.probability > 0.0 else 0.0,
        n_eff=opportunities,
        count=hits,
        # 间隔分布不按槽切分：单格样本太少，切了只会得到一个没意义的分位数。
        intervals=statistics.intervals,
        approximate=False,
    )


def _outgoing(
    edges: Mapping[tuple[str, str], EdgeStatistics], source: str
) -> dict[str, EdgeStatistics]:
    return {
        target: statistics
        for (edge_source, target), statistics in edges.items()
        if edge_source == source
    }


def _node_lift(tree: PredictionTree, action: str, slot: SlotKey) -> float:
    """该动作在该格的时刻提升；这个周几从没做过就是 1.0（不提升也不惩罚）。

    与 ``marginal_at`` 同源：分子取曲线在这个槽的边际率，分母取该动作的全天平均。没有曲线时
    退回全天平均，而全天平均相对自身的提升按定义就是 1。
    """

    baseline = tree.baselines.get(action, 0.0)
    if baseline <= 0.0:
        return 1.0
    return marginal_at(tree, slot, action) / baseline


def _candidate(
    tree: PredictionTree,
    action: str,
    slot: SlotKey,
    curve: DayCurve,
    cell: NodeStatistics | None,
    opportunities: float,
) -> NodeCandidate:
    """把曲线、基线与这一格的账本拼成一个候选。

    率全部来自曲线（唯一一份）；两个 lift 由率与已发布的基线现算；伴随值来自格子——没有格子
    就是"这一格从没发生过"，命中数 0、机会数取该槽的曝光。趋势取自曲线：它是这个行为的属性，
    不是这一格的属性。
    """

    marginal = curve.marginal[slot.slot]
    weekday_baseline = tree.weekday_baselines.get(action)
    return NodeCandidate(
        action=action,
        marginal=marginal,
        hazard=curve.hazard[slot.slot],
        cumulative=curve.cumulative[slot.slot],
        lift_all_day=_lift(marginal, tree.baselines.get(action, 0.0)),
        lift_weekday=_lift(
            marginal, weekday_baseline[slot.slot] if weekday_baseline is not None else 0.0
        ),
        n_eff=cell.n_eff if cell is not None else opportunities,
        trend=curve.trend,
        count=cell.counts.occurred_days if cell is not None else 0.0,
    )


def _lift(value: float, baseline: float) -> float:
    return value / baseline if baseline > 0.0 else 0.0


def _normalized_entropy(cells: Mapping[str, NodeStatistics]) -> float:
    """h_不规律：候选分布摊得越平越高。单一候选为 0，完全没有候选按满值算。"""

    weights = _seen(cells)
    if not weights:
        return 1.0
    if len(weights) < 2:
        return 0.0
    total = sum(weights)
    entropy = -sum((weight / total) * math.log(weight / total) for weight in weights)
    return entropy / math.log(len(weights))


def _escape(cells: Mapping[str, NodeStatistics]) -> float:
    """h_未见：Good-Turing 式逃逸质量，"只见过一次"的那部分占比。

    权重是衰减的，所以"恰好出现一次"没有精确定义；这里用"加权计数不超过一次"当近似——
    一个动作若只在这一格出现过一次，它的加权计数至多是 1（越久远越小）。

    **这个近似会低估逃逸**：一个 100 天前的孤例只贡献 0.06 的权重而不是 1，于是留白比真正的
    Good-Turing 更少。方向与"宁可多留白"相反，是已知并接受的偏差——衰减的本意就是让久远的
    观测说话更轻，逃逸质量跟着轻是一致的，而不是要靠它多留白。真要精确的 N₁ 得另存一份
    不衰减的出现次数，那是为一个伴随值加一整套账本。
    """

    weights = _seen(cells)
    if not weights:
        return 1.0
    singletons = sum(weight for weight in weights if weight <= 1.0)
    return min(1.0, singletons / sum(weights))


def _seen(cells: Mapping[str, NodeStatistics]) -> list[float]:
    """这一格**真的发生过**的那些动作的计数。

    格子上还会挂着只有 ``earlier_days`` 的记录（那天更早做过这个动作，于是当天其后每个槽
    都留了一笔账，用来算危险率的风险集与累积率）。它们的 ``occurred_days`` 是 0——不是候选，
    把它们算进留白会凭空拉低熵、也会稀释逃逸质量（实测把"两个候选五五开"的 1.0 压成 0.63）。
    """

    return [
        statistics.counts.occurred_days
        for statistics in cells.values()
        if statistics.counts.occurred_days > 0.0
    ]


def _require_tree(tree: PredictionTree) -> None:
    if not isinstance(tree, PredictionTree):
        raise PredictionTreeError("tree must be a PredictionTree")


__all__ = [
    "DayOutlook",
    "EdgeCandidate",
    "NodeCandidate",
    "RecurrenceStatus",
    "SlotOutlook",
    "cumulative_at",
    "day_outlook",
    "hazard_at",
    "marginal_at",
    "node_at",
    "parallels",
    "recurrence_status",
    "slot_at",
    "slot_outlook",
    "successors",
]
