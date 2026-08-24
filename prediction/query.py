"""对一代已发布的树提问。

四类问题对应四个函数：这个时刻会开始做什么（``slot_outlook``）、做完这件接着做什么
（``successors``）、跟这件同时会做什么（``parallels``）、这件事该做了没有
（``recurrence_status``）。累积率不单独成一类，它作为字段挂在节点候选上——"到这个点为止
今天通常做了没有"和"这个槽会不会开始做"来自同一批计数。

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
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime

from prediction.edges import NO_SUCCESSOR
from prediction.errors import PredictionTreeError
from prediction.model import (
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
    """一个格子的完整答案：全部候选 + 两条留白曲线。

    ``irregularity`` 是候选分布的归一化熵——"认识这些动作，但此刻没有赢家"；
    ``escape`` 是 Good-Turing 式的"可能发生没见过的事"。两条都高就该闭嘴。

    ``candidates`` 里会混有 ``count == 0`` 的条目：那是"当天更早做过这个动作"留下的账
    （危险率的风险集与累积率靠它），不是这一格的候选。它们照常返回是因为 ``cumulative``
    这个问题要用到它们，但两条留白曲线**不把它们算进去**。
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
    lift: float
    n_eff: float
    count: float
    intervals: IntervalQuantiles | None
    approximate: bool


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
    """这个格子里 ``action`` 开始的概率；格子上没有记录时退回该动作自己的全天平均。

    退回**不是** 0：三十天没在这个点见过不等于永不发生，给 0 会让任何依赖它的比值直接爆掉。
    退回的是最弱的那层先验（全天平均），因为已发布的树是稀疏的，邻域与跨周几两层的中间量
    没有随树落盘——这是稀疏存储换来的近似，方向偏保守（略高于真值）。
    """

    statistics = tree.nodes.get((slot, action))
    if statistics is not None:
        return statistics.marginal
    return tree.baselines.get(action, 0.0)


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
    return SlotOutlook(
        slot=slot,
        candidates=tuple(
            _candidate(action, cells[action]) for action in sorted(cells)
        ),
        irregularity=_normalized_entropy(cells),
        escape=_escape(cells),
    )


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

    存储只按"先封口的在前"存一个方向（沿行为树的 ``concurrent_with`` 纪律），但并行在语义上
    是对称的：查"看手机"必须也能查到"吃饭 ∥ 看手机"。对称闭包在**读侧**取，不在存储里存两份。
    """

    _require_tree(tree)
    outgoing = {
        (target if source == action else source): statistics
        for (source, target), statistics in tree.parallels.items()
        if action in (source, target)
    }
    return tuple(
        EdgeCandidate(
            target=target,
            probability=outgoing[target].probability,
            lift=outgoing[target].lift,
            n_eff=outgoing[target].n_eff,
            count=outgoing[target].count,
            intervals=outgoing[target].intervals,
            approximate=False,
        )
        for target in sorted(outgoing)
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
    """该动作在该格的时刻提升；树上没有这一格就是 1.0（不提升也不惩罚）。

    与 ``marginal_at`` 的兜底一致：没有格子证据时退回全天平均，而全天平均相对自身的提升
    按定义就是 1。
    """

    statistics = tree.nodes.get((slot, action))
    return statistics.lift_all_day if statistics is not None else 1.0


def _candidate(action: str, statistics: NodeStatistics) -> NodeCandidate:
    return NodeCandidate(
        action=action,
        marginal=statistics.marginal,
        hazard=statistics.hazard,
        cumulative=statistics.cumulative,
        lift_all_day=statistics.lift_all_day,
        lift_weekday=statistics.lift_weekday,
        n_eff=statistics.n_eff,
        trend=statistics.trend,
        count=statistics.counts.occurred_days,
    )


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
    "EdgeCandidate",
    "NodeCandidate",
    "RecurrenceStatus",
    "SlotOutlook",
    "marginal_at",
    "parallels",
    "recurrence_status",
    "slot_at",
    "slot_outlook",
    "successors",
]
