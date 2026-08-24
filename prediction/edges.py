"""转移与并行：配对、删失、边统计。

零 IO 纯函数。三条卫生里的第三条落在本模块：**转移删失**——窗口内观测有空洞时那一对既不算
转移也不算"什么都没做"，直接扔掉。少了它，观测断档会被误读成"他做完 A 就没再做别的"，
而 ``P(∅│A)`` 恰恰是提醒逻辑最依赖的那个数。

并行必须与转移分开（见 ``TODO(PRED-TREE-001)``）：一边吃饭一边看手机会被"下一个开始"的
配对逻辑记成"吃饭→看手机，间隔 5 分钟"的假因果，而且把真正的下一步（吃完饭洗碗）挤掉——
因为配对只取下一个。行为树的 ``concurrent_with`` 已经把这件事判好了，本层照读。
"""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Mapping, Sequence, Set as AbstractSet
from dataclasses import dataclass
from datetime import date, datetime

from prediction.config import PredictionTreeConfig
from prediction.errors import PredictionTreeError
from prediction.model import (
    EdgeStatistics,
    IntervalQuantiles,
    ObservedAction,
    ObservedGap,
    SlotKey,
)
from prediction.nodes import decay_weight

NO_SUCCESSOR = "∅"


@dataclass(frozen=True)
class EdgeLedger:
    """配对结果的原始账本。``censored`` 只用于可观测性，不参与任何概率。"""

    transitions: dict[tuple[str, str], float]
    no_successor: dict[str, float]
    parallels: dict[tuple[str, str], float]
    intervals: dict[tuple[str, str], list[tuple[float, float]]]
    slot_histogram: dict[tuple[str, str], dict[SlotKey, float]]
    censored: float


def pair(
    actions: Sequence[ObservedAction],
    gaps: Sequence[ObservedGap],
    concurrent: Sequence[tuple[int, int]],
    *,
    config: PredictionTreeConfig,
    reference: date,
) -> EdgeLedger:
    """把有序的行为流配对成边。

    ``concurrent`` 是行为树声明的并行对（按 ``actions`` 的下标），配对时跳过它们并单独记账。
    锚定用**开始时刻**——``ended_at`` 常常不可知（``status_basis = observation_lost``）。
    """

    # concurrent 的下标指向**排序后**的位置（source 存的就是 rank），所以有序是前置条件而不是
    # 调用方的礼节；builder 已经查过一遍，这里作为公开纯函数自己也要守住。
    require_ordered(actions)
    # ∅ 是本层的哨兵，不是动作名。真有一个 kind_token 叫 ∅ 时，derive 的第二个循环会用
    # "无后继"的账覆盖掉同键的真转移边（计数、间隔、直方图一起被换掉），而且悄无声息。
    # 这是"我们自己的产物是否自洽"，硬拒。
    if any(item.action == NO_SUCCESSOR for item in actions):
        raise PredictionTreeError(
            f"{NO_SUCCESSOR!r} is reserved for 'no successor' and cannot be an action name"
        )
    ordered = sorted(range(len(actions)), key=lambda index: actions[index].started_at)
    concurrent_pairs = {tuple(sorted(pair_)) for pair_ in concurrent}
    concurrent_partners: dict[int, set[int]] = {}
    for left, right in concurrent_pairs:
        concurrent_partners.setdefault(left, set()).add(right)
        concurrent_partners.setdefault(right, set()).add(left)
    window = config.transition_window_seconds
    gap_starts, gap_ends = _gap_spans(gaps)
    transitions: dict[tuple[str, str], float] = {}
    no_successor: dict[str, float] = {}
    parallels: dict[tuple[str, str], float] = {}
    intervals: dict[tuple[str, str], list[tuple[float, float]]] = {}
    slot_histogram: dict[tuple[str, str], dict[SlotKey, float]] = {}
    censored = 0.0

    # 并行边直接从行为树声明的对里数，**不经过后继搜索**：并行不是"下一件事"，
    # 它既不该受转移窗口约束（长时段重叠很常见），也不该因为中间插进一个真后继而消失。
    # 旧写法两条都犯了：吃饭 ∥ 看手机、中间插一次倒水，并行关系就凭空不见了。
    for left, right in sorted(concurrent_pairs):
        if not 0 <= left < len(actions) or not 0 <= right < len(actions):
            raise PredictionTreeError("concurrent pair references an action outside the batch")
        later = max(actions[left].day, actions[right].day)
        key = (actions[left].action, actions[right].action)
        parallels[key] = parallels.get(key, 0.0) + decay_weight(
            float((reference - later).days), config.decay_half_life_days
        )

    for position, index in enumerate(ordered):
        current = actions[index]
        weight = decay_weight(float((reference - current.day).days), config.decay_half_life_days)
        deadline = current.started_at.timestamp() + window
        partners: AbstractSet[int] = concurrent_partners.get(index, frozenset())
        successor: ObservedAction | None = None
        for follower_position in range(position + 1, len(ordered)):
            follower_index = ordered[follower_position]
            follower = actions[follower_index]
            if follower.started_at.timestamp() > deadline:
                break
            if follower_index in partners:
                continue  # 同时发生的那一条不是"下一件事"，跳过继续往后找
            successor = follower
            break

        slot_key = SlotKey.of(current.started_at, slot_minutes=config.slot_minutes)
        if successor is not None:
            key = (current.action, successor.action)
            transitions[key] = transitions.get(key, 0.0) + weight
            gap_seconds = successor.started_at.timestamp() - current.started_at.timestamp()
            intervals.setdefault(key, []).append((gap_seconds, weight))
            _tally(slot_histogram, key, slot_key, weight)
            continue

        if _window_fully_observed(current.started_at, deadline, gap_starts, gap_ends):
            no_successor[current.action] = no_successor.get(current.action, 0.0) + weight
            # ∅ 也上直方图：联合查询的分母是"该槽内 source 的未删失次数"，
            # 少了这一半，"这个槽做完 A 通常就收工"会被算成"这个槽做完 A 必然接着做 B"。
            _tally(slot_histogram, (current.action, NO_SUCCESSOR), slot_key, weight)
        else:
            # 删失：窗口内有空洞，我们不知道后来发生了什么——**不得**记成"什么都没做"。
            censored += weight
    return EdgeLedger(
        transitions=transitions,
        no_successor=no_successor,
        parallels=parallels,
        intervals=intervals,
        slot_histogram=slot_histogram,
        censored=censored,
    )


def _tally(
    histogram: dict[tuple[str, str], dict[SlotKey, float]],
    key: tuple[str, str],
    slot: SlotKey,
    weight: float,
) -> None:
    cell = histogram.setdefault(key, {})
    cell[slot] = cell.get(slot, 0.0) + weight


def _gap_spans(gaps: Sequence[ObservedGap]) -> tuple[list[float], list[float]]:
    """把空白折成两条按开始时刻排序的平行数组，供二分查询。

    逐条线性扫是 O(动作数 × 空白数)，一年万条 occurrence 配上数千段空白就是 10⁷ 次
    aware-datetime 转换——和"毫秒级重建"差几个数量级，而且每次都重算同一批时间戳。
    """

    ordered = sorted(
        ((gap.started_at.timestamp(), gap.ended_at.timestamp()) for gap in gaps),
    )
    return [start for start, _end in ordered], [end for _start, end in ordered]


def _window_fully_observed(
    started_at: datetime, deadline: float, starts: Sequence[float], ends: Sequence[float]
) -> bool:
    """窗口 ``[started_at, deadline)`` 内有没有任何一段空白与之相交。"""

    start = started_at.timestamp()
    # 只有开始时刻早于窗口右端的空白才可能相交；从那里往前找第一个结束时刻越过窗口左端的。
    limit = bisect_left(starts, deadline)
    return not any(ends[index] > start for index in range(limit))


def derive(ledger: EdgeLedger, *, config: PredictionTreeConfig) -> dict[tuple[str, str], EdgeStatistics]:
    """派生边概率、lift 与间隔分位数。

    分母**含 ∅**，且 ∅ 走与其余目标**完全相同**的收缩：先按各目标的整体份额构造先验，
    再用同一个伪计数往回收。这一点不能省——旧写法里转移边收缩、∅ 是裸比值，结果
    ``Σ_b P(b│a) + P(∅│a)`` 在生产档参数下只有 0.93（收缩借走的质量没有任何一方交出来），
    而 ``P(∅│a)`` 正是提醒逻辑最依赖的那个数。

    归一化的确切说法是：**对全部已知目标（含 ∅）求和为 1**。已发布的边是稀疏的，只列出
    该源真的去过的目标，因此列出来的那些加起来 ≤ 1；差额恰好是先验分给"该源从没去过的
    目标"的那部分质量，也就是这条边上的逃逸概率。

    lift 的分子分母必须同口径（都含 ∅），否则 lift 会被 ∅ 系统性拉偏。
    """

    outgoing: dict[str, float] = {}
    for (source, _target), count in ledger.transitions.items():
        outgoing[source] = outgoing.get(source, 0.0) + count
    for source, count in ledger.no_successor.items():
        outgoing[source] = outgoing.get(source, 0.0) + count

    shares = _target_shares(ledger, config)
    statistics: dict[tuple[str, str], EdgeStatistics] = {}
    for (source, target), count in ledger.transitions.items():
        statistics[(source, target)] = _edge(
            count=count,
            opportunities=outgoing.get(source, 0.0),
            share=shares.get(target, 0.0),
            config=config,
            intervals=quantiles(ledger.intervals.get((source, target), ())),
            histogram=ledger.slot_histogram.get((source, target), {}),
        )
    for source, count in ledger.no_successor.items():
        statistics[(source, NO_SUCCESSOR)] = _edge(
            count=count,
            opportunities=outgoing.get(source, 0.0),
            share=shares.get(NO_SUCCESSOR, 0.0),
            config=config,
            # ∅ 不是一个动作，没有"下一件事什么时候开始"可言。
            intervals=None,
            histogram=ledger.slot_histogram.get((source, NO_SUCCESSOR), {}),
        )
    return statistics


def _edge(
    *,
    count: float,
    opportunities: float,
    share: float,
    config: PredictionTreeConfig,
    intervals: IntervalQuantiles | None,
    histogram: Mapping[SlotKey, float],
) -> EdgeStatistics:
    probability = (count + config.shrink_edge * share) / (opportunities + config.shrink_edge)
    return EdgeStatistics(
        count=count,
        probability=probability,
        lift=probability / share if share > 0.0 else 0.0,
        n_eff=opportunities,
        intervals=intervals,
        slot_histogram=dict(histogram),
    )


def _target_shares(ledger: EdgeLedger, config: PredictionTreeConfig) -> dict[str, float]:
    """各目标（含 ∅）在全部转移机会里的整体份额，是收缩链的先验。

    平滑写成 ``(n + ε) / (总数 + 目标数·ε)`` 而不是 ``(n + ε) / (总数 + ε)``——后者的份额
    加起来不等于 1，收缩之后每条边都会少掉一块质量。
    """

    incoming: dict[str, float] = {}
    for (_source, target), count in ledger.transitions.items():
        incoming[target] = incoming.get(target, 0.0) + count
    for _source, count in ledger.no_successor.items():
        incoming[NO_SUCCESSOR] = incoming.get(NO_SUCCESSOR, 0.0) + count
    total = sum(incoming.values())
    if total <= 0.0:
        return {}
    epsilon = config.laplace_epsilon
    denominator = total + len(incoming) * epsilon
    return {target: (count + epsilon) / denominator for target, count in incoming.items()}


def derive_parallels(ledger: EdgeLedger) -> dict[tuple[str, str], EdgeStatistics]:
    """并行边：``P(与 A 同时做 B)``；它不是转移，不进转移的概率分布。"""

    totals: dict[str, float] = {}
    for (source, _target), count in ledger.parallels.items():
        totals[source] = totals.get(source, 0.0) + count
    return {
        (source, target): EdgeStatistics(
            count=count,
            probability=count / totals[source] if totals.get(source) else 0.0,
            lift=0.0,
            n_eff=totals.get(source, 0.0),
            intervals=None,
            slot_histogram={},
        )
        for (source, target), count in ledger.parallels.items()
    }


def quantiles(samples: Sequence[tuple[float, float]]) -> IntervalQuantiles | None:
    """加权分位数（秒）；样本为空返回 None，而不是编一个区间出来。"""

    if not samples:
        return None
    ordered = sorted(samples, key=lambda item: item[0])
    total = sum(weight for _value, weight in ordered)
    if total <= 0.0:
        return None
    return IntervalQuantiles(
        p10=_weighted_quantile(ordered, total, 0.10),
        p50=_weighted_quantile(ordered, total, 0.50),
        p90=_weighted_quantile(ordered, total, 0.90),
        sample_count=total,
    )


def _weighted_quantile(
    ordered: Sequence[tuple[float, float]], total: float, fraction: float
) -> float:
    target = total * fraction
    cumulative = 0.0
    for value, weight in ordered:
        cumulative += weight
        if cumulative >= target:
            return value
    return ordered[-1][0]


def require_ordered(actions: Sequence[ObservedAction]) -> None:
    """配对依赖时间序；调用方若给了乱序数据，这里立刻失败而不是静默算错。"""

    for previous, following in zip(actions, actions[1:], strict=False):
        if following.started_at < previous.started_at:
            raise PredictionTreeError("actions must be ordered by started_at")


__all__ = [
    "NO_SUCCESSOR",
    "EdgeLedger",
    "derive",
    "derive_parallels",
    "pair",
    "quantiles",
    "require_ordered",
]
