"""离线回测：十三个参数的定档工具，**不在每夜生产路径上**。

用法是留出回测——用截止日之前的数据建树，在截止日之后的若干天上评估。判据以**校准**为主
（预测 0.7 的格子实际是不是真的 70% 发生），log-loss 与 ECE 是同一件事的两个数字化侧面。
定档纪律见 ``TODO(PRED-TUNING-001)``：任何调整不涨指标不合入。

两条方法论教训写进了实现，别绕过去（它们是拿真实数据换来的）：

- **评估集是全部检验槽**，不是"发生过的槽"。只在发生槽上算，高频行为靠自相关就能刷出虚高的
  解释比例，而且这个虚高在真实数据上看不出来。这里因此对留出窗口里的**每一个日历日** ×
  每个槽 × 每个已知动作各出一个样本——"那天什么都没做"的日子恰恰是最该被预测到的阴性样本，
  按"有记录的天"收集会把它们整天丢掉，实测能把基础率虚高一倍。
- **被空白盖住的槽不进评估集**：那段时间他做了我们也看不见，把它当成"没发生"就是在惩罚
  观测质量。

置换检验也在这里，同样是离线回归工具：它回答"周几效应是不是真的"，不是每夜都要跑的东西。

未裁定的候选工具（用户提出，具体逻辑待讨论）：**反向马尔可夫计算或许适合放在回测里**——
离线回测天然"双锚都在"（未来已知），可以做遮蔽实验：在留出段人为挖 gap，用前向-后向桥接
推断被遮蔽的行为，与真值对比，并与"纯时刻先验"基线头对头。这正是"序列信息是否超越钟面"
的测量仪：若桥接赢不过时刻先验，生产侧就永远不需要它。生产路径已裁定不加反向算法。
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, timedelta

from prediction import builder, nodes, query
from prediction.config import PredictionTreeConfig
from prediction.errors import PredictionTreeError
from prediction.model import (
    MINUTES_PER_DAY,
    BehaviorSnapshot,
    ObservedAction,
    ObservedGap,
    PredictionTree,
    SlotKey,
)

_EPSILON = 1e-12


@dataclass(frozen=True)
class CalibrationBin:
    """一个概率区间里"说了多少"与"真发生了多少"的对照。"""

    lower: float
    upper: float
    count: int
    predicted: float
    observed: float

    @property
    def error(self) -> float:
        return abs(self.predicted - self.observed)


@dataclass(frozen=True)
class BacktestReport:
    """一次留出回测的成绩单。

    ``baseline_log_loss`` 是"只用整体基础率、完全不看时刻"的成绩：树必须显著赢过它，
    赢不过就说明这套参数下的时刻信息等于没用。
    """

    samples: int
    positives: int
    log_loss: float
    baseline_log_loss: float
    brier: float
    expected_calibration_error: float
    bins: tuple[CalibrationBin, ...]

    @property
    def bits_gained(self) -> float:
        """相对基线每样本省下的比特数；≤0 就是这套参数没学到东西。"""

        return (self.baseline_log_loss - self.log_loss) / math.log(2)


def split(snapshot: BehaviorSnapshot, *, cutoff: date) -> tuple[BehaviorSnapshot, BehaviorSnapshot]:
    """按截止日把快照劈成训练与留出两半（截止日当天归训练）。

    **横跨截止日的观测空白按日切开**，两边各拿自己那一段：整段留给训练侧会让留出侧的首日
    看起来"全程在看"，于是没观测到的槽被当成阴性样本打了分——正好是在惩罚观测质量。

    并行关系只在训练半边保留：留出半边只用来数"发生了没有"，不再重建任何东西。
    """

    if not isinstance(snapshot, BehaviorSnapshot):
        raise PredictionTreeError("snapshot must be a BehaviorSnapshot")
    if not isinstance(cutoff, date):
        raise PredictionTreeError("cutoff must be a date")
    train_indexes = [
        index for index, item in enumerate(snapshot.actions) if item.day <= cutoff
    ]
    position = {index: rank for rank, index in enumerate(train_indexes)}
    train = BehaviorSnapshot(
        actions=tuple(snapshot.actions[index] for index in train_indexes),
        gaps=_gaps_until(snapshot.gaps, cutoff),
        concurrent=tuple(
            (position[left], position[right])
            for left, right in snapshot.concurrent
            if left in position and right in position
        ),
        skipped_duplicates=snapshot.skipped_duplicates,
    )
    holdout = BehaviorSnapshot(
        actions=tuple(item for item in snapshot.actions if item.day > cutoff),
        gaps=_gaps_after(snapshot.gaps, cutoff),
        concurrent=(),
        skipped_duplicates=0,
    )
    return train, holdout


def _gaps_until(gaps: Sequence[ObservedGap], cutoff: date) -> tuple[ObservedGap, ...]:
    return tuple(
        piece
        for gap in gaps
        for day, piece in _by_day(gap)
        if day <= cutoff
    )


def _gaps_after(gaps: Sequence[ObservedGap], cutoff: date) -> tuple[ObservedGap, ...]:
    return tuple(
        piece
        for gap in gaps
        for day, piece in _by_day(gap)
        if day > cutoff
    )


def _by_day(gap: ObservedGap) -> list[tuple[date, ObservedGap]]:
    return [
        (day, piece)
        for day, pieces in nodes.group_gaps_by_day((gap,)).items()
        for piece in pieces
    ]


def backtest(
    snapshot: BehaviorSnapshot,
    *,
    config: PredictionTreeConfig,
    cutoff: date,
    built_at,
    through: date | None = None,
    bins: int = 10,
) -> BacktestReport:
    """建树 → 在留出窗口的每一天上逐槽逐动作打分 → 出成绩单。

    ``through`` 是留出窗口的最后一天，默认取快照里最晚的一天。它必须显式存在：留出窗口的
    边界不能从"有记录的天"推出来，否则窗口末尾那些什么都没发生的日子会被悄悄剪掉。
    """

    train, holdout = split(snapshot, cutoff=cutoff)
    if not train.actions:
        raise PredictionTreeError("the training half of the split has no actions")
    if not holdout.actions:
        raise PredictionTreeError("the holdout half of the split has no actions")
    last_day = through if through is not None else snapshot.latest_day
    if last_day is None or last_day <= cutoff:
        raise PredictionTreeError("the holdout window is empty")
    tree = builder.build(train, config=config, reference=cutoff, built_at=built_at)
    return score(
        tree,
        holdout,
        config=config,
        since=cutoff + timedelta(days=1),
        through=last_day,
        bins=bins,
    )


def score(
    tree: PredictionTree,
    holdout: BehaviorSnapshot,
    *,
    config: PredictionTreeConfig,
    since: date,
    through: date,
    bins: int = 10,
) -> BacktestReport:
    """在 ``[since, through]`` 这个留出窗口上评估一棵已建好的树。"""

    pairs = tuple(samples(tree, holdout, config=config, since=since, through=through))
    if not pairs:
        raise PredictionTreeError("the holdout window yields no evaluation samples")
    positives = sum(1 for _predicted, actual in pairs if actual)
    base_rate = positives / len(pairs)
    return BacktestReport(
        samples=len(pairs),
        positives=positives,
        log_loss=_log_loss(pairs),
        baseline_log_loss=_log_loss(tuple((base_rate, actual) for _p, actual in pairs)),
        brier=sum((p - float(actual)) ** 2 for p, actual in pairs) / len(pairs),
        expected_calibration_error=_expected_calibration_error(pairs, bins),
        bins=calibration(pairs, bins=bins),
    )


def samples(
    tree: PredictionTree,
    holdout: BehaviorSnapshot,
    *,
    config: PredictionTreeConfig,
    since: date,
    through: date,
) -> list[tuple[float, bool]]:
    """留出集：窗口里每一天 × 每个槽 × 每个已知动作一个 ``(预测概率, 是否发生)``。

    窗口由 ``since``/``through`` 显式给定，**不从记录里推**：什么都没发生的日子正是最该被
    预测到的阴性样本，按"有记录的天"收集会把它们整天丢掉，基础率因此虚高（实测两倍）。

    树上没见过的动作不进评估集——它们在树这一层本来就无从预测（那是留白该接住的事，
    不是这套参数的锅）。槽宽取**树自己带的**那一份，而不是传入的 config：树可能出自另一档
    参数，用错槽宽会静默算错。
    """

    if since > through:
        raise PredictionTreeError("the holdout window ends before it starts")
    slot_minutes = tree.slot_minutes
    slots = MINUTES_PER_DAY // slot_minutes
    occurred: set[tuple[date, int, str]] = set()
    for item in holdout.actions:
        slot = SlotKey.of(item.started_at, slot_minutes=slot_minutes)
        occurred.add((item.day, slot.slot, item.action))
    gaps_by_day = nodes.group_gaps_by_day(holdout.gaps)

    collected: list[tuple[float, bool]] = []
    day = since
    while day <= through:
        coverage = nodes.day_coverage(day, tuple(gaps_by_day.get(day, ())), slot_minutes)
        for slot_index in range(slots):
            if coverage[slot_index] <= 0.0:
                continue  # 被空白盖住的槽不进评估集
            key = SlotKey(weekday=day.weekday(), slot=slot_index)
            for action in tree.actions:
                collected.append(
                    (
                        query.marginal_at(tree, key, action),
                        (day, slot_index, action) in occurred,
                    )
                )
        day += timedelta(days=1)
    return collected


@dataclass(frozen=True)
class TimingCalibration:
    """危险率时刻分布的校准结果（PRED-RATES-001 附带条件①的前一半）。

    ``skipped`` 是"发布的危险率不足以构成像样分布"而放弃评估的 (日, 动作) 数——
    如实说"没法评"，比硬评诚实。
    """

    samples: int
    skipped: int
    median_abs_error_minutes: float
    coverage_p10_p90: float


@dataclass(frozen=True)
class CumulativeCalibration:
    """累积率"该完成线"的校准结果（PRED-RATES-001 附带条件①的后一半）。"""

    samples: int
    expected_calibration_error: float
    bins: tuple[CalibrationBin, ...]


def first_occurrence_timing(
    tree: PredictionTree,
    holdout: BehaviorSnapshot,
    *,
    since: date,
    through: date,
    min_mass: float = 0.5,
) -> TimingCalibration:
    """校准危险率的"预计时刻"：P(第 t 槽才第一次发生) = h(t)·Π_{s<t}(1−h(s))。

    只评**实际发生了**的 (日, 动作)：归一化后的链是"发生条件下几点"的分布，
    拿没发生的天评它是问错问题。缺失格的危险率按 0 计（树上没有"首次落在这"的证据）；
    链总质量低于 ``min_mass`` 的跳过并计入 ``skipped``。
    """

    slots = MINUTES_PER_DAY // tree.slot_minutes
    firsts = _first_slots(tree, holdout, since=since, through=through)
    errors: list[float] = []
    covered = 0
    skipped = 0
    for (day, action), actual in sorted(firsts.items()):
        weekday = day.weekday()
        distribution: list[float] = []
        mass = 0.0
        survive = 1.0
        for slot_index in range(slots):
            cell = tree.nodes.get((SlotKey(weekday=weekday, slot=slot_index), action))
            hazard = cell.hazard if cell is not None else 0.0
            probability = survive * hazard
            distribution.append(probability)
            mass += probability
            survive *= 1.0 - hazard
        if mass < min_mass:
            skipped += 1
            continue
        p10 = p50 = p90 = slots - 1
        accumulated = 0.0
        seen10 = seen50 = False
        for slot_index, probability in enumerate(distribution):
            accumulated += probability / mass
            if not seen10 and accumulated >= 0.1:
                p10, seen10 = slot_index, True
            if not seen50 and accumulated >= 0.5:
                p50, seen50 = slot_index, True
            if accumulated >= 0.9:
                p90 = slot_index
                break
        errors.append(abs(p50 - actual) * tree.slot_minutes)
        if p10 <= actual <= p90:
            covered += 1
    if not errors:
        raise PredictionTreeError("no (day, action) pair was evaluable for timing calibration")
    ordered = sorted(errors)
    middle = len(ordered) // 2
    median = ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2
    return TimingCalibration(
        samples=len(errors),
        skipped=skipped,
        median_abs_error_minutes=median,
        coverage_p10_p90=covered / len(errors),
    )


def cumulative_calibration(
    tree: PredictionTree,
    holdout: BehaviorSnapshot,
    *,
    since: date,
    through: date,
    bins: int = 10,
) -> CumulativeCalibration:
    """校准累积率："到这个槽为止通常做过了"说 0.9 的地方，留出段里是不是真有九成的天已做过。

    只评树上**发布了**的格子——含只有 ``earlier_days`` 的格子，它们正是累积率的主要载体。
    留出段的观测空白会让"实际做过没有"读偏低（做了没看见），沿用 covered≡1 的既定退化假设，
    定档时对带 gap 的留出段保持警惕。
    """

    slots = MINUTES_PER_DAY // tree.slot_minutes
    firsts = _first_slots(tree, holdout, since=since, through=through)
    pairs: list[tuple[float, bool]] = []
    day = since
    while day <= through:
        weekday = day.weekday()
        for slot_index in range(slots):
            key = SlotKey(weekday=weekday, slot=slot_index)
            for action in tree.actions:
                cell = tree.nodes.get((key, action))
                if cell is None:
                    continue
                actual_first = firsts.get((day, action))
                pairs.append(
                    (cell.cumulative, actual_first is not None and actual_first <= slot_index)
                )
        day += timedelta(days=1)
    if not pairs:
        raise PredictionTreeError("the holdout window yields no cumulative calibration pairs")
    return CumulativeCalibration(
        samples=len(pairs),
        expected_calibration_error=_expected_calibration_error(pairs, bins),
        bins=calibration(pairs, bins=bins),
    )


def _first_slots(
    tree: PredictionTree,
    holdout: BehaviorSnapshot,
    *,
    since: date,
    through: date,
) -> dict[tuple[date, str], int]:
    """窗口内每个 (日, 动作) 的实际首次发生槽位。"""

    if since > through:
        raise PredictionTreeError("the holdout window ends before it starts")
    firsts: dict[tuple[date, str], int] = {}
    for item in holdout.actions:
        if item.day < since or item.day > through:
            continue
        slot = SlotKey.of(item.started_at, slot_minutes=tree.slot_minutes).slot
        key = (item.day, item.action)
        if key not in firsts or slot < firsts[key]:
            firsts[key] = slot
    return firsts


def calibration(
    pairs: Sequence[tuple[float, bool]], *, bins: int = 10
) -> tuple[CalibrationBin, ...]:
    """校准曲线：把预测按概率分箱，比"说的"和"发生的"。"""

    if isinstance(bins, bool) or not isinstance(bins, int) or bins < 1:
        raise PredictionTreeError("bins must be a positive integer")
    buckets: list[list[tuple[float, bool]]] = [[] for _ in range(bins)]
    for predicted, actual in pairs:
        index = min(int(predicted * bins), bins - 1)
        buckets[index].append((predicted, actual))
    return tuple(
        CalibrationBin(
            lower=index / bins,
            upper=(index + 1) / bins,
            count=len(bucket),
            predicted=sum(p for p, _a in bucket) / len(bucket),
            observed=sum(1 for _p, a in bucket if a) / len(bucket),
        )
        for index, bucket in enumerate(buckets)
        if bucket
    )


def isotonic(pairs: Sequence[tuple[float, bool]]) -> tuple[tuple[float, float], ...]:
    """等渗回归（PAVA）：把预测单调地映射到实测频率。

    返回 ``(阈值, 校准后概率)`` 的升序折线，用 ``apply`` 查表。等渗只重排数值、不改变排序，
    所以它能修好校准而不损伤区分度——这正是执行门需要的那个概率。
    """

    ordered = sorted(pairs, key=lambda item: item[0])
    if not ordered:
        return ()
    # 先按**相同预测值**合并再跑 PAVA：等渗回归对同一个输入必须给同一个输出，
    # 不先合并的话同一个预测值会散成许多单点块，查表时命中的是其中第一个而不是它们的均值。
    grouped: list[tuple[float, float, float]] = []  # (预测值, 加权和, 权重)
    for predicted, actual in ordered:
        if grouped and grouped[-1][0] == predicted:
            value, total, weight = grouped.pop()
            grouped.append((value, total + float(actual), weight + 1.0))
        else:
            grouped.append((predicted, float(actual), 1.0))

    blocks: list[tuple[float, float, float]] = []  # (块的右端阈值, 加权和, 权重)
    for block in grouped:
        blocks.append(block)
        while len(blocks) > 1 and _mean(blocks[-2]) > _mean(blocks[-1]):
            threshold, total, weight = blocks.pop()
            _previous, previous_total, previous_weight = blocks.pop()
            blocks.append((threshold, previous_total + total, previous_weight + weight))
    return tuple((threshold, total / weight) for threshold, total, weight in blocks)


def _mean(block: tuple[float, float, float]) -> float:
    return block[1] / block[2]


def apply(curve: Sequence[tuple[float, float]], value: float) -> float:
    """按等渗折线校准一个概率；折线为空时原样返回。

    折线是**阶梯**而不是插值：落在两个块之间的值取右边那个块的拟合值。区间内没有观测，
    插值等于凭空造一个中间答案。
    """

    if not curve:
        return value
    for threshold, calibrated in curve:
        if value <= threshold:
            return calibrated
    return curve[-1][1]


def permutation_test(
    snapshot: BehaviorSnapshot,
    *,
    config: PredictionTreeConfig,
    cutoff: date,
    built_at,
    through: date | None = None,
    rounds: int = 200,
    seed: int = 0,
) -> float:
    """周几效应是不是真的：打乱"哪天"再看成绩还剩多少。

    置换保留每次行为的**时刻**、只重排它落在哪一天，因此它破坏的恰好是周几结构而不是钟面
    结构。返回 p 值 = 置换后成绩不差于真实成绩的比例；p 小说明周几维度确实带来了信息。

    这是**离线回归工具**，不进每夜生产路径。
    """

    if isinstance(rounds, bool) or not isinstance(rounds, int) or rounds < 1:
        raise PredictionTreeError("rounds must be a positive integer")
    observed = backtest(
        snapshot, config=config, cutoff=cutoff, built_at=built_at, through=through
    ).bits_gained
    generator = random.Random(seed)
    at_least_as_good = 0
    for _ in range(rounds):
        shuffled = _shuffle_days(snapshot, generator)
        try:
            permuted = backtest(
                shuffled, config=config, cutoff=cutoff, built_at=built_at, through=through
            ).bits_gained
        except PredictionTreeError:
            # 置换后切分退化（某一侧空了）。既不能当成"打赢了"也不能悄悄从分母里消失——
            # 后者会系统性压低 p 值、偏向显著。按**保守**方向计入分子。
            at_least_as_good += 1
            continue
        if permuted >= observed:
            at_least_as_good += 1
    return (at_least_as_good + 1) / (rounds + 1)


def _shuffle_days(snapshot: BehaviorSnapshot, generator: random.Random) -> BehaviorSnapshot:
    """把每条行为整体挪到另一个已观测日的同一时刻；周几结构被打乱，钟面结构不变。"""

    days = sorted({item.day for item in snapshot.actions})
    remapped = dict(zip(days, generator.sample(days, len(days)), strict=True))
    return BehaviorSnapshot(
        actions=tuple(
            sorted(
                (
                    ObservedAction(
                        action=item.action,
                        started_at=item.started_at
                        + timedelta(days=(remapped[item.day] - item.day).days),
                        day=remapped[item.day],
                    )
                    for item in snapshot.actions
                ),
                key=lambda item: item.started_at,
            )
        ),
        gaps=snapshot.gaps,
        concurrent=(),  # 下标已经失效；置换只用来看周几效应，转移边不参与本检验
        skipped_duplicates=0,
    )


def _log_loss(pairs: Sequence[tuple[float, bool]]) -> float:
    total = 0.0
    for predicted, actual in pairs:
        clamped = min(max(predicted, _EPSILON), 1.0 - _EPSILON)
        total -= math.log(clamped) if actual else math.log(1.0 - clamped)
    return total / len(pairs)


def _expected_calibration_error(pairs: Sequence[tuple[float, bool]], bins: int) -> float:
    curve = calibration(pairs, bins=bins)
    return sum(item.count * item.error for item in curve) / len(pairs)


__all__ = [
    "BacktestReport",
    "CalibrationBin",
    "CumulativeCalibration",
    "TimingCalibration",
    "apply",
    "backtest",
    "calibration",
    "cumulative_calibration",
    "first_occurrence_timing",
    "isotonic",
    "permutation_test",
    "samples",
    "score",
    "split",
]
