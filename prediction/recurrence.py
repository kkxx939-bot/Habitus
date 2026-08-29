"""复发间隔：同一动作相邻两次之间隔多久。

零 IO 纯函数。它是**月频行为的唯一解法**——因为它不分桶：一年剪 12 次头发就是 11 个间隔
样本全部用在同一个估计上，而"几号"那种分桶方式会把同样的样本摊到 31 个格子里，每格 0.4 个，
什么也算不出来（见 ``TODO(PRED-TREE-001)`` 的范围一节）。

它与钟面是互补的两种信号：复发间隔回答"**该不该做了**"，钟面回答"**什么时候说**"。
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date

from prediction.config import PredictionTreeConfig
from prediction.edges import quantiles
from prediction.model import ObservedAction, RecurrenceStatistics
from prediction.nodes import decay_weight

_SECONDS_PER_DAY = 86_400.0


def derive(
    actions: Sequence[ObservedAction],
    *,
    config: PredictionTreeConfig,
    reference: date,
) -> dict[str, RecurrenceStatistics]:
    """按动作算相邻发生的间隔分位数。

    超过 ``recurrence_window_days`` 的间隔**不计入**：那不是"复发"，是中断之后重新开始
    （搬家前后各剪过一次头发，中间隔了半年，把它算进中位数只会污染估计）。
    """

    window_seconds = config.recurrence_window_days * _SECONDS_PER_DAY
    by_action: dict[str, list[ObservedAction]] = {}
    for item in actions:
        by_action.setdefault(item.action, []).append(item)

    statistics: dict[str, RecurrenceStatistics] = {}
    for action, occurrences in by_action.items():
        ordered = sorted(occurrences, key=lambda item: item.started_at)
        samples: list[tuple[float, float]] = []
        for previous, following in zip(ordered, ordered[1:], strict=False):
            interval = following.started_at.timestamp() - previous.started_at.timestamp()
            if interval <= 0.0 or interval > window_seconds:
                continue
            # 用**独立的**复发证据窗，不用钟面的 τ：月频行为在 τ=60 下有效间隔样本
            # 封顶 3.4 个，分位数永远在噪声里——复发本来就是为低频行为设的，证据窗必须更长。
            weight = decay_weight(
                float((reference - following.day).days), config.recurrence_half_life_days
            )
            samples.append((interval, weight))
        computed = quantiles(samples)
        if computed is not None:
            statistics[action] = RecurrenceStatistics(intervals=computed)
    return statistics


def overdue_ratio(statistics: RecurrenceStatistics, elapsed_seconds: float) -> float:
    """距上次已过 ``elapsed_seconds``，相对中位间隔是几倍——上层据此判断"该做了没有"。

    本层只给这个比值，不划线：多少倍算"该提醒"取决于该行为的重要性，那是语义层的判断。
    """

    median = statistics.intervals.p50
    return elapsed_seconds / median if median > 0.0 else 0.0


__all__ = ["derive", "overdue_ratio"]
