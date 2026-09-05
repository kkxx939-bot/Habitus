"""一趟扫描把行为快照编排成整棵树。

本模块只做编排，全部计算在 ``nodes`` / ``edges`` / ``recurrence`` 三个纯函数模块里；
它自己不碰 IO（读取在 ``source``、发布在 ``store``），所以夜批的正确性可以在纯函数层穷举验证。

**每夜整棵重建，不做增量**：重建成本低（单人一年万条 occurrence 毫秒级），而增量会让口径漂移
——kinds 词表变了就要用新口径重数历史。副作用是树的数值不保证跨夜连续，上层不得假设
"昨天 0.8 今天还是 0.8"（见 ``TODO(PRED-TREE-001)``）。
"""

from __future__ import annotations

from datetime import UTC, date, datetime

from foundation.integrity import canonical_digest
from prediction import edges, nodes, recurrence
from prediction.config import PredictionTreeConfig
from prediction.errors import PredictionTreeError
from prediction.model import BehaviorSnapshot, PredictionTree

_SECONDS_PER_DAY = 86_400.0


def config_digest(config: PredictionTreeConfig) -> str:
    """参数指纹：换了估计参数就是另一套统计，混读没有意义。

    取 ``estimation_parameters()``（运维节奏参数改了不影响树上的任何一个数字，所以不进），
    再掺上 ``nodes.estimation_constants()``——**会改变发布数字、却没有住在配置里的那些常量**
    （趋势的周期档位、发布精度）。少了后半截，改一次档位会让新旧两代树带着同一个指纹并存到
    保留代数用完，而"一次查询钉住一代"防的正是这种混读。
    """

    return canonical_digest(
        {**config.estimation_parameters(), **nodes.estimation_constants()}
    )


def build(
    snapshot: BehaviorSnapshot,
    *,
    config: PredictionTreeConfig,
    reference: date,
    built_at: datetime,
) -> PredictionTree:
    """从快照重建整棵树。

    ``reference`` 是衰减的基准日（通常是重建当天）；``built_at`` 只作发布元数据，
    不参与任何计算——保证同一输入同一参数产出同一棵树（时间戳除外）。
    """

    if not isinstance(snapshot, BehaviorSnapshot):
        raise PredictionTreeError("snapshot must be a BehaviorSnapshot")
    if not isinstance(built_at, datetime) or built_at.utcoffset() is None:
        raise PredictionTreeError("built_at must be a timezone-aware datetime")

    # 复发间隔要先算：趋势的两个证据窗按行为**自身的周期**缩放，而周期就是复发的中位间隔。
    # 它只依赖行为流本身、不依赖节点账本，所以提前算没有代价。
    recurrences = recurrence.derive(snapshot.actions, config=config, reference=reference)
    periods = {
        action: statistics.intervals.p50 / _SECONDS_PER_DAY
        for action, statistics in recurrences.items()
    }
    # **空白账在这里消解一次，然后同一份交给全部消费者。** 曝光与转移删失问的是不同的问题
    # （"那一刻在不在看" vs "这段区间干不干净"），但它们必须从**同一份**空白出发——否则同一条
    # 记录会被读成两回事，而每个新接进来的消费者都要重新面对同一道选择题。规则见
    # ``nodes.reconcile_gaps``；本函数是唯一的分发点，所以消解只能放在这里。
    gaps = nodes.reconcile_gaps(snapshot.actions, snapshot.gaps)
    ledger = nodes.accumulate(snapshot.actions, gaps, config=config, reference=reference)
    trends = nodes.pooled_trends(
        snapshot.actions,
        gaps,
        config=config,
        reference=reference,
        periods=periods,
    )
    completion = nodes.completion_curves(
        snapshot.actions, gaps, config=config, reference=reference
    )
    derived = nodes.derive_all(
        ledger, config=config, trends=trends, completion=completion
    )
    edge_ledger = edges.pair(
        snapshot.actions,
        gaps,
        snapshot.concurrent,
        config=config,
        reference=reference,
    )
    return PredictionTree(
        built_at=built_at.astimezone(UTC),
        reference_day=reference,
        config_digest=config_digest(config),
        slot_minutes=config.slot_minutes,
        nodes=derived.cells,
        curves=derived.curves,
        weekday_baselines=derived.weekday_baselines,
        edges=edges.derive(edge_ledger, config=config),
        parallels=edges.derive_parallels(edge_ledger),
        parallel_totals=edges.parallel_totals(edge_ledger),
        recurrences=recurrences,
        exposure=ledger.exposure,
        baselines=nodes.all_day_marginals(ledger, config=config),
        actions=ledger.actions,
        observed_days=ledger.observed_days,
        censored_transitions=edge_ledger.censored,
    )


__all__ = ["build", "config_digest"]
