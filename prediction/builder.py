"""一趟扫描把行为快照编排成整棵树。

本模块只做编排，全部计算在 ``nodes`` / ``edges`` / ``recurrence`` 三个纯函数模块里；
它自己不碰 IO（读取在 ``source``、发布在 ``store``），所以夜批的正确性可以在纯函数层穷举验证。

**每夜整棵重建，不做增量**：重建成本低（单人一年万条 occurrence 毫秒级），而增量会让口径漂移
——kinds 词表变了就要用新口径重数历史。副作用是树的数值不保证跨夜连续，上层不得假设
"昨天 0.8 今天还是 0.8"（见 ``TODO(PRED-TREE-001)``）。
"""

from __future__ import annotations

from datetime import date, datetime, timezone

from foundation.integrity import canonical_digest
from prediction import edges, nodes, recurrence
from prediction.config import PredictionTreeConfig
from prediction.errors import PredictionTreeError
from prediction.model import BehaviorSnapshot, PredictionTree


def config_digest(config: PredictionTreeConfig) -> str:
    """参数指纹：换了估计参数就是另一套统计，混读没有意义。

    只取 ``estimation_parameters()``——运维节奏参数改了不影响树上的任何一个数字。
    """

    return canonical_digest(config.estimation_parameters())


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

    ledger = nodes.accumulate(
        snapshot.actions, snapshot.gaps, config=config, reference=reference
    )
    edge_ledger = edges.pair(
        snapshot.actions,
        snapshot.gaps,
        snapshot.concurrent,
        config=config,
        reference=reference,
    )
    return PredictionTree(
        built_at=built_at.astimezone(timezone.utc),
        reference_day=reference,
        config_digest=config_digest(config),
        slot_minutes=config.slot_minutes,
        nodes=nodes.derive(ledger, config=config),
        edges=edges.derive(edge_ledger, config=config),
        parallels=edges.derive_parallels(edge_ledger),
        recurrences=recurrence.derive(
            snapshot.actions, config=config, reference=reference
        ),
        exposure=ledger.exposure,
        baselines=nodes.all_day_marginals(ledger, config=config),
        actions=ledger.actions,
        observed_days=ledger.observed_days,
        censored_transitions=edge_ledger.censored,
    )


__all__ = ["build", "config_digest"]
