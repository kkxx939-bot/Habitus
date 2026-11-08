"""L2 生命周期维护扫描里"这一个节点这一轮做什么"的纯决策。

只看热度排名、退休资格和本轮已用名额，不读写任何存储；执行与名额计数仍由维护编排负责。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from habitus.memory.recall import MemoryRecallRanking, MemoryTemperature


class MemoryMaintenanceAction(Enum):
    SKIP = "skip"
    # 热度状态已标退休：创建 RETIRE operation 并续跑删除
    RETIRE = "retire"
    # COLD_2 且到了该类退休期：标候选或宽限期满标退休
    CONSIDER_RETIREMENT = "consider_retirement"
    # COLD_2、未压缩且本轮还有名额：送去压缩（同一版内容判过 KEEP 的由执行方跳过）
    COMPACT = "compact"


@dataclass(frozen=True)
class MemoryMaintenanceQuota:
    max_retirements: int
    max_compactions: int


def plan_maintenance_action(
    ranking: MemoryRecallRanking,
    *,
    retirement_eligible: bool,
    retired_count: int,
    compaction_attempts: int,
    quota: MemoryMaintenanceQuota,
) -> MemoryMaintenanceAction:
    if ranking.state.retired_at is not None:
        if retired_count >= quota.max_retirements:
            return MemoryMaintenanceAction.SKIP
        return MemoryMaintenanceAction.RETIRE
    if ranking.temperature is not MemoryTemperature.COLD_2:
        return MemoryMaintenanceAction.SKIP
    if retirement_eligible:
        if retired_count >= quota.max_retirements:
            return MemoryMaintenanceAction.SKIP
        return MemoryMaintenanceAction.CONSIDER_RETIREMENT
    if ranking.state.compacted_at is not None:
        return MemoryMaintenanceAction.SKIP
    if compaction_attempts >= quota.max_compactions:
        return MemoryMaintenanceAction.SKIP
    return MemoryMaintenanceAction.COMPACT


__all__ = ["MemoryMaintenanceAction", "MemoryMaintenanceQuota", "plan_maintenance_action"]
