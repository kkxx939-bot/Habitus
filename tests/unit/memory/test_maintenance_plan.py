"""L2 生命周期维护的单节点纯决策：退休优先于压缩，两类名额各自封顶。"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest

from habitus.memory.compaction.maintenance_plan import (
    MemoryMaintenanceAction,
    MemoryMaintenanceQuota,
    plan_maintenance_action,
)
from habitus.memory.recall import MemoryTemperature

AT = datetime(2026, 8, 3, tzinfo=UTC)
QUOTA = MemoryMaintenanceQuota(max_retirements=2, max_compactions=3)


def ranking(
    temperature: MemoryTemperature | None,
    *,
    retired: bool = False,
    compacted: bool = False,
) -> Any:
    state = SimpleNamespace(
        retired_at=AT if retired else None,
        compacted_at=AT if compacted else None,
    )
    return SimpleNamespace(state=state, temperature=temperature)


@pytest.mark.parametrize(
    ("case", "eligible", "retired_count", "attempts", "expected"),
    [
        (ranking(MemoryTemperature.HOT, retired=True), False, 0, 0, MemoryMaintenanceAction.RETIRE),
        (ranking(MemoryTemperature.HOT, retired=True), False, 2, 0, MemoryMaintenanceAction.SKIP),
        (ranking(MemoryTemperature.HOT), False, 0, 0, MemoryMaintenanceAction.SKIP),
        (ranking(None), False, 0, 0, MemoryMaintenanceAction.SKIP),
        (ranking(MemoryTemperature.COLD_2), True, 1, 0, MemoryMaintenanceAction.CONSIDER_RETIREMENT),
        (ranking(MemoryTemperature.COLD_2), True, 2, 0, MemoryMaintenanceAction.SKIP),
        (ranking(MemoryTemperature.COLD_2, compacted=True), True, 0, 0, MemoryMaintenanceAction.CONSIDER_RETIREMENT),
        (ranking(MemoryTemperature.COLD_2, compacted=True), False, 0, 0, MemoryMaintenanceAction.SKIP),
        (ranking(MemoryTemperature.COLD_2), False, 2, 2, MemoryMaintenanceAction.COMPACT),
        (ranking(MemoryTemperature.COLD_2), False, 0, 3, MemoryMaintenanceAction.SKIP),
    ],
)
def test_plan_maintenance_action(case, eligible, retired_count, attempts, expected):
    assert (
        plan_maintenance_action(
            case,
            retirement_eligible=eligible,
            retired_count=retired_count,
            compaction_attempts=attempts,
            quota=QUOTA,
        )
        is expected
    )
