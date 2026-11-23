"""Conversation、Job 联合生命周期的状态组合与安全门矩阵。"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from habitus.memory.conversation import ConversationAddress
from habitus.memory.workflow import (
    ConversationLifecycleMaintenanceResult,
    ConversationLifecycleManager,
)
from habitus.pre.conversation import ConversationBatch
from tests.helpers import BASE_TIME, closed_turn
from tests.unit.workflow.test_lifecycle_manager import lifecycle_manager


def test_constructor_rejects_summary_config_different_from_compactor(tmp_path: Path) -> None:
    manager = lifecycle_manager(tmp_path, compaction_enabled=False)
    with pytest.raises(ValueError, match="share one lifecycle config"):
        ConversationLifecycleManager(
            manager.compactor,
            manager.journal,
            manager.segment_store,
            manager.range_store,
            manager.jobs,
            manager.retirement_store,
            summary_config=replace(manager.summary_config, cleanup_batch_size=1),
        )


@pytest.mark.parametrize("now", [BASE_TIME.replace(tzinfo=None)])
def test_maintain_once_rejects_invalid_explicit_time(tmp_path: Path, now: datetime) -> None:
    manager = lifecycle_manager(tmp_path, compaction_enabled=False)
    address = ConversationAddress("conversation-1", date(2026, 7, 1))
    with pytest.raises(ValueError, match="lifecycle now"):
        asyncio.run(manager.maintain_once(address, now=now))


def test_maintain_once_normalizes_offset_time_to_utc(tmp_path: Path) -> None:
    manager = lifecycle_manager(tmp_path, compaction_enabled=False)
    address = ConversationAddress("conversation-1", date(2026, 7, 1))
    offset = BASE_TIME.astimezone(timezone(timedelta(hours=8)))
    result = asyncio.run(manager.maintain_once(address, now=offset))
    assert result.compaction.summary is None


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("released_history_segment_ids", ("",), "must contain"),
        ("deleted_memory_job_sequences", (0,), "positive integers"),
    ],
)
def test_maintenance_result_rejects_invalid_fields(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    manager = lifecycle_manager(tmp_path, compaction_enabled=False)
    address = ConversationAddress("conversation-1", date(2026, 7, 1))
    valid = asyncio.run(manager.maintain_once(address, now=BASE_TIME))
    arguments = {name: getattr(valid, name) for name in valid.__dataclass_fields__}
    arguments[field] = value
    with pytest.raises(TypeError, match=message):
        ConversationLifecycleMaintenanceResult(**arguments)  # type: ignore[arg-type]


def test_history_released_means_sealed_and_file_gone(tmp_path: Path) -> None:
    """任务清理认"原文已释放"：这一段已经封存过、原文文件不在了；没封存的段（STAGED 失败没发布）不算释放。"""

    manager = lifecycle_manager(tmp_path, compaction_enabled=False)
    address = ConversationAddress("conversation-1", date(2026, 7, 1))
    retention = manager.retention
    unsealed = "000000000000-000000000001"
    assert not retention.history_released(
        address, unsealed, manager.journal.read_state(address).archived_through_sequence
    )

    manager.journal.append(address, ConversationBatch("conversation-1", closed_turn()))
    sealed = manager.journal.seal(address, through_sequence=1).segment
    archived = manager.journal.read_state(address).archived_through_sequence
    assert not retention.history_released(address, sealed.segment_id, archived)
    assert manager.journal.release_history(address, sealed.segment_id, sealed.digest)
    assert retention.history_released(address, sealed.segment_id, archived)
    assert not retention.history_released(address, "000000000002-000000000003", archived)
