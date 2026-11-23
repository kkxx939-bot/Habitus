"""裁定 108：摘要压缩被模型以"这一条输入有问题"拒收后，同一组来源不再重问；来源变了才再试。"""

from __future__ import annotations

import asyncio
from datetime import date, timedelta
from pathlib import Path

import pytest

from habitus.memory.conversation import ConversationAddress
from habitus.memory.conversation.compaction import (
    ConversationSegmentSummaryCompactionConfig,
    ConversationSummaryCompactionConfig,
)
from habitus.model_client import ModelContentSafetyError, ModelTransportError
from habitus.pre.conversation import ConversationBatch
from tests.helpers import BASE_TIME, closed_turn, segment_summary
from tests.unit.workflow.test_lifecycle_manager import lifecycle_manager


def _manager(tmp_path: Path):  # type: ignore[no-untyped-def]
    config = ConversationSummaryCompactionConfig(
        enabled=True,
        segment_to_range=ConversationSegmentSummaryCompactionConfig(min_age_days=1, min_source_count=2),
    )
    manager = lifecycle_manager(tmp_path, compaction_enabled=True, compaction_config=config)
    return manager


def _summaries(manager, address: ConversationAddress, starts: tuple[int, ...]) -> None:  # type: ignore[no-untyped-def]
    for start in starts:
        manager.journal.append(address, ConversationBatch(address.conversation_id, closed_turn(start_sequence=start)))
        source = manager.journal.seal(address, through_sequence=start + 1).segment
        manager.segment_store.create(address, source, segment_summary(source))


@pytest.mark.parametrize("error", [ModelContentSafetyError("blocked")])
def test_a_refused_source_group_is_not_asked_again_until_its_sources_change(tmp_path: Path, error: Exception) -> None:
    manager = _manager(tmp_path)
    compactor = manager.compactor
    address = ConversationAddress("conversation-1", date(2026, 7, 1))
    _summaries(manager, address, (0, 2))
    asked: list[int] = []

    async def generate(plan):  # type: ignore[no-untyped-def]
        asked.append(len(plan.sources))
        raise error

    compactor.generator.generate = generate  # type: ignore[method-assign]
    later = BASE_TIME + timedelta(days=30)
    first = asyncio.run(compactor.compact_once(address, now=later))
    assert first.summary is None and "content_safety" in first.reason
    second = asyncio.run(compactor.compact_once(address, now=later))
    assert asked == [2] and second.summary is None  # 同一组不再问
    _summaries(manager, address, (4,))  # 多了一份来源：这一组变了，再试一次
    asyncio.run(compactor.compact_once(address, now=later + timedelta(days=30)))
    assert asked == [2, 3]


def test_a_waiting_failure_is_not_recorded_and_is_raised_for_the_next_cycle(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    compactor = manager.compactor
    address = ConversationAddress("conversation-1", date(2026, 7, 1))
    _summaries(manager, address, (0, 2))

    async def generate(_plan):  # type: ignore[no-untyped-def]
        raise ModelTransportError("down")

    compactor.generator.generate = generate  # type: ignore[method-assign]
    with pytest.raises(ModelTransportError):
        asyncio.run(compactor.compact_once(address, now=BASE_TIME + timedelta(days=30)))
    assert compactor.refusals.keys(address) == frozenset()
