"""Segment Summary 的来源绑定、不可变存储和两阶段连续压缩规划测试。"""

from datetime import date, timedelta
from pathlib import Path

import pytest

from memory.conversation import (
    ConversationAddress,
    ConversationJournalConfig,
    ConversationLayout,
    ConversationRangeSummaryCompactionConfig,
    ConversationSegmentSummaryCompactionConfig,
    ConversationSummaryCompactionConfig,
    ConversationSummaryCompactionError,
    ConversationSummaryCompactionPlanner,
    ConversationSummaryFrontier,
    ConversationSummaryStore,
)
from pre.conversation import ConversationRangeSummaryStage
from tests.helpers import BASE_TIME, closed_turn, segment, segment_summary


def sources():
    first_segment = segment(
        segment_id="000000000000-000000000001",
        messages=closed_turn(start_sequence=0),
    )
    second_segment = segment(
        segment_id="000000000002-000000000003",
        messages=closed_turn(start_sequence=2),
    )
    return segment_summary(first_segment), segment_summary(second_segment)


def test_summary_store_is_immutable_idempotent_and_revalidates_source_digest(tmp_path: Path) -> None:
    config = ConversationJournalConfig()
    layout = ConversationLayout(
        tmp_path,
        max_conversation_tree_entries=config.max_conversation_tree_entries,
    )
    store = ConversationSummaryStore(layout)
    address = ConversationAddress("conversation-1", date(2026, 7, 1))
    source = segment(
        segment_id="000000000000-000000000001",
        messages=closed_turn(start_sequence=0),
    )
    summary = segment_summary(source)

    assert store.create(address, source, summary) == summary
    assert store.create(address, source, summary) == summary
    assert store.list(address) == (summary,)

    changed = segment(
        segment_id=source.segment_id,
        messages=closed_turn(start_sequence=0, prompt="已经修改的原文"),
    )
    with pytest.raises(Exception, match="does not match"):
        store.read(address, changed)


def test_summary_store_rejects_noncanonical_or_path_mismatched_content(tmp_path: Path) -> None:
    config = ConversationJournalConfig()
    layout = ConversationLayout(
        tmp_path,
        max_conversation_tree_entries=config.max_conversation_tree_entries,
    )
    store = ConversationSummaryStore(layout)
    address = ConversationAddress("conversation-1", date(2026, 7, 1))
    source = segment(
        segment_id="000000000000-000000000001",
        messages=closed_turn(start_sequence=0),
    )
    summary = segment_summary(source)
    store.create(address, source, summary)
    path = layout.summary_path(address, source.segment_id)
    path.write_text(path.read_text().replace("{", "{\n", 1))
    with pytest.raises(Exception, match="valid conversation summary"):
        store.read(address, source)


def test_compaction_planner_selects_only_old_contiguous_sources_by_count() -> None:
    first, second = sources()
    config = ConversationSummaryCompactionConfig(
        segment_to_range=ConversationSegmentSummaryCompactionConfig(
            min_age_days=0,
            min_source_count=2,
            max_wait_days=180,
            max_source_count=10,
            max_source_chars=100_000,
        ),
        range_to_archive=ConversationRangeSummaryCompactionConfig(
            min_age_days=180,
            min_source_count=2,
            max_source_count=10,
            max_source_chars=100_000,
        ),
    )
    plan = ConversationSummaryCompactionPlanner(config).plan(
        ConversationSummaryFrontier((first, second), (), ()),
        ConversationRangeSummaryStage.RANGE,
        now=BASE_TIME + timedelta(days=1),
    )
    assert plan is not None
    assert plan.sources == (first, second)
    assert plan.trigger == "source_count"
    assert tuple(ref.summary_id for ref in plan.source_refs) == (
        first.segment_id,
        second.segment_id,
    )


def test_compaction_frontier_and_plan_reject_overlap_gap_and_cross_conversation() -> None:
    first, second = sources()
    overlapping = segment_summary(
        segment(segment_id="000000000001-000000000002", messages=closed_turn(start_sequence=1))
    )
    with pytest.raises(ConversationSummaryCompactionError, match="overlapping"):
        ConversationSummaryFrontier((first, overlapping), (), ())

    gapped = segment_summary(
        segment(segment_id="000000000004-000000000005", messages=closed_turn(start_sequence=4))
    )
    from memory.conversation import ConversationSummaryCompactionPlan

    with pytest.raises(ValueError, match="contiguous"):
        ConversationSummaryCompactionPlan(
            ConversationRangeSummaryStage.RANGE, (first, gapped), "source_count"
        )
    other = segment_summary(
        segment(
            conversation_id="conversation-2",
            segment_id="000000000002-000000000003",
            messages=closed_turn(start_sequence=2),
        )
    )
    with pytest.raises(ValueError, match="one conversation"):
        ConversationSummaryCompactionPlan(
            ConversationRangeSummaryStage.RANGE, (first, other), "source_count"
        )
