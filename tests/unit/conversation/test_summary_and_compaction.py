"""Segment Summary 的来源绑定、不可变存储和两阶段连续压缩规划测试。"""

import hashlib
import json
from datetime import date, timedelta
from pathlib import Path

import pytest

from habitus.memory.conversation import (
    ConversationAddress,
    ConversationJournalConfig,
    ConversationLayout,
    ConversationRangeSummaryCompactionConfig,
    ConversationSegmentSummaryCompactionConfig,
    ConversationSummaryCompactionConfig,
    ConversationSummaryCompactionError,
    ConversationSummaryCompactionPlanner,
    ConversationSummaryFrontier,
    ConversationSummaryRetirementError,
    ConversationSummaryRetirementStore,
    ConversationSummaryStore,
)
from habitus.memory.conversation.indexing import ConversationSummaryReference, ConversationSummaryStage
from habitus.pre.conversation import ConversationRangeSummaryStage
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
    path.write_bytes(path.read_bytes().replace(b"{", b"{\n", 1))
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


def test_recently_used_summary_splits_eligible_chain_without_hot_cold_ranking() -> None:
    first, second = sources()
    config = ConversationSummaryCompactionConfig(
        segment_to_range=ConversationSegmentSummaryCompactionConfig(
            min_age_days=0,
            min_source_count=2,
            max_wait_days=180,
            max_source_count=10,
            max_source_chars=100_000,
        )
    )
    plan = ConversationSummaryCompactionPlanner(config).plan(
        ConversationSummaryFrontier((first, second), (), ()),
        ConversationRangeSummaryStage.RANGE,
        now=BASE_TIME + timedelta(days=1),
        protected_source_ids=frozenset({first.segment_id}),
    )
    assert plan is None


def test_compaction_frontier_and_plan_reject_overlap_gap_and_cross_conversation() -> None:
    first, second = sources()
    overlapping = segment_summary(
        segment(segment_id="000000000001-000000000002", messages=closed_turn(start_sequence=1))
    )
    with pytest.raises(ConversationSummaryCompactionError, match="overlapping"):
        ConversationSummaryFrontier((first, overlapping), (), ())

    gapped = segment_summary(segment(segment_id="000000000004-000000000005", messages=closed_turn(start_sequence=4)))
    from habitus.memory.conversation import ConversationSummaryCompactionPlan

    with pytest.raises(ValueError, match="contiguous"):
        ConversationSummaryCompactionPlan(ConversationRangeSummaryStage.RANGE, (first, gapped), "source_count")
    other = segment_summary(
        segment(
            conversation_id="conversation-2",
            segment_id="000000000002-000000000003",
            messages=closed_turn(start_sequence=2),
        )
    )
    with pytest.raises(ValueError, match="one conversation"):
        ConversationSummaryCompactionPlan(ConversationRangeSummaryStage.RANGE, (first, other), "source_count")


_ARCHIVE_ID = "000000000000-000000000003"


@pytest.mark.parametrize(
    ("field", "bad", "message"),
    [
        ("start_sequence", "0", "retirement manifest start_sequence must be non-negative"),
        ("end_sequence", 3.0, "retirement manifest end_sequence must be positive"),
        ("expected_use_version", True, "retirement manifest expected_use_version must be non-negative"),
        ("archive.identity", 1, "summary reference identity is invalid"),
        ("archive.digest", None, "summary retirement digest must be lowercase SHA-256"),
    ],
)
def test_retirement_manifest_read_rejects_wrongly_typed_fields(
    tmp_path: Path, field: str, bad: object, message: str
) -> None:
    """磁盘上类型错的退休清单字段在 ``from_dict`` 解码时就被拒（W16：类型检查在边界）。"""

    address = ConversationAddress("conversation-1", date(2026, 7, 1))
    reference = ConversationSummaryReference(address, ConversationSummaryStage.ARCHIVE, _ARCHIVE_ID)
    archive: dict[str, object] = {"identity": reference.identity, "digest": "a" * 64}
    value: dict[str, object] = {
        "schema": "conversation_summary_retirement_v1",
        "archive": archive,
        "start_sequence": 0,
        "end_sequence": 3,
        "ranges": [],
        "segments": [],
        "expected_use_version": 0,
        "phase": "retiring",
        "prepared_at": "2026-07-01T00:00:00Z",
        "updated_at": "2026-07-01T00:00:00Z",
    }
    if field.startswith("archive."):
        archive[field.removeprefix("archive.")] = bad
    else:
        value[field] = bad
    store = ConversationSummaryRetirementStore(tmp_path)
    store.initialize()
    digest = hashlib.sha256(reference.identity.encode("utf-8")).hexdigest()
    (store.root / f"{digest}.json").write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ConversationSummaryRetirementError, match="failed to read") as caught:
        store.read(reference)
    assert isinstance(caught.value.__cause__, ValueError)
    assert message in str(caught.value.__cause__)
