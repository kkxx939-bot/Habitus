from __future__ import annotations

from datetime import date, timedelta

import pytest

from habitus.memory.conversation import (
    ConversationAddress,
    ConversationLayout,
    ConversationRangeSummaryStore,
    ConversationSummaryCompactionConfig,
    ConversationSummaryExpander,
    ConversationSummaryExpansionError,
    ConversationSummaryStore,
)
from habitus.memory.conversation.indexing import ConversationSummaryMatch, summary_reference
from habitus.pre.conversation import (
    ConversationRangeSummary,
    ConversationRangeSummaryStage,
    ConversationSummarySourceRef,
)
from tests.helpers import closed_turn, segment, segment_summary, summary_content


def test_range_fallback_expands_verified_child_summaries_without_changing_frontier(tmp_path) -> None:
    layout = ConversationLayout(tmp_path, max_conversation_tree_entries=10_000)
    segment_store = ConversationSummaryStore(layout)
    range_store = ConversationRangeSummaryStore(layout)
    address = ConversationAddress("conversation-1", date(2026, 7, 1))
    first_source = segment(
        conversation_id="conversation-1",
        segment_id="000000000000-000000000001",
        messages=closed_turn(start_sequence=0),
    )
    second_source = segment(
        conversation_id="conversation-1",
        segment_id="000000000002-000000000003",
        messages=closed_turn(start_sequence=2, prompt="改为先给结论"),
    )
    children = (segment_summary(first_source), segment_summary(second_source))
    for source, summary in zip((first_source, second_source), children, strict=True):
        segment_store.create(address, source, summary)
    content = summary_content()
    parent = ConversationRangeSummary(
        conversation_id="conversation-1",
        range_id="000000000000-000000000003",
        stage=ConversationRangeSummaryStage.RANGE,
        source_refs=tuple(ConversationSummarySourceRef.from_summary(item) for item in children),
        start_sequence=0,
        end_sequence=3,
        started_at=children[0].started_at,
        ended_at=children[-1].ended_at,
        generated_at=children[-1].generated_at + timedelta(seconds=1),
        starts_mid_turn=False,
        ends_mid_turn=False,
        **content.to_dict(),
    )
    range_store.create(address, parent, children)
    match = ConversationSummaryMatch(
        summary_reference(address, parent),
        parent,
        "parent compact summary",
        0.9,
        0.9,
    )

    expanded = ConversationSummaryExpander(segment_store, range_store).expand(
        match,
        max_chars=20_000,
    )
    assert expanded.reference == match.reference
    assert expanded.summary == parent
    assert expanded.content.count("<conversation_summary_source") == 2
    assert children[0].overview in expanded.content
    assert children[1].overview in expanded.content
    assert range_store.read(address, ConversationRangeSummaryStage.RANGE, parent.range_id) == parent
    with pytest.raises(ConversationSummaryExpansionError, match="max_source_reads"):
        ConversationSummaryExpander(
            segment_store,
            range_store,
            max_source_reads=1,
        ).expand(match, max_chars=20_000)


def test_default_expansion_read_bound_covers_default_legal_archive_fanout(tmp_path) -> None:
    layout = ConversationLayout(tmp_path, max_conversation_tree_entries=10_000)
    expander = ConversationSummaryExpander(
        ConversationSummaryStore(layout),
        ConversationRangeSummaryStore(layout),
    )
    config = ConversationSummaryCompactionConfig()
    legal_reads = config.range_to_archive.max_source_count * (
        1 + config.segment_to_range.max_source_count
    )

    assert expander.max_source_reads >= legal_reads
