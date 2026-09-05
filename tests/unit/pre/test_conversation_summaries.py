"""Conversation 宽语义摘要、来源绑定和两级压缩 Schema 测试。"""

from dataclasses import replace
from datetime import timedelta

import pytest

from habitus.pre.conversation.summaries import (
    ConversationRangeSummary,
    ConversationRangeSummaryStage,
    ConversationSegmentSummary,
    ConversationSummaryContent,
    ConversationSummarySchemaError,
    ConversationSummarySourceKind,
    ConversationSummarySourceRef,
)
from tests.helpers import closed_turn, segment, segment_summary, summary_content


def test_summary_content_requires_complete_strict_shape_and_preserves_history_dimensions() -> None:
    content = summary_content()
    restored = ConversationSummaryContent.model_validate(content.to_dict())

    assert restored == content
    schema = ConversationSummaryContent.model_json_schema()
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {"overview", "chronology", "corrections", "ending_state", "open_threads"}

    with pytest.raises(ConversationSummarySchemaError, match="exactly"):
        ConversationSummaryContent.model_validate({**content.to_dict(), "memory_candidates": []})


@pytest.mark.parametrize(
    "kwargs",
    [
        {"overview": ""},
        {"chronology": ()},
        {"ending_state": " "},
        {"open_threads": tuple("x" * 2001 for _ in range(1))},
    ],
)
def test_summary_content_rejects_empty_or_unbounded_semantics(kwargs: dict[str, object]) -> None:
    values = summary_content().to_dict()
    values.update(kwargs)
    with pytest.raises(ConversationSummarySchemaError):
        ConversationSummaryContent.model_validate(values)


def test_segment_summary_round_trip_and_source_binding_detects_wrong_segment() -> None:
    source = segment()
    summary = segment_summary(source)
    restored = ConversationSegmentSummary.from_dict(summary.to_dict())

    restored.require_matches_source(source)
    assert restored.digest == summary.digest

    other = segment(segment_id="segment-2")
    with pytest.raises(ConversationSummarySchemaError, match="does not match"):
        restored.require_matches_source(other)


def test_segment_summary_rejects_generation_before_source_end_and_bad_digest() -> None:
    valid = segment_summary()
    with pytest.raises(ConversationSummarySchemaError, match="time range"):
        replace(valid, generated_at=valid.ended_at - timedelta(seconds=1))
    with pytest.raises(ConversationSummarySchemaError, match="SHA-256"):
        replace(valid, source_message_digest="INVALID")


def _summary_for_range(index: int) -> ConversationSegmentSummary:
    start_sequence = index * 2
    end_sequence = start_sequence + 1
    source = segment(
        segment_id=f"{start_sequence:012d}-{end_sequence:012d}",
        messages=closed_turn(start_sequence=start_sequence),
    )
    return segment_summary(source)


def test_range_summary_requires_two_contiguous_same_stage_sources() -> None:
    sources = (_summary_for_range(0), _summary_for_range(1))
    refs = tuple(ConversationSummarySourceRef.from_summary(item) for item in sources)
    content = summary_content()
    summary = ConversationRangeSummary(
        conversation_id="conversation-1",
        range_id="000000000000-000000000003",
        stage=ConversationRangeSummaryStage.RANGE,
        source_refs=refs,
        start_sequence=0,
        end_sequence=3,
        started_at=sources[0].started_at,
        ended_at=sources[-1].ended_at,
        generated_at=sources[-1].ended_at + timedelta(seconds=1),
        **content.to_dict(),
    )

    summary.require_matches_sources(sources)
    assert ConversationRangeSummary.from_dict(summary.to_dict()) == summary

    with pytest.raises(ConversationSummarySchemaError, match="between 2"):
        replace(summary, source_refs=(refs[0],), end_sequence=1, range_id="000000000000-000000000001")


def test_range_summary_rejects_gaps_wrong_source_kind_and_wrong_identity() -> None:
    first = _summary_for_range(0)
    third = _summary_for_range(2)
    refs = tuple(ConversationSummarySourceRef.from_summary(item) for item in (first, third))
    with pytest.raises(ConversationSummarySchemaError, match="contiguous"):
        ConversationRangeSummary(
            conversation_id="conversation-1",
            range_id="000000000000-000000000005",
            stage=ConversationRangeSummaryStage.RANGE,
            source_refs=refs,
            start_sequence=0,
            end_sequence=5,
            started_at=first.started_at,
            ended_at=third.ended_at,
            generated_at=third.ended_at + timedelta(seconds=1),
            **summary_content().to_dict(),
        )

    wrong_kind = ConversationSummarySourceRef(
        kind=ConversationSummarySourceKind.RANGE,
        summary_id="000000000000-000000000001",
        digest=first.digest,
        start_sequence=0,
        end_sequence=1,
    )
    with pytest.raises(ConversationSummarySchemaError, match="source kind"):
        ConversationRangeSummary(
            conversation_id="conversation-1",
            range_id="000000000000-000000000003",
            stage=ConversationRangeSummaryStage.RANGE,
            source_refs=(wrong_kind, ConversationSummarySourceRef.from_summary(_summary_for_range(1))),
            start_sequence=0,
            end_sequence=3,
            started_at=first.started_at,
            ended_at=_summary_for_range(1).ended_at,
            generated_at=_summary_for_range(1).ended_at + timedelta(seconds=1),
            **summary_content().to_dict(),
        )
