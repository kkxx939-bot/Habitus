"""Conversation 摘要内容、来源绑定、连续范围和两级压缩矩阵。"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta

import pytest

import pre.conversation.summaries.model as summary_model
from pre.conversation.summaries import (
    ConversationRangeSummary,
    ConversationRangeSummaryStage,
    ConversationSegmentSummary,
    ConversationSummaryContent,
    ConversationSummarySchemaError,
    ConversationSummarySourceKind,
    ConversationSummarySourceRef,
)
from tests.helpers import BASE_TIME, closed_turn, segment, segment_summary, summary_content


def _content_values(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "overview": "overview",
        "chronology": ("step one",),
        "corrections": (),
        "ending_state": "ending",
        "open_threads": (),
    }
    values.update(overrides)
    return values


@pytest.mark.parametrize("field", ["overview", "ending_state"])
@pytest.mark.parametrize("value", ["text", " 中文 ", "multi\nline"])
def test_summary_required_text_accepts_any_non_whitespace_only_value(field: str, value: str) -> None:
    values = _content_values(**{field: value})
    assert getattr(ConversationSummaryContent(**values), field) == value  # type: ignore[arg-type]


@pytest.mark.parametrize("field", ["overview", "ending_state"])
@pytest.mark.parametrize("value", ["", " ", "\n\t", None, 1, [], {}])
def test_summary_required_text_rejects_empty_or_non_text_values(field: str, value: object) -> None:
    with pytest.raises(ConversationSummarySchemaError):
        ConversationSummaryContent(**_content_values(**{field: value}))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "values"),
    [
        ("chronology", ["one"]),
        ("chronology", ("one", "two")),
        ("corrections", []),
        ("corrections", ["fixed"]),
        ("open_threads", ()),
        ("open_threads", ("todo",)),
    ],
)
def test_summary_text_sequences_accept_list_or_tuple_and_freeze_to_tuple(field: str, values: object) -> None:
    content = ConversationSummaryContent(**_content_values(**{field: values}))  # type: ignore[arg-type]
    assert isinstance(getattr(content, field), tuple)


@pytest.mark.parametrize("field", ["chronology", "corrections", "open_threads"])
@pytest.mark.parametrize("value", [None, "text", {}, 1, ("",), (" ",), (1,), ({},)])
def test_summary_text_sequences_reject_non_lists_and_invalid_items(field: str, value: object) -> None:
    with pytest.raises(ConversationSummarySchemaError):
        ConversationSummaryContent(**_content_values(**{field: value}))  # type: ignore[arg-type]


def test_summary_chronology_is_required_but_other_sequences_may_be_empty() -> None:
    with pytest.raises(ConversationSummarySchemaError, match="at least one"):
        ConversationSummaryContent(**_content_values(chronology=()))  # type: ignore[arg-type]
    assert ConversationSummaryContent(**_content_values(corrections=(), open_threads=()))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "value", "accepted"),
    [
        ("overview", "x" * 16_000, True),
        ("overview", "x" * 16_001, False),
        ("ending_state", "x" * 4_000, True),
        ("ending_state", "x" * 4_001, False),
        ("chronology", tuple("x" for _ in range(128)), True),
        ("chronology", tuple("x" for _ in range(129)), False),
        ("corrections", tuple("x" for _ in range(128)), True),
        ("corrections", tuple("x" for _ in range(129)), False),
        ("open_threads", tuple("x" for _ in range(128)), True),
        ("open_threads", tuple("x" for _ in range(129)), False),
        ("chronology", ("x" * 2_000,), True),
        ("chronology", ("x" * 2_001,), False),
        ("corrections", ("x" * 2_001,), False),
        ("open_threads", ("x" * 2_001,), False),
    ],
)
def test_summary_content_enforces_each_character_and_list_boundary(field: str, value: object, accepted: bool) -> None:
    if accepted:
        assert ConversationSummaryContent(**_content_values(**{field: value}))  # type: ignore[arg-type]
    else:
        with pytest.raises(ConversationSummarySchemaError, match="exceeds"):
            ConversationSummaryContent(**_content_values(**{field: value}))  # type: ignore[arg-type]


def test_summary_content_schema_matches_runtime_bounds_and_closed_shape() -> None:
    schema = ConversationSummaryContent.model_json_schema()
    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    properties = schema["properties"]
    assert properties["overview"]["maxLength"] == 16_000
    assert properties["ending_state"]["maxLength"] == 4_000
    for field in ("chronology", "corrections", "open_threads"):
        assert properties[field]["maxItems"] == 128
        assert properties[field]["items"]["maxLength"] == 2_000


@pytest.mark.parametrize("value", [None, [], "summary", 1, True])
def test_summary_content_model_validate_requires_mapping(value: object) -> None:
    with pytest.raises(ConversationSummarySchemaError, match="object"):
        ConversationSummaryContent.model_validate(value)


@pytest.mark.parametrize("field", ["overview", "chronology", "corrections", "ending_state", "open_threads"])
def test_summary_content_model_validate_rejects_each_missing_field(field: str) -> None:
    payload = summary_content().to_dict()
    payload.pop(field)
    with pytest.raises(ConversationSummarySchemaError, match="exactly"):
        ConversationSummaryContent.model_validate(payload)


@pytest.mark.parametrize("extra", ["memory", "owner", "confidence", "tags", "source"])
def test_summary_content_model_validate_rejects_unknown_fields(extra: str) -> None:
    with pytest.raises(ConversationSummarySchemaError, match="exactly"):
        ConversationSummaryContent.model_validate({**summary_content().to_dict(), extra: True})


def _segment_summary(index: int = 0, *, conversation_id: str = "conversation-1") -> ConversationSegmentSummary:
    start = index * 2
    source = segment(
        conversation_id=conversation_id,
        segment_id=f"{start:012d}-{start + 1:012d}",
        messages=closed_turn(start_sequence=start),
    )
    return segment_summary(source)


@pytest.mark.parametrize("field", ["conversation_id", "segment_id"])
@pytest.mark.parametrize("value", ["conversation-1", "中文", "a.b-c_1", "000000000000-000000000001"])
def test_segment_summary_accepts_safe_identifiers(field: str, value: str) -> None:
    assert getattr(replace(_segment_summary(), **{field: value}), field) == value


@pytest.mark.parametrize("field", ["conversation_id", "segment_id"])
@pytest.mark.parametrize("value", ["", ".", "..", "../escape", "a/b", "a\\b", " leading", "trailing ", None, 1])
def test_segment_summary_rejects_unsafe_identifiers(field: str, value: object) -> None:
    with pytest.raises(ConversationSummarySchemaError):
        replace(_segment_summary(), **{field: value})


@pytest.mark.parametrize("field", ["start_sequence", "end_sequence"])
@pytest.mark.parametrize("value", [0, 1, 10**12])
def test_segment_summary_accepts_non_negative_sequence_values(field: str, value: int) -> None:
    base = _segment_summary()
    updates = {field: value}
    if field == "start_sequence" and value > base.end_sequence:
        updates["end_sequence"] = value
    if field == "end_sequence" and value < base.start_sequence:
        updates["start_sequence"] = value
    assert getattr(replace(base, **updates), field) == value


@pytest.mark.parametrize("field", ["start_sequence", "end_sequence"])
@pytest.mark.parametrize("value", [-1, True, False, 1.0, "1", None])
def test_segment_summary_rejects_negative_boolean_or_non_integer_sequence(field: str, value: object) -> None:
    with pytest.raises(ConversationSummarySchemaError):
        replace(_segment_summary(), **{field: value})


def test_segment_summary_rejects_reversed_sequence_range() -> None:
    with pytest.raises(ConversationSummarySchemaError, match="range"):
        replace(_segment_summary(), start_sequence=2, end_sequence=1)


@pytest.mark.parametrize("field", ["started_at", "ended_at", "generated_at"])
@pytest.mark.parametrize("value", [BASE_TIME, BASE_TIME.isoformat(), BASE_TIME.isoformat().replace("+00:00", "Z")])
def test_segment_summary_normalizes_all_temporal_fields(field: str, value: datetime | str) -> None:
    base = _segment_summary()
    updates: dict[str, object] = {field: value}
    if field == "started_at":
        updates["ended_at"] = max(base.ended_at, BASE_TIME)
    if field == "ended_at":
        updates["started_at"] = min(base.started_at, BASE_TIME)
        updates["generated_at"] = max(base.generated_at, BASE_TIME)
    if field == "generated_at":
        updates["ended_at"] = min(base.ended_at, BASE_TIME)
    assert isinstance(getattr(replace(base, **updates), field), datetime)


@pytest.mark.parametrize("field", ["started_at", "ended_at", "generated_at"])
@pytest.mark.parametrize("value", [datetime(2026, 7, 1), "not-a-date", None, 1])
def test_segment_summary_rejects_naive_invalid_or_non_temporal_fields(field: str, value: object) -> None:
    with pytest.raises(ConversationSummarySchemaError):
        replace(_segment_summary(), **{field: value})


def test_segment_summary_rejects_started_after_end_or_generation_before_end() -> None:
    base = _segment_summary()
    with pytest.raises(ConversationSummarySchemaError, match="time range"):
        replace(base, started_at=base.ended_at + timedelta(seconds=1))
    with pytest.raises(ConversationSummarySchemaError, match="time range"):
        replace(base, generated_at=base.ended_at - timedelta(seconds=1))


def test_segment_summary_source_binding_detects_each_changed_source_dimension() -> None:
    source = segment(segment_id="000000000000-000000000001")
    value = segment_summary(source)
    value.require_matches_source(source)
    mutations = [
        replace(value, conversation_id="other"),
        replace(value, segment_id="other"),
        replace(value, source_message_digest="a" * 64),
        replace(value, start_sequence=value.start_sequence + 1),
        replace(value, end_sequence=value.end_sequence + 1),
        replace(value, started_at=value.started_at - timedelta(seconds=1)),
        replace(value, ended_at=value.ended_at + timedelta(seconds=1), generated_at=value.generated_at + timedelta(seconds=2)),
    ]
    for mutated in mutations:
        with pytest.raises(ConversationSummarySchemaError, match="does not match"):
            mutated.require_matches_source(source)
    with pytest.raises(TypeError):
        value.require_matches_source(object())  # type: ignore[arg-type]


def test_segment_summary_round_trip_is_exact_and_digest_stable() -> None:
    value = _segment_summary()
    restored = ConversationSegmentSummary.from_dict(value.to_dict())
    assert restored == value
    assert restored.digest == value.digest


@pytest.mark.parametrize("extra", ["owner", "memory", "confidence"])
def test_segment_summary_from_dict_rejects_unknown_fields(extra: str) -> None:
    value = _segment_summary().to_dict()
    with pytest.raises(ConversationSummarySchemaError, match="unknown"):
        ConversationSegmentSummary.from_dict({**value, extra: True})


@pytest.mark.parametrize(
    "field",
    [
        "conversation_id",
        "segment_id",
        "source_message_digest",
        "start_sequence",
        "end_sequence",
        "started_at",
        "ended_at",
        "generated_at",
        "overview",
        "chronology",
        "corrections",
        "ending_state",
        "open_threads",
    ],
)
def test_segment_summary_from_dict_rejects_each_missing_field(field: str) -> None:
    payload = _segment_summary().to_dict()
    payload.pop(field)
    with pytest.raises(ConversationSummarySchemaError, match="missing"):
        ConversationSegmentSummary.from_dict(payload)


def test_segment_summary_from_dict_rejects_wrong_schema_version() -> None:
    with pytest.raises(ConversationSummarySchemaError, match="unsupported"):
        ConversationSegmentSummary.from_dict({**_segment_summary().to_dict(), "schema_version": "v0"})


@pytest.mark.parametrize("stage", list(ConversationRangeSummaryStage))
def test_range_summary_stage_maps_to_exact_source_kind(stage: ConversationRangeSummaryStage) -> None:
    expected = ConversationSummarySourceKind.SEGMENT if stage is ConversationRangeSummaryStage.RANGE else ConversationSummarySourceKind.RANGE
    assert stage.source_kind is expected


@pytest.mark.parametrize("kind", list(ConversationSummarySourceKind))
def test_summary_source_reference_round_trip_preserves_kind(kind: ConversationSummarySourceKind) -> None:
    value = ConversationSummarySourceRef(kind, "000000000000-000000000001", "a" * 64, 0, 1)
    assert ConversationSummarySourceRef.from_dict(value.to_dict()) == value


@pytest.mark.parametrize("kind", ["segment_summary", "range_summary", "unknown", None, 1])
def test_summary_source_constructor_requires_enum_instance_not_raw_value(kind: object) -> None:
    with pytest.raises(ConversationSummarySchemaError, match="kind"):
        ConversationSummarySourceRef(kind, "000000000000-000000000001", "a" * 64, 0, 1)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("start", "end"),
    [
        (0, 0),
        (0, 1),
        (999_999_999_999, 999_999_999_999),
    ],
)
def test_summary_source_reference_accepts_bounded_sequence_identity(start: int, end: int) -> None:
    identity = f"{start:012d}-{end:012d}"
    value = ConversationSummarySourceRef(ConversationSummarySourceKind.SEGMENT, identity, "a" * 64, start, end)
    assert value.summary_id == identity


@pytest.mark.parametrize(
    ("start", "end"),
    [
        (-1, 0),
        (0, -1),
        (2, 1),
        (1_000_000_000_000, 1_000_000_000_000),
        (True, 1),
        (0, 1.0),
    ],
)
def test_summary_source_reference_rejects_invalid_sequence_range(start: object, end: object) -> None:
    with pytest.raises(ConversationSummarySchemaError):
        ConversationSummarySourceRef(ConversationSummarySourceKind.SEGMENT, "000000000000-000000000001", "a" * 64, start, end)  # type: ignore[arg-type]


@pytest.mark.parametrize("summary_id", ["wrong", "000000000000-000000000002", "00000000000-000000000001"])
def test_summary_source_reference_identity_must_equal_sequence_range(summary_id: str) -> None:
    with pytest.raises(ConversationSummarySchemaError, match="does not match"):
        ConversationSummarySourceRef(ConversationSummarySourceKind.SEGMENT, summary_id, "a" * 64, 0, 1)


@pytest.mark.parametrize("payload", [None, [], "ref", 1])
def test_summary_source_reference_from_dict_requires_object(payload: object) -> None:
    with pytest.raises(ConversationSummarySchemaError, match="object"):
        ConversationSummarySourceRef.from_dict(payload)


@pytest.mark.parametrize("field", ["kind", "summary_id", "digest", "start_sequence", "end_sequence"])
def test_summary_source_reference_from_dict_requires_exact_fields(field: str) -> None:
    payload = ConversationSummarySourceRef(
        ConversationSummarySourceKind.SEGMENT,
        "000000000000-000000000001",
        "a" * 64,
        0,
        1,
    ).to_dict()
    payload.pop(field)
    with pytest.raises(ConversationSummarySchemaError, match="fields"):
        ConversationSummarySourceRef.from_dict(payload)


def _range_summary(
    indices: tuple[int, ...] = (0, 1),
    *,
    conversation_id: str = "conversation-1",
) -> tuple[ConversationRangeSummary, tuple[ConversationSegmentSummary, ...]]:
    sources = tuple(_segment_summary(index, conversation_id=conversation_id) for index in indices)
    refs = tuple(ConversationSummarySourceRef.from_summary(source) for source in sources)
    start = sources[0].start_sequence
    end = sources[-1].end_sequence
    value = ConversationRangeSummary(
        conversation_id=conversation_id,
        range_id=f"{start:012d}-{end:012d}",
        stage=ConversationRangeSummaryStage.RANGE,
        source_refs=refs,
        start_sequence=start,
        end_sequence=end,
        started_at=sources[0].started_at,
        ended_at=sources[-1].ended_at,
        generated_at=sources[-1].ended_at + timedelta(seconds=1),
        **summary_content().to_dict(),
    )
    return value, sources


def _archive_summary() -> tuple[ConversationRangeSummary, tuple[ConversationRangeSummary, ...]]:
    first, _ = _range_summary((0, 1))
    second, _ = _range_summary((2, 3))
    sources = (first, second)
    refs = tuple(ConversationSummarySourceRef.from_summary(source) for source in sources)
    value = ConversationRangeSummary(
        conversation_id="conversation-1",
        range_id="000000000000-000000000007",
        stage=ConversationRangeSummaryStage.ARCHIVE,
        source_refs=refs,
        start_sequence=0,
        end_sequence=7,
        started_at=first.started_at,
        ended_at=second.ended_at,
        generated_at=second.ended_at + timedelta(seconds=1),
        **summary_content().to_dict(),
    )
    return value, sources


@pytest.mark.parametrize("count", [2, 3, 10, 100])
def test_range_summary_accepts_two_to_one_thousand_contiguous_sources(count: int) -> None:
    value, sources = _range_summary(tuple(range(count)))
    value.require_matches_sources(sources)
    assert len(value.source_refs) == count


@pytest.mark.parametrize("count", [0, 1, 1001])
def test_range_summary_rejects_source_count_outside_bounds(count: int) -> None:
    valid, _ = _range_summary()
    if count == 0:
        refs: tuple[ConversationSummarySourceRef, ...] = ()
    elif count == 1:
        refs = valid.source_refs[:1]
    else:
        first = valid.source_refs[0]
        refs = tuple(
            ConversationSummarySourceRef(
                first.kind,
                f"{index:012d}-{index:012d}",
                first.digest,
                index,
                index,
            )
            for index in range(count)
        )
    with pytest.raises(ConversationSummarySchemaError, match="between 2 and 1000"):
        replace(valid, source_refs=refs)


def test_range_summary_requires_tuple_of_source_refs() -> None:
    valid, _ = _range_summary()
    with pytest.raises(ConversationSummarySchemaError):
        replace(valid, source_refs=list(valid.source_refs))  # type: ignore[arg-type]
    with pytest.raises(ConversationSummarySchemaError):
        replace(valid, source_refs=(valid.source_refs[0], object()))  # type: ignore[arg-type]


def test_range_and_archive_require_stage_specific_source_kind() -> None:
    valid, _ = _range_summary()
    wrong = replace(valid.source_refs[0], kind=ConversationSummarySourceKind.RANGE)
    with pytest.raises(ConversationSummarySchemaError, match="source kind"):
        replace(valid, source_refs=(wrong, valid.source_refs[1]))
    archive, _ = _archive_summary()
    wrong_archive = replace(archive.source_refs[0], kind=ConversationSummarySourceKind.SEGMENT)
    with pytest.raises(ConversationSummarySchemaError, match="source kind"):
        replace(archive, source_refs=(wrong_archive, archive.source_refs[1]))


@pytest.mark.parametrize("indices", [(0, 2), (1, 0), (0, 1, 3)])
def test_range_summary_rejects_gapped_or_reordered_sources(indices: tuple[int, ...]) -> None:
    with pytest.raises(ConversationSummarySchemaError, match="contiguous"):
        _range_summary(indices)


def test_range_summary_identity_and_coverage_must_match_source_edges() -> None:
    valid, _ = _range_summary()
    with pytest.raises(ConversationSummarySchemaError, match="range_id"):
        replace(valid, range_id="000000000000-000000000002")
    with pytest.raises(ConversationSummarySchemaError, match="coverage"):
        replace(valid, start_sequence=1, range_id="000000000001-000000000003")
    with pytest.raises(ConversationSummarySchemaError, match="coverage"):
        replace(valid, end_sequence=2, range_id="000000000000-000000000002")


def test_range_summary_rejects_invalid_time_order() -> None:
    valid, _ = _range_summary()
    with pytest.raises(ConversationSummarySchemaError, match="time range"):
        replace(valid, started_at=valid.ended_at + timedelta(seconds=1))
    with pytest.raises(ConversationSummarySchemaError, match="time range"):
        replace(valid, generated_at=valid.ended_at - timedelta(seconds=1))


def test_range_summary_binding_detects_count_type_conversation_reference_and_time_mismatch() -> None:
    valid, sources = _range_summary()
    valid.require_matches_sources(sources)
    with pytest.raises(ConversationSummarySchemaError, match="count"):
        valid.require_matches_sources(sources[:1])
    with pytest.raises(ConversationSummarySchemaError, match="count"):
        valid.require_matches_sources(list(sources))  # type: ignore[arg-type]
    with pytest.raises(ConversationSummarySchemaError, match="source type"):
        valid.require_matches_sources((object(), object()))  # type: ignore[arg-type]
    other_sources = tuple(replace(source, conversation_id="other") for source in sources)
    with pytest.raises(ConversationSummarySchemaError, match="another conversation"):
        valid.require_matches_sources(other_sources)
    changed_refs = (replace(sources[0], overview="changed"), sources[1])
    with pytest.raises(ConversationSummarySchemaError, match="bindings"):
        valid.require_matches_sources(changed_refs)
    changed_time = replace(valid, started_at=valid.started_at - timedelta(seconds=1))
    with pytest.raises(ConversationSummarySchemaError, match="time coverage"):
        changed_time.require_matches_sources(sources)


def test_archive_summary_requires_range_stage_sources_and_cannot_compress_archive_again() -> None:
    archive, sources = _archive_summary()
    archive.require_matches_sources(sources)
    with pytest.raises(ConversationSummarySchemaError, match="cannot be compressed again"):
        ConversationSummarySourceRef.from_summary(archive)
    with pytest.raises(ConversationSummarySchemaError, match="cannot be compressed again"):
        summary_model._summary_source_ref(archive)


def test_range_summary_round_trip_is_exact_and_digest_stable_for_both_stages() -> None:
    for value in (_range_summary()[0], _archive_summary()[0]):
        restored = ConversationRangeSummary.from_dict(value.to_dict())
        assert restored == value
        assert restored.digest == value.digest


@pytest.mark.parametrize("field", [
    "schema_version",
    "conversation_id",
    "range_id",
    "stage",
    "source_refs",
    "start_sequence",
    "end_sequence",
    "started_at",
    "ended_at",
    "generated_at",
    "overview",
    "chronology",
    "corrections",
    "ending_state",
    "open_threads",
])
def test_range_summary_from_dict_requires_exact_field_set(field: str) -> None:
    payload = _range_summary()[0].to_dict()
    payload.pop(field)
    with pytest.raises(ConversationSummarySchemaError, match="fields"):
        ConversationRangeSummary.from_dict(payload)


def test_range_summary_from_dict_rejects_wrong_schema_stage_and_source_container() -> None:
    payload = _range_summary()[0].to_dict()
    with pytest.raises(ConversationSummarySchemaError, match="unsupported"):
        ConversationRangeSummary.from_dict({**payload, "schema_version": "v0"})
    with pytest.raises(ConversationSummarySchemaError, match="stage"):
        ConversationRangeSummary.from_dict({**payload, "stage": "unknown"})
    with pytest.raises(ConversationSummarySchemaError, match="source_refs"):
        ConversationRangeSummary.from_dict({**payload, "source_refs": "not-list"})


@pytest.mark.parametrize("source", [None, object(), "summary", 1])
def test_summary_source_ref_factory_rejects_unsupported_source_type(source: object) -> None:
    with pytest.raises(TypeError):
        ConversationSummarySourceRef.from_summary(source)  # type: ignore[arg-type]
