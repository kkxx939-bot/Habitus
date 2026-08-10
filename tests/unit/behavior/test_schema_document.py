"""Behavior Schema、Markdown 正文和末尾结构字段一致性测试。"""

from copy import deepcopy
from datetime import datetime, timedelta, timezone
from urllib.parse import quote

import pytest
from behavior_test_payloads import (
    episode_storage_payload,
    event_payload,
    event_uri,
    outcome_payload,
    outcome_uri,
)

from behavior.document import BehaviorDocumentCodec, BehaviorDocumentIntegrityError, BehaviorDocumentMetadata
from behavior.model import BehaviorKind
from behavior.schema import BehaviorOperationMode, BehaviorSchemaError, BehaviorSchemaRegistry
from behavior.uri import BehaviorURI


def test_default_registry_declares_exact_tree_and_operation_modes() -> None:
    registry = BehaviorSchemaRegistry.load_default()
    assert [schema.kind for schema in registry.all()] == list(BehaviorKind)
    assert registry.get(BehaviorKind.EVENT).operation_mode is BehaviorOperationMode.ADD_ONLY
    assert registry.get(BehaviorKind.OUTCOME).operation_mode is BehaviorOperationMode.APPEND_ONLY
    assert registry.get(BehaviorKind.EPISODE).operation_mode is BehaviorOperationMode.ADD_ONLY


def test_event_schema_rejects_unknown_fields_and_broken_action_order() -> None:
    registry = BehaviorSchemaRegistry.load_default()
    unknown = event_payload()
    unknown["behavior_name"] = "home-arrival"
    with pytest.raises(BehaviorSchemaError, match="unknown fields"):
        registry.validate(BehaviorKind.EVENT, unknown)

    broken = event_payload()
    broken["actions"][0]["sequence"] = 2
    with pytest.raises(BehaviorSchemaError, match="contiguous"):
        registry.validate(BehaviorKind.EVENT, broken)

    early_onset = event_payload()
    early_onset["onset_available_at"] = early_onset["started_at"] - timedelta(seconds=1)
    with pytest.raises(BehaviorSchemaError, match="onset_available_at"):
        registry.validate(BehaviorKind.EVENT, early_onset)

    early_action_identity = event_payload()
    early_action_identity["actions"][0]["started_at"] = early_action_identity["started_at"]
    early_action_identity["actions"][0]["available_at"] = (
        early_action_identity["started_at"] - timedelta(seconds=1)
    )
    with pytest.raises(BehaviorSchemaError, match="available_at"):
        registry.validate(BehaviorKind.EVENT, early_action_identity)


def test_outcome_schema_requires_exact_mirrored_event_uri() -> None:
    registry = BehaviorSchemaRegistry.load_default()
    target_event_uri = event_uri()
    normalized = registry.validate(BehaviorKind.OUTCOME, outcome_payload(target_event_uri))
    assert normalized["event_uri"] == target_event_uri

    mismatched = outcome_payload(target_event_uri)
    mismatched["event_name"] = "另一个事件"
    with pytest.raises(BehaviorSchemaError, match="mirror"):
        registry.validate(BehaviorKind.OUTCOME, mismatched)


def test_episode_schema_validates_event_order_phases_and_transitions() -> None:
    registry = BehaviorSchemaRegistry.load_default()
    first = event_uri()
    second = event_uri("主人洗手", minute=40)
    outcome = outcome_uri()
    normalized = registry.validate(
        BehaviorKind.EPISODE,
        episode_storage_payload(first, second, outcome),
    )
    assert normalized["ordered_event_uris"] == (first, second)
    assert normalized["phases"][0]["started_at"] == datetime(2026, 8, 8, 10, 30, tzinfo=timezone.utc)
    assert normalized["phases"][0]["ended_at"] == datetime(2026, 8, 8, 10, 32, tzinfo=timezone.utc)

    reversed_transition = episode_storage_payload(first, second, outcome)
    reversed_transition["transitions"][0]["from_event_uri"] = second
    reversed_transition["transitions"][0]["to_event_uri"] = first
    with pytest.raises(BehaviorSchemaError, match="real Event order"):
        registry.validate(BehaviorKind.EPISODE, reversed_transition)

    reversed_phase_time = episode_storage_payload(first, second, outcome)
    reversed_phase_time["phases"][0]["ended_at"] = datetime(2026, 8, 8, 10, 29, tzinfo=timezone.utc)
    with pytest.raises(BehaviorSchemaError, match="cannot precede"):
        registry.validate(BehaviorKind.EPISODE, reversed_phase_time)


def test_document_codec_round_trip_and_tamper_rejection() -> None:
    registry = BehaviorSchemaRegistry.load_default()
    codec = BehaviorDocumentCodec(registry)
    metadata = BehaviorDocumentMetadata.initial(datetime(2026, 8, 8, tzinfo=timezone.utc))
    document = codec.build(BehaviorKind.EVENT, event_payload(), metadata=metadata)
    raw = codec.encode(document)
    assert raw.count("M2BOS_BEHAVIOR_FIELDS") == 1
    assert codec.decode(raw, expected_address=document.address) == document

    tampered = raw.replace("主人已经回家", "主人尚未回家", 1)
    with pytest.raises(BehaviorDocumentIntegrityError, match="body"):
        codec.decode(tampered, expected_address=document.address)

    wrong_address = registry.address_for(
        BehaviorKind.EVENT,
        event_payload("主人回家后直接洗手"),
    )
    with pytest.raises(BehaviorDocumentIntegrityError, match="physical tree"):
        codec.decode(raw, expected_address=wrong_address)


def test_document_fields_are_deeply_immutable() -> None:
    registry = BehaviorSchemaRegistry.load_default()
    codec = BehaviorDocumentCodec(registry)
    source = event_payload()
    document = codec.build(
        BehaviorKind.EVENT,
        source,
        metadata=BehaviorDocumentMetadata.initial(datetime.now(timezone.utc)),
    )
    source["actions"][0]["semantics"] = "外部篡改"
    assert document.fields["actions"][0]["semantics"] == "打开空调"
    assert document.fields["event_date"] == "2026-08-08"
    assert document.fields["started_at"] == "2026-08-08T10:30:00.000000+00:00"
    assert isinstance(registry.validate(BehaviorKind.EVENT, document.fields)["started_at"], datetime)
    with pytest.raises(TypeError):
        document.fields["semantic_summary"] = "篡改"  # type: ignore[index]


def test_episode_renderer_does_not_write_prediction_results() -> None:
    registry = BehaviorSchemaRegistry.load_default()
    first = event_uri()
    second = event_uri("主人洗手", minute=40)
    outcome = outcome_uri()
    payload = episode_storage_payload(first, second, outcome)
    rendered = registry.render_markdown(BehaviorKind.EPISODE, payload)
    assert "## Event Timeline" in rendered
    assert "prediction" not in rendered.casefold()
    assert (
        BehaviorURI.from_address(registry.address_for(BehaviorKind.EPISODE, payload)).to_address().kind
        is BehaviorKind.EPISODE
    )


def test_event_schema_rejects_action_times_outside_or_reversed_within_event() -> None:
    registry = BehaviorSchemaRegistry.load_default()
    outside = event_payload()
    outside["actions"][0]["started_at"] = outside["ended_at"] + timedelta(minutes=1)
    outside["actions"][0]["ended_at"] = outside["ended_at"] + timedelta(minutes=2)
    outside["actions"][0]["available_at"] = outside["actions"][0]["started_at"]
    with pytest.raises(BehaviorSchemaError, match="Event time window"):
        registry.validate(BehaviorKind.EVENT, outside)

    reversed_actions = event_payload()
    first = reversed_actions["actions"][0]
    first["started_at"] = datetime(2026, 8, 8, 10, 31, tzinfo=timezone.utc)
    first["ended_at"] = datetime(2026, 8, 8, 10, 31, 30, tzinfo=timezone.utc)
    first["available_at"] = first["started_at"]
    second = deepcopy(first)
    second.update(
        {
            "action_id": "act_0002",
            "sequence": 2,
            "started_at": datetime(2026, 8, 8, 10, 30, tzinfo=timezone.utc),
            "ended_at": datetime(2026, 8, 8, 10, 30, 30, tzinfo=timezone.utc),
            "available_at": datetime(2026, 8, 8, 10, 30, tzinfo=timezone.utc),
        }
    )
    reversed_actions["actions"].append(second)
    with pytest.raises(BehaviorSchemaError, match="Action order"):
        registry.validate(BehaviorKind.EVENT, reversed_actions)


def test_episode_schema_rejects_cross_phase_reverse_order() -> None:
    registry = BehaviorSchemaRegistry.load_default()
    first = event_uri("first", minute=30)
    second = event_uri("second", minute=40)
    outcome = outcome_uri("first", minute=30)
    payload = episode_storage_payload(first, second, outcome)
    payload["phases"][0]["event_uris"] = [second]
    payload["phases"][1]["event_uris"] = [first]
    with pytest.raises(BehaviorSchemaError, match="Phase order"):
        registry.validate(BehaviorKind.EPISODE, payload)


def test_outcome_schema_canonicalizes_uri_and_sorts_delayed_records() -> None:
    registry = BehaviorSchemaRegistry.load_default()
    event_name = "主人回家后查看并打开空调"
    target_event_uri = event_uri(event_name)
    encoded_uri = target_event_uri.replace(event_name, quote(event_name))
    payload = outcome_payload(encoded_uri)
    delayed = deepcopy(payload["outcomes"][0])
    delayed["outcome_id"] = "out_delayed"
    delayed["occurred_at"] = datetime(2026, 8, 8, 10, 34, tzinfo=timezone.utc)
    payload["outcomes"].append(delayed)

    normalized = registry.validate(BehaviorKind.OUTCOME, payload)

    assert normalized["event_uri"] == target_event_uri
    assert [item["outcome_id"] for item in normalized["outcomes"]] == ["out_delayed", "out_0001"]


def test_schema_rejects_non_document_episode_uri_and_invalid_parameter_unicode() -> None:
    registry = BehaviorSchemaRegistry.load_default()
    first = event_uri("first", minute=30)
    second = event_uri("second", minute=40)
    outcome = outcome_uri("first", minute=30)
    invalid_episode = episode_storage_payload(first, second, outcome)
    invalid_episode["ordered_event_uris"][0] = "behavior://behaviors/events"
    with pytest.raises(BehaviorSchemaError, match="Event document"):
        registry.validate(BehaviorKind.EPISODE, invalid_episode)

    invalid_parameters = event_payload()
    invalid_parameters["actions"][0]["parameters"] = {"invalid": "\ud800"}
    with pytest.raises(BehaviorSchemaError, match="UTF-8 JSON"):
        registry.validate(BehaviorKind.EVENT, invalid_parameters)


def test_episode_requires_weak_phase_coverage_with_total_event_classification() -> None:
    registry = BehaviorSchemaRegistry.load_default()
    first = event_uri("mainline", minute=30)
    contextual = event_uri("查看手机", minute=40)
    outcome = outcome_uri("mainline", minute=30)
    payload = episode_storage_payload(first, contextual, outcome)
    payload["phases"] = [payload["phases"][0]]
    payload["unphased_events"] = [
        {
            "event_uri": contextual,
            "role": "contextual",
            "reason": "与环境调整主线没有直接关系",
        }
    ]

    normalized = registry.validate(BehaviorKind.EPISODE, payload)

    assert normalized["unphased_events"][0]["event_uri"] == contextual
    assert normalized["phases"][0]["confidence"] == 0.96

    silently_omitted = deepcopy(payload)
    silently_omitted["unphased_events"] = []
    with pytest.raises(BehaviorSchemaError, match="classified"):
        registry.validate(BehaviorKind.EPISODE, silently_omitted)

    double_classified = deepcopy(payload)
    double_classified["phases"][0]["event_uris"].append(contextual)
    with pytest.raises(BehaviorSchemaError, match="both phased and unphased"):
        registry.validate(BehaviorKind.EPISODE, double_classified)

    empty_phase = deepcopy(payload)
    empty_phase["phases"][0]["event_uris"] = []
    with pytest.raises(BehaviorSchemaError, match="at least one Event"):
        registry.validate(BehaviorKind.EPISODE, empty_phase)

    invalid_role = deepcopy(payload)
    invalid_role["unphased_events"][0]["role"] = "ignored"
    with pytest.raises(BehaviorSchemaError, match="unphased Event role"):
        registry.validate(BehaviorKind.EPISODE, invalid_role)


def test_behavior_time_preserves_local_offset_and_requires_matching_local_date() -> None:
    registry = BehaviorSchemaRegistry.load_default()
    payload = event_payload("local-arrival")
    payload["event_date"] = "2026-08-09"
    payload["started_at"] = "2026-08-09T00:30:00.000000+08:00"
    payload["ended_at"] = "2026-08-09T00:32:00.000000+08:00"
    payload["onset_available_at"] = payload["started_at"]
    payload["actions"][0]["available_at"] = payload["started_at"]

    materialized = registry.materialize(BehaviorKind.EVENT, payload)

    assert materialized.storage_fields["started_at"] == "2026-08-09T00:30:00.000000+08:00"
    assert materialized.address.identity_name.endswith("--20260809T003000000000+0800")

    payload["event_date"] = "2026-08-08"
    with pytest.raises(BehaviorSchemaError, match="local started_at date"):
        registry.validate(BehaviorKind.EVENT, payload)


def test_episode_outcome_snapshot_shape_is_strict_and_system_metadata_is_not_rendered() -> None:
    registry = BehaviorSchemaRegistry.load_default()
    first = event_uri("first", minute=30)
    second = event_uri("second", minute=40)
    target_outcome = outcome_uri("first", minute=30)
    payload = episode_storage_payload(first, second, target_outcome)

    rendered = registry.render_markdown(BehaviorKind.EPISODE, payload)

    assert "0" * 64 not in rendered
    invalid_digest = deepcopy(payload)
    invalid_digest["outcome_snapshots"][0]["digest"] = "SHA256:not-a-digest"
    with pytest.raises(BehaviorSchemaError, match="SHA-256"):
        registry.validate(BehaviorKind.EPISODE, invalid_digest)

    mismatched_uri = deepcopy(payload)
    mismatched_uri["outcome_snapshots"][0]["uri"] = outcome_uri("second", minute=40)
    with pytest.raises(BehaviorSchemaError, match="match outcome_uris"):
        registry.validate(BehaviorKind.EPISODE, mismatched_uri)


@pytest.mark.parametrize("method_name", ["get", "validate", "address_for", "render_markdown"])
def test_registry_unknown_kind_uses_one_schema_error_contract(method_name: str) -> None:
    registry = BehaviorSchemaRegistry.load_default()
    method = getattr(registry, method_name)
    arguments = ("unknown",) if method_name == "get" else ("unknown", {})
    with pytest.raises(BehaviorSchemaError, match="unknown behavior type"):
        method(*arguments)
