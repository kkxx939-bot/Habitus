"""Prediction 节点、URI、Schema 与文档规范化测试。"""

from copy import deepcopy
from datetime import date, datetime, timezone
from urllib.parse import quote

import pytest
from prediction_test_payloads import transition_payload

from prediction import (
    PredictionAddress,
    PredictionDirectory,
    PredictionDocumentCodec,
    PredictionKind,
    PredictionLevel,
    PredictionSchemaError,
    PredictionSchemaRegistry,
    PredictionURI,
    PredictionURIError,
    derive_sample_identity,
)


def _codec() -> PredictionDocumentCodec:
    return PredictionDocumentCodec(PredictionSchemaRegistry.load_default())


def test_prediction_address_uri_and_directory_round_trip_are_canonical() -> None:
    address = PredictionAddress(PredictionKind.TRANSITION, date(2026, 8, 8), "a" * 64)
    uri = PredictionURI.from_address(address)

    assert str(uri) == f"prediction://samples/transitions/2026/08/08/aa/{'a' * 64}.md"
    assert uri.to_address() == address
    assert uri.containing_directory == PredictionDirectory.branch(PredictionKind.TRANSITION, 2026, 8, 8, "aa")
    assert str(PredictionURI.from_layer(uri.containing_directory, PredictionLevel.ABSTRACT)).endswith(
        "/.abstract.md"
    )
    assert PredictionURI.root().is_root


@pytest.mark.parametrize(
    "uri",
    (
        "prediction://samples/transitions/٢٠٢٦/08/08/aa/" + "a" * 64 + ".md",
        "prediction://samples/transitions/2026/08/08/aa/" + "A" * 64 + ".md",
        "prediction://samples/transitions/2026/08/08/aa/../" + "a" * 64 + ".md",
        "prediction://samples/transitions/2026/08/08/zz/" + "a" * 64 + ".md",
        "prediction://samples/transitions/2026/08/08/" + "a" * 64 + ".md",
        "prediction://models/2026/08/08/" + "a" * 64 + ".md",
    ),
)
def test_prediction_uri_rejects_noncanonical_or_unconfirmed_paths(uri: str) -> None:
    with pytest.raises(PredictionURIError):
        PredictionURI(uri)


def test_schema_materializes_strong_values_into_canonical_immutable_document() -> None:
    codec = _codec()
    document = codec.build(PredictionKind.TRANSITION, transition_payload())
    encoded = codec.encode(document)
    decoded = codec.decode(encoded, expected_address=document.address)

    assert decoded == document
    assert document.fields["sample_date"] == "2026-08-08"
    assert document.fields["anchor"]["cutoff_at"] == "2026-08-08T10:30:00.000000Z"
    assert "HABITUS_PREDICTION_FIELDS" in encoded
    assert "室内温度为 29 摄氏度" in document.markdown_body


def test_schema_rejects_anchor_time_contradiction_and_modality_overlap() -> None:
    registry = PredictionSchemaRegistry.load_default()
    payload = transition_payload()
    payload["anchor"]["precision"] = "bounded"
    with pytest.raises(PredictionSchemaError, match="bounded"):
        registry.validate(PredictionKind.TRANSITION, payload)

    payload = transition_payload()
    payload["input"]["observation_frame"]["coverage"]["missing_modalities"].append("vision")
    with pytest.raises(PredictionSchemaError, match="both available and missing"):
        registry.validate(PredictionKind.TRANSITION, payload)


def test_bounded_anchor_uses_lower_bound_as_its_visibility_fence() -> None:
    payload = transition_payload()
    cutoff = payload["anchor"]["cutoff_at"]
    payload["anchor"].update(
        {
            "prefix_length": 1,
            "previous_step_ref": "action:act_0000",
            "decision_basis": "previous_step_ordered",
            "cutoff_at": None,
            "precision": "bounded",
            "lower_bound_at": cutoff,
            "upper_bound_at": None,
        }
    )
    payload["identity_material"]["prefix_length"] = 1
    identity = derive_sample_identity(
        PredictionKind.TRANSITION,
        payload["identity_material"],
        payload["provenance"]["projection_version"],
        payload["materialization_context"],
    )
    payload["logical_sample_id"] = identity.logical_sample_id
    payload["materialization_id"] = identity.materialization_id
    payload["input"]["behavior_history"]["active_behaviors"] = [
        {
            "step_kind": "action",
            "step_ref": "action:act_0000",
            "source_uri": None,
            "local_id": "act_0000",
            "sequence": 1,
            "semantics": "已知前一步",
            "actor": "主人",
            "behavior_type": "daily_activity",
            "target_refs": [],
            "status": "active",
            "started_at": cutoff,
            "ended_at": None,
            "available_at": cutoff,
        }
    ]
    payload["input"]["observation_frame"]["facts"][0]["available_at"] = datetime(
        2026,
        8,
        8,
        10,
        31,
        tzinfo=timezone.utc,
    )

    with pytest.raises(PredictionSchemaError, match="later than the prediction cutoff"):
        PredictionSchemaRegistry.load_default().validate(PredictionKind.TRANSITION, payload)


def test_schema_rejects_post_cutoff_label_shape_as_transition_input_contract_violation() -> None:
    registry = PredictionSchemaRegistry.load_default()
    payload = transition_payload()
    payload["label"]["target_kind"] = "terminal"
    payload["prediction_scope"]["target_level"] = "terminal"
    payload["prediction_scope"]["prediction_mode"] = "termination"

    with pytest.raises(PredictionSchemaError, match="terminal transition"):
        registry.validate(PredictionKind.TRANSITION, payload)


def test_schema_rejects_prediction_mode_that_contradicts_a_well_formed_terminal_label() -> None:
    payload = transition_payload()
    payload["label"].update(
        {
            "target_kind": "terminal",
            "source_ref": None,
            "terminal": {"status": "completed", "reason": None},
        }
    )
    payload["prediction_scope"]["target_level"] = "terminal"

    with pytest.raises(PredictionSchemaError, match="termination mode"):
        PredictionSchemaRegistry.load_default().validate(PredictionKind.TRANSITION, payload)


def test_schema_rejects_container_observed_without_an_exact_cutoff() -> None:
    payload = transition_payload()
    payload["anchor"].update(
        {
            "decision_basis": "container_observed",
            "precision": "order_only",
            "cutoff_at": None,
        }
    )

    with pytest.raises(PredictionSchemaError, match="container anchor requires exact precision"):
        PredictionSchemaRegistry.load_default().validate(PredictionKind.TRANSITION, payload)


def test_schema_canonicalizes_uri_aliases_before_provenance_uniqueness_check() -> None:
    payload = transition_payload()
    duplicated = deepcopy(payload["provenance"]["source_bindings"][0])
    duplicated["uri"] = duplicated["uri"].replace("主人回家", quote("主人回家", safe=""))
    payload["provenance"]["source_bindings"].append(duplicated)

    with pytest.raises(PredictionSchemaError, match="source bindings must be unique"):
        PredictionSchemaRegistry.load_default().validate(PredictionKind.TRANSITION, payload)


def test_schema_rejects_label_source_outside_provenance_closure() -> None:
    payload = transition_payload()
    unrelated_ref = (
        "behavior://behaviors/events/2026/08/08/另一事件"
        "--20260808T103000000000%2B0000.md#action:act_0001"
    )
    payload["label"]["source_ref"] = unrelated_ref
    payload["identity_material"]["target_ref"] = unrelated_ref
    identity = derive_sample_identity(
        PredictionKind.TRANSITION,
        payload["identity_material"],
        payload["provenance"]["projection_version"],
        payload["materialization_context"],
    )
    payload["logical_sample_id"] = identity.logical_sample_id
    payload["materialization_id"] = identity.materialization_id

    with pytest.raises(PredictionSchemaError, match="label sources must be closed"):
        PredictionSchemaRegistry.load_default().validate(PredictionKind.TRANSITION, payload)


def test_schema_rejects_non_utf8_json_parameter_before_persistence() -> None:
    payload = deepcopy(transition_payload())
    payload["label"]["parameters"] = {"bad": "\ud800"}

    with pytest.raises(PredictionSchemaError, match="canonical UTF-8 JSON"):
        _codec().build(PredictionKind.TRANSITION, payload)


@pytest.mark.parametrize("invalid", (float("nan"), float("inf"), float("-inf"), 10**1000))
def test_schema_rejects_non_finite_or_unrepresentable_numbers(invalid: float | int) -> None:
    payload = transition_payload()
    payload["label"]["delay_seconds"] = invalid

    with pytest.raises(PredictionSchemaError, match="finite non-negative number"):
        _codec().build(PredictionKind.TRANSITION, payload)


def test_sample_date_forbids_datetime_even_when_it_is_a_date_subclass() -> None:
    with pytest.raises(TypeError, match="without a time"):
        PredictionAddress(
            PredictionKind.TRANSITION,
            datetime(2026, 8, 8, tzinfo=timezone.utc),
            "a" * 64,
        )
