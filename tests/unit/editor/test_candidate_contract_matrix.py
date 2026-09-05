"""结构化记忆候选、临时身份、关系与 page_id 的严格契约矩阵。"""

from __future__ import annotations

from itertools import combinations
from types import MappingProxyType

import pytest

from habitus.memory.document import MemoryLinkType
from habitus.memory.editor.candidate import (
    MemoryCandidate,
    MemoryCandidateBatch,
    MemoryCandidateError,
    MemoryIdentityProposal,
    MemoryIdentityProposalBasis,
    MemoryIdentityProposalType,
    MemoryRelationAction,
    MemoryRelationCandidate,
)
from habitus.memory.editor.page_id import (
    MemoryPageIdError,
    MemoryPageIdMap,
    validate_new_page_id,
    validate_page_id,
    validate_unique_page_ids,
)
from habitus.memory.model import MemoryAddress, MemoryKind
from habitus.memory.uri import MemoryURI, MemoryURIError
from tests.helpers import document, memory_fields, snapshot_batch

FIELD_FOR_KIND = {
    MemoryKind.PROFILE: "profile",
    MemoryKind.PREFERENCE: "preferences",
    MemoryKind.ENTITY: "entities",
    MemoryKind.TOOL: "tools",
    MemoryKind.EVENT: "events",
    MemoryKind.INTENTION: "intentions",
}
ALL_FIELDS = (*FIELD_FOR_KIND.values(), "identity_proposals", "relations")
KINDS_WITH_OPTIONAL_STRING_FIELDS = (
    MemoryKind.ENTITY,
    MemoryKind.TOOL,
    MemoryKind.EVENT,
    MemoryKind.INTENTION,
)


def _empty_output() -> dict[str, object]:
    return {field: [] for field in ALL_FIELDS}


def _raw_candidate(kind: MemoryKind, page_id: object = 100) -> dict[str, object]:
    result = {"page_id": page_id, **memory_fields(kind)}
    if kind is MemoryKind.EVENT:
        result["event_date"] = "2026-07-01"
    if kind is MemoryKind.INTENTION:
        result["confirmed"] = True
    return result


@pytest.mark.parametrize("value", [1, 2, 98, 99, 100, 101, 2**31, 2**63])
def test_page_id_accepts_all_positive_integer_ranges(value: int) -> None:
    assert validate_page_id(value) == value


@pytest.mark.parametrize("value", [0, -1, -100, True, False, 1.0, "1", None, [], {}])
def test_page_id_rejects_non_positive_bool_and_non_integer(value: object) -> None:
    with pytest.raises(MemoryPageIdError):
        validate_page_id(value)


@pytest.mark.parametrize("value", [100, 101, 999, 2**63])
def test_new_page_id_accepts_only_new_range(value: int) -> None:
    assert validate_new_page_id(value) == value


@pytest.mark.parametrize("value", [1, 2, 98, 99, 0, -1, True, 100.0, "100", None])
def test_new_page_id_rejects_existing_or_invalid_range(value: object) -> None:
    with pytest.raises(MemoryPageIdError):
        validate_new_page_id(value)


@pytest.mark.parametrize("values", [(1, 1), (99, 99), (100, 100), (1, 100, 1), (100, 101, 100)])
def test_unique_page_id_validator_rejects_duplicates_across_ranges(values: tuple[int, ...]) -> None:
    with pytest.raises(MemoryPageIdError, match="duplicate"):
        validate_unique_page_ids(values)


@pytest.mark.parametrize("values", [(), (1,), (99, 100), (1, 2, 100, 101)])
def test_unique_page_id_validator_accepts_unique_ordered_or_empty_values(values: tuple[int, ...]) -> None:
    validate_unique_page_ids(values)


@pytest.mark.parametrize("kind", tuple(MemoryKind))
@pytest.mark.parametrize("page_id", [1, 99, 100, 101, 2**31])
def test_candidate_accepts_every_kind_and_page_range(kind: MemoryKind, page_id: int) -> None:
    candidate = MemoryCandidate(
        page_id,
        kind,
        memory_fields(kind),
        confirmed=True if kind is MemoryKind.INTENTION else None,
    )
    assert candidate.page_id == page_id
    assert candidate.address.kind is kind
    assert isinstance(candidate.fields, MappingProxyType)
    assert candidate.to_dict()["page_id"] == page_id


@pytest.mark.parametrize("kind", tuple(MemoryKind))
@pytest.mark.parametrize("confirmed", [True, False])
def test_confirmed_control_field_is_owned_only_by_intention(kind: MemoryKind, confirmed: bool) -> None:
    if kind is MemoryKind.INTENTION:
        assert MemoryCandidate(100, kind, memory_fields(kind), confirmed=confirmed).confirmed is confirmed
    else:
        with pytest.raises(MemoryCandidateError, match="only Intention"):
            MemoryCandidate(100, kind, memory_fields(kind), confirmed=confirmed)


@pytest.mark.parametrize("confirmed", [None, 0, 1, "true", [], {}])
def test_intention_candidate_requires_explicit_boolean_confirmation(confirmed: object) -> None:
    with pytest.raises(MemoryCandidateError, match="boolean"):
        MemoryCandidate(100, MemoryKind.INTENTION, memory_fields(MemoryKind.INTENTION), confirmed=confirmed)  # type: ignore[arg-type]


@pytest.mark.parametrize("kind", tuple(MemoryKind))
@pytest.mark.parametrize("value", [None, [], "fields", 1, True])
def test_candidate_rejects_non_mapping_business_fields(kind: MemoryKind, value: object) -> None:
    with pytest.raises(TypeError):
        MemoryCandidate(100, kind, value, confirmed=True if kind is MemoryKind.INTENTION else None)  # type: ignore[arg-type]


@pytest.mark.parametrize("kind", tuple(MemoryKind))
def test_candidate_rejects_unknown_business_or_system_field(kind: MemoryKind) -> None:
    with pytest.raises(MemoryCandidateError, match="invalid"):
        MemoryCandidate(
            100,
            kind,
            {**memory_fields(kind), "revision": 1},
            confirmed=True if kind is MemoryKind.INTENTION else None,
        )


@pytest.mark.parametrize("kind", KINDS_WITH_OPTIONAL_STRING_FIELDS)
def test_candidate_rejects_whitespace_in_every_present_optional_string(kind: MemoryKind) -> None:
    fields = memory_fields(kind)
    schema = MemoryCandidateBatch.model_json_schema()["properties"][FIELD_FOR_KIND[kind]]["items"]  # type: ignore[index]
    optional = set(schema["properties"]) - set(schema["required"]) - {"confirmed"}  # type: ignore[index]
    field = sorted(optional)[0]
    fields[field] = " "
    with pytest.raises(MemoryCandidateError, match="empty string"):
        MemoryCandidate(
            100,
            kind,
            fields,
            confirmed=True if kind is MemoryKind.INTENTION else None,
        )


@pytest.mark.parametrize("action", tuple(MemoryRelationAction))
@pytest.mark.parametrize("link_type", tuple(MemoryLinkType))
@pytest.mark.parametrize(("source", "target"), [(1, 2), (99, 100), (100, 101), (2**31, 2**31 + 1)])
def test_relation_candidate_accepts_every_action_type_and_page_range(
    action: MemoryRelationAction,
    link_type: MemoryLinkType,
    source: int,
    target: int,
) -> None:
    candidate = MemoryRelationCandidate(action, source, target, link_type)
    assert candidate.action is action
    assert candidate.link_type is link_type
    if link_type.is_symmetric:
        assert candidate.from_page_id < candidate.to_page_id
    else:
        assert (candidate.from_page_id, candidate.to_page_id) == (source, target)
    assert candidate.to_dict()["link_type"] == link_type.value


@pytest.mark.parametrize("link_type", tuple(MemoryLinkType))
@pytest.mark.parametrize("page_id", [1, 99, 100, 2**31])
def test_relation_candidate_rejects_self_reference(link_type: MemoryLinkType, page_id: int) -> None:
    with pytest.raises(MemoryCandidateError, match="same page"):
        MemoryRelationCandidate(MemoryRelationAction.ADD, page_id, page_id, link_type)


@pytest.mark.parametrize("action", ["unknown", "create", "delete", "", None, 1, True])
def test_relation_candidate_rejects_unknown_action(action: object) -> None:
    with pytest.raises((MemoryCandidateError, TypeError)):
        MemoryRelationCandidate(action, 1, 2, MemoryLinkType.RELATED_TO)  # type: ignore[arg-type]


@pytest.mark.parametrize("link_type", ["unknown", "parent", "", None, 1, True])
def test_relation_candidate_rejects_unknown_link_type(link_type: object) -> None:
    with pytest.raises((MemoryCandidateError, TypeError)):
        MemoryRelationCandidate(MemoryRelationAction.ADD, 1, 2, link_type)  # type: ignore[arg-type]


@pytest.mark.parametrize("position", ["from_page_id", "to_page_id"])
@pytest.mark.parametrize("value", [0, -1, True, 1.0, "1", None, [], {}])
def test_relation_candidate_rejects_invalid_page_ids(position: str, value: object) -> None:
    kwargs = {
        "action": MemoryRelationAction.ADD,
        "from_page_id": 1,
        "to_page_id": 2,
        "link_type": MemoryLinkType.RELATED_TO,
    }
    kwargs[position] = value
    with pytest.raises(MemoryCandidateError):
        MemoryRelationCandidate(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize("source", [1, 50, 99])
@pytest.mark.parametrize("target", [1, 99, 100, 101])
def test_same_memory_identity_proposal_requires_distinct_valid_target(source: int, target: int) -> None:
    if source == target:
        with pytest.raises(MemoryCandidateError, match="differ"):
            MemoryIdentityProposal(
                MemoryIdentityProposalType.SAME_MEMORY,
                source,
                target,
                MemoryIdentityProposalBasis.DUPLICATE_IDENTITY,
            )
    else:
        proposal = MemoryIdentityProposal(
            MemoryIdentityProposalType.SAME_MEMORY,
            source,
            target,
            MemoryIdentityProposalBasis.DUPLICATE_IDENTITY,
        )
        assert proposal.target_page_id == target


@pytest.mark.parametrize("basis", [MemoryIdentityProposalBasis.EXPLICIT_FORGET, MemoryIdentityProposalBasis.FULLY_INVALIDATED])
@pytest.mark.parametrize("source", [1, 50, 99])
def test_remove_memory_identity_proposal_accepts_only_explicit_removal_basis(
    basis: MemoryIdentityProposalBasis,
    source: int,
) -> None:
    proposal = MemoryIdentityProposal(MemoryIdentityProposalType.REMOVE_MEMORY, source, None, basis)
    assert proposal.target_page_id is None
    assert proposal.to_dict()["basis"] == basis.value


@pytest.mark.parametrize("source", [100, 101, 2**31])
def test_identity_proposal_source_must_be_existing_page_range(source: int) -> None:
    with pytest.raises(MemoryCandidateError, match="fully read existing"):
        MemoryIdentityProposal(
            MemoryIdentityProposalType.REMOVE_MEMORY,
            source,
            None,
            MemoryIdentityProposalBasis.EXPLICIT_FORGET,
        )


@pytest.mark.parametrize("basis", [MemoryIdentityProposalBasis.EXPLICIT_FORGET, MemoryIdentityProposalBasis.FULLY_INVALIDATED])
def test_same_memory_rejects_removal_basis(basis: MemoryIdentityProposalBasis) -> None:
    with pytest.raises(MemoryCandidateError, match="duplicate_identity"):
        MemoryIdentityProposal(MemoryIdentityProposalType.SAME_MEMORY, 1, 2, basis)


def test_remove_memory_rejects_duplicate_basis_and_target() -> None:
    with pytest.raises(MemoryCandidateError, match="explicit_forget"):
        MemoryIdentityProposal(
            MemoryIdentityProposalType.REMOVE_MEMORY,
            1,
            None,
            MemoryIdentityProposalBasis.DUPLICATE_IDENTITY,
        )
    with pytest.raises(MemoryCandidateError, match="cannot contain"):
        MemoryIdentityProposal(
            MemoryIdentityProposalType.REMOVE_MEMORY,
            1,
            2,
            MemoryIdentityProposalBasis.EXPLICIT_FORGET,
        )


@pytest.mark.parametrize("field", FIELD_FOR_KIND.values())
@pytest.mark.parametrize("container", [[], {}, "candidate", None, 1])
def test_batch_constructor_requires_tuple_per_memory_kind(field: str, container: object) -> None:
    kwargs = {field: container}
    with pytest.raises(TypeError):
        MemoryCandidateBatch(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize("field", FIELD_FOR_KIND.values())
def test_batch_constructor_rejects_candidate_in_wrong_kind_bucket(field: str) -> None:
    expected = next(kind for kind, name in FIELD_FOR_KIND.items() if name == field)
    wrong = next(kind for kind in MemoryKind if kind is not expected)
    candidate = MemoryCandidate(
        100,
        wrong,
        memory_fields(wrong),
        confirmed=True if wrong is MemoryKind.INTENTION else None,
    )
    with pytest.raises(MemoryCandidateError, match="wrong type"):
        MemoryCandidateBatch(**{field: (candidate,)})


@pytest.mark.parametrize("field", ["identity_proposals", "relations"])
@pytest.mark.parametrize("value", [[], {}, "items", None, 1])
def test_batch_constructor_requires_tuple_for_control_collections(field: str, value: object) -> None:
    with pytest.raises(TypeError):
        MemoryCandidateBatch(**{field: value})  # type: ignore[arg-type]


@pytest.mark.parametrize("kind", tuple(MemoryKind))
def test_batch_rejects_duplicate_address_even_with_different_page_id(kind: MemoryKind) -> None:
    first = MemoryCandidate(100, kind, memory_fields(kind), confirmed=True if kind is MemoryKind.INTENTION else None)
    second = MemoryCandidate(101, kind, memory_fields(kind), confirmed=True if kind is MemoryKind.INTENTION else None)
    with pytest.raises(MemoryCandidateError, match="duplicate address"):
        MemoryCandidateBatch(**{FIELD_FOR_KIND[kind]: (first, second)})


@pytest.mark.parametrize(("left_kind", "right_kind"), tuple(combinations(MemoryKind, 2)))
def test_batch_rejects_duplicate_page_id_across_all_kind_combinations(
    left_kind: MemoryKind,
    right_kind: MemoryKind,
) -> None:
    left = MemoryCandidate(100, left_kind, memory_fields(left_kind), confirmed=True if left_kind is MemoryKind.INTENTION else None)
    right = MemoryCandidate(100, right_kind, memory_fields(right_kind), confirmed=True if right_kind is MemoryKind.INTENTION else None)
    with pytest.raises(MemoryCandidateError, match="duplicate page_id"):
        MemoryCandidateBatch(
            **{
                FIELD_FOR_KIND[left_kind]: (left,),
                FIELD_FOR_KIND[right_kind]: (right,),
            }
        )


@pytest.mark.parametrize("missing", ALL_FIELDS)
def test_model_validate_requires_every_top_level_array(missing: str) -> None:
    output = _empty_output()
    output.pop(missing)
    with pytest.raises(MemoryCandidateError, match="missing"):
        MemoryCandidateBatch.model_validate(output)


@pytest.mark.parametrize("unknown", ["uri", "revision", "operations", "deletes", "links", "backlinks", "skills", "topics"])
def test_model_validate_rejects_unknown_top_level_output(unknown: str) -> None:
    output = _empty_output()
    output[unknown] = []
    with pytest.raises(MemoryCandidateError, match="unknown"):
        MemoryCandidateBatch.model_validate(output)


@pytest.mark.parametrize("value", [None, [], (), "output", 1, True])
def test_model_validate_requires_object_root(value: object) -> None:
    with pytest.raises(MemoryCandidateError, match="object"):
        MemoryCandidateBatch.model_validate(value)


@pytest.mark.parametrize("field", ALL_FIELDS)
@pytest.mark.parametrize("value", [None, {}, "items", 1, True])
def test_model_validate_requires_array_for_every_output_field(field: str, value: object) -> None:
    output = _empty_output()
    output[field] = value
    with pytest.raises(MemoryCandidateError, match="array"):
        MemoryCandidateBatch.model_validate(output)


@pytest.mark.parametrize("kind", tuple(MemoryKind))
@pytest.mark.parametrize("item", [None, [], "candidate", 1, True])
def test_model_validate_rejects_non_object_candidate_item(kind: MemoryKind, item: object) -> None:
    output = _empty_output()
    output[FIELD_FOR_KIND[kind]] = [item]
    with pytest.raises(MemoryCandidateError, match="non-object"):
        MemoryCandidateBatch.model_validate(output)


@pytest.mark.parametrize("kind", tuple(MemoryKind))
@pytest.mark.parametrize("page_id", [None, 0, -1, True, 1.0, "100"])
def test_model_validate_rejects_missing_or_invalid_candidate_page_id(kind: MemoryKind, page_id: object) -> None:
    output = _empty_output()
    raw = _raw_candidate(kind, page_id)
    if page_id is None:
        raw.pop("page_id")
    output[FIELD_FOR_KIND[kind]] = [raw]
    with pytest.raises(MemoryCandidateError):
        MemoryCandidateBatch.model_validate(output)


@pytest.mark.parametrize("kind", tuple(MemoryKind))
def test_model_validate_round_trip_is_exact_for_each_kind(kind: MemoryKind) -> None:
    output = _empty_output()
    output[FIELD_FOR_KIND[kind]] = [_raw_candidate(kind)]
    batch = MemoryCandidateBatch.model_validate(output)
    assert batch.to_dict() == output
    assert batch.iter_candidates() == getattr(batch, FIELD_FOR_KIND[kind])


@pytest.mark.parametrize("field", ["action", "from_page_id", "to_page_id", "link_type"])
def test_relation_parser_rejects_each_missing_field(field: str) -> None:
    output = _empty_output()
    relation = {"action": "add", "from_page_id": 100, "to_page_id": 101, "link_type": "related_to"}
    relation.pop(field)
    output["relations"] = [relation]
    with pytest.raises(MemoryCandidateError, match="missing"):
        MemoryCandidateBatch.model_validate(output)


@pytest.mark.parametrize("field", ["uri", "confidence", "backlink", "source", "target"])
def test_relation_parser_rejects_unknown_fields(field: str) -> None:
    output = _empty_output()
    output["relations"] = [{
        "action": "add",
        "from_page_id": 100,
        "to_page_id": 101,
        "link_type": "related_to",
        field: "forbidden",
    }]
    with pytest.raises(MemoryCandidateError, match="unknown"):
        MemoryCandidateBatch.model_validate(output)


@pytest.mark.parametrize("field", ["proposal_type", "source_page_id", "target_page_id", "basis"])
def test_identity_parser_rejects_each_missing_field(field: str) -> None:
    output = _empty_output()
    proposal = {
        "proposal_type": "same_memory",
        "source_page_id": 1,
        "target_page_id": 100,
        "basis": "duplicate_identity",
    }
    proposal.pop(field)
    output["identity_proposals"] = [proposal]
    with pytest.raises(MemoryCandidateError, match="missing"):
        MemoryCandidateBatch.model_validate(output)


@pytest.mark.parametrize("field", ["uri", "revision", "delete", "confidence", "reasoning"])
def test_identity_parser_rejects_unknown_fields(field: str) -> None:
    output = _empty_output()
    output["identity_proposals"] = [{
        "proposal_type": "same_memory",
        "source_page_id": 1,
        "target_page_id": 100,
        "basis": "duplicate_identity",
        field: "forbidden",
    }]
    with pytest.raises(MemoryCandidateError, match="unknown"):
        MemoryCandidateBatch.model_validate(output)


def test_candidate_json_schema_exactly_covers_yaml_fields_and_control_fields() -> None:
    schema = MemoryCandidateBatch.model_json_schema()
    assert schema["additionalProperties"] is False
    assert tuple(schema["required"]) == ALL_FIELDS
    assert set(schema["properties"]) == set(ALL_FIELDS)
    for kind, field_name in FIELD_FOR_KIND.items():
        item = schema["properties"][field_name]["items"]
        expected = {"page_id", *(memory_fields(kind).keys())}
        if kind is MemoryKind.INTENTION:
            expected.add("confirmed")
        assert expected <= set(item["properties"])
        assert item["additionalProperties"] is False


def test_page_id_map_register_copy_and_multi_binding_are_isolated() -> None:
    first = MemoryURI.from_address(document(MemoryKind.PREFERENCE).address)
    second = MemoryURI.from_address(document(MemoryKind.ENTITY).address)
    page_ids = MemoryPageIdMap()
    assert page_ids.register_existing(first) == 1
    assert page_ids.register_existing(first) == 1
    page_ids.register_new(first, 100)
    page_ids.register_new(second, 101)
    copied = page_ids.copy()
    copied.register_new(second, 102)
    assert page_ids.page_ids_for(second) == frozenset({101})
    assert copied.page_ids_for(second) == frozenset({101, 102})
    assert page_ids.page_id_for(first) == 1
    assert page_ids.existing_items() == ((1, str(first)),)


@pytest.mark.parametrize("page_id", [1, 50, 99])
def test_page_id_map_protects_existing_binding_from_redirect(page_id: int) -> None:
    page_ids = MemoryPageIdMap()
    uris = [MemoryURI.from_address(document(MemoryKind.PREFERENCE).address)]
    for index in range(1, page_id):
        uris.append(MemoryURI.from_address(MemoryAddress.preference(f"主题-{index}")))
    for uri in uris:
        page_ids.register_existing(uri)
    with pytest.raises(MemoryPageIdError, match="redirected"):
        page_ids.register_resolved(MemoryURI.from_address(document(MemoryKind.ENTITY).address), page_id)


def test_page_id_map_from_snapshots_maps_only_found_documents_in_stable_order() -> None:
    profile = document(MemoryKind.PROFILE)
    preference = document(MemoryKind.PREFERENCE)
    batch = snapshot_batch(preference, profile)
    page_ids = MemoryPageIdMap.from_snapshots(batch)
    identities = tuple(item.identity for item in batch.snapshots)
    assert page_ids.items() == tuple(enumerate(identities, start=1))
    assert all(page_ids.resolve(index) == identity for index, identity in page_ids.items())


@pytest.mark.parametrize("value", [None, [], {}, "snapshots", 1, True])
def test_page_id_map_from_snapshots_rejects_wrong_batch_type(value: object) -> None:
    with pytest.raises(TypeError):
        MemoryPageIdMap.from_snapshots(value)  # type: ignore[arg-type]


def test_page_id_map_rejects_directory_and_layer_uris() -> None:
    page_ids = MemoryPageIdMap()
    for uri in (MemoryURI.root(), MemoryURI("memory://preferences/.abstract.md")):
        with pytest.raises(MemoryURIError):
            page_ids.register_existing(uri)
