"""VikingDB 记录编解码、元数据包络、物理索引键和过滤器的完整矩阵。"""

from __future__ import annotations

import hashlib
import math
from dataclasses import replace

import pytest

from habitus.infrastructure.vector import VectorStoreFilter, VectorStoreIntegrityError
from habitus.infrastructure.vector.adapters import vikingdb as protocol
from tests.unit.infrastructure.test_vikingdb_protocol import record, search_item


@pytest.mark.parametrize("scope", ["default/memory", "project/collection", "中文/记忆"])
@pytest.mark.parametrize("identity", ["memory://profile.md", "memory://preferences/回答风格.md", "x", ""])
def test_point_id_is_deterministic_scoped_and_distinct_from_metadata_id(scope: str, identity: str) -> None:
    first = protocol._point_id(scope, identity)
    second = protocol._point_id(scope, identity)
    assert first == second
    assert first != protocol._metadata_id(scope, identity)
    assert len(first) == 36


@pytest.mark.parametrize(
    "names",
    [
        (),
        [],
        ("a",),
        ("state", "claim"),
        ("A", "a"),
        ("name_1", "name-2"),
        ("x" * 64,),
    ],
)
def test_metadata_names_accepts_ordered_unique_ascii_resource_names(names: object) -> None:
    assert protocol._metadata_names(names) == tuple(names)


@pytest.mark.parametrize(
    "invalid",
    [
        "state",
        b"state",
        None,
        1,
        True,
        {},
        set(),
        ("",),
        (" ",),
        (" state",),
        ("state ",),
        ("a/b",),
        ("a.b",),
        ("中文",),
        ("x" * 65,),
        ("same", "same"),
        (1,),
        (None,),
    ],
)
def test_metadata_names_rejects_wrong_container_invalid_name_or_duplicate(invalid: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        protocol._metadata_names(invalid)


@pytest.mark.parametrize(
    "attributes",
    [
        {
            "uri": "memory://profile.md",
            "level": 2,
            "directory_key": "memory://",
            "parent_key": "memory://",
            "scope_roots": ("memory://",),
            "kind": "profile",
            "revision": 1,
        },
        {
            "uri": "memory://events/2026/07/28/a.md",
            "level": 0,
            "directory_key": "memory://events/2026/07/28/",
            "parent_key": "memory://events/2026/07/",
            "scope_roots": ("memory://", "memory://events/"),
            "kind": "event",
            "revision": 99,
            "optional": False,
        },
        {},
    ],
)
@pytest.mark.parametrize("scope", ["default/memory", "project/collection"])
def test_memory_record_round_trip_preserves_logical_value_and_derives_physical_fields(
    attributes: dict[str, object],
    scope: str,
) -> None:
    source = record()
    source = replace(source, attributes=attributes)
    encoded = protocol._item_from_record(source, scope=scope)
    wrapped = {
        "id": encoded["id"],
        "fields": {key: value for key, value in encoded.items() if key != "id"},
    }
    restored = protocol._record_from_item(wrapped, scope=scope)
    assert restored == source
    assert encoded["identity_key"] == hashlib.sha256(source.identity.encode()).hexdigest()
    assert encoded["record_type"] == "memory"


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("record_type", "metadata"),
        ("identity", "memory://other.md"),
        ("identity_key", "wrong"),
        ("scan_order", -1),
        ("uri_key", "wrong"),
        ("level", 999),
        ("directory_key_hash", "wrong"),
        ("parent_key_hash", "wrong"),
        ("scope_root_keys", []),
        ("kind", "other"),
        ("revision", 999),
        ("attributes_json", "not-json"),
        ("attributes_json", "[]"),
        ("vector", None),
        ("vector", "1,2"),
        ("vector", [0, 0]),
        ("content", None),
        ("content_digest", "wrong"),
    ],
)
def test_record_decoder_rejects_every_tampered_identity_index_or_value_field(
    field: str,
    invalid: object,
) -> None:
    encoded = search_item(record())
    encoded["fields"][field] = invalid
    with pytest.raises(VectorStoreIntegrityError):
        protocol._record_from_item(encoded, scope="default/memory")


@pytest.mark.parametrize("invalid", [{}, {"id": "x"}, {"fields": None}, {"fields": []}])
def test_record_decoder_requires_fields_object(invalid: dict[str, object]) -> None:
    with pytest.raises(VectorStoreIntegrityError, match="fields object"):
        protocol._record_from_item(invalid, scope="default/memory")


@pytest.mark.parametrize(
    "value",
    [
        {},
        {"dimension": 2},
        {"digest": "a" * 64, "generation": 3},
        {"nested": {"items": [1, None, False]}},
        {"中文": "值"},
    ],
)
@pytest.mark.parametrize("dimension", [1, 2, 8])
def test_metadata_record_round_trip_binds_scope_name_and_value(
    value: dict[str, object],
    dimension: int,
) -> None:
    scope = "default/memory"
    name = "state"
    point_id = protocol._metadata_id(scope, name)
    encoded = protocol._metadata_item(
        point_id=point_id,
        identity=f"metadata:{scope}:{name}",
        name=name,
        scope=scope,
        value=value,
        dimension=dimension,
    )
    wrapped = {
        "id": encoded["id"],
        "fields": {key: item for key, item in encoded.items() if key != "id"},
    }
    assert protocol._metadata_from_item(wrapped, scope=scope, name=name) == value
    assert len(encoded["vector"]) == dimension
    assert encoded["vector"][0] == 1.0


@pytest.mark.parametrize("invalid", [None, [], (), "value", 1, True, {1: "x"}])
def test_metadata_item_requires_string_keyed_mapping(invalid: object) -> None:
    with pytest.raises(TypeError):
        protocol._metadata_item(
            point_id="id",
            identity="identity",
            name="state",
            scope="scope",
            value=invalid,
            dimension=2,
        )


@pytest.mark.parametrize(
    "mutation",
    [
        ("id", "wrong"),
        ("record_type", "memory"),
        ("identity", "wrong"),
        ("identity_key", "wrong"),
        ("scan_order", -1),
        ("content", "wrong"),
        ("content_digest", "wrong"),
        ("attributes_json", "[]"),
        ("uri_key", "wrong"),
        ("scope_root_keys", []),
        ("metadata_json", "not-json"),
        ("metadata_json", "[]"),
        ("metadata_json", '{"name":"wrong","scope":"default/memory","value":{}}'),
        ("metadata_json", '{"name":"state","scope":"wrong","value":{}}'),
        ("metadata_json", '{"name":"state","scope":"default/memory","value":[]}'),
        ("metadata_json", '{"name":"state","scope":"default/memory","value":{},"extra":1}'),
    ],
)
def test_metadata_decoder_rejects_tampered_identity_index_or_envelope(
    mutation: tuple[str, object],
) -> None:
    scope = "default/memory"
    name = "state"
    encoded = protocol._metadata_item(
        point_id=protocol._metadata_id(scope, name),
        identity=f"metadata:{scope}:{name}",
        name=name,
        scope=scope,
        value={"dimension": 2},
        dimension=2,
    )
    wrapped = {
        "id": encoded["id"],
        "fields": {key: item for key, item in encoded.items() if key != "id"},
    }
    field, invalid = mutation
    if field == "id":
        wrapped["id"] = invalid
    else:
        wrapped["fields"][field] = invalid
    with pytest.raises(VectorStoreIntegrityError):
        protocol._metadata_from_item(wrapped, scope=scope, name=name)


@pytest.mark.parametrize("dimension", [1, 2, 3, 128])
def test_sentinel_vector_has_declared_dimension_and_unit_norm(dimension: int) -> None:
    value = protocol._sentinel_vector(dimension)
    assert len(value) == dimension
    assert math.sqrt(sum(item * item for item in value)) == 1.0


@pytest.mark.parametrize("invalid", [0, -1, True, False, 1.5, "1", None, [], {}])
def test_sentinel_vector_rejects_invalid_dimension(invalid: object) -> None:
    with pytest.raises(ValueError):
        protocol._sentinel_vector(invalid)


@pytest.mark.parametrize(
    ("response", "expected"),
    [({}, {}), ({"result": None}, {}), ({"result": {}}, {}), ({"result": {"data": []}}, {"data": []})],
)
def test_data_result_normalizes_absent_or_mapping_result(
    response: dict[str, object],
    expected: dict[str, object],
) -> None:
    assert protocol._data_result(response) == expected


@pytest.mark.parametrize("invalid", [[], (), "result", 0, 1, True, object()])
def test_data_result_rejects_non_mapping_result(invalid: object) -> None:
    with pytest.raises(VectorStoreIntegrityError):
        protocol._data_result({"result": invalid})


@pytest.mark.parametrize(
    ("response", "expected"),
    [({}, {}), ({"Result": None}, {}), ({"Result": {}}, {}), ({"data": {"x": 1}}, {"x": 1})],
)
def test_console_result_supports_public_and_private_response_envelopes(
    response: dict[str, object],
    expected: dict[str, object],
) -> None:
    assert protocol._console_result(response) == expected


@pytest.mark.parametrize("invalid", [[], (), "result", 0, 1, True, object()])
def test_console_result_rejects_non_mapping_result(invalid: object) -> None:
    with pytest.raises(VectorStoreIntegrityError):
        protocol._console_result({"Result": invalid})


@pytest.mark.parametrize("items", [[], [{"id": "1"}], [{"id": "1"}, {"id": "2"}]])
def test_search_items_accepts_list_of_mapping_items(items: list[dict[str, object]]) -> None:
    assert protocol._search_items({"result": {"data": items}}) == tuple(items)


@pytest.mark.parametrize("invalid", [None, {}, (), "items", 1, True, [1], [{"id": "1"}, 2]])
def test_search_items_rejects_non_list_or_non_mapping_item(invalid: object) -> None:
    with pytest.raises(VectorStoreIntegrityError):
        protocol._search_items({"result": {"data": invalid}})


@pytest.mark.parametrize(
    ("items", "allowed"),
    [([], set()), ([{"id": "a"}], {"a"}), ([{"id": "a"}, {"id": "b"}], {"a", "b", "c"})],
)
def test_fetched_id_validator_accepts_requested_unique_subset(
    items: list[dict[str, object]],
    allowed: set[str],
) -> None:
    protocol._validate_fetched_ids(items, allowed=allowed, label="test")


@pytest.mark.parametrize(
    "items",
    [
        [{"id": None}],
        [{"id": 1}],
        [{"id": "outside"}],
        [{"id": "a"}, {"id": "a"}],
    ],
)
def test_fetched_id_validator_rejects_non_text_unrequested_or_duplicate_id(
    items: list[dict[str, object]],
) -> None:
    with pytest.raises(VectorStoreIntegrityError):
        protocol._validate_fetched_ids(items, allowed={"a"}, label="test")


@pytest.mark.parametrize(
    ("field", "value", "physical_field"),
    [
        ("uri", "memory://profile.md", "uri_key"),
        ("directory_key", "memory://", "directory_key_hash"),
        ("parent_key", "memory://", "parent_key_hash"),
        ("scope_roots", "memory://", "scope_root_keys"),
        ("level", 2, "level"),
        ("kind", "profile", "kind"),
        ("revision", 1, "revision"),
    ],
)
def test_filter_compiler_maps_each_logical_field_to_declared_physical_index(
    field: str,
    value: object,
    physical_field: str,
) -> None:
    compiled = protocol._compile_filter(VectorStoreFilter({field: value}, {}))
    conditions = compiled["conds"] if compiled["op"] == "and" else [compiled]
    assert conditions[0] == {"op": "must", "field": "record_type", "conds": ["memory"]}
    assert any(item["field"] == physical_field for item in conditions)


@pytest.mark.parametrize(
    "filters",
    [
        VectorStoreFilter({}, {}),
        VectorStoreFilter({}, {"kind": ("profile", "event")}),
        VectorStoreFilter(
            {"level": 2},
            {"scope_roots": ("memory://", "memory://events/")},
        ),
    ],
)
def test_filter_compiler_always_limits_results_to_memory_records(filters: VectorStoreFilter) -> None:
    rendered = str(protocol._compile_filter(filters))
    assert "record_type" in rendered
    assert "memory" in rendered


@pytest.mark.parametrize("field", ["unknown", "content", "identity", "metadata_json", "record_type"])
def test_filter_compiler_rejects_field_without_declared_scalar_index(field: str) -> None:
    with pytest.raises(ValueError, match="no declared scalar index"):
        protocol._compile_filter(VectorStoreFilter({field: "x"}, {}))


@pytest.mark.parametrize("field", ["uri", "directory_key", "parent_key", "scope_roots"])
@pytest.mark.parametrize("invalid", [None, 0, 1, True, False, (), [], {}, object()])
def test_hashed_filter_fields_require_text_values(field: str, invalid: object) -> None:
    with pytest.raises(ValueError, match="must be strings"):
        protocol._physical_filter(field, (invalid,))


@pytest.mark.parametrize(
    "attributes",
    [
        {},
        {"uri": "memory://profile.md"},
        {"scope_roots": []},
        {"scope_roots": ["memory://", "memory://events/"]},
        {"level": 0, "revision": 0, "kind": "system"},
        {"level": 2, "revision": 99, "kind": "中文"},
    ],
)
def test_physical_index_fields_derives_stable_bounded_scalar_values(attributes: dict[str, object]) -> None:
    value = protocol._physical_index_fields(attributes)
    assert set(value) == {
        "uri_key",
        "level",
        "directory_key_hash",
        "parent_key_hash",
        "scope_root_keys",
        "kind",
        "revision",
    }
    assert all(len(item) == 64 for item in value["scope_root_keys"])


@pytest.mark.parametrize("field", ["uri", "directory_key", "parent_key"])
@pytest.mark.parametrize("invalid", [None, 0, 1, True, (), [], {}, object()])
def test_physical_index_text_attributes_reject_non_text(field: str, invalid: object) -> None:
    with pytest.raises(ValueError):
        protocol._physical_index_fields({field: invalid})


@pytest.mark.parametrize("invalid", [None, "root", 1, True, {}, ["ok", 1], ("ok", None)])
def test_physical_index_scope_roots_require_text_sequence(invalid: object) -> None:
    with pytest.raises(ValueError):
        protocol._physical_index_fields({"scope_roots": invalid})


@pytest.mark.parametrize("field", ["level", "revision"])
@pytest.mark.parametrize("invalid", [None, "1", 1.5, True, False, (), [], {}])
def test_physical_index_integer_fields_reject_non_integer(field: str, invalid: object) -> None:
    with pytest.raises(ValueError):
        protocol._physical_index_fields({field: invalid})


@pytest.mark.parametrize("invalid", [None, 1, True, (), [], {}, "中" * 129])
def test_physical_index_kind_requires_text_within_utf8_limit(invalid: object) -> None:
    with pytest.raises(ValueError):
        protocol._physical_index_fields({"kind": invalid})


@pytest.mark.parametrize("value", ["", "a", "memory://profile.md", "中文", "x" * 10_000])
def test_index_key_is_deterministic_fixed_width_sha256(value: str) -> None:
    assert protocol._index_key(value) == hashlib.sha256(value.encode()).hexdigest()
    assert len(protocol._index_key(value)) == 64


@pytest.mark.parametrize("invalid", [None, 0, 1, True, (), [], {}, object()])
def test_index_key_rejects_non_text_source(invalid: object) -> None:
    with pytest.raises(TypeError):
        protocol._index_key(invalid)


@pytest.mark.parametrize("value", ["", "a", "b", "memory://profile.md", "中文"])
def test_scan_order_is_deterministic_non_negative_signed_int64(value: str) -> None:
    order = protocol._scan_order(value)
    assert order == protocol._scan_order(value)
    assert 0 <= order < 2**63


def test_scan_order_batch_rejects_non_record_and_simulated_hash_collision(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(TypeError):
        protocol._validate_scan_order_batch((record(), "bad"))
    monkeypatch.setattr(protocol, "_scan_order", lambda _identity: 1)
    with pytest.raises(ValueError, match="collide"):
        protocol._validate_scan_order_batch((record("a"), record("b")))


@pytest.mark.parametrize(
    ("values", "size", "expected"),
    [
        ((), 1, ()),
        ((1,), 1, ((1,),)),
        ((1, 2, 3), 1, ((1,), (2,), (3,))),
        ((1, 2, 3), 2, ((1, 2), (3,))),
        ((1, 2, 3), 10, ((1, 2, 3),)),
    ],
)
def test_batching_preserves_order_and_tail(values: tuple[int, ...], size: int, expected: tuple[tuple[int, ...], ...]) -> None:
    assert protocol._batches(values, size) == expected


@pytest.mark.parametrize(
    "value",
    [{}, {"a": 1}, {"中文": [1, None, False]}, {"nested": {"b": 2, "a": 1}}],
)
def test_json_text_is_canonical_compact_and_unicode_preserving(value: dict[str, object]) -> None:
    encoded = protocol._json_text(value, "test")
    assert " " not in encoded
    assert "\\u" not in encoded


@pytest.mark.parametrize("invalid", [{"x": object()}, {"x": set()}, {"x": math.nan}, {"x": math.inf}])
def test_json_text_rejects_non_json_or_non_finite_value(invalid: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="JSON serializable"):
        protocol._json_text(invalid, "test")


def test_bounded_text_enforces_utf8_bytes_not_character_count() -> None:
    assert protocol._bounded_text("x" * (1024 * 1024), "test")
    with pytest.raises(ValueError, match="one-megabyte"):
        protocol._bounded_text("中" * 400_000, "test")
