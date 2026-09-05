"""向量公共模型、路由、容量和 VikingDB 严格配置的边界矩阵。"""

from __future__ import annotations

import hashlib
import math
from dataclasses import replace

import pytest

from habitus.infrastructure.vector import (
    VectorPublicationSnapshot,
    VectorStoreConfig,
    VectorStoreFilter,
    VectorStoreMatch,
    VectorStoreRecord,
    VectorStoreRequirements,
    VectorStoreRouteConfig,
    VectorStoreState,
)
from habitus.infrastructure.vector.adapters.vikingdb_config import (
    VikingDBVectorStoreConfig,
    bounded_retry_after,
    credential_template_names,
    render_credential_template,
)
from habitus.model_client import EmbeddingVector


def _record(**overrides: object) -> VectorStoreRecord:
    content = overrides.pop("content", "memory content")
    digest_source = content if isinstance(content, str) else "invalid content"
    values: dict[str, object] = {
        "identity": "memory://profile.md",
        "vector": EmbeddingVector((1.0, 0.0)),
        "content": content,
        "content_digest": hashlib.sha256(digest_source.encode()).hexdigest(),
        "attributes": {"kind": "profile", "level": 2, "scope_roots": ["memory://"]},
    }
    values.update(overrides)
    return VectorStoreRecord(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize("identity", ["memory://profile.md", "中文身份", "a", "https://example.test/item"])
def test_vector_record_accepts_normalized_identity(identity: str) -> None:
    assert _record(identity=identity).identity == identity


@pytest.mark.parametrize("identity", ["", " ", " leading", "trailing ", "a\nb", "a\tb", None, 1])
def test_vector_record_rejects_invalid_identity(identity: object) -> None:
    with pytest.raises(ValueError):
        _record(identity=identity)


@pytest.mark.parametrize("content", ["content", " 中文 ", "\nvalue\n"])
def test_vector_record_accepts_non_empty_content_and_exact_digest(content: str) -> None:
    assert _record(content=content).content == content


@pytest.mark.parametrize("content", ["", " ", "\n\t", None, 1])
def test_vector_record_rejects_empty_or_non_text_content(content: object) -> None:
    with pytest.raises(ValueError):
        _record(content=content)


@pytest.mark.parametrize("vector", [None, (1.0, 0.0), [1.0, 0.0], "vector"])
def test_vector_record_requires_embedding_vector(vector: object) -> None:
    with pytest.raises(TypeError):
        _record(vector=vector)


@pytest.mark.parametrize("digest", ["", "0" * 64, "A" * 64, None])
def test_vector_record_rejects_non_matching_content_digest(digest: object) -> None:
    with pytest.raises(ValueError, match="content_digest"):
        _record(content_digest=digest)


@pytest.mark.parametrize(
    "attributes",
    [
        {"text": "value"},
        {"integer": 1},
        {"float": 0.5},
        {"boolean": True},
        {"tuple": ("a", "b")},
        {"list": ["a", "b"]},
        {"mixed": ["a", 2, False]},
    ],
)
def test_vector_record_normalizes_supported_attribute_values(attributes: dict[str, object]) -> None:
    value = _record(attributes=attributes)
    assert dict(value.attributes)


@pytest.mark.parametrize(
    "attributes",
    [
        None,
        [],
        {"": "value"},
        {" field": "value"},
        {"field ": "value"},
        {"line\nbreak": "value"},
        {"empty": []},
        {"duplicate": ["a", "a"]},
        {"nested": {"a": 1}},
        {"none": None},
        {"nan": math.nan},
        {"inf": math.inf},
    ],
)
def test_vector_record_rejects_unsupported_or_ambiguous_attributes(attributes: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        _record(attributes=attributes)


@pytest.mark.parametrize(
    ("attributes", "equals", "one_of", "expected"),
    [
        ({"kind": "profile"}, {"kind": "profile"}, {}, True),
        ({"kind": "profile"}, {"kind": "event"}, {}, False),
        ({"kind": "profile"}, {}, {"kind": ("profile", "event")}, True),
        ({"scope": ("memory://", "memory://profile.md")}, {}, {"scope": ("memory://profile.md",)}, True),
        ({"scope": ("memory://",)}, {}, {"scope": ("memory://event",)}, False),
        ({"kind": "profile"}, {"missing": "x"}, {}, False),
        ({"kind": "profile", "level": 2}, {"kind": "profile"}, {"level": (1, 2)}, True),
    ],
)
def test_vector_filter_has_deterministic_cross_backend_matching(
    attributes: dict[str, object],
    equals: dict[str, object],
    one_of: dict[str, tuple[object, ...]],
    expected: bool,
) -> None:
    condition = VectorStoreFilter(equals, one_of)  # type: ignore[arg-type]
    assert condition.matches(attributes) is expected  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("equals", "one_of"),
    [
        ({"kind": ("profile",)}, {}),
        ({}, None),
        ({}, {"kind": ()}),
        ({}, {"kind": ["profile"]}),
        ({}, {"kind": ("profile", "profile")}),
        ({"kind": "profile"}, {"kind": ("profile",)}),
        ({}, {"": ("profile",)}),
        ({}, {"kind": (None,)}),
    ],
)
def test_vector_filter_rejects_invalid_operator_shapes(equals: object, one_of: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        VectorStoreFilter(equals, one_of)  # type: ignore[arg-type]


@pytest.mark.parametrize("score", [-1, -0.5, 0, 0.5, 1, 1.0])
def test_vector_match_accepts_finite_cosine_boundaries(score: int | float) -> None:
    assert VectorStoreMatch(_record(), score).score == float(score)


@pytest.mark.parametrize("score", [-1.00001, 1.00001, math.nan, math.inf, -math.inf])
def test_vector_match_rejects_out_of_range_or_non_finite_score(score: float) -> None:
    with pytest.raises(ValueError):
        VectorStoreMatch(_record(), score)


@pytest.mark.parametrize("score", [True, False, "0.5", None, []])
def test_vector_match_rejects_non_numeric_score(score: object) -> None:
    with pytest.raises(TypeError):
        VectorStoreMatch(_record(), score)  # type: ignore[arg-type]


def test_vector_match_requires_valid_record() -> None:
    with pytest.raises(TypeError):
        VectorStoreMatch(object(), 0.5)  # type: ignore[arg-type]


def _state(**overrides: object) -> VectorStoreState:
    values: dict[str, object] = {
        "schema_version": "v1",
        "embedding_fingerprint": "sha256:embedding",
        "dimension": 2,
        "checkpoint": 0,
        "generation": 1,
        "record_count": 0,
        "ready": True,
    }
    values.update(overrides)
    return VectorStoreState(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize("field", ["schema_version", "embedding_fingerprint"])
@pytest.mark.parametrize("invalid", ["", " ", " leading", "trailing ", None, 1])
def test_vector_state_rejects_invalid_identity_fields(field: str, invalid: object) -> None:
    with pytest.raises(ValueError):
        _state(**{field: invalid})


@pytest.mark.parametrize(
    ("field", "minimum"),
    [("dimension", 1), ("checkpoint", 0), ("generation", 1), ("record_count", 0)],
)
def test_vector_state_accepts_numeric_minimums(field: str, minimum: int) -> None:
    assert getattr(_state(**{field: minimum}), field) == minimum


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("dimension", 0),
        ("dimension", -1),
        ("checkpoint", -1),
        ("generation", 0),
        ("record_count", -1),
        ("dimension", True),
        ("checkpoint", 1.0),
        ("generation", "1"),
        ("record_count", None),
    ],
)
def test_vector_state_rejects_invalid_numeric_fields(field: str, invalid: object) -> None:
    with pytest.raises(ValueError):
        _state(**{field: invalid})


@pytest.mark.parametrize("ready", [None, 0, 1, "true"])
def test_vector_state_requires_boolean_ready(ready: object) -> None:
    with pytest.raises(TypeError):
        _state(ready=ready)


@pytest.mark.parametrize(
    ("state", "claim_exists", "building"),
    [
        (None, False, False),
        (None, True, False),
        (None, True, True),
        (_state(), False, False),
        (_state(), True, True),
    ],
)
def test_vector_publication_snapshot_accepts_coherent_states(
    state: VectorStoreState | None,
    claim_exists: bool,
    building: bool,
) -> None:
    assert VectorPublicationSnapshot(state, claim_exists, building).building is building


@pytest.mark.parametrize(
    ("state", "claim_exists", "building", "error"),
    [
        (object(), False, False, TypeError),
        (None, 1, False, TypeError),
        (None, False, 1, TypeError),
        (None, False, True, ValueError),
    ],
)
def test_vector_publication_snapshot_rejects_incoherent_states(
    state: object,
    claim_exists: object,
    building: object,
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        VectorPublicationSnapshot(state, claim_exists, building)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "base_url",
    [
        "",
        "https://vector.example.com",
        "https://vector.example.com/api/v2",
        "http://localhost:8080",
        "http://127.0.0.1:8080",
        "http://[::1]:8080",
    ],
)
def test_vector_route_accepts_secure_remote_or_loopback_url(base_url: str) -> None:
    assert VectorStoreRouteConfig(base_url=base_url).base_url == base_url.rstrip("/")


@pytest.mark.parametrize(
    "base_url",
    [
        "ftp://vector.example.com",
        "http://vector.example.com",
        "https://user@vector.example.com",
        "https://user:pass@vector.example.com",
        "https://vector.example.com?token=x",
        "https://vector.example.com#fragment",
        "//vector.example.com",
        "not-a-url",
    ],
)
def test_vector_route_rejects_insecure_or_credential_bearing_url(base_url: str) -> None:
    with pytest.raises(ValueError):
        VectorStoreRouteConfig(base_url=base_url)


@pytest.mark.parametrize("field", ["provider", "adapter"])
@pytest.mark.parametrize("value", ["", " ", "1invalid", "bad/name", "bad name", None, 1])
def test_vector_route_rejects_invalid_provider_or_adapter(field: str, value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        VectorStoreRouteConfig(**{field: value})


@pytest.mark.parametrize("credential_ref", ["", "vikingdb", "VikingDB.Main", "private-vector"])
def test_vector_route_normalizes_credential_reference(credential_ref: str) -> None:
    route = VectorStoreRouteConfig(credential_ref=f" {credential_ref} ")
    assert route.credential_ref == credential_ref.lower()


@pytest.mark.parametrize(
    "credential_ref",
    [
        None,
        [],
        "1bad",
        "bad/name",
        "bad name",
        "$credential",
    ],
)
def test_vector_route_rejects_invalid_credential_reference(credential_ref: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        VectorStoreRouteConfig(credential_ref=credential_ref)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "extra_headers",
    [
        {"X-Trace": "value"},
        {"X-One": "1", "x-two": "2"},
        {},
    ],
)
def test_vector_route_accepts_non_secret_static_headers(extra_headers: dict[str, str]) -> None:
    assert dict(VectorStoreRouteConfig(extra_headers=extra_headers).extra_headers) == extra_headers


@pytest.mark.parametrize(
    "extra_headers",
    [
        None,
        [],
        {"": "value"},
        {1: "value"},
        {"X": 1},
        {"X\nBad": "value"},
        {"X:Bad": "value"},
        {"X": "line\nbreak"},
        {"Authorization": "secret"},
        {"Proxy-Authorization": "secret"},
        {"Host": "override"},
        {"X-Test": "one", "x-test": "two"},
    ],
)
def test_vector_route_rejects_invalid_reserved_or_duplicate_headers(extra_headers: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        VectorStoreRouteConfig(extra_headers=extra_headers)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "valid"),
    [
        ("timeout_seconds", 0.001),
        ("timeout_seconds", 600),
        ("retry_base_delay_seconds", 0.001),
        ("retry_base_delay_seconds", 60),
        ("retry_max_delay_seconds", 0.001),
        ("retry_max_delay_seconds", 300),
        ("max_retries", 0),
        ("max_retries", 10),
        ("max_concurrent", 1),
        ("max_concurrent", 4096),
        ("max_response_bytes", 1024),
        ("max_response_bytes", 64 * 1024 * 1024),
    ],
)
def test_vector_route_accepts_operational_boundaries(field: str, valid: int | float) -> None:
    values = {field: valid}
    if field == "retry_base_delay_seconds" and valid == 60:
        values["retry_max_delay_seconds"] = 60
    if field == "retry_max_delay_seconds" and valid == 0.001:
        values["retry_base_delay_seconds"] = 0.001
    assert getattr(VectorStoreRouteConfig(**values), field) == valid


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("timeout_seconds", 0),
        ("timeout_seconds", 601),
        ("timeout_seconds", math.nan),
        ("retry_base_delay_seconds", 0),
        ("retry_max_delay_seconds", 301),
        ("max_retries", -1),
        ("max_retries", 11),
        ("max_retries", True),
        ("max_concurrent", 0),
        ("max_concurrent", 4097),
        ("max_response_bytes", 1023),
        ("max_response_bytes", 64 * 1024 * 1024 + 1),
    ],
)
def test_vector_route_rejects_operational_values_outside_boundaries(field: str, invalid: object) -> None:
    with pytest.raises(ValueError):
        VectorStoreRouteConfig(**{field: invalid})


def test_vector_route_rejects_retry_max_below_base() -> None:
    with pytest.raises(ValueError, match="cannot be below"):
        VectorStoreRouteConfig(retry_base_delay_seconds=2, retry_max_delay_seconds=1)


@pytest.mark.parametrize("value", [None, [], "route", 1])
def test_vector_route_from_mapping_rejects_non_object(value: object) -> None:
    with pytest.raises(TypeError):
        VectorStoreRouteConfig.from_mapping(value)


def test_vector_route_from_mapping_rejects_unknown_fields() -> None:
    with pytest.raises(ValueError, match="unknown"):
        VectorStoreRouteConfig.from_mapping({"unknown": True})


@pytest.mark.parametrize("collection", ["memory", "conversation_summaries", "a.b-c_1", "1collection"])
def test_vector_store_config_accepts_collection_names(collection: str) -> None:
    assert VectorStoreConfig(collection=collection).collection == collection


@pytest.mark.parametrize(
    "collection",
    ["", " ", ".hidden", "-leading", "bad/name", "bad name", "a" * 256, None, 1],
)
def test_vector_store_config_rejects_invalid_collection_names(collection: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        VectorStoreConfig(collection=collection)  # type: ignore[arg-type]


@pytest.mark.parametrize("options", [{}, {"schema_mode": "managed"}, {"nested": {"a": [1, True, None]}}])
def test_vector_store_config_accepts_json_options(options: dict[str, object]) -> None:
    assert dict(VectorStoreConfig(options=options).options) == options


@pytest.mark.parametrize(
    "options",
    [None, [], {"": 1}, {1: "value"}, {"set": {1, 2}}, {"nan": math.nan}, {"object": object()}],
)
def test_vector_store_config_rejects_non_json_options(options: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        VectorStoreConfig(options=options)  # type: ignore[arg-type]


def test_vector_store_config_requires_route_value_object() -> None:
    with pytest.raises(TypeError):
        VectorStoreConfig(route=object())  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "valid"),
    [
        ("dimension", 1),
        ("dimension", 65_536),
        ("max_records", 1),
        ("max_records", 100_000_000),
        ("max_search_hits", 1),
        ("max_search_hits", 1_000_000),
        ("max_record_chars", 1),
        ("max_record_chars", 1_000_000),
    ],
)
def test_vector_requirements_accept_boundaries(field: str, valid: int) -> None:
    values = {"dimension": 2, "max_records": 1_000_000, "max_search_hits": 1, "max_record_chars": 100}
    values[field] = valid
    if field == "max_records" and valid == 1:
        values["max_search_hits"] = 1
    assert getattr(VectorStoreRequirements(**values), field) == valid


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("dimension", 0),
        ("dimension", 65_537),
        ("max_records", 0),
        ("max_records", 100_000_001),
        ("max_search_hits", 0),
        ("max_search_hits", 1_000_001),
        ("max_record_chars", 0),
        ("max_record_chars", 1_000_001),
        ("dimension", True),
        ("max_records", 1.0),
    ],
)
def test_vector_requirements_reject_out_of_range_or_non_integer_values(field: str, invalid: object) -> None:
    values: dict[str, object] = {
        "dimension": 2,
        "max_records": 100,
        "max_search_hits": 10,
        "max_record_chars": 100,
    }
    values[field] = invalid
    with pytest.raises(ValueError):
        VectorStoreRequirements(**values)  # type: ignore[arg-type]


def test_vector_requirements_reject_search_capacity_above_record_capacity() -> None:
    with pytest.raises(ValueError, match="cannot exceed"):
        VectorStoreRequirements(2, 10, 11, 100)


@pytest.mark.parametrize("auth_mode", ["api_key", "ak_sk", "private_headers"])
def test_vikingdb_config_accepts_each_authentication_mode(auth_mode: str) -> None:
    overrides: dict[str, object] = {"auth_mode": auth_mode}
    if auth_mode == "private_headers":
        overrides.update(region="", credential_headers={"X-Token": "Bearer {token}"})
    config = VikingDBVectorStoreConfig(**overrides)  # type: ignore[arg-type]
    assert config.auth_mode == auth_mode


@pytest.mark.parametrize("auth_mode", ["", "bearer", "oauth", None, 1])
def test_vikingdb_config_rejects_unknown_authentication_mode(auth_mode: object) -> None:
    with pytest.raises(ValueError):
        VikingDBVectorStoreConfig(auth_mode=auth_mode)  # type: ignore[arg-type]


@pytest.mark.parametrize("schema_mode", ["managed", "precreated"])
def test_vikingdb_config_accepts_schema_modes_with_valid_authentication(schema_mode: str) -> None:
    auth_mode = "ak_sk" if schema_mode == "managed" else "api_key"
    assert VikingDBVectorStoreConfig(auth_mode=auth_mode, schema_mode=schema_mode).schema_mode == schema_mode


@pytest.mark.parametrize("schema_mode", ["", "automatic", "legacy", None, 1])
def test_vikingdb_config_rejects_unknown_schema_mode(schema_mode: object) -> None:
    with pytest.raises(ValueError):
        VikingDBVectorStoreConfig(schema_mode=schema_mode)  # type: ignore[arg-type]


@pytest.mark.parametrize("auth_mode", ["api_key", "private_headers"])
def test_vikingdb_managed_schema_requires_ak_sk(auth_mode: str) -> None:
    overrides: dict[str, object] = {"auth_mode": auth_mode, "schema_mode": "managed"}
    if auth_mode == "private_headers":
        overrides.update(region="", credential_headers={"X-Token": "{token}"})
    with pytest.raises(ValueError, match="requires ak_sk"):
        VikingDBVectorStoreConfig(**overrides)  # type: ignore[arg-type]


@pytest.mark.parametrize("field", ["project_name", "index_name"])
@pytest.mark.parametrize("value", ["default", "project-1", "index_name", "a.b", "9name", " value "])
def test_vikingdb_resource_names_are_validated_and_trimmed(field: str, value: str) -> None:
    config = VikingDBVectorStoreConfig(**{field: value})
    assert getattr(config, field) == value.strip()


@pytest.mark.parametrize("field", ["project_name", "index_name"])
@pytest.mark.parametrize("value", ["", " ", "-leading", ".leading", "bad/name", "bad name", "a" * 129, None, 1])
def test_vikingdb_rejects_invalid_resource_names(field: str, value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        VikingDBVectorStoreConfig(**{field: value})


@pytest.mark.parametrize("region", ["cn-beijing", "CN-BEIJING", " ap-southeast-1 ", "cn-guangzhou", "cn-shanghai"])
def test_vikingdb_normalizes_public_region(region: str) -> None:
    assert VikingDBVectorStoreConfig(region=region).region == region.strip().lower()


@pytest.mark.parametrize("region", ["", "beijing", "cn_beijing", "CN Beijing", None, 1])
def test_vikingdb_public_auth_rejects_invalid_region(region: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        VikingDBVectorStoreConfig(region=region)  # type: ignore[arg-type]


def test_vikingdb_unknown_public_region_requires_explicit_endpoints() -> None:
    with pytest.raises(ValueError, match="console_url"):
        VikingDBVectorStoreConfig(auth_mode="ak_sk", region="custom-region-1")
    config = VikingDBVectorStoreConfig(
        auth_mode="ak_sk",
        region="custom-region-1",
        console_url="https://console.example.com",
    )
    with pytest.raises(ValueError, match="route.base_url"):
        config.data_url(VectorStoreRouteConfig())
    assert config.data_url(VectorStoreRouteConfig(base_url="https://data.example.com")) == "https://data.example.com"


@pytest.mark.parametrize(
    "console_url",
    [
        "https://console.example.com",
        "https://console.example.com/",
        "http://localhost:8080",
        "http://127.0.0.1:8080",
    ],
)
def test_vikingdb_ak_sk_accepts_secure_or_loopback_console_origin(console_url: str) -> None:
    config = VikingDBVectorStoreConfig(auth_mode="ak_sk", console_url=console_url)
    assert config.console_url == console_url.rstrip("/")


@pytest.mark.parametrize(
    "console_url",
    [
        "http://remote.example.com",
        "ftp://console.example.com",
        "https://user@console.example.com",
        "https://console.example.com/path",
        "https://console.example.com?token=x",
        "https://console.example.com#fragment",
    ],
)
def test_vikingdb_rejects_invalid_console_origin(console_url: str) -> None:
    with pytest.raises(ValueError):
        VikingDBVectorStoreConfig(auth_mode="ak_sk", console_url=console_url)


@pytest.mark.parametrize("auth_mode", ["api_key", "private_headers"])
def test_vikingdb_non_ak_sk_rejects_console_url(auth_mode: str) -> None:
    overrides: dict[str, object] = {"auth_mode": auth_mode, "console_url": "https://console.example.com"}
    if auth_mode == "private_headers":
        overrides.update(region="", credential_headers={"X-Token": "{token}"})
    with pytest.raises(ValueError, match="only valid"):
        VikingDBVectorStoreConfig(**overrides)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "headers",
    [
        {"X-Token": "Bearer {token}"},
        {"X-AK": "{access_key}", "X-Signature": "v1 {signature}"},
        {"X-Combined": "{tenant}:{token}"},
    ],
)
def test_vikingdb_private_headers_accept_controlled_templates(headers: dict[str, str]) -> None:
    config = VikingDBVectorStoreConfig(auth_mode="private_headers", region="", credential_headers=headers)
    assert dict(config.credential_headers) == headers


@pytest.mark.parametrize(
    "headers",
    [
        None,
        [],
        {"": "{token}"},
        {1: "{token}"},
        {"X:Bad": "{token}"},
        {"X\nBad": "{token}"},
        {"X": ""},
        {"X": 1},
        {"X": "no-placeholder"},
        {"X": "{token"},
        {"X": "token}"},
        {"X": "{token}\nvalue"},
        {"Host": "{token}"},
        {"Proxy-Authorization": "{token}"},
        {"X-Test": "{one}", "x-test": "{two}"},
    ],
)
def test_vikingdb_private_headers_reject_invalid_templates(headers: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        VikingDBVectorStoreConfig(auth_mode="private_headers", region="", credential_headers=headers)  # type: ignore[arg-type]


def test_vikingdb_public_auth_rejects_private_credential_headers() -> None:
    with pytest.raises(ValueError, match="only valid"):
        VikingDBVectorStoreConfig(credential_headers={"X-Token": "{token}"})


@pytest.mark.parametrize(
    ("field", "minimum", "maximum"),
    [
        ("upsert_batch_size", 1, 100),
        ("fetch_batch_size", 1, 100),
        ("delete_batch_size", 1, 100),
        ("search_page_size", 1, 5000),
        ("scan_page_size", 1, 5000),
        ("max_search_hits", 1, 1_000_000),
        ("max_records", 1, 100_000_000),
    ],
)
def test_vikingdb_integer_capacity_accepts_both_boundaries(field: str, minimum: int, maximum: int) -> None:
    base = {"max_records": 100_000_000, "max_search_hits": 1}
    for value in (minimum, maximum):
        values = {**base, field: value}
        if field == "max_records" and value == 1:
            values["max_search_hits"] = 1
        assert getattr(VikingDBVectorStoreConfig(**values), field) == value


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("upsert_batch_size", 0),
        ("upsert_batch_size", 101),
        ("fetch_batch_size", 0),
        ("fetch_batch_size", 101),
        ("delete_batch_size", 0),
        ("delete_batch_size", 101),
        ("search_page_size", 0),
        ("search_page_size", 5001),
        ("scan_page_size", 0),
        ("scan_page_size", 5001),
        ("max_search_hits", 0),
        ("max_search_hits", 1_000_001),
        ("max_records", 0),
        ("max_records", 100_000_001),
        ("fetch_batch_size", True),
        ("scan_page_size", 1.0),
    ],
)
def test_vikingdb_integer_capacity_rejects_out_of_range_or_non_integer(field: str, invalid: object) -> None:
    with pytest.raises(ValueError):
        VikingDBVectorStoreConfig(**{field: invalid})


def test_vikingdb_search_capacity_cannot_exceed_record_capacity() -> None:
    with pytest.raises(ValueError, match="cannot exceed"):
        VikingDBVectorStoreConfig(max_records=10, max_search_hits=11)


@pytest.mark.parametrize(
    ("field", "minimum", "maximum"),
    [
        ("index_sync_timeout_seconds", 0.1, 3600.0),
        ("index_sync_poll_interval_seconds", 0.01, 60.0),
    ],
)
def test_vikingdb_sync_timing_accepts_boundaries(field: str, minimum: float, maximum: float) -> None:
    for value in (minimum, maximum):
        values: dict[str, object] = {field: value}
        if field == "index_sync_timeout_seconds" and value == 0.1:
            values["index_sync_poll_interval_seconds"] = 0.01
        if field == "index_sync_poll_interval_seconds" and value == 60:
            values["index_sync_timeout_seconds"] = 60
        assert getattr(VikingDBVectorStoreConfig(**values), field) == value


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("index_sync_timeout_seconds", 0.09),
        ("index_sync_timeout_seconds", 3600.1),
        ("index_sync_timeout_seconds", math.nan),
        ("index_sync_timeout_seconds", True),
        ("index_sync_poll_interval_seconds", 0.009),
        ("index_sync_poll_interval_seconds", 60.1),
        ("index_sync_poll_interval_seconds", math.inf),
        ("index_sync_poll_interval_seconds", "1"),
    ],
)
def test_vikingdb_sync_timing_rejects_invalid_values(field: str, invalid: object) -> None:
    with pytest.raises(ValueError):
        VikingDBVectorStoreConfig(**{field: invalid})


def test_vikingdb_poll_interval_cannot_exceed_timeout() -> None:
    with pytest.raises(ValueError, match="cannot exceed"):
        VikingDBVectorStoreConfig(index_sync_timeout_seconds=1, index_sync_poll_interval_seconds=2)


@pytest.mark.parametrize("value", [None, [], "options", 1])
def test_vikingdb_from_mapping_requires_mapping(value: object) -> None:
    with pytest.raises(TypeError):
        VikingDBVectorStoreConfig.from_mapping(value)  # type: ignore[arg-type]


def test_vikingdb_from_mapping_rejects_unknown_fields() -> None:
    with pytest.raises(ValueError, match="unknown"):
        VikingDBVectorStoreConfig.from_mapping({"unknown": True})


@pytest.mark.parametrize(
    ("config", "route", "expected"),
    [
        (VikingDBVectorStoreConfig(region="cn-beijing"), VectorStoreRouteConfig(), "https://api-vikingdb.vikingdb.cn-beijing.volces.com"),
        (VikingDBVectorStoreConfig(region="cn-shanghai"), VectorStoreRouteConfig(), "https://api-vikingdb.vikingdb.cn-shanghai.volces.com"),
        (VikingDBVectorStoreConfig(), VectorStoreRouteConfig(base_url="https://custom.example.com"), "https://custom.example.com"),
        (
            VikingDBVectorStoreConfig(auth_mode="private_headers", region="", credential_headers={"X": "{token}"}),
            VectorStoreRouteConfig(base_url="http://localhost:8080"),
            "http://localhost:8080",
        ),
    ],
)
def test_vikingdb_data_url_resolution(config: VikingDBVectorStoreConfig, route: VectorStoreRouteConfig, expected: str) -> None:
    assert config.data_url(route) == expected


def test_vikingdb_data_url_requires_route_type() -> None:
    with pytest.raises(TypeError):
        VikingDBVectorStoreConfig().data_url(object())  # type: ignore[arg-type]


@pytest.mark.parametrize("auth_mode", ["api_key", "private_headers"])
def test_vikingdb_console_url_requires_ak_sk(auth_mode: str) -> None:
    overrides: dict[str, object] = {"auth_mode": auth_mode}
    if auth_mode == "private_headers":
        overrides.update(region="", credential_headers={"X": "{token}"})
    with pytest.raises(ValueError):
        VikingDBVectorStoreConfig(**overrides).resolved_console_url()  # type: ignore[arg-type]


def test_vikingdb_console_url_uses_explicit_or_region_endpoint() -> None:
    assert VikingDBVectorStoreConfig(auth_mode="ak_sk").resolved_console_url() == "https://vikingdb.cn-beijing.volcengineapi.com"
    explicit = VikingDBVectorStoreConfig(auth_mode="ak_sk", console_url="https://console.example.com")
    assert explicit.resolved_console_url() == "https://console.example.com"


@pytest.mark.parametrize(
    ("template", "names"),
    [
        ("Bearer {token}", ("token",)),
        ("{ACCESS_KEY}:{secret_key}", ("access_key", "secret_key")),
        ("prefix {tenant.id} suffix", ("tenant.id",)),
        ("no placeholders", ()),
        ("{token}-{token}", ("token", "token")),
    ],
)
def test_credential_template_extracts_normalized_names(template: str, names: tuple[str, ...]) -> None:
    assert credential_template_names(template) == names


@pytest.mark.parametrize("template", [None, 1, [], {}])
def test_credential_template_requires_text(template: object) -> None:
    with pytest.raises(TypeError):
        credential_template_names(template)  # type: ignore[arg-type]


def test_credential_template_renders_only_declared_placeholders() -> None:
    assert render_credential_template("Bearer {token}", {"token": "secret"}) == "Bearer secret"
    assert render_credential_template("{one}:{two}", {"one": "1", "two": "2"}) == "1:2"
    with pytest.raises(ValueError, match="missing"):
        render_credential_template("{one}:{two}", {"one": "1"})


@pytest.mark.parametrize(
    ("value", "maximum", "expected"),
    [
        (None, 10, None),
        ("", 10, None),
        ("invalid", 10, None),
        ("nan", 10, None),
        ("inf", 10, None),
        ("-1", 10, None),
        ("0", 10, 0.0),
        ("0.5", 10, 0.5),
        ("10", 10, 10.0),
        ("999", 10, 10.0),
    ],
)
def test_retry_after_is_finite_non_negative_and_bounded(value: str | None, maximum: float, expected: float | None) -> None:
    assert bounded_retry_after(value, maximum) == expected


def test_vikingdb_requirements_validate_record_search_and_response_capacities() -> None:
    config = VikingDBVectorStoreConfig(max_records=100, max_search_hits=20, fetch_batch_size=1, search_page_size=1, scan_page_size=1)
    route = VectorStoreRouteConfig(max_response_bytes=20_000)
    config.validate_requirements(VectorStoreRequirements(2, 100, 20, 100), route)
    with pytest.raises(ValueError, match="max_records"):
        config.validate_requirements(VectorStoreRequirements(2, 101, 20, 100), route)
    with pytest.raises(ValueError, match="max_search_hits"):
        config.validate_requirements(VectorStoreRequirements(2, 100, 21, 100), route)
    with pytest.raises(ValueError, match="page sizes"):
        replace(config, fetch_batch_size=100).validate_requirements(
            VectorStoreRequirements(2, 100, 20, 10_000),
            route,
        )


@pytest.mark.parametrize("requirements", [None, object(), {}])
def test_vikingdb_requirements_require_contract_type(requirements: object) -> None:
    with pytest.raises(TypeError):
        VikingDBVectorStoreConfig().validate_requirements(requirements, VectorStoreRouteConfig())  # type: ignore[arg-type]


@pytest.mark.parametrize("route", [None, object(), {}])
def test_vikingdb_requirements_require_route_type(route: object) -> None:
    with pytest.raises(TypeError):
        VikingDBVectorStoreConfig().validate_requirements(VectorStoreRequirements(2, 10, 5, 100), route)  # type: ignore[arg-type]
