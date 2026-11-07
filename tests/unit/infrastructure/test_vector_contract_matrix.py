"""向量公共模型、路由与容量的边界矩阵。"""

from __future__ import annotations

import hashlib
import math

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
        ({}, {"kind": ()}),
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
@pytest.mark.parametrize("invalid", ["", " ", " leading", "trailing "])
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
    ],
)
def test_vector_state_rejects_invalid_numeric_fields(field: str, invalid: object) -> None:
    with pytest.raises(ValueError):
        _state(**{field: invalid})


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
@pytest.mark.parametrize("value", ["", " ", "1invalid", "bad/name", "bad name"])
def test_vector_route_rejects_invalid_provider_or_adapter(field: str, value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        VectorStoreRouteConfig(**{field: value})


@pytest.mark.parametrize("credential_ref", ["", "vector", "Vector.Main", "private-vector"])
def test_vector_route_normalizes_credential_reference(credential_ref: str) -> None:
    route = VectorStoreRouteConfig(credential_ref=f" {credential_ref} ")
    assert route.credential_ref == credential_ref.lower()


@pytest.mark.parametrize(
    "credential_ref",
    [
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
        {"": "value"},
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


@pytest.mark.parametrize(
    ("field", "bad", "error", "message"),
    [
        ("provider", 1, TypeError, "provider and adapter must be strings"),
        ("adapter", ["local_sqlite"], TypeError, "provider and adapter must be strings"),
        ("base_url", 1, TypeError, "base_url must be a string"),
        ("credential_ref", None, TypeError, "credential_ref must be a string"),
        ("timeout_seconds", "30", ValueError, "timeout_seconds must be numeric"),
        ("retry_base_delay_seconds", True, ValueError, "retry_base_delay_seconds must be numeric"),
        ("retry_max_delay_seconds", None, ValueError, "retry_max_delay_seconds must be numeric"),
        ("max_retries", True, ValueError, "max_retries must be between 0 and 10"),
        ("max_concurrent", 2.0, ValueError, "max_concurrent must be between 1 and 4096"),
        ("max_response_bytes", "2048", ValueError, "max_response_bytes must be between 1024 and 67108864"),
        ("extra_headers", ["X-Trace"], TypeError, "extra_headers must be an object"),
        ("extra_headers", {"X-Trace": 1}, ValueError, "extra_headers must contain string keys and values"),
        ("extra_headers", {1: "value"}, ValueError, "extra_headers must contain string keys and values"),
    ],
)
def test_vector_route_from_mapping_rejects_wrong_yaml_types(
    field: str, bad: object, error: type[Exception], message: str
) -> None:
    """YAML 里类型错的路由字段在 ``from_mapping`` 就被拒（W16：类型检查在边界，不靠值对象构造器）。"""

    with pytest.raises(error, match=message):
        VectorStoreRouteConfig.from_mapping({field: bad})


@pytest.mark.parametrize(
    ("field", "bad", "error", "message"),
    [
        ("collection", 1, TypeError, "collection must be a string"),
        ("options", ["schema_mode"], TypeError, "options must be an object"),
        ("options", {1: "value"}, ValueError, "options keys must be non-empty strings"),
        ("route", {"max_retries": "2"}, ValueError, "max_retries must be between 0 and 10"),
    ],
)
def test_vector_store_config_from_mapping_rejects_wrong_yaml_types(
    field: str, bad: object, error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        VectorStoreConfig.from_mapping({field: bad})


def test_vector_store_config_from_mapping_keeps_defaults_for_absent_fields() -> None:
    assert VectorStoreConfig.from_mapping({}) == VectorStoreConfig()
    assert VectorStoreConfig.from_mapping({"route": {"max_retries": 3}}).route.max_retries == 3


@pytest.mark.parametrize("collection", ["memory", "conversation_summaries", "a.b-c_1", "1collection"])
def test_vector_store_config_accepts_collection_names(collection: str) -> None:
    assert VectorStoreConfig(collection=collection).collection == collection


@pytest.mark.parametrize(
    "collection",
    ["", " ", ".hidden", "-leading", "bad/name", "bad name", "a" * 256],
)
def test_vector_store_config_rejects_invalid_collection_names(collection: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        VectorStoreConfig(collection=collection)  # type: ignore[arg-type]


@pytest.mark.parametrize("options", [{}, {"schema_mode": "managed"}, {"nested": {"a": [1, True, None]}}])
def test_vector_store_config_accepts_json_options(options: dict[str, object]) -> None:
    assert dict(VectorStoreConfig(options=options).options) == options


@pytest.mark.parametrize(
    "options",
    [{"": 1}, {"set": {1, 2}}, {"nan": math.nan}, {"object": object()}],
)
def test_vector_store_config_rejects_non_json_options(options: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        VectorStoreConfig(options=options)  # type: ignore[arg-type]


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
