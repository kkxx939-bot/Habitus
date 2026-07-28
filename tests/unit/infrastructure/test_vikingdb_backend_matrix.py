"""VikingDB Backend 的 Schema、批处理、分页和可见性状态机矩阵。"""

from __future__ import annotations

import asyncio

import pytest

from infrastructure.vector import (
    VectorStoreFilter,
    VectorStoreIntegrityError,
    VectorStoreRouteConfig,
)
from infrastructure.vector.adapters import vikingdb as protocol
from infrastructure.vector.adapters.vikingdb import VikingDBBackend
from infrastructure.vector.adapters.vikingdb_client import VikingDBRestClient
from infrastructure.vector.adapters.vikingdb_config import VikingDBVectorStoreConfig
from ModelClient import EmbeddingVector
from tests.unit.infrastructure.test_vikingdb_protocol import record, search_item


def backend(
    *,
    config: VikingDBVectorStoreConfig | None = None,
) -> VikingDBBackend:
    selected = config or VikingDBVectorStoreConfig(auth_mode="api_key")
    credentials = {
        "api_key": "secret",
    }
    if selected.auth_mode == "ak_sk":
        credentials = {"access_key": "ak", "secret_key": "sk"}
    elif selected.auth_mode == "private_headers":
        credentials = {"token": "secret"}
    route = VectorStoreRouteConfig(
        base_url="https://vector.example.com",
        max_retries=0,
        max_response_bytes=16 * 1024 * 1024,
    )
    client = VikingDBRestClient(route, selected, credentials=credentials)
    return VikingDBBackend("volcengine", "memory", selected, client)


def wrapped_record(value, *, scope: str = "default/memory") -> dict[str, object]:
    encoded = protocol._item_from_record(value, scope=scope)
    return {
        "id": encoded["id"],
        "fields": {key: item for key, item in encoded.items() if key != "id"},
    }


def collection_metadata(dimension: int = 2) -> dict[str, object]:
    fields = []
    for name, field_type in protocol._FIELD_TYPES.items():
        item: dict[str, object] = {"FieldName": name, "FieldType": field_type}
        if name == "id":
            item["IsPrimaryKey"] = True
        if name == "vector":
            item["Dim"] = dimension
        fields.append(item)
    return {"Fields": fields}


def index_metadata(**overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "IndexName": "default",
        "VectorIndex": {"Distance": "cosine"},
        "ScalarIndex": list(protocol._SCALAR_INDEX_FIELDS),
    }
    value.update(overrides)
    return value


@pytest.mark.parametrize("field", ["provider_name", "collection"])
@pytest.mark.parametrize("invalid", ["", None, 0, True, (), [], {}, object()])
def test_backend_requires_non_empty_provider_and_collection(
    field: str,
    invalid: object,
) -> None:
    current = backend()
    values = {
        "provider_name": "volcengine",
        "collection": "memory",
    }
    values[field] = invalid
    try:
        with pytest.raises((TypeError, ValueError)):
            VikingDBBackend(values["provider_name"], values["collection"], current.config, current._client)
    finally:
        asyncio.run(current.close())


@pytest.mark.parametrize("invalid", [None, "config", {}, [], 1, True, object()])
def test_backend_requires_vikingdb_config(invalid: object) -> None:
    current = backend()
    try:
        with pytest.raises(TypeError):
            VikingDBBackend("provider", "memory", invalid, current._client)
    finally:
        asyncio.run(current.close())


@pytest.mark.parametrize("invalid", [None, "client", {}, [], 1, True, object()])
def test_backend_requires_vikingdb_rest_client(invalid: object) -> None:
    config = VikingDBVectorStoreConfig(auth_mode="api_key")
    with pytest.raises(TypeError):
        VikingDBBackend("provider", "memory", config, invalid)


def test_backend_exposes_declared_capacity_scope_and_identity_payloads() -> None:
    current = backend(config=VikingDBVectorStoreConfig(max_records=123, max_search_hits=45))
    try:
        assert current.max_records == 123
        assert current.max_search_hits == 45
        assert current._scope == "default/memory"
        assert current._data_identity() == {"project": "default", "collection_name": "memory"}
        assert current._data_identity(include_index=True)["index_name"] == "default"
        assert current._console_identity() == {"ProjectName": "default", "CollectionName": "memory"}
        assert current._console_identity(include_index=True)["IndexName"] == "default"
    finally:
        asyncio.run(current.close())


def test_precreated_initialize_rejects_missing_collection() -> None:
    current = backend()

    async def missing():
        return None

    current._collection_metadata = missing
    try:
        with pytest.raises(VectorStoreIntegrityError, match="does not exist"):
            asyncio.run(current.initialize())
        assert not current._initialized
    finally:
        asyncio.run(current.close())


def test_managed_initialize_defers_missing_collection_creation_until_dimension_is_known() -> None:
    config = VikingDBVectorStoreConfig(auth_mode="ak_sk", schema_mode="managed")
    current = backend(config=config)

    async def missing():
        return None

    current._collection_metadata = missing
    try:
        asyncio.run(current.initialize())
        asyncio.run(current.initialize())
        assert current._initialized
    finally:
        asyncio.run(current.close())


def test_initialize_validates_existing_collection_and_index_once() -> None:
    current = backend()
    calls = {"collection": 0, "index": 0}

    async def metadata():
        calls["collection"] += 1
        return collection_metadata()

    async def require_index():
        calls["index"] += 1

    current._collection_metadata = metadata
    current._require_index = require_index
    try:
        asyncio.run(current.initialize())
        asyncio.run(current.initialize())
        assert calls == {"collection": 1, "index": 1}
    finally:
        asyncio.run(current.close())


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.update(Fields=None), "Fields schema"),
        (lambda value: value["Fields"].pop(), "missing fields"),
        (lambda value: value["Fields"][0].update(FieldType="int64"), "incompatible types"),
        (lambda value: value["Fields"][0].update(IsPrimaryKey=False), "primary key"),
        (
            lambda value: next(item for item in value["Fields"] if item["FieldName"] == "vector").update(Dim=3),
            "dimension",
        ),
    ],
    ids=["bad-fields", "missing-field", "wrong-type", "not-primary", "wrong-dimension"],
)
def test_collection_metadata_rejects_every_schema_incompatibility(mutation, message: str) -> None:
    current = backend()
    metadata = collection_metadata()
    mutation(metadata)
    try:
        with pytest.raises(VectorStoreIntegrityError, match=message):
            current._validate_collection_metadata(metadata, expected_dimension=2)
    finally:
        asyncio.run(current.close())


@pytest.mark.parametrize(
    "metadata",
    [
        {},
        collection_metadata(2),
    ],
)
def test_collection_metadata_accepts_api_key_trust_boundary_or_complete_schema(
    metadata: dict[str, object],
) -> None:
    current = backend()
    try:
        current._validate_collection_metadata(metadata, expected_dimension=2)
    finally:
        asyncio.run(current.close())


@pytest.mark.parametrize(
    ("metadata", "message"),
    [
        (None, "does not exist"),
        ({"IndexName": "other", "VectorIndex": {"Distance": "cosine"}, "ScalarIndex": list(protocol._SCALAR_INDEX_FIELDS)}, "another index"),
        ({"IndexName": "default", "VectorIndex": None, "ScalarIndex": []}, "no VectorIndex"),
        ({"IndexName": "default", "VectorIndex": {"Distance": "l2"}, "ScalarIndex": list(protocol._SCALAR_INDEX_FIELDS)}, "cosine"),
        ({"IndexName": "default", "VectorIndex": {"Distance": "cosine"}, "ScalarIndex": None}, "ScalarIndex"),
        ({"IndexName": "default", "VectorIndex": {"Distance": "cosine"}, "ScalarIndex": ["record_type"]}, "missing scalar fields"),
    ],
)
def test_require_index_rejects_missing_or_incompatible_metadata(
    metadata: object,
    message: str,
) -> None:
    current = backend()

    async def read_index():
        return metadata

    current._index_metadata = read_index
    try:
        with pytest.raises(VectorStoreIntegrityError, match=message):
            asyncio.run(current._require_index())
    finally:
        asyncio.run(current.close())


@pytest.mark.parametrize("metadata", [{}, index_metadata(), index_metadata(VectorIndex={"Distance": "cos"})])
def test_require_index_accepts_api_key_trust_boundary_and_cosine_alias(metadata: dict[str, object]) -> None:
    current = backend()

    async def read_index():
        return metadata

    current._index_metadata = read_index
    try:
        asyncio.run(current._require_index())
    finally:
        asyncio.run(current.close())


def test_read_empty_identity_sequence_skips_initialization_and_remote_fetch() -> None:
    current = backend()
    try:
        assert asyncio.run(current.read(())) == ()
        assert not current._initialized
    finally:
        asyncio.run(current.close())


@pytest.mark.parametrize("batch_size", [1, 2, 3])
def test_read_batches_remote_ids_and_restores_requested_order(batch_size: int) -> None:
    config = VikingDBVectorStoreConfig(fetch_batch_size=batch_size)
    current = backend(config=config)
    current._initialized = True
    values = tuple(record(f"memory://preferences/{name}.md") for name in ("a", "b", "c", "d"))
    by_point = {protocol._point_id(current._scope, item.identity): wrapped_record(item) for item in values}
    calls: list[tuple[str, ...]] = []

    async def fetch(ids: tuple[str, ...]):
        calls.append(ids)
        return tuple(by_point[point_id] for point_id in reversed(ids))

    current._fetch = fetch
    try:
        restored = asyncio.run(current.read(tuple(item.identity for item in values)))
        assert restored == values
        assert len(calls) == (len(values) + batch_size - 1) // batch_size
    finally:
        asyncio.run(current.close())


def test_read_omits_missing_remote_identity_without_reordering_found_values() -> None:
    current = backend()
    current._initialized = True
    first = record("a")
    third = record("c")

    async def fetch(_ids):
        return (wrapped_record(third), wrapped_record(first))

    current._fetch = fetch
    try:
        assert asyncio.run(current.read(("a", "b", "c"))) == (first, third)
    finally:
        asyncio.run(current.close())


@pytest.mark.parametrize("batch_size", [1, 2, 3])
def test_upsert_and_delete_apply_configured_batch_sizes(batch_size: int) -> None:
    config = VikingDBVectorStoreConfig(upsert_batch_size=batch_size, delete_batch_size=batch_size)
    current = backend(config=config)
    upserted: list[tuple[dict[str, object], ...]] = []
    deleted: list[tuple[str, ...]] = []

    async def upsert(items):
        upserted.append(items)

    async def data(path, body):
        if path.endswith("/delete"):
            deleted.append(tuple(body["ids"]))
        return {"result": {}}

    current._upsert = upsert
    current._client.data = data
    values = tuple(record(name) for name in ("a", "b", "c", "d"))
    try:
        asyncio.run(current._upsert_records(values))
        asyncio.run(current.delete(tuple(item.identity for item in values)))
        assert [len(items) for items in upserted] == [batch_size] * (len(values) // batch_size) + ([len(values) % batch_size] if len(values) % batch_size else [])
        assert [len(items) for items in deleted] == [batch_size] * (len(values) // batch_size) + ([len(values) % batch_size] if len(values) % batch_size else [])
        assert tuple(point_id for batch in deleted for point_id in batch) == tuple(
            protocol._point_id(current._scope, item.identity) for item in values
        )
    finally:
        asyncio.run(current.close())


def test_delete_all_and_empty_upsert_use_explicit_remote_protocol() -> None:
    current = backend()
    calls: list[tuple[str, dict[str, object]]] = []

    async def data(path, body):
        calls.append((path, body))
        return {"result": {}}

    current._client.data = data
    try:
        asyncio.run(current._upsert(()))
        assert calls == []
        asyncio.run(current.delete_all())
        assert calls[0][0].endswith("/delete")
        assert calls[0][1]["del_all"] is True
    finally:
        asyncio.run(current.close())


@pytest.mark.parametrize("complete", [True, False])
def test_wait_visible_routes_complete_and_incremental_publication(complete: bool) -> None:
    current = backend()
    calls: list[tuple[str, object, object]] = []

    async def full(upserts):
        calls.append(("full", upserts, ()))

    async def incremental(upserts, deletes):
        calls.append(("incremental", upserts, deletes))

    current._wait_full_index = full
    current._wait_incremental_index = incremental
    upserts = (record("a"),)
    deletes = () if complete else ("b",)
    try:
        asyncio.run(current.wait_visible(upserts, deletes, complete=complete))
        assert calls[0][0] == ("full" if complete else "incremental")
    finally:
        asyncio.run(current.close())


def test_complete_publication_rejects_explicit_deletes() -> None:
    current = backend()
    try:
        with pytest.raises(ValueError, match="cannot contain explicit deletes"):
            asyncio.run(current.wait_visible((record("a"),), ("b",), complete=True))
    finally:
        asyncio.run(current.close())


@pytest.mark.parametrize("score", [-10, -1, -0.5, 0, 0.5, 1, 10])
def test_search_paginates_deduplicates_and_clamps_finite_scores(score: float) -> None:
    config = VikingDBVectorStoreConfig(search_page_size=1)
    current = backend(config=config)
    current._initialized = True
    values = (record("a"), record("b"))
    pages = iter(
        [
            {"result": {"data": [search_item(values[0], score=score)]}},
            {"result": {"data": [search_item(values[1], score=score)]}},
        ]
    )

    async def data(_path, _body):
        return next(pages)

    current._client.data = data
    try:
        matches = asyncio.run(
            current.search(
                EmbeddingVector((1, 0)),
                filters=VectorStoreFilter({}, {}),
                limit=2,
            )
        )
        assert tuple(item.record for item in matches) == values
        assert all(item.score == max(-1.0, min(1.0, score)) for item in matches)
    finally:
        asyncio.run(current.close())


@pytest.mark.parametrize("score", [None, "1", True, False, [], {}, float("nan"), float("inf")])
def test_search_rejects_missing_non_numeric_or_non_finite_score(score: object) -> None:
    current = backend(config=VikingDBVectorStoreConfig(search_page_size=1))
    current._initialized = True
    item = search_item(record(), score=0.5)
    item["score"] = score

    async def data(_path, _body):
        return {"result": {"data": [item]}}

    current._client.data = data
    try:
        with pytest.raises(VectorStoreIntegrityError):
            asyncio.run(
                current.search(
                    EmbeddingVector((1, 0)),
                    filters=VectorStoreFilter({}, {}),
                    limit=1,
                )
            )
    finally:
        asyncio.run(current.close())


@pytest.mark.parametrize("failure", ["duplicate-id", "missing-order", "order-collision"])
def test_scalar_scan_rejects_unstable_pagination_or_scan_order(failure: str) -> None:
    current = backend(config=VikingDBVectorStoreConfig(scan_page_size=2))
    current._initialized = True
    first = wrapped_record(record("a"))
    second = wrapped_record(record("b"))
    if failure == "duplicate-id":
        second = {"id": first["id"], "fields": dict(second["fields"])}
    elif failure == "missing-order":
        second["fields"].pop("scan_order")
    else:
        second["fields"]["scan_order"] = first["fields"]["scan_order"]

    async def data(_path, _body):
        return {"result": {"data": [first, second]}}

    current._client.data = data
    try:
        with pytest.raises(VectorStoreIntegrityError):
            asyncio.run(current._scan_raw(filter_payload={}, limit=2, output_fields=("identity",)))
    finally:
        asyncio.run(current.close())


def test_scan_returns_records_in_stable_scalar_order() -> None:
    current = backend(config=VikingDBVectorStoreConfig(scan_page_size=2))
    current._initialized = True
    values = sorted((record("a"), record("b")), key=lambda item: protocol._scan_order(item.identity))
    items = [wrapped_record(item) for item in values]

    async def data(_path, _body):
        return {"result": {"data": items}}

    current._client.data = data
    try:
        assert asyncio.run(current.scan(filters=VectorStoreFilter({}, {}), limit=2)) == tuple(values)
    finally:
        asyncio.run(current.close())


def test_incremental_validation_rejects_remote_scan_order_collision() -> None:
    current = backend()
    current._initialized = True
    value = record("a")
    collision = wrapped_record(record("other"))
    collision["fields"]["scan_order"] = protocol._scan_order(value.identity)

    async def data(_path, _body):
        return {"result": {"data": [collision]}}

    current._client.data = data
    try:
        with pytest.raises(VectorStoreIntegrityError, match="collision"):
            asyncio.run(current.validate_records((value,), replacing=False))
        asyncio.run(current.validate_records((value,), replacing=True))
    finally:
        asyncio.run(current.close())


def test_ensure_collection_routes_create_recreate_or_validate_by_schema_mode() -> None:
    async def scenario() -> None:
        managed = backend(config=VikingDBVectorStoreConfig(auth_mode="ak_sk", schema_mode="managed"))
        calls: list[tuple[str, int | None]] = []

        async def missing():
            return None

        async def create(dimension):
            calls.append(("create", dimension))

        managed._collection_metadata = missing
        managed._create_collection = create
        await managed._ensure_collection(2, published_dimension=None)
        assert calls == [("create", 2)]
        await managed.close()

        precreated = backend()
        precreated._collection_metadata = missing
        with pytest.raises(VectorStoreIntegrityError, match="precreated"):
            await precreated._ensure_collection(2, published_dimension=None)
        await precreated.close()

    asyncio.run(scenario())


def test_dimension_change_recreates_only_managed_collection() -> None:
    async def scenario() -> None:
        managed = backend(config=VikingDBVectorStoreConfig(auth_mode="ak_sk", schema_mode="managed"))
        calls: list[str] = []

        async def metadata():
            return collection_metadata(2)

        async def drop():
            calls.append("drop")

        async def create(_dimension):
            calls.append("create")

        managed._collection_metadata = metadata
        managed._drop_collection = drop
        managed._create_collection = create
        await managed._ensure_collection(3, published_dimension=2)
        assert calls == ["drop", "create"]
        await managed.close()

        precreated = backend()
        precreated._collection_metadata = metadata
        with pytest.raises(VectorStoreIntegrityError, match="dimension changed"):
            await precreated._ensure_collection(3, published_dimension=2)
        await precreated.close()

    asyncio.run(scenario())
