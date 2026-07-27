"""以 VikingDB V2 协议实现厂商原始向量 Backend。"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import uuid
from collections.abc import Mapping, Sequence
from typing import TypeVar, cast

from infrastructure.vector.adapters.vikingdb_client import (
    VikingDBNotFoundError,
    VikingDBRestClient,
)
from infrastructure.vector.adapters.vikingdb_config import VikingDBVectorStoreConfig
from infrastructure.vector.factory import VectorStoreBuildContext, VectorStoreFactory
from infrastructure.vector.model import (
    VectorStoreBusyError,
    VectorStoreConflictError,
    VectorStoreFilter,
    VectorStoreIntegrityError,
    VectorStoreMatch,
    VectorStoreRecord,
    VectorValue,
)
from ModelClient import EmbeddingVector

_POINT_NAMESPACE = uuid.UUID("1bd3197e-6322-5ed2-b4c1-f583cf35fec5")
_RECORD_TYPE_MEMORY = "memory"
_RECORD_TYPE_METADATA = "m2bos_metadata"
_DESCRIPTION = "m2bOS rebuildable memory vector index"
_TEXT_FIELD_BYTE_LIMIT = 1024 * 1024
_BatchItem = TypeVar("_BatchItem")
_FILTERABLE_FIELDS = {
    "uri",
    "level",
    "directory_key",
    "parent_key",
    "scope_roots",
    "kind",
    "revision",
}
_OUTPUT_FIELDS = (
    "record_type",
    "identity",
    "identity_key",
    "scan_order",
    "vector",
    "content",
    "content_digest",
    "attributes_json",
    "metadata_json",
    "uri_key",
    "level",
    "directory_key_hash",
    "parent_key_hash",
    "scope_root_keys",
    "kind",
    "revision",
)
_SCALAR_INDEX_FIELDS = (
    "record_type",
    "identity_key",
    "scan_order",
    "uri_key",
    "level",
    "directory_key_hash",
    "parent_key_hash",
    "scope_root_keys",
    "kind",
    "revision",
)
_FIELD_TYPES = {
    "id": "string",
    "record_type": "string",
    "identity": "text",
    "identity_key": "string",
    "scan_order": "int64",
    "vector": "vector",
    "content": "text",
    "content_digest": "string",
    "attributes_json": "text",
    "metadata_json": "text",
    "uri_key": "string",
    "level": "int64",
    "directory_key_hash": "string",
    "parent_key_hash": "string",
    "scope_root_keys": "list<string>",
    "kind": "string",
    "revision": "int64",
}


class VikingDBBackend:
    """只实现 VikingDB 物理协议、Schema、编解码和索引可见性。"""

    adapter_name = "vikingdb"

    def __init__(
        self,
        provider_name: str,
        collection: str,
        config: VikingDBVectorStoreConfig,
        client: VikingDBRestClient,
    ) -> None:
        if not isinstance(provider_name, str) or not provider_name:
            raise ValueError("vikingdb provider_name must be non-empty")
        if not isinstance(collection, str) or not collection:
            raise ValueError("vikingdb collection must be non-empty")
        if not isinstance(config, VikingDBVectorStoreConfig):
            raise TypeError("config must be VikingDBVectorStoreConfig")
        if not isinstance(client, VikingDBRestClient):
            raise TypeError("client must be VikingDBRestClient")
        self.provider_name = provider_name
        self.collection = collection
        self.config = config
        self._client = client
        self._initialize_lock = asyncio.Lock()
        self._initialized = False

    @property
    def max_records(self) -> int:
        return self.config.max_records

    @property
    def max_search_hits(self) -> int:
        return self.config.max_search_hits

    async def initialize(self) -> None:
        if self._initialized:
            return
        async with self._initialize_lock:
            if self._initialized:
                return
            metadata = await self._collection_metadata()
            if metadata is None:
                if self.config.schema_mode != "managed":
                    raise VectorStoreIntegrityError(
                        "configured VikingDB collection does not exist; precreate its schema and index"
                    )
                self._initialized = True
                return
            self._validate_collection_metadata(metadata, expected_dimension=None)
            await self._require_index()
            self._initialized = True

    async def read_metadata(
        self,
        names: tuple[str, ...],
    ) -> Mapping[str, Mapping[str, object]]:
        await self.initialize()
        if await self._collection_metadata() is None:
            return {}
        normalized = _metadata_names(names)
        items = await self._fetch(tuple(_metadata_id(self._scope, name) for name in normalized))
        values: dict[str, Mapping[str, object]] = {}
        by_id = {str(item.get("id")): item for item in items}
        for name in normalized:
            item = by_id.get(_metadata_id(self._scope, name))
            if item is not None:
                values[name] = _metadata_from_item(item, scope=self._scope, name=name)
        return values

    async def write_metadata(
        self,
        values: Mapping[str, Mapping[str, object]],
        *,
        dimension: int,
    ) -> None:
        if not isinstance(values, Mapping):
            raise TypeError("VikingDB metadata values must be an object")
        names = _metadata_names(tuple(values))
        await self._upsert(
            tuple(
                _metadata_item(
                    point_id=_metadata_id(self._scope, name),
                    identity=f"metadata:{self._scope}:{name}",
                    name=name,
                    scope=self._scope,
                    value=values[name],
                    dimension=dimension,
                )
                for name in names
            )
        )

    async def ensure_schema(
        self,
        dimension: int,
        *,
        published_dimension: int | None,
    ) -> None:
        await self.initialize()
        await self._ensure_collection(dimension, published_dimension=published_dimension)

    async def read(self, identities: tuple[str, ...]) -> tuple[VectorStoreRecord, ...]:
        if not identities:
            return ()
        await self.initialize()
        values: dict[str, VectorStoreRecord] = {}
        for batch in _batches(identities, self.config.fetch_batch_size):
            for item in await self._fetch(tuple(_point_id(self._scope, identity) for identity in batch)):
                record = _record_from_item(item, scope=self._scope)
                values[record.identity] = record
        return tuple(values[identity] for identity in identities if identity in values)

    async def delete_all(self) -> None:
        await self._delete_all_data()

    async def upsert(self, records: tuple[VectorStoreRecord, ...]) -> None:
        await self._upsert_records(records)

    async def delete(self, identities: tuple[str, ...]) -> None:
        await self._delete_ids(tuple(_point_id(self._scope, identity) for identity in identities))

    async def validate_records(
        self,
        records: tuple[VectorStoreRecord, ...],
        *,
        replacing: bool,
    ) -> None:
        _validate_scan_order_batch(records)
        if not replacing:
            await self._assert_incremental_scan_orders(records)

    async def wait_visible(
        self,
        upserts: tuple[VectorStoreRecord, ...],
        deletes: tuple[str, ...],
        *,
        complete: bool,
    ) -> None:
        if complete:
            if deletes:
                raise ValueError("a complete VikingDB publication cannot contain explicit deletes")
            await self._wait_full_index(upserts)
            return
        await self._wait_incremental_index(upserts, deletes)

    async def search(
        self,
        query_vector: EmbeddingVector,
        *,
        filters: VectorStoreFilter,
        limit: int,
    ) -> tuple[VectorStoreMatch, ...]:
        await self.initialize()
        matches: list[VectorStoreMatch] = []
        seen: set[str] = set()
        offset = 0
        while len(matches) < limit:
            page_size = min(self.config.search_page_size, limit - len(matches))
            result = await self._client.data(
                "/api/vikingdb/data/search/vector",
                {
                    **self._data_identity(include_index=True),
                    "dense_vector": list(query_vector.values),
                    "filter": _compile_filter(filters),
                    "output_fields": list(_OUTPUT_FIELDS),
                    "limit": page_size,
                    "offset": offset,
                },
            )
            page = _search_items(result)
            if not page:
                break
            for item in page:
                point_id = item.get("id")
                if not isinstance(point_id, str) or not point_id or point_id in seen:
                    raise VectorStoreIntegrityError("VikingDB vector pagination is unstable")
                seen.add(point_id)
                raw_score = item.get("score")
                if isinstance(raw_score, bool) or not isinstance(raw_score, int | float):
                    raise VectorStoreIntegrityError("VikingDB search result has no numeric score")
                score = float(raw_score)
                if not math.isfinite(score):
                    raise VectorStoreIntegrityError("VikingDB search result score is not finite")
                matches.append(
                    VectorStoreMatch(
                        record=_record_from_item(item, scope=self._scope),
                        score=max(-1.0, min(1.0, score)),
                    )
                )
            offset += len(page)
            if len(page) < page_size:
                break
        return tuple(matches)

    async def scan(
        self,
        *,
        filters: VectorStoreFilter,
        limit: int,
    ) -> tuple[VectorStoreRecord, ...]:
        await self.initialize()
        items = await self._scan_items(filters=filters, limit=limit)
        return tuple(_record_from_item(item, scope=self._scope) for item in items)

    async def close(self) -> None:
        await self._client.close()

    @property
    def _scope(self) -> str:
        return f"{self.config.project_name}/{self.collection}"

    def _data_identity(self, *, include_index: bool = False) -> dict[str, object]:
        value: dict[str, object] = {
            "project": self.config.project_name,
            "collection_name": self.collection,
        }
        if include_index:
            value["index_name"] = self.config.index_name
        return value

    def _console_identity(self, *, include_index: bool = False) -> dict[str, object]:
        value: dict[str, object] = {
            "ProjectName": self.config.project_name,
            "CollectionName": self.collection,
        }
        if include_index:
            value["IndexName"] = self.config.index_name
        return value

    async def _collection_metadata(self) -> Mapping[str, object] | None:
        if self.config.auth_mode == "api_key":
            try:
                await self._fetch((_metadata_id(self._scope, "state"), _metadata_id(self._scope, "claim")))
            except VikingDBNotFoundError:
                return None
            return {}
        try:
            if self.config.auth_mode == "ak_sk":
                response = await self._client.public_console(
                    "GetVikingdbCollection",
                    self._console_identity(),
                )
            else:
                response = await self._client.private_console(
                    "/api/vikingdb/GetCollection",
                    self._console_identity(),
                )
        except VikingDBNotFoundError:
            return None
        result = _console_result(response)
        return result or None

    async def _index_metadata(self) -> Mapping[str, object] | None:
        if self.config.auth_mode == "api_key":
            return {}
        try:
            if self.config.auth_mode == "ak_sk":
                response = await self._client.public_console(
                    "GetVikingdbIndex",
                    self._console_identity(include_index=True),
                )
            else:
                response = await self._client.private_console(
                    "/api/vikingdb/GetIndex",
                    self._console_identity(include_index=True),
                )
        except VikingDBNotFoundError:
            return None
        result = _console_result(response)
        return result or None

    async def _require_index(self) -> None:
        metadata = await self._index_metadata()
        if metadata is None:
            raise VectorStoreIntegrityError("configured VikingDB index does not exist")
        if not metadata:
            # API Key 数据面无法读取 Index 元数据，只能由预创建配置承担 Schema 信任边界。
            return
        name = metadata.get("IndexName")
        if name is not None and name != self.config.index_name:
            raise VectorStoreIntegrityError("VikingDB index metadata identifies another index")
        vector = metadata.get("VectorIndex")
        if not isinstance(vector, Mapping):
            raise VectorStoreIntegrityError("VikingDB index metadata has no VectorIndex")
        distance = vector.get("Distance")
        if str(distance).casefold() not in {"cosine", "cos"}:
            raise VectorStoreIntegrityError("VikingDB index must use cosine distance")
        raw_scalar = metadata.get("ScalarIndex")
        if not isinstance(raw_scalar, list) or any(not isinstance(item, str) for item in raw_scalar):
            raise VectorStoreIntegrityError("VikingDB ScalarIndex metadata is invalid")
        missing = sorted(set(_SCALAR_INDEX_FIELDS) - set(cast(list[str], raw_scalar)))
        if missing:
            raise VectorStoreIntegrityError(f"VikingDB index is missing scalar fields: {missing}")

    async def _ensure_collection(
        self,
        dimension: int,
        *,
        published_dimension: int | None,
    ) -> None:
        metadata = await self._collection_metadata()
        if metadata is None:
            if self.config.schema_mode != "managed":
                raise VectorStoreIntegrityError("precreated VikingDB collection is missing")
            await self._create_collection(dimension)
            return
        if published_dimension is not None and published_dimension != dimension:
            if self.config.schema_mode != "managed":
                raise VectorStoreIntegrityError(
                    "precreated VikingDB vector dimension changed; recreate the collection externally"
                )
            await self._drop_collection()
            await self._create_collection(dimension)
            return
        self._validate_collection_metadata(metadata, expected_dimension=dimension)
        await self._require_index()

    async def _create_collection(self, dimension: int) -> None:
        if self.config.schema_mode != "managed" or self.config.auth_mode != "ak_sk":
            raise VectorStoreIntegrityError("this VikingDB route cannot manage collection schema")
        fields: list[dict[str, object]] = [
            {"FieldName": "id", "FieldType": "string", "IsPrimaryKey": True},
            {"FieldName": "record_type", "FieldType": "string"},
            {"FieldName": "identity", "FieldType": "text"},
            {"FieldName": "identity_key", "FieldType": "string"},
            {"FieldName": "scan_order", "FieldType": "int64"},
            {"FieldName": "vector", "FieldType": "vector", "Dim": dimension},
            {"FieldName": "content", "FieldType": "text"},
            {"FieldName": "content_digest", "FieldType": "string"},
            {"FieldName": "attributes_json", "FieldType": "text"},
            {"FieldName": "metadata_json", "FieldType": "text"},
            {"FieldName": "uri_key", "FieldType": "string"},
            {"FieldName": "level", "FieldType": "int64"},
            {"FieldName": "directory_key_hash", "FieldType": "string"},
            {"FieldName": "parent_key_hash", "FieldType": "string"},
            {"FieldName": "scope_root_keys", "FieldType": "list<string>"},
            {"FieldName": "kind", "FieldType": "string"},
            {"FieldName": "revision", "FieldType": "int64"},
        ]
        body = {
            **self._console_identity(),
            "Description": _DESCRIPTION,
            "PrimaryKey": "id",
            "Fields": fields,
        }
        try:
            await self._client.public_console("CreateVikingdbCollection", body)
        except VectorStoreConflictError:
            metadata = await self._collection_metadata()
            if metadata is None:
                raise
            self._validate_collection_metadata(metadata, expected_dimension=dimension)
        await self._wait_for_schema(dimension=dimension, require_index=False)
        index_body = {
            **self._console_identity(include_index=True),
            "VectorIndex": {
                "IndexType": "hnsw",
                "Distance": "cosine",
                "Quant": "int8",
            },
            "ScalarIndex": list(_SCALAR_INDEX_FIELDS),
        }
        try:
            await self._client.public_console("CreateVikingdbIndex", index_body)
        except VectorStoreConflictError:
            pass
        await self._wait_for_schema(dimension=dimension, require_index=True)

    async def _wait_for_schema(self, *, dimension: int, require_index: bool) -> None:
        for attempt in range(self._client.route.max_retries + 1):
            metadata = await self._collection_metadata()
            index_metadata = await self._index_metadata() if require_index else {}
            if metadata is not None and (not require_index or index_metadata is not None):
                self._validate_collection_metadata(metadata, expected_dimension=dimension)
                if require_index:
                    await self._require_index()
                return
            if attempt < self._client.route.max_retries:
                await asyncio.sleep(
                    min(
                        self._client.route.retry_base_delay_seconds * (2**attempt),
                        self._client.route.retry_max_delay_seconds,
                    )
                )
        target = "collection and index" if require_index else "collection"
        raise VectorStoreBusyError(f"created VikingDB {target} did not become readable")

    async def _drop_collection(self) -> None:
        try:
            await self._client.public_console(
                "DeleteVikingdbCollection",
                self._console_identity(),
            )
        except VikingDBNotFoundError:
            return
        for attempt in range(self._client.route.max_retries + 1):
            if await self._collection_metadata() is None:
                return
            if attempt < self._client.route.max_retries:
                await asyncio.sleep(
                    min(
                        self._client.route.retry_base_delay_seconds * (2**attempt),
                        self._client.route.retry_max_delay_seconds,
                    )
                )
        raise VectorStoreBusyError("deleted VikingDB collection is still visible")

    def _validate_collection_metadata(
        self,
        metadata: Mapping[str, object],
        *,
        expected_dimension: int | None,
    ) -> None:
        if not metadata:
            return
        raw_fields = metadata.get("Fields")
        if not isinstance(raw_fields, list) or any(not isinstance(item, Mapping) for item in raw_fields):
            raise VectorStoreIntegrityError("VikingDB collection metadata has no valid Fields schema")
        fields = {
            str(item.get("FieldName")): item
            for item in cast(list[Mapping[str, object]], raw_fields)
        }
        required = set(_FIELD_TYPES)
        missing = sorted(required - set(fields))
        if missing:
            raise VectorStoreIntegrityError(f"VikingDB collection schema is missing fields: {missing}")
        incompatible = sorted(
            name
            for name, expected_type in _FIELD_TYPES.items()
            if fields[name].get("FieldType") != expected_type
        )
        if incompatible:
            raise VectorStoreIntegrityError(
                f"VikingDB collection fields have incompatible types: {incompatible}"
            )
        primary = fields["id"]
        if primary.get("IsPrimaryKey") is not True:
            raise VectorStoreIntegrityError("VikingDB id field must be the string primary key")
        vector = fields["vector"]
        if expected_dimension is not None and vector.get("Dim") != expected_dimension:
            raise VectorStoreIntegrityError("VikingDB vector dimension does not match configuration")

    async def _fetch(self, ids: tuple[str, ...]) -> tuple[Mapping[str, object], ...]:
        if not ids:
            return ()
        response = await self._client.data(
            "/api/vikingdb/data/fetch_in_collection",
            {
                **self._data_identity(),
                "ids": list(ids),
            },
        )
        result = _data_result(response)
        raw = result.get("fetch", [])
        if not isinstance(raw, list) or any(not isinstance(item, Mapping) for item in raw):
            raise VectorStoreIntegrityError("VikingDB fetch response has an invalid shape")
        items = tuple(cast(list[Mapping[str, object]], raw))
        _validate_fetched_ids(items, allowed=set(ids), label="collection")
        return items

    async def _upsert_records(self, records: Sequence[VectorStoreRecord]) -> None:
        for batch in _batches(records, self.config.upsert_batch_size):
            await self._upsert(tuple(_item_from_record(record, scope=self._scope) for record in batch))

    async def _upsert(self, items: tuple[Mapping[str, object], ...]) -> None:
        if not items:
            return
        await self._client.data(
            "/api/vikingdb/data/upsert",
            {
                **self._data_identity(),
                "data": [dict(item) for item in items],
                "ttl": 0,
            },
        )

    async def _delete_ids(self, ids: tuple[str, ...]) -> None:
        for batch in _batches(ids, self.config.delete_batch_size):
            await self._client.data(
                "/api/vikingdb/data/delete",
                {
                    **self._data_identity(),
                    "ids": list(batch),
                },
            )

    async def _delete_all_data(self) -> None:
        await self._client.data(
            "/api/vikingdb/data/delete",
            {
                **self._data_identity(),
                "del_all": True,
            },
        )

    async def _list_memory_digests(self, *, limit: int) -> dict[str, str]:
        items = await self._scan_raw(
            filter_payload=_record_type_filter(),
            limit=limit,
            output_fields=("content_digest",),
        )
        values: dict[str, str] = {}
        for item in items:
            point_id = item.get("id")
            fields = item.get("fields")
            digest = fields.get("content_digest") if isinstance(fields, Mapping) else None
            if not isinstance(point_id, str) or not isinstance(digest, str):
                raise VectorStoreIntegrityError("VikingDB index digest scan is invalid")
            values[point_id] = digest
        return values

    async def _scan_items(
        self,
        *,
        filters: VectorStoreFilter,
        limit: int,
    ) -> tuple[Mapping[str, object], ...]:
        return await self._scan_raw(
            filter_payload=_compile_filter(filters),
            limit=limit,
            output_fields=_OUTPUT_FIELDS,
        )

    async def _scan_raw(
        self,
        *,
        filter_payload: Mapping[str, object],
        limit: int,
        output_fields: Sequence[str],
    ) -> tuple[Mapping[str, object], ...]:
        items: list[Mapping[str, object]] = []
        seen: set[str] = set()
        seen_orders: dict[int, str] = {}
        offset = 0
        while len(items) < limit:
            page_size = min(self.config.scan_page_size, limit - len(items))
            response = await self._client.data(
                "/api/vikingdb/data/search/scalar",
                {
                    **self._data_identity(include_index=True),
                    "field": "scan_order",
                    "order": "asc",
                    "filter": dict(filter_payload),
                    "output_fields": list(dict.fromkeys((*output_fields, "scan_order"))),
                    "limit": page_size,
                    "offset": offset,
                },
            )
            page = _search_items(response)
            if not page:
                break
            for item in page:
                point_id = item.get("id")
                if not isinstance(point_id, str) or not point_id or point_id in seen:
                    raise VectorStoreIntegrityError("VikingDB scalar pagination is unstable")
                fields = item.get("fields")
                order = fields.get("scan_order") if isinstance(fields, Mapping) else None
                if isinstance(order, bool) or not isinstance(order, int):
                    raise VectorStoreIntegrityError("VikingDB scalar result lost its scan_order")
                collision = seen_orders.get(order)
                if collision is not None and collision != point_id:
                    raise VectorStoreIntegrityError("VikingDB scan_order collision was detected")
                seen.add(point_id)
                seen_orders[order] = point_id
                items.append(item)
            offset += len(page)
            if len(page) < page_size:
                break
        return tuple(items[:limit])

    async def _wait_full_index(self, records: Sequence[VectorStoreRecord]) -> None:
        expected = {
            _point_id(self._scope, record.identity): record.content_digest
            for record in records
        }
        deadline = asyncio.get_running_loop().time() + self.config.index_sync_timeout_seconds
        while True:
            actual = await self._list_memory_digests(limit=self.config.max_records + 1)
            if len(actual) > self.config.max_records:
                raise VectorStoreIntegrityError("VikingDB index exceeds the configured record capacity")
            if actual == expected:
                return
            if asyncio.get_running_loop().time() >= deadline:
                raise VectorStoreBusyError("VikingDB index did not converge after full replacement")
            await asyncio.sleep(self.config.index_sync_poll_interval_seconds)

    async def _wait_incremental_index(
        self,
        upserts: Sequence[VectorStoreRecord],
        deletes: Sequence[str],
    ) -> None:
        expected = {
            _point_id(self._scope, record.identity): record.content_digest
            for record in upserts
        }
        deleted_ids = {_point_id(self._scope, identity) for identity in deletes}
        changed_ids = tuple((*expected, *sorted(deleted_ids)))
        if not changed_ids:
            return
        deadline = asyncio.get_running_loop().time() + self.config.index_sync_timeout_seconds
        while True:
            indexed = await self._fetch_index(changed_ids)
            digests: dict[str, str] = {}
            for item in indexed:
                point_id = item.get("id")
                fields = item.get("fields")
                digest = fields.get("content_digest") if isinstance(fields, Mapping) else None
                if not isinstance(point_id, str) or not isinstance(digest, str):
                    raise VectorStoreIntegrityError("VikingDB index fetch returned invalid fields")
                digests[point_id] = digest
            if all(digests.get(point_id) == digest for point_id, digest in expected.items()) and not (
                set(digests) & deleted_ids
            ):
                return
            if asyncio.get_running_loop().time() >= deadline:
                raise VectorStoreBusyError("VikingDB index did not converge after incremental update")
            await asyncio.sleep(self.config.index_sync_poll_interval_seconds)

    async def _assert_incremental_scan_orders(
        self,
        upserts: Sequence[VectorStoreRecord],
    ) -> None:
        if not upserts:
            return
        for batch in _batches(upserts, 4999):
            expected = {
                _scan_order(record.identity): _point_id(self._scope, record.identity)
                for record in batch
            }
            response = await self._client.data(
                "/api/vikingdb/data/search/scalar",
                {
                    **self._data_identity(include_index=True),
                    "field": "scan_order",
                    "order": "asc",
                    "filter": {
                        "op": "and",
                        "conds": [
                            _record_type_filter(),
                            {
                                "op": "must",
                                "field": "scan_order",
                                "conds": list(expected),
                            },
                        ],
                    },
                    "output_fields": ["scan_order"],
                    "limit": len(expected) + 1,
                    "offset": 0,
                },
            )
            for item in _search_items(response):
                point_id = item.get("id")
                fields = item.get("fields")
                order = fields.get("scan_order") if isinstance(fields, Mapping) else None
                if not isinstance(point_id, str) or isinstance(order, bool) or not isinstance(order, int):
                    raise VectorStoreIntegrityError("VikingDB scan-order collision query is invalid")
                allowed_id = expected.get(order)
                if allowed_id is None or allowed_id != point_id:
                    raise VectorStoreIntegrityError("VikingDB scan_order collision was detected")

    async def _fetch_index(self, ids: tuple[str, ...]) -> tuple[Mapping[str, object], ...]:
        items: list[Mapping[str, object]] = []
        allowed = set(ids)
        seen: set[str] = set()
        for batch in _batches(ids, self.config.fetch_batch_size):
            response = await self._client.data(
                "/api/vikingdb/data/fetch_in_index",
                {
                    **self._data_identity(include_index=True),
                    "ids": list(batch),
                    "output_fields": ["content_digest"],
                },
            )
            result = _data_result(response)
            raw = result.get("fetch", result.get("items", []))
            if not isinstance(raw, list) or any(not isinstance(item, Mapping) for item in raw):
                raise VectorStoreIntegrityError("VikingDB index fetch response has an invalid shape")
            page = cast(list[Mapping[str, object]], raw)
            _validate_fetched_ids(page, allowed=allowed, seen=seen, label="index")
            items.extend(page)
        return tuple(items)

def build_vikingdb_backend(context: VectorStoreBuildContext) -> VikingDBBackend:
    config = VikingDBVectorStoreConfig.from_mapping(context.config.options)
    config.validate_requirements(context.requirements, context.config.route)
    client = VikingDBRestClient(
        context.config.route,
        config,
        credentials=context.credentials,
    )
    return VikingDBBackend(
        context.config.provider,
        context.config.collection,
        config,
        client,
    )


def register_builtin_vector_adapters(factory: VectorStoreFactory | None = None) -> VectorStoreFactory:
    target = factory or VectorStoreFactory()
    target.register_adapter("vikingdb", build_vikingdb_backend)
    return target


def _point_id(scope: str, identity: str) -> str:
    return str(uuid.uuid5(_POINT_NAMESPACE, f"memory:{scope}:{identity}"))


def _metadata_id(scope: str, name: str) -> str:
    return str(uuid.uuid5(_POINT_NAMESPACE, f"metadata:{scope}:{name}"))


def _metadata_names(names: Sequence[str]) -> tuple[str, ...]:
    if isinstance(names, str | bytes) or not isinstance(names, Sequence):
        raise TypeError("VikingDB metadata names must be a sequence")
    result: list[str] = []
    seen: set[str] = set()
    for name in names:
        if (
            not isinstance(name, str)
            or not name
            or name != name.strip()
            or len(name) > 64
            or any(not (character.isascii() and (character.isalnum() or character in "_-")) for character in name)
        ):
            raise ValueError("VikingDB metadata name is invalid")
        if name in seen:
            raise ValueError("VikingDB metadata names must be unique")
        seen.add(name)
        result.append(name)
    return tuple(result)


def _item_from_record(record: VectorStoreRecord, *, scope: str) -> dict[str, object]:
    attributes = {
        name: list(value) if isinstance(value, tuple) else value
        for name, value in record.attributes.items()
    }
    return {
        "id": _point_id(scope, record.identity),
        "record_type": _RECORD_TYPE_MEMORY,
        "identity": _bounded_text(record.identity, "identity"),
        "identity_key": _index_key(record.identity),
        "scan_order": _scan_order(record.identity),
        "vector": list(record.vector.values),
        "content": _bounded_text(record.content, "content"),
        "content_digest": record.content_digest,
        "attributes_json": _json_text(attributes, "attributes_json"),
        "metadata_json": "{}",
        **_physical_index_fields(attributes),
    }


def _record_from_item(item: Mapping[str, object], *, scope: str) -> VectorStoreRecord:
    fields = item.get("fields")
    if not isinstance(fields, Mapping):
        raise VectorStoreIntegrityError("VikingDB record is missing its fields object")
    if fields.get("record_type") != _RECORD_TYPE_MEMORY:
        raise VectorStoreIntegrityError("VikingDB result is not an m2bOS memory record")
    raw_attributes = fields.get("attributes_json")
    if not isinstance(raw_attributes, str):
        raise VectorStoreIntegrityError("VikingDB record has no attributes_json")
    try:
        decoded_attributes = json.loads(raw_attributes)
    except json.JSONDecodeError as exc:
        raise VectorStoreIntegrityError("VikingDB attributes_json is invalid") from exc
    if not isinstance(decoded_attributes, dict):
        raise VectorStoreIntegrityError("VikingDB attributes_json root must be an object")
    identity = fields.get("identity")
    if (
        not isinstance(identity, str)
        or item.get("id") != _point_id(scope, identity)
        or fields.get("identity_key") != _index_key(identity)
        or fields.get("scan_order") != _scan_order(identity)
    ):
        raise VectorStoreIntegrityError("VikingDB point identity or hash is invalid")
    expected_index_fields = _physical_index_fields(decoded_attributes)
    for field_name, expected in expected_index_fields.items():
        if fields.get(field_name) != expected:
            raise VectorStoreIntegrityError(
                f"VikingDB indexed field does not match attributes_json: {field_name}"
            )
    raw_vector = fields.get("vector")
    if not isinstance(raw_vector, Sequence) or isinstance(raw_vector, str | bytes):
        raise VectorStoreIntegrityError("VikingDB record is missing its dense vector")
    try:
        return VectorStoreRecord(
            identity=identity,
            vector=EmbeddingVector(tuple(cast(Sequence[float], raw_vector))),
            content=cast(str, fields.get("content")),
            content_digest=cast(str, fields.get("content_digest")),
            attributes=cast(Mapping[str, VectorValue], decoded_attributes),
        )
    except (TypeError, ValueError) as exc:
        raise VectorStoreIntegrityError("VikingDB memory record fields are invalid") from exc


def _metadata_item(
    *,
    point_id: str,
    identity: str,
    name: str,
    scope: str,
    value: Mapping[str, object],
    dimension: int,
) -> dict[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError("VikingDB metadata value must be a string-keyed object")
    text = _json_text(
        {
            "name": name,
            "scope": scope,
            "value": dict(value),
        },
        "metadata_json",
    )
    empty_digest = hashlib.sha256(b"{}").hexdigest()
    empty_attributes: dict[str, object] = {}
    return {
        "id": point_id,
        "record_type": _RECORD_TYPE_METADATA,
        "identity": _bounded_text(identity, "metadata identity"),
        "identity_key": _index_key(identity),
        "scan_order": _scan_order(identity),
        "vector": _sentinel_vector(dimension),
        "content": "{}",
        "content_digest": empty_digest,
        "attributes_json": "{}",
        "metadata_json": text,
        **_physical_index_fields(empty_attributes),
    }


def _metadata_from_item(
    item: Mapping[str, object],
    *,
    scope: str,
    name: str,
) -> Mapping[str, object]:
    fields = item.get("fields")
    expected_id = _metadata_id(scope, name)
    expected_identity = f"metadata:{scope}:{name}"
    empty_digest = hashlib.sha256(b"{}").hexdigest()
    if (
        item.get("id") != expected_id
        or not isinstance(fields, Mapping)
        or fields.get("record_type") != _RECORD_TYPE_METADATA
        or fields.get("identity") != expected_identity
        or fields.get("identity_key") != _index_key(expected_identity)
        or fields.get("scan_order") != _scan_order(expected_identity)
        or fields.get("content") != "{}"
        or fields.get("content_digest") != empty_digest
        or fields.get("attributes_json") != "{}"
    ):
        raise VectorStoreIntegrityError("VikingDB metadata record identity is invalid")
    for field_name, expected in _physical_index_fields({}).items():
        if fields.get(field_name) != expected:
            raise VectorStoreIntegrityError("VikingDB metadata index fields are invalid")
    raw = fields.get("metadata_json")
    if not isinstance(raw, str):
        raise VectorStoreIntegrityError("VikingDB metadata record has no JSON payload")
    try:
        metadata = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise VectorStoreIntegrityError("VikingDB metadata JSON is invalid") from exc
    if not isinstance(metadata, dict) or set(metadata) != {"name", "scope", "value"}:
        raise VectorStoreIntegrityError("VikingDB metadata envelope is invalid")
    value = metadata.get("value")
    if metadata.get("name") != name or metadata.get("scope") != scope or not isinstance(value, dict):
        raise VectorStoreIntegrityError("VikingDB metadata ownership is invalid")
    if any(not isinstance(key, str) for key in value):
        raise VectorStoreIntegrityError("VikingDB metadata value contains a non-string key")
    return cast(Mapping[str, object], value)


def _sentinel_vector(dimension: int) -> list[float]:
    _positive_int(dimension, "dimension")
    return [1.0, *([0.0] * (dimension - 1))]


def _data_result(response: Mapping[str, object]) -> Mapping[str, object]:
    result = response.get("result")
    if result is None:
        return {}
    if not isinstance(result, Mapping):
        raise VectorStoreIntegrityError("VikingDB data response result must be an object")
    return result


def _console_result(response: Mapping[str, object]) -> Mapping[str, object]:
    result = response.get("Result", response.get("data"))
    if result is None:
        return {}
    if not isinstance(result, Mapping):
        raise VectorStoreIntegrityError("VikingDB console result must be an object")
    return result


def _search_items(response: Mapping[str, object]) -> tuple[Mapping[str, object], ...]:
    raw = _data_result(response).get("data", [])
    if not isinstance(raw, list) or any(not isinstance(item, Mapping) for item in raw):
        raise VectorStoreIntegrityError("VikingDB search response has an invalid shape")
    return tuple(cast(list[Mapping[str, object]], raw))


def _validate_fetched_ids(
    items: Sequence[Mapping[str, object]],
    *,
    allowed: set[str],
    label: str,
    seen: set[str] | None = None,
) -> None:
    """拒绝远端 Fetch 返回未请求、重复或非字符串主键。"""

    observed = set() if seen is None else seen
    for item in items:
        point_id = item.get("id")
        if not isinstance(point_id, str) or point_id not in allowed or point_id in observed:
            raise VectorStoreIntegrityError(f"VikingDB {label} fetch returned an invalid point id")
        observed.add(point_id)


def _compile_filter(filters: VectorStoreFilter) -> dict[str, object]:
    clauses: list[dict[str, object]] = [
        {"op": "must", "field": "record_type", "conds": [_RECORD_TYPE_MEMORY]}
    ]
    for field, value in filters.equals.items():
        _filter_field(field)
        physical_field, physical_values = _physical_filter(field, (value,))
        clauses.append({"op": "must", "field": physical_field, "conds": list(physical_values)})
    for field, values in filters.one_of.items():
        _filter_field(field)
        physical_field, physical_values = _physical_filter(field, values)
        clauses.append({"op": "must", "field": physical_field, "conds": list(physical_values)})
    if len(clauses) == 1:
        return clauses[0]
    return {"op": "and", "conds": clauses}


def _record_type_filter() -> dict[str, object]:
    return {"op": "must", "field": "record_type", "conds": [_RECORD_TYPE_MEMORY]}


def _filter_field(field: str) -> None:
    if field not in _FILTERABLE_FIELDS:
        raise ValueError(f"VikingDB filter field has no declared scalar index: {field}")


def _physical_index_fields(attributes: Mapping[str, object]) -> dict[str, object]:
    """把完整逻辑属性转换成不超过 VikingDB string 上限的索引键。"""

    uri = _optional_text_attribute(attributes, "uri")
    directory = _optional_text_attribute(attributes, "directory_key")
    parent = _optional_text_attribute(attributes, "parent_key")
    roots_value = attributes.get("scope_roots", [])
    if not isinstance(roots_value, list | tuple) or any(not isinstance(item, str) for item in roots_value):
        raise ValueError("vector scope_roots must contain strings")
    roots = tuple(cast(Sequence[str], roots_value))
    level = attributes.get("level", -1)
    revision = attributes.get("revision", 0)
    kind = attributes.get("kind", "system")
    if isinstance(level, bool) or not isinstance(level, int):
        raise ValueError("vector level must be an integer")
    if isinstance(revision, bool) or not isinstance(revision, int):
        raise ValueError("vector revision must be an integer")
    if not isinstance(kind, str) or len(kind.encode("utf-8")) > 256:
        raise ValueError("vector kind must be a string of at most 256 UTF-8 bytes")
    return {
        "uri_key": _index_key(uri),
        "level": level,
        "directory_key_hash": _index_key(directory),
        "parent_key_hash": _index_key(parent),
        "scope_root_keys": [_index_key(root) for root in roots] or [_index_key("")],
        "kind": kind,
        "revision": revision,
    }


def _physical_filter(
    field: str,
    values: Sequence[object],
) -> tuple[str, tuple[object, ...]]:
    hashed_fields = {
        "uri": "uri_key",
        "directory_key": "directory_key_hash",
        "parent_key": "parent_key_hash",
        "scope_roots": "scope_root_keys",
    }
    physical = hashed_fields.get(field, field)
    if field in hashed_fields:
        if any(not isinstance(value, str) for value in values):
            raise ValueError(f"vector filter {field} values must be strings")
        return physical, tuple(_index_key(cast(str, value)) for value in values)
    return physical, tuple(values)


def _optional_text_attribute(attributes: Mapping[str, object], field: str) -> str:
    value = attributes.get(field, "")
    if not isinstance(value, str):
        raise ValueError(f"vector {field} must be a string")
    return value


def _index_key(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("vector index key source must be a string")
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _scan_order(value: str) -> int:
    """使用 63 位稳定排序键；批次规划和回读都会检查碰撞。"""

    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def _validate_scan_order_batch(records: Sequence[VectorStoreRecord]) -> None:
    """VikingDB 标量分页依赖 63 位排序键，因此写入前必须拒绝批内碰撞。"""

    scan_orders: dict[int, str] = {}
    for record in records:
        if not isinstance(record, VectorStoreRecord):
            raise TypeError("VikingDB record batch contains an invalid item")
        order = _scan_order(record.identity)
        collision = scan_orders.get(order)
        if collision is not None and collision != record.identity:
            raise ValueError("vector record identities collide on the VikingDB scan key")
        scan_orders[order] = record.identity


def _positive_int(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"vector {name} must be a positive integer")

def _batches(values: Sequence[_BatchItem], size: int) -> tuple[Sequence[_BatchItem], ...]:
    return tuple(values[offset : offset + size] for offset in range(0, len(values), size))


def _json_text(value: Mapping[str, object], label: str) -> str:
    try:
        encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"VikingDB {label} must be JSON serializable") from exc
    return _bounded_text(encoded, label)


def _bounded_text(value: str, label: str) -> str:
    if len(value.encode("utf-8")) > _TEXT_FIELD_BYTE_LIMIT:
        raise ValueError(f"VikingDB {label} exceeds the one-megabyte text field limit")
    return value


__all__ = [
    "VikingDBBackend",
    "build_vikingdb_backend",
    "register_builtin_vector_adapters",
]
