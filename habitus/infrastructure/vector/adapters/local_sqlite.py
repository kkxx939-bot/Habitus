"""嵌在进程内的本地向量 Backend：标准库 SQLite 存记录，numpy 暴力算余弦。

每个 Collection 一个 SQLite 文件（``{data_root}/{collection}.sqlite3``）。打开时把全部向量读进一个
float32 矩阵，upsert / delete 提交成功后同步改矩阵；另一连接提交过（``PRAGMA data_version`` 变了）就整块重读，
所以矩阵永远不比文件旧。写入在同一个 SQLite 事务里完成，提交即可见，``wait_visible`` 无事可做。
发布编排（generation、checkpoint、building/ready、并发锁）仍全部在 ``PublishedVectorStore``。
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import sqlite3
import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar, cast

import numpy as np
import numpy.typing as npt

from habitus.foundation.embedding import EmbeddingVector
from habitus.infrastructure.vector.config import VectorStoreConfig
from habitus.infrastructure.vector.factory import VectorStoreBuildContext, VectorStoreFactory
from habitus.infrastructure.vector.model import (
    VectorStoreBusyError,
    VectorStoreConflictError,
    VectorStoreError,
    VectorStoreFilter,
    VectorStoreIntegrityError,
    VectorStoreMatch,
    VectorStoreRecord,
    VectorValue,
)

ADAPTER_NAME = "local_sqlite"
_FORMAT = "habitus_local_vector_v1"
_OPTION_FIELDS = frozenset({"busy_timeout_seconds"})
_FLOAT32 = np.dtype("<f4")
_COLUMNS = {
    "schema_info": ("name", "value"),
    "records": ("identity", "vector", "content", "content_digest", "attributes_json"),
    "metadata": ("name", "value_json"),
}
#: 单条 SQL 里 ``IN (...)`` 的参数个数上限，远低于 SQLite 默认的 32766。
_SELECT_BATCH = 500
#: 另一连接持锁、磁盘满、IO 错误：腾出锁或空间后同一份输入再来就能成，记忆 Job 按"等依赖恢复"处理（用户 11-06 定）
_BUSY_CODES = frozenset({sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED, sqlite3.SQLITE_FULL, sqlite3.SQLITE_IOERR})
_CORRUPT_CODES = frozenset({sqlite3.SQLITE_CORRUPT, sqlite3.SQLITE_NOTADB})
_Result = TypeVar("_Result")
_Matrix = npt.NDArray[np.float32]


@dataclass(frozen=True)
class LocalSQLiteVectorStoreConfig:
    """本地后端的严格 options；只有 SQLite 等写锁的时长这一项可调。"""

    busy_timeout_seconds: float = 5.0

    def __post_init__(self) -> None:
        value = self.busy_timeout_seconds
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError("local_sqlite busy_timeout_seconds must be numeric")
        number = float(value)
        if not math.isfinite(number) or not 0.001 <= number <= 600.0:
            raise ValueError("local_sqlite busy_timeout_seconds must be between 0.001 and 600")
        object.__setattr__(self, "busy_timeout_seconds", number)

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> LocalSQLiteVectorStoreConfig:
        if not isinstance(value, Mapping):
            raise TypeError("local_sqlite vector options must be an object")
        unknown = sorted(set(value) - _OPTION_FIELDS)
        if unknown:
            raise ValueError(f"local_sqlite vector options contain unknown fields: {unknown}")
        return cls(**cast(dict[str, float], dict(value)))

    @staticmethod
    def validate_store(store: VectorStoreConfig) -> None:
        """本地文件不走网络、不要凭据：这几项写了就是配错了。

        Collection 名就是文件名；macOS 与 Windows 的文件系统不分大小写，``Memory`` 与 ``memory`` 会落到同一个文件，
        所以只收小写名。
        """

        if not isinstance(store, VectorStoreConfig):
            raise TypeError("store must be VectorStoreConfig")
        route = store.route
        if store.collection != store.collection.lower():
            raise ValueError("local_sqlite vector collection must be lowercase (it is the file name)")
        if route.base_url:
            raise ValueError("local_sqlite vector store does not accept route.base_url")
        if route.credential_ref:
            raise ValueError("local_sqlite vector store does not accept route.credential_ref")
        if route.extra_headers:
            raise ValueError("local_sqlite vector store does not accept route.extra_headers")


class _VectorCache:
    """文件里全部记录的向量矩阵与属性；行序无意义，按 identity 定位。"""

    def __init__(self, dimension: int | None) -> None:
        self.dimension = dimension
        self.identities: list[str] = []
        self.attributes: list[Mapping[str, VectorValue]] = []
        self.positions: dict[str, int] = {}
        self.matrix: _Matrix = np.zeros((0, dimension or 0), dtype=_FLOAT32)

    def upsert(self, rows: Sequence[tuple[str, _Matrix, Mapping[str, VectorValue]]]) -> None:
        appended: list[_Matrix] = []
        # 同一批里同一 identity 出现多次时与 INSERT OR REPLACE 一致：后写的赢
        latest = {row[0]: row for row in rows}
        for identity, vector, attributes in latest.values():
            position = self.positions.get(identity)
            if position is None:
                self.positions[identity] = len(self.identities)
                self.identities.append(identity)
                self.attributes.append(attributes)
                appended.append(vector)
            else:
                self.matrix[position] = vector
                self.attributes[position] = attributes
        if appended:
            self.matrix = np.vstack((self.matrix, np.stack(appended)))

    def delete(self, identities: Sequence[str]) -> None:
        removed = {self.positions[identity] for identity in identities if identity in self.positions}
        if not removed:
            return
        keep = [index for index in range(len(self.identities)) if index not in removed]
        self.identities = [self.identities[index] for index in keep]
        self.attributes = [self.attributes[index] for index in keep]
        self.matrix = self.matrix[np.asarray(keep, dtype=np.intp)].reshape(len(keep), self.matrix.shape[1])
        self.positions = {identity: index for index, identity in enumerate(self.identities)}

    def matching(self, filters: VectorStoreFilter) -> list[int]:
        return [index for index, attributes in enumerate(self.attributes) if filters.matches(attributes)]


class LocalSQLiteVectorBackend:
    """只实现本地 SQLite 文件的物理读写、编解码和 numpy 余弦检索。"""

    adapter_name = ADAPTER_NAME

    def __init__(
        self,
        provider_name: str,
        collection: str,
        path: Path,
        config: LocalSQLiteVectorStoreConfig,
        *,
        max_records: int,
        max_search_hits: int,
    ) -> None:
        self.provider_name = provider_name
        self.collection = collection
        self.path = path
        self.config = config
        self._max_records = max_records
        self._max_search_hits = max_search_hits
        self._lock = threading.Lock()
        self._connection: sqlite3.Connection | None = None
        self._cache = _VectorCache(None)
        self._data_version: int | None = None

    @property
    def max_records(self) -> int:
        return self._max_records

    @property
    def max_search_hits(self) -> int:
        return self._max_search_hits

    async def initialize(self) -> None:
        await self._call(self._open)

    async def read_metadata(self, names: tuple[str, ...]) -> Mapping[str, Mapping[str, object]]:
        normalized = _metadata_names(names)
        return await self._call(lambda: self._read_metadata(normalized))

    async def write_metadata(self, values: Mapping[str, Mapping[str, object]], *, dimension: int) -> None:
        if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 1:
            raise ValueError("vector metadata dimension must be a positive integer")
        encoded = _encoded_metadata(values)
        await self._call(lambda: self._write_metadata(encoded))

    async def ensure_schema(self, dimension: int, *, published_dimension: int | None) -> None:
        await self._call(lambda: self._ensure_schema(dimension, published_dimension))

    async def read(self, identities: tuple[str, ...]) -> tuple[VectorStoreRecord, ...]:
        if not identities:
            return ()
        return await self._call(lambda: self._select_ordered(identities))

    async def delete_all(self) -> None:
        await self._call(self._delete_all)

    async def upsert(self, records: tuple[VectorStoreRecord, ...]) -> None:
        if records:
            await self._call(lambda: self._upsert(records))

    async def delete(self, identities: tuple[str, ...]) -> None:
        if identities:
            await self._call(lambda: self._delete(identities))

    async def validate_records(self, records: tuple[VectorStoreRecord, ...], *, replacing: bool) -> None:
        await self._call(lambda: self._assert_dimensions(records))

    async def wait_visible(
        self,
        upserts: tuple[VectorStoreRecord, ...],
        deletes: tuple[str, ...],
        *,
        complete: bool,
    ) -> None:
        """SQLite 提交即可见，没有异步索引要等。"""

    async def search(
        self,
        query_vector: EmbeddingVector,
        *,
        filters: VectorStoreFilter,
        limit: int,
    ) -> tuple[VectorStoreMatch, ...]:
        return await self._call(lambda: self._search(query_vector, filters, limit))

    async def scan(self, *, filters: VectorStoreFilter, limit: int) -> tuple[VectorStoreRecord, ...]:
        return await self._call(lambda: self._scan(filters, limit))

    async def close(self) -> None:
        await self._call(self._close)

    async def _call(self, operation: Callable[[], _Result]) -> _Result:
        return await asyncio.to_thread(self._locked, operation)

    def _locked(self, operation: Callable[[], _Result]) -> _Result:
        with self._lock:
            try:
                return operation()
            except sqlite3.Error as exc:
                self._data_version = None
                raise _translated(exc) from exc
            except BaseException:
                self._data_version = None
                raise

    def _open(self) -> None:
        if self._connection is None:
            self._connection = _connect(self.path, self.config.busy_timeout_seconds)
            self._sync()

    def _close(self) -> None:
        connection, self._connection = self._connection, None
        self._cache = _VectorCache(None)
        self._data_version = None
        if connection is not None:
            connection.close()

    def _db(self) -> sqlite3.Connection:
        if self._connection is None:
            self._open()
        assert self._connection is not None
        return self._connection

    def _sync(self) -> None:
        """另一连接提交过或本实例出过错，就从文件整块重读矩阵。"""

        connection = self._db()
        version = int(connection.execute("PRAGMA data_version").fetchone()[0])
        if version != self._data_version:
            self._cache = _load_cache(connection)
            self._data_version = version

    def _read_metadata(self, names: tuple[str, ...]) -> dict[str, Mapping[str, object]]:
        connection = self._db()
        values: dict[str, Mapping[str, object]] = {}
        for batch in _batches(names):
            for name, raw in connection.execute(
                f"SELECT name, value_json FROM metadata WHERE name IN ({', '.join('?' * len(batch))})", batch
            ):
                values[str(name)] = _json_object(raw, "metadata")
        return {name: values[name] for name in names if name in values}

    def _write_metadata(self, encoded: Sequence[tuple[str, str]]) -> None:
        with _transaction(self._db()) as connection:
            connection.executemany("INSERT OR REPLACE INTO metadata(name, value_json) VALUES (?, ?)", encoded)

    def _ensure_schema(self, dimension: int, published_dimension: int | None) -> None:
        if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 1:
            raise ValueError("vector schema dimension must be a positive integer")
        with _transaction(self._db()) as connection:
            self._sync()
            stored = self._cache.dimension
            if stored == dimension:
                return
            if stored is not None and published_dimension == dimension:
                raise VectorStoreIntegrityError("local vector file dimension does not match the published state")
            # 首次建库，或维度变了：旧向量与发布元数据一起作废，由随后的全量发布重建
            connection.execute("DELETE FROM records")
            connection.execute("DELETE FROM metadata")
            connection.execute(
                "INSERT OR REPLACE INTO schema_info(name, value) VALUES ('dimension', ?)",
                (str(dimension),),
            )
        self._cache = _VectorCache(dimension)

    def _delete_all(self) -> None:
        with _transaction(self._db()) as connection:
            self._sync()
            connection.execute("DELETE FROM records")
        self._cache = _VectorCache(self._cache.dimension)

    def _upsert(self, records: tuple[VectorStoreRecord, ...]) -> None:
        with _transaction(self._db()) as connection:
            self._sync()
            self._assert_dimensions(records)
            rows = [(record, _vector_array(record.vector)) for record in records]
            connection.executemany(
                "INSERT OR REPLACE INTO records(identity, vector, content, content_digest, attributes_json)"
                " VALUES (?, ?, ?, ?, ?)",
                [
                    (record.identity, vector.tobytes(), record.content, record.content_digest, _attributes_json(record))
                    for record, vector in rows
                ],
            )
        self._cache.upsert([(record.identity, vector, record.attributes) for record, vector in rows])

    def _delete(self, identities: tuple[str, ...]) -> None:
        with _transaction(self._db()) as connection:
            self._sync()
            for batch in _batches(identities):
                connection.execute(f"DELETE FROM records WHERE identity IN ({', '.join('?' * len(batch))})", batch)
        self._cache.delete(identities)

    def _assert_dimensions(self, records: Sequence[VectorStoreRecord]) -> None:
        self._sync()
        if records and self._cache.dimension is None:
            raise VectorStoreIntegrityError("local vector file has no schema dimension yet")
        if any(record.vector.dimension != self._cache.dimension for record in records):
            raise ValueError("vector record dimension does not match the local vector file")

    def _search(
        self,
        query_vector: EmbeddingVector,
        filters: VectorStoreFilter,
        limit: int,
    ) -> tuple[VectorStoreMatch, ...]:
        self._sync()
        ranked = _ranked(self._cache, query_vector, filters, limit)
        records = self._select_ordered(tuple(identity for _score, identity in ranked))
        if len(records) != len(ranked):
            # 排序与取记录之间另一连接提交了删除：发布在读的过程中变了，可重试，不是文件损坏
            raise VectorStoreConflictError("local vector file changed during search")
        return tuple(
            VectorStoreMatch(record=record, score=max(-1.0, min(1.0, score)))
            for (score, _identity), record in zip(ranked, records, strict=True)
        )

    def _scan(self, filters: VectorStoreFilter, limit: int) -> tuple[VectorStoreRecord, ...]:
        self._sync()
        cache = self._cache
        selected = sorted(cache.identities[index] for index in cache.matching(filters))[:limit]
        return self._select_ordered(tuple(selected))

    def _select_ordered(self, identities: tuple[str, ...]) -> tuple[VectorStoreRecord, ...]:
        self._sync()
        connection = self._db()
        dimension = self._cache.dimension
        found: dict[str, VectorStoreRecord] = {}
        for batch in _batches(identities):
            for row in connection.execute(
                "SELECT identity, vector, content, content_digest, attributes_json FROM records"
                f" WHERE identity IN ({', '.join('?' * len(batch))})",
                batch,
            ):
                record = _record_from_row(row, dimension)
                found[record.identity] = record
        return tuple(found[identity] for identity in identities if identity in found)


def build_local_sqlite_backend(context: VectorStoreBuildContext) -> LocalSQLiteVectorBackend:
    config = LocalSQLiteVectorStoreConfig.from_mapping(context.config.options)
    config.validate_store(context.config)
    if context.data_root is None:
        raise VectorStoreError("local_sqlite vector store requires a data_root under the storage root")
    return LocalSQLiteVectorBackend(
        context.config.provider,
        context.config.collection,
        Path(context.data_root) / f"{context.config.collection}.sqlite3",
        config,
        max_records=context.requirements.max_records,
        max_search_hits=context.requirements.max_search_hits,
    )


def register_builtin_vector_adapters(factory: VectorStoreFactory | None = None) -> VectorStoreFactory:
    target = factory or VectorStoreFactory()
    target.register_adapter(ADAPTER_NAME, build_local_sqlite_backend)
    return target


def _connect(path: Path, busy_timeout_seconds: float) -> sqlite3.Connection:
    """打开（必要时新建）一个 Collection 文件；WAL 让读不挡写，FULL 同步让发布元数据按提交顺序耐久。

    文件里是记忆原文：目录 0700、库文件 0600，与存储根下其他 SQLite 一致。主库要在连接前就建成 0600，
    SQLite 随后按主库的权限创建 ``-wal`` / ``-shm``。
    """

    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    path.touch(mode=0o600, exist_ok=True)
    os.chmod(path, 0o600)
    connection = sqlite3.connect(path, timeout=busy_timeout_seconds, isolation_level=None, check_same_thread=False)
    try:
        connection.execute("PRAGMA journal_mode=WAL").fetchall()
        connection.execute("PRAGMA synchronous=FULL")
        with _transaction(connection):
            _create_schema(connection)
    except BaseException:
        connection.close()
        raise
    return connection


def _ranked(
    cache: _VectorCache,
    query_vector: EmbeddingVector,
    filters: VectorStoreFilter,
    limit: int,
) -> list[tuple[float, str]]:
    """先按属性过滤，再对剩下的行算余弦（向量都已归一化，点积即余弦）；同分按 identity 升序，结果确定。"""

    candidates = cache.matching(filters)
    if not candidates:
        return []
    if query_vector.dimension != cache.dimension:
        raise VectorStoreIntegrityError("query vector dimension does not match the local vector file")
    scores = cache.matrix[np.asarray(candidates, dtype=np.intp)] @ _vector_array(query_vector)
    return sorted(
        ((float(score), cache.identities[index]) for score, index in zip(scores.tolist(), candidates, strict=True)),
        key=lambda item: (-item[0], item[1]),
    )[:limit]


@contextmanager
def _transaction(connection: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """``BEGIN IMMEDIATE`` 先拿写锁：等不到锁在开头就报 busy，不在半路。"""

    connection.execute("BEGIN IMMEDIATE")
    try:
        yield connection
        connection.execute("COMMIT")
    except BaseException:
        if connection.in_transaction:
            connection.execute("ROLLBACK")
        raise


def _create_schema(connection: sqlite3.Connection) -> None:
    connection.execute("CREATE TABLE IF NOT EXISTS schema_info(name TEXT PRIMARY KEY, value TEXT NOT NULL)")
    connection.execute(
        "CREATE TABLE IF NOT EXISTS records("
        "identity TEXT PRIMARY KEY, vector BLOB NOT NULL, content TEXT NOT NULL,"
        " content_digest TEXT NOT NULL, attributes_json TEXT NOT NULL)"
    )
    connection.execute("CREATE TABLE IF NOT EXISTS metadata(name TEXT PRIMARY KEY, value_json TEXT NOT NULL)")
    for table, columns in _COLUMNS.items():
        if tuple(row[1] for row in connection.execute(f"PRAGMA table_info({table})")) != columns:
            raise VectorStoreIntegrityError(f"local vector file table {table} has an unexpected layout")
    connection.execute("INSERT OR IGNORE INTO schema_info(name, value) VALUES ('format', ?)", (_FORMAT,))
    row = connection.execute("SELECT value FROM schema_info WHERE name = 'format'").fetchone()
    if row is None or row[0] != _FORMAT:
        raise VectorStoreIntegrityError("local vector file has an unknown format")


def _load_cache(connection: sqlite3.Connection) -> _VectorCache:
    row = connection.execute("SELECT value FROM schema_info WHERE name = 'dimension'").fetchone()
    dimension = None if row is None else _stored_dimension(row[0])
    cache = _VectorCache(dimension)
    rows = connection.execute("SELECT identity, vector, attributes_json FROM records").fetchall()
    if rows and dimension is None:
        raise VectorStoreIntegrityError("local vector file has records but no schema dimension")
    if rows:
        assert dimension is not None
        cache.upsert(
            [
                (str(identity), _blob_vector(blob, dimension), _decoded_attributes(attributes))
                for identity, blob, attributes in rows
            ]
        )
    return cache


def _stored_dimension(value: object) -> int:
    if not isinstance(value, str) or not value.isdigit() or int(value) < 1:
        raise VectorStoreIntegrityError("local vector file has an invalid schema dimension")
    return int(value)


def _record_from_row(row: tuple[object, ...], dimension: int | None) -> VectorStoreRecord:
    identity, blob, content, digest, attributes = row
    if dimension is None:
        raise VectorStoreIntegrityError("local vector file has records but no schema dimension")
    vector = _blob_vector(blob, dimension)
    try:
        return VectorStoreRecord(
            identity=cast(str, identity),
            vector=EmbeddingVector(tuple(vector.tolist())),
            content=cast(str, content),
            content_digest=cast(str, digest),
            attributes=_decoded_attributes(attributes),
        )
    except (TypeError, ValueError) as exc:
        raise VectorStoreIntegrityError("local vector record fields are invalid") from exc


def _blob_vector(blob: object, dimension: int) -> _Matrix:
    if not isinstance(blob, bytes) or len(blob) != dimension * _FLOAT32.itemsize:
        raise VectorStoreIntegrityError("local vector record has a vector of the wrong size")
    return np.frombuffer(blob, dtype=_FLOAT32).astype(np.float32)


def _vector_array(vector: EmbeddingVector) -> _Matrix:
    return np.asarray(vector.values, dtype=_FLOAT32)


def _attributes_json(record: VectorStoreRecord) -> str:
    plain = {name: list(value) if isinstance(value, tuple) else value for name, value in record.attributes.items()}
    return json.dumps(plain, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _decoded_attributes(raw: object) -> dict[str, VectorValue]:
    value = _json_object(raw, "attributes_json")
    return {name: tuple(item) if isinstance(item, list) else cast(VectorValue, item) for name, item in value.items()}


def _encoded_metadata(values: Mapping[str, Mapping[str, object]]) -> list[tuple[str, str]]:
    if not isinstance(values, Mapping):
        raise TypeError("vector metadata values must be an object")
    names = _metadata_names(tuple(values))
    encoded: list[tuple[str, str]] = []
    for name in names:
        value = values[name]
        if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
            raise TypeError("vector metadata value must be a string-keyed object")
        try:
            text = json.dumps(dict(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("vector metadata value must be JSON serializable") from exc
        encoded.append((name, text))
    return encoded


def _json_object(raw: object, label: str) -> dict[str, object]:
    if not isinstance(raw, str):
        raise VectorStoreIntegrityError(f"local vector {label} is not text")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise VectorStoreIntegrityError(f"local vector {label} is invalid JSON") from exc
    if not isinstance(value, dict):
        raise VectorStoreIntegrityError(f"local vector {label} root must be an object")
    return value


def _metadata_names(names: Sequence[str]) -> tuple[str, ...]:
    if isinstance(names, str | bytes) or not isinstance(names, Sequence):
        raise TypeError("vector metadata names must be a sequence")
    if any(not isinstance(name, str) or not name or name != name.strip() for name in names):
        raise ValueError("vector metadata name must be normalized non-empty text")
    if len(set(names)) != len(names):
        raise ValueError("vector metadata names must be unique")
    return tuple(names)


def _batches(values: Sequence[str]) -> Iterator[tuple[str, ...]]:
    for offset in range(0, len(values), _SELECT_BATCH):
        yield tuple(values[offset : offset + _SELECT_BATCH])


def _translated(exc: sqlite3.Error) -> VectorStoreError:
    code = getattr(exc, "sqlite_errorcode", None)
    primary = code & 0xFF if isinstance(code, int) else None
    if primary in _BUSY_CODES:
        return VectorStoreBusyError(f"local vector file is temporarily unavailable: {exc}")
    if primary in _CORRUPT_CODES:
        return VectorStoreIntegrityError(f"local vector file is corrupt: {exc}")
    return VectorStoreError(f"local vector file operation failed: {exc}")


__all__ = [
    "LocalSQLiteVectorBackend",
    "LocalSQLiteVectorStoreConfig",
    "build_local_sqlite_backend",
    "register_builtin_vector_adapters",
]
