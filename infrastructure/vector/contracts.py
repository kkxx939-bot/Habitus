"""向量数据库 Adapter 必须实现的完整耐久契约。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

from infrastructure.vector.model import (
    VectorStoreFilter,
    VectorStoreMatch,
    VectorStoreRecord,
    VectorStoreState,
)
from ModelClient import EmbeddingVector


class VectorStore(Protocol):
    """支持远程 Collection、可重放写入和受控检索的向量存储。"""

    @property
    def adapter_name(self) -> str: ...

    @property
    def provider_name(self) -> str: ...

    @property
    def collection(self) -> str: ...

    async def initialize(self) -> None: ...

    async def state(self) -> VectorStoreState | None: ...

    async def read(self, identities: tuple[str, ...]) -> Sequence[VectorStoreRecord]: ...

    async def replace_all(
        self,
        records: tuple[VectorStoreRecord, ...],
        *,
        schema_version: str,
        embedding_fingerprint: str,
        dimension: int,
        checkpoint: int,
        expected_generation: int | None,
    ) -> VectorStoreState: ...

    async def apply(
        self,
        upserts: tuple[VectorStoreRecord, ...],
        deletes: tuple[str, ...],
        *,
        checkpoint: int,
        expected_generation: int,
        expected_checkpoint: int,
    ) -> VectorStoreState: ...

    async def search(
        self,
        query_vector: EmbeddingVector,
        *,
        filters: VectorStoreFilter,
        limit: int,
    ) -> Sequence[VectorStoreMatch]: ...

    async def scan(
        self,
        *,
        filters: VectorStoreFilter,
        limit: int,
    ) -> Sequence[VectorStoreRecord]: ...

    async def close(self) -> None: ...


class RawVectorBackend(Protocol):
    """厂商 Adapter 必须实现的物理 Collection 操作，不包含发布编排。"""

    @property
    def adapter_name(self) -> str: ...

    @property
    def provider_name(self) -> str: ...

    @property
    def collection(self) -> str: ...

    @property
    def max_records(self) -> int: ...

    @property
    def max_search_hits(self) -> int: ...

    async def initialize(self) -> None: ...

    async def read_metadata(
        self,
        names: tuple[str, ...],
    ) -> Mapping[str, Mapping[str, object]]: ...

    async def write_metadata(
        self,
        values: Mapping[str, Mapping[str, object]],
        *,
        dimension: int,
    ) -> None: ...

    async def ensure_schema(
        self,
        dimension: int,
        *,
        published_dimension: int | None,
    ) -> None: ...

    async def read(self, identities: tuple[str, ...]) -> Sequence[VectorStoreRecord]: ...

    async def delete_all(self) -> None: ...

    async def upsert(self, records: tuple[VectorStoreRecord, ...]) -> None: ...

    async def delete(self, identities: tuple[str, ...]) -> None: ...

    async def validate_records(
        self,
        records: tuple[VectorStoreRecord, ...],
        *,
        replacing: bool,
    ) -> None: ...

    async def wait_visible(
        self,
        upserts: tuple[VectorStoreRecord, ...],
        deletes: tuple[str, ...],
        *,
        complete: bool,
    ) -> None: ...

    async def search(
        self,
        query_vector: EmbeddingVector,
        *,
        filters: VectorStoreFilter,
        limit: int,
    ) -> Sequence[VectorStoreMatch]: ...

    async def scan(
        self,
        *,
        filters: VectorStoreFilter,
        limit: int,
    ) -> Sequence[VectorStoreRecord]: ...

    async def close(self) -> None: ...


__all__ = ["RawVectorBackend", "VectorStore"]
