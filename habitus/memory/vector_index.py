"""Memory 与 Conversation Summary 两个持久向量索引共用的应用侧发布骨架。

下层 ``infrastructure/vector/publication`` 已经统一了代次、检查点与失败重放；这里只统一应用侧
"真相源快照 → 复用或重新编码 → 全量/增量发布 → 只读审计"的流程。真相源遍历、记录属性、扫描过滤、
差异报告类型和逐字报错文字由具体索引声明，本模块不依赖任何记忆子包。
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar, Generic, Protocol, TypeVar

from habitus.infrastructure.vector import (
    VectorStore,
    VectorStoreConflictError,
    VectorStoreFilter,
    VectorStoreRecord,
    VectorStoreState,
)
from habitus.model_client import Embedder, EmbeddingVector


class VectorIndexSource(Protocol):
    """一条尚未生成向量的规范真相源快照。"""

    @property
    def identity(self) -> str: ...

    @property
    def content(self) -> str: ...

    @property
    def content_digest(self) -> str: ...


class VectorIndexBounds(Protocol):
    """发布骨架读取的有界运行参数。"""

    @property
    def max_records(self) -> int: ...

    @property
    def max_search_hits(self) -> int: ...

    @property
    def embedding_batch_size(self) -> int: ...

    @property
    def stale_retries(self) -> int: ...


@dataclass(frozen=True)
class VectorIndexMessages:
    """具体索引逐字保留的报错文字。"""

    rebuild_checkpoint: str
    rebuild_conflict: str
    audit_not_ready: str
    audit_record_count: str
    embedding_count: str
    embedding_dimension: str
    materialization_lost: str
    search_limit: str


SourceT = TypeVar("SourceT", bound=VectorIndexSource)
ReportT = TypeVar("ReportT")
ConfigT = TypeVar("ConfigT", bound=VectorIndexBounds)


class PersistentVectorIndex(ABC, Generic[SourceT, ReportT, ConfigT]):
    """远程索引是可重建派生数据；具体索引的真相源始终是唯一内容来源。"""

    _schema_version: ClassVar[str]
    _messages: ClassVar[VectorIndexMessages]
    _error_type: ClassVar[type[Exception]]

    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        *,
        dimension: int,
        embedding_fingerprint: str,
        config: ConfigT,
    ) -> None:
        self.embedder = embedder
        self.store = store
        self.dimension = dimension
        self.embedding_fingerprint = embedding_fingerprint
        self.config = config
        self._lock = asyncio.Lock()

    async def ensure_ready(self) -> VectorStoreState:
        """验证远端集合；缺失或模型语义变化时从真相源完整重建。"""

        await self.store.initialize()
        async with self._lock:
            state = await self.store.state()
            if self._state_matches(state):
                assert state is not None
                return state
            checkpoint = 0 if state is None else state.checkpoint
            return await self._rebuild_locked(checkpoint=checkpoint)

    async def rebuild(self, *, checkpoint: int | None = None) -> VectorStoreState:
        """管理入口：完整扫描真相源并发布一代新的远程索引。"""

        await self.store.initialize()
        async with self._lock:
            current = await self.store.state()
            selected = (0 if current is None else current.checkpoint) if checkpoint is None else checkpoint
            if selected < 0:
                raise ValueError(self._messages.rebuild_checkpoint)
            return await self._rebuild_locked(checkpoint=selected)

    async def audit_consistency(self) -> ReportT:
        """无副作用比较全部身份和内容摘要，不触发重建。"""

        state = await self.store.state()
        if not self._state_matches(state):
            raise self._error_type(self._messages.audit_not_ready)
        assert state is not None
        expected = {source.identity: source for source in self._walk_sources()}
        indexed = {
            record.identity: record
            for record in await self.store.scan(
                filters=self._scan_filter(),
                limit=self.config.max_records,
            )
        }
        if len(indexed) != state.record_count:
            raise self._error_type(self._messages.audit_record_count)
        expected_ids = set(expected)
        indexed_ids = set(indexed)
        stale = tuple(
            sorted(
                identity
                for identity in expected_ids & indexed_ids
                if expected[identity].content_digest != indexed[identity].content_digest
            )
        )
        return self._consistency_report(
            expected_count=len(expected),
            indexed_count=len(indexed),
            missing_identities=tuple(sorted(expected_ids - indexed_ids)),
            stale_identities=stale,
            orphan_identities=tuple(sorted(indexed_ids - expected_ids)),
        )

    async def close(self) -> None:
        await self.store.close()

    async def _ready_state_locked(self) -> VectorStoreState:
        """增量同步每轮开头读取已发布状态；语义不匹配时先在锁内完整重建。"""

        state = await self.store.state()
        if not self._state_matches(state):
            state = await self._rebuild_locked(checkpoint=0 if state is None else state.checkpoint)
        assert state is not None
        return state

    async def _rebuild_locked(self, *, checkpoint: int) -> VectorStoreState:
        for _ in range(self.config.stale_retries + 1):
            state = await self.store.state()
            sources = self._walk_sources()
            reusable: dict[str, VectorStoreRecord] = {}
            if self._state_matches(state):
                assert state is not None
                self._require_rebuild_capacity(state)
                reusable = {
                    record.identity: record
                    for record in await self.store.scan(
                        filters=self._scan_filter(),
                        limit=self.config.max_records,
                    )
                }
            records = await self._materialize(sources, existing=reusable)
            try:
                return await self.store.replace_all(
                    records,
                    schema_version=self._schema_version,
                    embedding_fingerprint=self.embedding_fingerprint,
                    dimension=self.dimension,
                    checkpoint=checkpoint,
                    expected_generation=None if state is None else state.generation,
                )
            except VectorStoreConflictError:
                # generation CAS 冲突：重读状态后在有界次数内重建，用尽后由循环外明确报错。
                continue
        raise self._error_type(self._messages.rebuild_conflict)

    async def _materialize(
        self,
        sources: Sequence[SourceT],
        *,
        existing: dict[str, VectorStoreRecord],
    ) -> tuple[VectorStoreRecord, ...]:
        resolved: dict[str, VectorStoreRecord] = {}
        missing: list[SourceT] = []
        for source in sources:
            current = existing.get(source.identity)
            if (
                current is not None
                and current.content_digest == source.content_digest
                and current.vector.dimension == self.dimension
            ):
                resolved[source.identity] = self._record(source, current.vector)
            else:
                missing.append(source)
        for offset in range(0, len(missing), self.config.embedding_batch_size):
            batch = missing[offset : offset + self.config.embedding_batch_size]
            vectors = await self.embedder.embed_documents(tuple(source.content for source in batch))
            if len(vectors) != len(batch):
                raise self._error_type(self._messages.embedding_count)
            for source, vector in zip(batch, vectors, strict=True):
                if vector.dimension != self.dimension:
                    raise self._error_type(self._messages.embedding_dimension)
                resolved[source.identity] = self._record(source, vector)
        if len(resolved) != len(sources):
            raise AssertionError(self._messages.materialization_lost)
        return tuple(resolved[identity] for identity in sorted(resolved))

    def _state_matches(self, state: VectorStoreState | None) -> bool:
        return bool(
            state is not None
            and state.ready
            and state.schema_version == self._schema_version
            and state.embedding_fingerprint == self.embedding_fingerprint
            and state.dimension == self.dimension
        )

    def _limit(self, value: int) -> int:
        if not 1 <= value <= self.config.max_search_hits:
            raise ValueError(self._messages.search_limit)
        return value

    def _require_rebuild_capacity(self, state: VectorStoreState) -> None:
        """复用已发布记录前的容量检查；默认不限制，由需要的具体索引覆盖。"""

    @abstractmethod
    def _walk_sources(self) -> Sequence[SourceT]:
        """有界遍历完整真相源，供重建与审计使用。"""

    @abstractmethod
    def _scan_filter(self) -> VectorStoreFilter:
        """重建复用与审计扫描该索引全部记录时使用的过滤。"""

    @abstractmethod
    def _record(self, source: SourceT, vector: EmbeddingVector) -> VectorStoreRecord:
        """把一条真相源快照与向量组装成远程记录。"""

    @abstractmethod
    def _consistency_report(
        self,
        *,
        expected_count: int,
        indexed_count: int,
        missing_identities: tuple[str, ...],
        stale_identities: tuple[str, ...],
        orphan_identities: tuple[str, ...],
    ) -> ReportT:
        """构造具体索引的差异报告。"""


__all__ = [
    "PersistentVectorIndex",
    "VectorIndexBounds",
    "VectorIndexMessages",
    "VectorIndexSource",
]
