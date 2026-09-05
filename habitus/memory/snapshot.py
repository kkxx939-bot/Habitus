"""使用严格 ``memory://`` L2 URI 读取完整记忆快照。"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from pathlib import Path
from typing import Protocol, TypeAlias, cast

from habitus.foundation.integrity.canonical_json import canonical_json
from habitus.infrastructure.editor.snapshot import (
    SnapshotBatch,
    SnapshotReadConfig,
    SnapshotReader,
    SnapshotReadLimitError,
    SnapshotState,
    VersionedSnapshot,
)
from habitus.memory.document import MemoryDocument
from habitus.memory.tree import MemoryTree
from habitus.memory.uri import MemoryURI

MemorySnapshot: TypeAlias = VersionedSnapshot[MemoryDocument]
MemorySnapshotBatch: TypeAlias = SnapshotBatch[MemoryDocument]


class _VisibilityEntry(Protocol):
    uri: MemoryURI
    before: MemoryDocument | None


class _VisibilityRecord(Protocol):
    entries: tuple[_VisibilityEntry, ...]


class _VisibilityJournal(Protocol):
    root: Path

    def visibility_generation(self) -> int: ...

    def pending(self) -> tuple[_VisibilityRecord, ...]: ...


class MemorySnapshotConsistencyError(RuntimeError):
    """并发事务持续变化，无法在有界重试内形成一致快照。"""


class MemorySnapshotReader:
    """把记忆领域的 URI 和文档模型适配到公共版本快照机制。"""

    def __init__(
        self,
        tree: MemoryTree,
        *,
        config: SnapshotReadConfig | None = None,
    ) -> None:
        if not isinstance(tree, MemoryTree):
            raise TypeError("tree must be a MemoryTree")
        self.tree = tree
        self._reader = SnapshotReader[MemoryDocument](
            load=self._load,
            revision_of=lambda document: document.metadata.revision,
            serialize=self._serialize,
            config=config,
        )
        self._visibility_journal: _VisibilityJournal | None = None

    @property
    def config(self) -> SnapshotReadConfig:
        """返回当前批量读取使用的显式资源边界。"""

        return self._reader.config

    def read(self, uri: MemoryURI | str) -> MemorySnapshot:
        """读取一个 L2 URI；目录和 L0/L1 URI 会被严格拒绝。"""

        parsed = self._document_uri(uri)
        batch = self._read_consistent((str(parsed),))
        snapshot = batch.get(str(parsed))
        assert snapshot is not None
        return snapshot

    def read_many(self, uris: Iterable[MemoryURI | str]) -> MemorySnapshotBatch:
        """先验证全部 L2 URI，再执行去重且有界的批量读取。"""

        if isinstance(uris, str) or not isinstance(uris, Iterable):
            raise TypeError("uris must be an iterable of MemoryURI or string values")
        identities_list: list[str] = []
        for uri in uris:
            identities_list.append(str(self._document_uri(uri)))
            if len(identities_list) > self.config.max_items:
                raise SnapshotReadLimitError("snapshot batch exceeds its configured item limit")
        identities = tuple(identities_list)
        return self._read_consistent(identities)

    def _read_physical(self, uri: MemoryURI | str) -> MemorySnapshot:
        """仅供持有事务租约的提交与恢复流程读取物理状态。"""

        parsed = self._document_uri(uri)
        return self._reader.read(str(parsed))

    def _read_many_physical(self, uris: Iterable[MemoryURI | str]) -> MemorySnapshotBatch:
        """仅供持有事务租约的提交与恢复流程批量读取物理状态。"""

        if isinstance(uris, str) or not isinstance(uris, Iterable):
            raise TypeError("uris must be an iterable of MemoryURI or string values")
        return self._reader.read_many(str(self._document_uri(uri)) for uri in uris)

    def bind_visibility_journal(self, journal: object) -> None:
        """把公开读取绑定到唯一的耐久事务可见性日志。"""

        if (
            not isinstance(getattr(journal, "root", None), Path)
            or not callable(getattr(journal, "visibility_generation", None))
            or not callable(getattr(journal, "pending", None))
        ):
            raise TypeError("journal must implement the memory visibility journal contract")
        resolved = cast(_VisibilityJournal, journal)
        if self._visibility_journal is not None and self._visibility_journal.root != resolved.root:
            raise ValueError("snapshot reader is already bound to another visibility journal")
        self._visibility_journal = resolved

    def _read_consistent(self, identities: tuple[str, ...]) -> MemorySnapshotBatch:
        journal = self._visibility_journal
        tree_journal = self.tree.visibility_journal
        if journal is None and tree_journal is not None:
            self.bind_visibility_journal(tree_journal)
            journal = self._visibility_journal
        if journal is None:
            return self._reader.read_many(identities)
        for _attempt in range(16):
            generation_before = journal.visibility_generation()
            pending_before = journal.pending()
            physical = self._reader.read_many(identities)
            pending_after = journal.pending()
            generation_after = journal.visibility_generation()
            if generation_before == generation_after and pending_before == pending_after:
                return self._overlay_prepared(physical, pending_before)
        raise MemorySnapshotConsistencyError(
            "memory transactions changed continuously while reading one logical snapshot"
        )

    def _overlay_prepared(
        self,
        physical: MemorySnapshotBatch,
        pending: tuple[_VisibilityRecord, ...],
    ) -> MemorySnapshotBatch:
        snapshots = {snapshot.identity: snapshot for snapshot in physical.snapshots}
        for record in pending:
            for entry in record.entries:
                identity = str(entry.uri)
                if identity not in snapshots:
                    continue
                snapshots[identity] = self._snapshot_for(identity, entry.before)
        ordered = tuple(snapshots[identity] for identity in sorted(snapshots))
        total_bytes = sum(snapshot.size_bytes for snapshot in ordered)
        if total_bytes > self.config.max_total_bytes:
            raise SnapshotReadLimitError("snapshot batch exceeds its configured total byte limit")
        return SnapshotBatch(ordered, total_bytes)

    def _snapshot_for(
        self,
        identity: str,
        document: MemoryDocument | None,
    ) -> MemorySnapshot:
        if document is None:
            return VersionedSnapshot.missing(identity)
        payload = self._serialize(document)
        if len(payload) > self.config.max_item_bytes:
            raise SnapshotReadLimitError("snapshot item exceeds its configured byte limit")
        return VersionedSnapshot(
            identity=identity,
            state=SnapshotState.FOUND,
            value=document,
            revision=document.metadata.revision,
            source_digest=hashlib.sha256(payload).hexdigest(),
            size_bytes=len(payload),
        )

    def _load(self, identity: str) -> MemoryDocument:
        uri = self._document_uri(identity)
        return self.tree._read_physical(uri.to_address())

    @staticmethod
    def _document_uri(value: MemoryURI | str) -> MemoryURI:
        parsed = MemoryURI.parse(value)
        parsed.to_address()
        return parsed

    @staticmethod
    def _serialize(document: MemoryDocument) -> bytes:
        """序列化完整规范内容，生成与读取版本绑定的稳定摘要输入。"""

        payload = {
            "memory_type": document.kind.value,
            "address": str(MemoryURI.from_address(document.address)),
            "revision": document.metadata.revision,
            "created_at": document.metadata.created_at,
            "updated_at": document.metadata.updated_at,
            "last_confirmed_at": document.metadata.last_confirmed_at,
            "fields": document.fields,
            "markdown_body": document.markdown_body,
            "links": [link.to_dict() for link in document.links],
            "backlinks": [link.to_dict() for link in document.backlinks],
        }
        return canonical_json(payload).encode("utf-8")


__all__ = [
    "MemorySnapshot",
    "MemorySnapshotBatch",
    "MemorySnapshotConsistencyError",
    "MemorySnapshotReader",
]
