"""行为树唯一的公开写入策略：纯 add-only 发布。

树的每个字都从一个门进（观测 → 融合 → 归约 → 此处落盘）；本模块只承担落盘一环——
锁内检查、原子创建、完整读回。旧的 Outcome CAS 追加与 Episode 引用校验随
``TODO(BHV-TREE-REBUILD-001)`` 取消追加通道一并退役：封口窗口使晚到引用在机械上不可能，
可变性没有生产者。

调用方只有归约：``reduction.runner`` 发布新文档；已落树的文档不改（词表没有迁移，裁定 101）。
语义树只读 ``behavior.tree``，不许触达本模块（架构测试钉死）。
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from habitus.behavior.document import BehaviorDocument
from habitus.behavior.model import BehaviorDirectory, BehaviorKind
from habitus.behavior.tree.config import BehaviorWriteConfig
from habitus.behavior.tree.store import BehaviorTree, BehaviorTreeConflictError
from habitus.behavior.uri import BehaviorURI
from habitus.infrastructure.store.contracts.lock import LockStore
from habitus.infrastructure.store.contracts.path_lock import PathLock
from habitus.infrastructure.tree.lock_key import root_lock_prefix, uri_lock_key
from habitus.infrastructure.tree.metadata import DocumentMetadata


class BehaviorPublishConflictError(RuntimeError):
    """add-only 地址已绑定到**不同**内容。

    逐字节相同内容的重放是幂等成功，不抛本错误——归约层按死规则⑤（stage 之后只有确定性
    落盘）崩溃重试时，同地址同字节必须能通过；底层 ``atomic_create_bytes`` 正为此设计。
    """


class BehaviorReadBackError(RuntimeError):
    """耐久写入后的完整读回结果与预期不一致。"""


@dataclass(frozen=True)
class BehaviorDocumentLockKeyspace:
    root: Path
    _prefix: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        normalized = self.root.expanduser().absolute().resolve(strict=False)
        object.__setattr__(self, "root", normalized)
        object.__setattr__(self, "_prefix", root_lock_prefix("behavior-document", normalized))

    def key(self, uri: BehaviorURI | str) -> str:
        """地址带支，两支写同一天同名的事件也是两把锁。"""

        return uri_lock_key(self._prefix, str(BehaviorURI.parse(uri)))


class BehaviorDocumentWriter:
    """纯 add-only 发布器；占用同一 habitus 树的写入方必须共享 LockStore。"""

    def __init__(
        self,
        tree: BehaviorTree,
        lock_store: LockStore,
        *,
        clock: Callable[[], datetime] | None = None,
        config: BehaviorWriteConfig | None = None,
    ) -> None:
        self.tree = tree
        self.clock = clock or (lambda: datetime.now(UTC))
        self.config = config or BehaviorWriteConfig()
        self._path_lock = PathLock(lock_store)
        self._keyspace = BehaviorDocumentLockKeyspace(tree.root)

    def publish(self, kind: BehaviorKind | str, payload: Mapping[str, Any]) -> BehaviorDocument:
        """发布一个 L2；逐字节相同的重放幂等成功，不同内容撞同地址即冲突。"""

        normalized_kind = BehaviorKind(kind)
        document = self.tree.document_codec.build(
            normalized_kind,
            payload,
            metadata=DocumentMetadata.initial(self._timestamp()),
        )
        uri = self.tree.uri_for(document.address)
        directory_uri = BehaviorURI.from_directory(BehaviorDirectory.for_address(document.address), self.tree.lane)
        # 目录 URI 一并入锁：容量不变量（max_children_per_directory）跨文档，只锁文档
        # 会让两次并发发布各自通过容量检查后共同越界，且越界目录在读侧直接完整性拒绝。
        # 现有两种锁存储的 fenced 本身就全局串行（进程内一把锁、SQLite 的 BEGIN IMMEDIATE），这把目录锁在它们下面是
        # 防御：换成按键并发的锁存储时容量检查仍然成立。
        with self._fenced_uris(uri, directory_uri):
            try:
                self.tree.create(document)
            except BehaviorTreeConflictError as exc:
                raise BehaviorPublishConflictError(
                    f"behavior address is already bound to different content: {uri}"
                ) from exc
            self._require_read_back(document)
        return document

    @contextmanager
    def _fenced_uris(self, *uris: BehaviorURI) -> Iterator[None]:
        keys = tuple(sorted({self._keyspace.key(uri) for uri in uris}))
        with ExitStack() as stack:
            guards = tuple(
                stack.enter_context(self._path_lock.acquire(key, ttl_seconds=self.config.lock_ttl_seconds))
                for key in keys
            )
            with self._path_lock.fenced(guards):
                yield
        # 文档只写一次（add-only）：锁放了之后这几行再也用不上，空闲的删掉（26 号清单 7.1）；别人正持有的不动
        for key in keys:
            self._path_lock.lock_store.discard_idle(key)

    def _require_read_back(self, expected: BehaviorDocument) -> None:
        actual = self.tree.read(expected.address)
        if actual != expected:
            raise BehaviorReadBackError(f"behavior document read-back mismatch: {self.tree.uri_for(expected.address)}")

    def _timestamp(self) -> datetime:
        value = self.clock()
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("behavior writer clock must return a timezone-aware datetime")
        return value.astimezone(UTC)


__all__ = [
    "BehaviorDocumentWriter",
    "BehaviorPublishConflictError",
    "BehaviorReadBackError",
]
