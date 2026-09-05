"""行为树唯一的公开写入策略：纯 add-only 发布。

树的每个字都从一个门进（观测 → 融合 → 归约 → 此处落盘）；本模块只承担落盘一环——
锁内检查、原子创建、完整读回。旧的 Outcome CAS 追加与 Episode 引用校验随
``TODO(BHV-TREE-REBUILD-001)`` 取消追加通道一并退役：封口窗口使晚到引用在机械上不可能，
可变性没有生产者。

## 链接只指向已存在的目标

前向链接（concurrent_with / results_from）的目标必须已经落盘——封口按依赖拓扑排序是归约层
的职责，这里只做最后一道确定性检查：目标不存在即拒绝，不可变文档里不允许出现悬空引用
（沿融合层 ``without_unresolvable_relations`` 的同一纪律）。
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from habitus.behavior.document import BehaviorDocument, BehaviorDocumentMetadata
from habitus.behavior.document.link import BehaviorLinkType, BehaviorStoredLink
from habitus.behavior.model import BehaviorAddress, BehaviorDirectory, BehaviorKind
from habitus.behavior.tree import BehaviorTree, BehaviorTreeConflictError
from habitus.behavior.uri import BehaviorURI
from habitus.infrastructure.store.contracts.lock import LockStore
from habitus.infrastructure.store.contracts.path_lock import PathLock


class BehaviorPublishConflictError(RuntimeError):
    """add-only 地址已绑定到**不同**内容，或链接目标不存在。

    逐字节相同内容的重放是幂等成功，不抛本错误——归约层按死规则⑤（stage 之后只有确定性
    落盘）崩溃重试时，同地址同字节必须能通过；底层 ``atomic_create_bytes`` 正为此设计。
    """


class BehaviorReadBackError(RuntimeError):
    """耐久写入后的完整读回结果与预期不一致。"""


@dataclass(frozen=True)
class BehaviorWriteConfig:
    lock_ttl_seconds: int = 30

    def __post_init__(self) -> None:
        if (
            isinstance(self.lock_ttl_seconds, bool)
            or not isinstance(self.lock_ttl_seconds, int)
            or not 1 <= self.lock_ttl_seconds <= 3_600
        ):
            raise ValueError("lock_ttl_seconds must be between 1 and 3600")


@dataclass(frozen=True)
class BehaviorDocumentLockKeyspace:
    root: Path
    _prefix: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.root, Path):
            raise TypeError("root must be a Path")
        normalized = self.root.expanduser().absolute().resolve(strict=False)
        object.__setattr__(self, "root", normalized)
        digest = hashlib.sha256(str(normalized).encode("utf-8")).hexdigest()[:24]
        object.__setattr__(self, "_prefix", f"behavior-document:{digest}")

    def key(self, uri: BehaviorURI | str) -> str:
        parsed = BehaviorURI.parse(uri)
        digest = hashlib.sha256(str(parsed).encode("utf-8")).hexdigest()
        return f"{self._prefix}:{digest}"


class BehaviorDocumentWriter:
    """纯 add-only 发布器；占用同一 behavior-root 的写入方必须共享 LockStore。"""

    def __init__(
        self,
        tree: BehaviorTree,
        lock_store: LockStore,
        *,
        clock: Callable[[], datetime] | None = None,
        config: BehaviorWriteConfig | None = None,
    ) -> None:
        if not isinstance(tree, BehaviorTree):
            raise TypeError("tree must be a BehaviorTree")
        required = ("acquire", "renew", "fenced", "release")
        if any(not callable(getattr(lock_store, name, None)) for name in required):
            raise TypeError("lock_store must implement the LockStore contract")
        if clock is not None and not callable(clock):
            raise TypeError("clock must be callable")
        if config is not None and not isinstance(config, BehaviorWriteConfig):
            raise TypeError("config must be BehaviorWriteConfig")
        self.tree = tree
        self.clock = clock or (lambda: datetime.now(UTC))
        self.config = config or BehaviorWriteConfig()
        self._path_lock = PathLock(lock_store)
        self._keyspace = BehaviorDocumentLockKeyspace(tree.root)

    def publish(
        self,
        kind: BehaviorKind | str,
        payload: Mapping[str, Any],
        *,
        links: Sequence[tuple[BehaviorLinkType | str, BehaviorURI | str]] = (),
    ) -> BehaviorDocument:
        """发布一个 L2；逐字节相同的重放幂等成功，不同内容撞同地址即冲突，链接目标必须已落盘。"""

        normalized_kind = BehaviorKind(kind)
        document = self.tree.document_codec.build(
            normalized_kind,
            payload,
            metadata=BehaviorDocumentMetadata.initial(self._timestamp()),
        )
        uri = BehaviorURI.from_address(document.address)
        directory_uri = BehaviorURI.from_directory(BehaviorDirectory.for_address(document.address))
        stored_links = self._stored_links(normalized_kind, uri, links)
        if stored_links:
            document = self.tree.document_codec.build(
                normalized_kind,
                payload,
                metadata=document.metadata,
                links=stored_links,
            )
        # 目录 URI 一并入锁：容量不变量（max_children_per_directory）跨文档，只锁文档
        # 会让两次并发发布各自通过容量检查后共同越界，且越界目录在读侧直接完整性拒绝。
        with self._fenced_uris(uri, directory_uri, *(link.to_uri for link in stored_links)):
            for link in stored_links:
                if not self.tree.exists(link.to_uri.to_address()):
                    raise BehaviorPublishConflictError(
                        f"behavior link targets a document that does not exist: {link.to_uri}"
                    )
            try:
                self.tree.create(document)
            except BehaviorTreeConflictError as exc:
                raise BehaviorPublishConflictError(
                    f"behavior address is already bound to different content: {uri}"
                ) from exc
            self._require_read_back(document)
        return document

    def restamp_kind_token(self, address: BehaviorAddress, token: str) -> BehaviorDocument:
        """把一条 occurrence 的 ``kind_token`` 改成 ``token``：与发布同一条通道（同一 codec、同一把
        文档锁、同样的读回校验），只是走 ``tree.replace`` 而不是 ``create``。

        用途只有一个：词表把两类并成一类之后，树上旧 token 的 occurrence 重打为新 token（原始名、
        地址、链接、语义面全部不动）。正文渲染含类型，所以这一天的概览 digest 会变、下次刷新重生成。
        """

        uri = BehaviorURI.from_address(address)
        with self._fenced_uris(uri):
            current = self.tree.read(address)
            if current.kind is not BehaviorKind.OCCURRENCE:
                raise BehaviorPublishConflictError(f"only occurrences carry a kind_token: {uri}")
            if current.fields.get("kind_token") == token:
                return current
            payload = {**current.fields, "kind_token": token}
            metadata = replace(
                current.metadata,
                revision=current.metadata.revision + 1,
                updated_at=max(self._timestamp(), current.metadata.updated_at),
            )
            document = self.tree.document_codec.build(
                current.kind, payload, metadata=metadata, links=current.links
            )
            try:
                self.tree.replace(document)
            except BehaviorTreeConflictError as exc:
                raise BehaviorPublishConflictError(f"behavior kind_token restamp rejected: {uri}") from exc
            self._require_read_back(document)
        return document

    def _stored_links(
        self,
        kind: BehaviorKind,
        from_uri: BehaviorURI,
        links: Sequence[tuple[BehaviorLinkType | str, BehaviorURI | str]],
    ) -> tuple[BehaviorStoredLink, ...]:
        if isinstance(links, str | bytes) or not isinstance(links, Sequence):
            raise TypeError("links must be a sequence of (link_type, target_uri) pairs")
        if links and kind is not BehaviorKind.OCCURRENCE:
            raise ValueError("only occurrences carry forward links")
        resolved: list[BehaviorStoredLink] = []
        for raw_type, raw_target in links:
            target = BehaviorURI.parse(raw_target)
            if target.to_address().kind is not BehaviorKind.OCCURRENCE:
                raise ValueError("behavior links must target occurrence documents")
            resolved.append(
                BehaviorStoredLink(
                    from_uri=from_uri,
                    to_uri=target,
                    link_type=BehaviorLinkType(raw_type),
                )
            )
        return tuple(resolved)

    @contextmanager
    def _fenced_uris(self, *uris: BehaviorURI) -> Iterator[None]:
        keys = tuple(sorted({self._keyspace.key(uri) for uri in uris}))
        with ExitStack() as stack:
            guards = tuple(
                stack.enter_context(
                    self._path_lock.acquire(key, ttl_seconds=self.config.lock_ttl_seconds)
                )
                for key in keys
            )
            with self._path_lock.fenced(guards):
                yield

    def _require_read_back(self, expected: BehaviorDocument) -> None:
        actual = self.tree.read(expected.address)
        if actual != expected:
            raise BehaviorReadBackError(
                f"behavior document read-back mismatch: {BehaviorURI.from_address(expected.address)}"
            )

    def _timestamp(self) -> datetime:
        value = self.clock()
        if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("behavior writer clock must return a timezone-aware datetime")
        return value.astimezone(UTC)


__all__ = [
    "BehaviorDocumentWriter",
    "BehaviorPublishConflictError",
    "BehaviorReadBackError",
    "BehaviorWriteConfig",
]
