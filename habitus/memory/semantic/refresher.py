"""按照目录依赖关系自下向上，用模板刷新可重建的 L0/L1。"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from habitus.infrastructure.store.contracts.path_lock import PathLock
from habitus.infrastructure.tree.lock_key import root_lock_prefix
from habitus.infrastructure.tree.overview import Overview, OverviewCard, OverviewChild
from habitus.memory.model import MemoryAddress, MemoryDirectory, MemoryLevel
from habitus.memory.schema import MemorySchemaRegistry
from habitus.memory.semantic.config import MemorySemanticConfig
from habitus.memory.semantic.model import (
    MemorySemanticRefreshResult,
    MemorySemanticRefreshStatus,
)
from habitus.memory.tree.store import MemoryTree


class MemorySemanticRefreshError(RuntimeError):
    """目录概览无法在边界内安全生成或发布。"""


class MemorySemanticRefresher:
    """读取目录里的直接 L2 与子目录，按模板拼出 L1 并确定性生成 L0；不调模型。"""

    def __init__(
        self,
        tree: MemoryTree,
        schema_registry: MemorySchemaRegistry,
        path_lock: PathLock,
        *,
        config: MemorySemanticConfig | None = None,
    ) -> None:
        self.tree = tree
        self.schema_registry = schema_registry
        self.path_lock = path_lock
        self.config = config or MemorySemanticConfig()
        self._lock_prefix = root_lock_prefix("memory-semantic", self.tree.root)

    def refresh_for(self, address: MemoryAddress) -> tuple[MemorySemanticRefreshResult, ...]:
        """从发生变化的 L2 父目录开始，依次刷新到记忆根目录。"""

        self.tree.initialize()
        return tuple(self.refresh_directory(directory) for directory in MemoryDirectory.for_address(address).lineage())

    def refresh_for_many(
        self,
        addresses: tuple[MemoryAddress, ...],
    ) -> tuple[MemorySemanticRefreshResult, ...]:
        """合并多个 L2 的祖先目录，并按最深目录优先各刷新一次。"""

        self.tree.initialize()
        directories = {
            directory for address in addresses for directory in MemoryDirectory.for_address(address).lineage()
        }
        ordered = tuple(sorted(directories, key=lambda directory: (-len(directory.parts), directory.parts)))
        return tuple(self.refresh_directory(directory) for directory in ordered)

    def refresh_directory(self, directory: MemoryDirectory) -> MemorySemanticRefreshResult:
        """在目录锁内读快照、拼模板、与现有层比较后写入；空目录删除派生层。"""

        with self._stable_directory(directory):
            if not self.tree.directory_exists(directory):
                return MemorySemanticRefreshResult(directory, MemorySemanticRefreshStatus.MISSING)
            overview = self._overview(directory)
            if overview.total_items == 0:
                deleted = self.tree.delete_layers(directory)
                return MemorySemanticRefreshResult(
                    directory,
                    MemorySemanticRefreshStatus.DELETED if deleted else MemorySemanticRefreshStatus.UNCHANGED,
                )
            rendered = overview.render()
            abstract = overview.abstract()
            if self._layers_equal(directory, abstract, rendered):
                return self._result(directory, MemorySemanticRefreshStatus.UNCHANGED)
            self.tree.write_layers(directory, abstract=abstract, overview=rendered)
            return self._result(directory, MemorySemanticRefreshStatus.WRITTEN)

    def rebuild(
        self,
        directory: MemoryDirectory | None = None,
    ) -> tuple[MemorySemanticRefreshResult, ...]:
        """有界遍历指定子树，并按照最深目录优先的顺序完整重建（补"写完卡片、刷概览之前崩了"的情况）。"""

        self.tree.initialize()
        root = directory or MemoryDirectory.root()
        if not self.tree.directory_exists(root):
            return ()
        pending = [root]
        discovered: list[MemoryDirectory] = []
        while pending:
            current = pending.pop()
            discovered.append(current)
            if len(discovered) > self.config.max_rebuild_directories:
                raise MemorySemanticRefreshError("memory semantic rebuild exceeded its configured directory bound")
            pending.extend(reversed(self.tree.child_directories(current, limit=self._enumeration_limit)))
        return tuple(self.refresh_directory(current) for current in reversed(discovered))

    @property
    def _enumeration_limit(self) -> int:
        return min(10_000, self.tree.tree_config.max_children_per_directory)

    def _overview(self, directory: MemoryDirectory) -> Overview:
        cards: list[OverviewCard] = []
        for address in self.tree.direct_addresses(directory, limit=self._enumeration_limit):
            document = self.tree.read(address)
            schema = self.schema_registry.get(document.kind)
            cards.append(
                OverviewCard(
                    filename=self.tree.path_for(address).name,
                    description=schema.overview_line(document.fields),
                    created_at=document.metadata.created_at,
                    updated_at=document.metadata.updated_at,
                    revision=document.metadata.revision,
                )
            )
        children = tuple(
            OverviewChild(
                child.parts[-1],
                len(self.tree.direct_addresses(child, limit=self._enumeration_limit))
                + len(self.tree.child_directories(child, limit=self._enumeration_limit)),
            )
            for child in self.tree.child_directories(directory, limit=self._enumeration_limit)
        )
        cards.sort(key=lambda card: (-card.updated_at.timestamp(), card.filename))
        listed = cards[: self.config.max_direct_entries]
        while listed:
            overview = Overview(directory.label, tuple(listed), children, len(cards) - len(listed))
            if len(overview.render()) <= self.config.max_overview_chars:
                return overview
            listed.pop()
        overview = Overview(directory.label, (), children, len(cards))
        if len(overview.render()) > self.config.max_overview_chars:
            raise MemorySemanticRefreshError("memory overview cannot fit its configured character bound")
        return overview

    @contextmanager
    def _stable_directory(self, directory: MemoryDirectory) -> Iterator[None]:
        wait_timeout = float(min(30, self.config.lock_ttl_seconds))
        with self.path_lock.acquire(
            self._lock_key(directory),
            ttl_seconds=self.config.lock_ttl_seconds,
            wait_timeout_seconds=wait_timeout,
            retry_delay_seconds=0.02,
        ) as directory_guard:
            with self.path_lock.fenced((directory_guard,)):
                yield

    def _layers_equal(self, directory: MemoryDirectory, abstract: str, overview: str) -> bool:
        if not self.tree.layer_exists(directory, MemoryLevel.ABSTRACT):
            return False
        if not self.tree.layer_exists(directory, MemoryLevel.OVERVIEW):
            return False
        return (
            self.tree.read_layer_bounded(
                directory,
                MemoryLevel.ABSTRACT,
                max_bytes=self.config.max_abstract_chars * 4 + 4,
            )
            == abstract
            and self.tree.read_layer_bounded(
                directory,
                MemoryLevel.OVERVIEW,
                max_bytes=self.config.max_overview_chars * 4 + 4,
            )
            == overview
        )

    def _result(self, directory: MemoryDirectory, status: MemorySemanticRefreshStatus) -> MemorySemanticRefreshResult:
        return MemorySemanticRefreshResult(
            directory=directory,
            status=status,
            abstract_path=self.tree.layer_path(directory, MemoryLevel.ABSTRACT),
            overview_path=self.tree.layer_path(directory, MemoryLevel.OVERVIEW),
        )

    def _lock_key(self, directory: MemoryDirectory) -> str:
        suffix = "/".join(directory.parts) or "/"
        return f"{self._lock_prefix}:{suffix}"


__all__ = [
    "MemorySemanticRefreshError",
    "MemorySemanticRefresher",
]
