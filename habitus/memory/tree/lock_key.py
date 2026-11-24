"""卡片文档的锁键：同一树根上同一张卡片的地址始终是同一把锁（派生写法见 ``infrastructure/tree/lock_key``）。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from habitus.infrastructure.tree.lock_key import root_lock_prefix, uri_lock_key
from habitus.memory.uri import MemoryURI


@dataclass(frozen=True)
class MemoryDocumentLockKeyspace:
    """使同一树根下同一 L2 URI 始终使用同一锁键。"""

    root: Path
    _prefix: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        normalized = self.root.expanduser().absolute().resolve(strict=False)
        object.__setattr__(self, "root", normalized)
        object.__setattr__(self, "_prefix", root_lock_prefix("memory-document", normalized))

    def key(self, uri: MemoryURI | str) -> str:
        """返回指定 L2 URI 的稳定且不泄漏路径的锁键。"""

        parsed = MemoryURI.parse(uri)
        parsed.to_address()
        return uri_lock_key(self._prefix, str(parsed))


__all__ = ["MemoryDocumentLockKeyspace"]
