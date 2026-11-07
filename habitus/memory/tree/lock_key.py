"""记忆树的锁键：同一记忆根上的文档写入、派生刷新与生命周期状态迁移共用的命名空间。"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

from habitus.memory.uri import MemoryURI


def memory_root_digest(root: Path) -> str:
    """返回记忆根的稳定短摘要；锁键靠它区分不同记忆根，又不把路径写进键里。"""

    normalized = root.expanduser().absolute().resolve(strict=False)
    return hashlib.sha256(str(normalized).encode("utf-8")).hexdigest()[:24]


def memory_root_lock_prefix(namespace: str, root: Path) -> str:
    """返回 ``<namespace>:<记忆根摘要>``；同一记忆根下各类锁键都以它开头。"""

    if not namespace or ":" in namespace:
        raise ValueError("lock namespace must be non-empty text without ':'")
    return f"{namespace}:{memory_root_digest(root)}"


@dataclass(frozen=True)
class MemoryDocumentLockKeyspace:
    """使同一记忆根下同一 L2 URI 始终使用同一锁键。"""

    root: Path
    _prefix: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        normalized = self.root.expanduser().absolute().resolve(strict=False)
        object.__setattr__(self, "root", normalized)
        object.__setattr__(self, "_prefix", memory_root_lock_prefix("memory-document", normalized))

    def key(self, uri: MemoryURI | str) -> str:
        """返回指定 L2 URI 的稳定且不泄漏路径的锁键。"""

        parsed = MemoryURI.parse(uri)
        parsed.to_address()
        uri_digest = hashlib.sha256(str(parsed).encode("utf-8")).hexdigest()
        return f"{self._prefix}:{uri_digest}"

    def transaction_key(self) -> str:
        """返回同一记忆根全部多文档发布共享的串行化锁键。"""

        return f"{self._prefix}:transaction"


__all__ = ["MemoryDocumentLockKeyspace", "memory_root_digest", "memory_root_lock_prefix"]
