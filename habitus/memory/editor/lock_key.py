"""为所有 Memory Editor 文档写入生成共享的排他锁键。"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

from habitus.memory.uri import MemoryURI


@dataclass(frozen=True)
class MemoryDocumentLockKeyspace:
    """使同一记忆根下同一 L2 URI 始终使用同一锁键。"""

    root: Path
    _prefix: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.root, Path):
            raise TypeError("root must be a Path")
        normalized = self.root.expanduser().absolute().resolve(strict=False)
        object.__setattr__(self, "root", normalized)
        root_digest = hashlib.sha256(str(normalized).encode("utf-8")).hexdigest()[:24]
        object.__setattr__(self, "_prefix", f"memory-document:{root_digest}")

    def key(self, uri: MemoryURI | str) -> str:
        """返回指定 L2 URI 的稳定且不泄漏路径的锁键。"""

        parsed = MemoryURI.parse(uri)
        parsed.to_address()
        uri_digest = hashlib.sha256(str(parsed).encode("utf-8")).hexdigest()
        return f"{self._prefix}:{uri_digest}"

    def transaction_key(self) -> str:
        """返回同一记忆根全部多文档发布共享的串行化锁键。"""

        return f"{self._prefix}:transaction"


__all__ = ["MemoryDocumentLockKeyspace"]
