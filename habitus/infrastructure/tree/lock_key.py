"""由树根派生的锁键：同一树根上的文档写入、派生刷新、归约等各类锁共用这一种写法，不把路径写进键里。"""

from __future__ import annotations

import hashlib
from pathlib import Path


def root_digest(root: Path) -> str:
    """树根的稳定短摘要。"""

    normalized = root.expanduser().absolute().resolve(strict=False)
    return hashlib.sha256(str(normalized).encode("utf-8")).hexdigest()[:24]


def root_lock_prefix(namespace: str, root: Path) -> str:
    """``<namespace>:<树根摘要>``；同一树根下的各类锁键都以它开头。"""

    if not namespace or ":" in namespace:
        raise ValueError("lock namespace must be non-empty text without ':'")
    return f"{namespace}:{root_digest(root)}"


def uri_lock_key(prefix: str, uri: str) -> str:
    """``<前缀>:<sha256(规范地址)>``：同一地址始终是同一把锁。"""

    return f"{prefix}:{hashlib.sha256(uri.encode('utf-8')).hexdigest()}"


__all__ = ["root_digest", "root_lock_prefix", "uri_lock_key"]
