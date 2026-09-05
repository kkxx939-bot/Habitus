"""为一次 benchmark 生成不会污染正式数据的独立存储与远程集合配置。"""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

from habitus.config import HabitusConfig


def isolated_config(
    config: HabitusConfig,
    *,
    storage_root: str | Path,
    collection_scope: str,
) -> HabitusConfig:
    """保持模型和运维参数不变，只替换存储根与两个远程集合。"""

    if not isinstance(config, HabitusConfig):
        raise TypeError("config must be HabitusConfig")
    root = Path(storage_root).expanduser().resolve()
    if not isinstance(collection_scope, str) or not collection_scope.strip():
        raise ValueError("collection_scope must be non-empty text")
    scope = hashlib.sha256(collection_scope.strip().encode("utf-8")).hexdigest()[:20]
    memory_store = replace(
        config.memory.vector_store,
        collection=f"habitus-benchmark-{scope}-memory",
    )
    summary_store = replace(
        config.conversation.summary_vector_store,
        collection=f"habitus-benchmark-{scope}-summary",
    )
    return replace(
        config,
        storage=replace(config.storage, root=root),
        memory=replace(config.memory, vector_store=memory_store),
        conversation=replace(config.conversation, summary_vector_store=summary_store),
    )


def require_empty_directory(path: str | Path, *, label: str) -> Path:
    """创建或确认一个空目录，防止性能结果混入上次运行状态。"""

    selected = Path(path).expanduser().resolve()
    if selected.exists():
        if not selected.is_dir():
            raise ValueError(f"{label} must be a directory")
        if any(selected.iterdir()):
            raise ValueError(f"{label} must be empty")
    else:
        selected.mkdir(parents=True)
    return selected


__all__ = ["isolated_config", "require_empty_directory"]
