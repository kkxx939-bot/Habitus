"""两种消费者输出存储共用的"删掉一份源的全部输出"：释放会话源时用，不依赖信封还在。"""

from __future__ import annotations

from pathlib import Path

from habitus.conversation.source.address import remove_if_empty
from habitus.conversation.source.model import ConversationSourceError
from habitus.infrastructure.store.filesystem import DurablePathIntegrityError, durable_unlink, list_real_directory


def discard_output_directory(root: Path, directory: Path, *, max_entries: int) -> int:
    """删 ``directory`` 里的全部普通文件（含写到一半的临时文件），再删空目录与空的上一级；返回删了几个。"""

    try:
        entries = list_real_directory(directory, artifact_root=root, max_entries=max_entries)
    except DurablePathIntegrityError as exc:
        raise ConversationSourceError("consumer output directory is invalid or exceeds its bound") from exc
    removed = 0
    for entry in entries:
        if not entry.is_file():
            raise ConversationSourceError("consumer output directory contains an unsupported entry")
        removed += int(durable_unlink(directory / entry.name, artifact_root=root))
    remove_if_empty(directory, artifact_root=root)
    remove_if_empty(directory.parent, artifact_root=root)
    return removed


__all__ = ["discard_output_directory"]
