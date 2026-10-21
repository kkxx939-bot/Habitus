"""行为侧"处理完了没有"的账坏了的会话源（BROKEN_OUTCOME / CORRUPTED，26 号清单 2.1.7；裁定 67 第 5 条）。

这类源的信封是好的，坏的是行为侧的结果 / 输出对不上——它不会自愈，不处置的话过不了释放门槛第 2 条，同一地址后面的源按
第 3 条也一份都放不了。处置：交给行为侧补救（``salvager``，组合根注入：逐轮看写没写过，没写过的落「没读懂」、那天还没
结算的进每晚重试的堆），然后记一个"已补救"标记，Behavior 这一侧就按终态算、信封可以释放；释放时标记一起删。
先补救、后记标记：中途崩溃，下一轮这份源还是坏的，再补救一遍（逐轮幂等）。

Memory 是有序 Consumer，坏了要人工修复，不在这里补救。
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path

from habitus.conversation.source.model import ConversationSourceEnvelope
from habitus.conversation.source.receipt import ConversationSourceConsumer
from habitus.infrastructure.store.filesystem import (
    DurablePathIntegrityError,
    ImmutableArtifactConflictError,
    atomic_create_bytes,
    durable_unlink,
)

SALVAGED_CONSUMERS = frozenset({ConversationSourceConsumer.BEHAVIOR_SESSION})


class SourceSalvage:
    def __init__(self, root: str | Path, *, salvager: Callable[[ConversationSourceEnvelope], object]) -> None:
        if not callable(salvager):
            raise TypeError("salvager must be callable")
        self.root = Path(root).expanduser().absolute()
        self.directory = self.root / "lifecycle" / "salvaged"
        self.salvager = salvager

    def salvages(self, consumer: ConversationSourceConsumer) -> bool:
        return ConversationSourceConsumer(consumer) in SALVAGED_CONSUMERS

    def salvage(self, envelope: ConversationSourceEnvelope, consumer: ConversationSourceConsumer) -> None:
        if not self.salvages(consumer):
            raise ValueError(f"{ConversationSourceConsumer(consumer).value} sources are not salvaged")
        self.salvager(envelope)
        payload = json.dumps(
            {"source_id": envelope.source_id, "consumer": ConversationSourceConsumer(consumer).value}, sort_keys=True
        ).encode("utf-8")
        try:
            atomic_create_bytes(self._path(envelope.source_id, consumer), payload, artifact_root=self.root)
        except (ImmutableArtifactConflictError, DurablePathIntegrityError):
            pass  # 已经记过

    def contains(self, source_id: str, consumer: ConversationSourceConsumer) -> bool:
        return self.salvages(consumer) and self._path(source_id, consumer).is_file()

    def discard(self, source_id: str) -> int:
        """这份源释放了：它的标记一起删。"""

        return sum(
            1
            for consumer in SALVAGED_CONSUMERS
            if durable_unlink(self._path(source_id, consumer), artifact_root=self.root)
        )

    def _path(self, source_id: str, consumer: ConversationSourceConsumer) -> Path:
        key = f"{source_id}/{ConversationSourceConsumer(consumer).value}"
        return self.directory / f"{hashlib.sha256(key.encode('utf-8')).hexdigest()[:32]}.json"


__all__ = ["SALVAGED_CONSUMERS", "SourceSalvage"]
