"""会话源的地址：同一个 (conversation_id, started_on) 下的源严格按序交付，释放、有序检查、水位都按地址算。

conversation 包不 import memory，所以不复用 memory 的地址类，只在这里定义同一对值。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from habitus.conversation.source.model import ConversationSourceEnvelope, ConversationSourceError
from habitus.foundation.integrity import canonical_digest
from habitus.infrastructure.store.filesystem import durable_rmdir

_ADDRESS_SCHEMA = "conversation_source_address_v1"


@dataclass(frozen=True, order=True)
class SourceAddress:
    conversation_id: str
    started_on: date

    def __post_init__(self) -> None:
        if not isinstance(self.conversation_id, str) or not self.conversation_id:
            raise ConversationSourceError("source address needs a non-empty conversation_id")
        if not isinstance(self.started_on, date):
            raise ConversationSourceError("source address needs a calendar date")

    @classmethod
    def of(cls, envelope: ConversationSourceEnvelope) -> SourceAddress:
        return cls(envelope.conversation_id, envelope.started_on)

    @property
    def digest(self) -> str:
        """目录名：会话号可以是任意文本，不直接当路径。"""

        return canonical_digest(
            {
                "schema": _ADDRESS_SCHEMA,
                "conversation_id": self.conversation_id,
                "started_on": self.started_on.isoformat(),
            }
        )[:32]

    def to_dict(self) -> dict[str, str]:
        return {"conversation_id": self.conversation_id, "started_on": self.started_on.isoformat()}

    @classmethod
    def from_dict(cls, value: object) -> SourceAddress:
        if not isinstance(value, dict):
            raise ConversationSourceError("source address must be an object")
        try:
            return cls(str(value["conversation_id"]), date.fromisoformat(str(value["started_on"])))
        except (KeyError, ValueError) as exc:
            raise ConversationSourceError("source address is invalid") from exc


def entry_count(directory: Path) -> int:
    """目录里有几项（不在就是 0）：写入时的容量闸只要个数，不解码、不校验每一项。"""

    try:
        with os.scandir(directory) as entries:
            return sum(1 for _entry in entries)
    except FileNotFoundError:
        return 0


def remove_if_empty(path: Path, *, artifact_root: Path) -> bool:
    """目录空了就删；不空或不存在都返回 False，不当错误。"""

    try:
        return durable_rmdir(path, artifact_root=artifact_root)
    except OSError:
        return False  # 还有别的项（并发写入或下一份源）：留着


__all__ = ["SourceAddress", "remove_if_empty"]
