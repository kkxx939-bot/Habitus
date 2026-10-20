"""Behavior 坏掉的会话源进隔离区（26 号清单 2.1.7）：信封副本落进隔离区，Behavior 这一侧就算终态。

Behavior 的 Outcome 损坏（BROKEN_OUTCOME / CORRUPTED）不会自愈；不隔离的话它过不了释放门槛第 2 条，同一地址后面的
源按第 3 条也一份都放不了。Memory 是有序 Consumer，坏了要人工修复，不在这里隔离。
"""

from __future__ import annotations

from habitus.conversation.source.model import ConversationSourceEnvelope, encode_durable_record
from habitus.conversation.source.receipt import ConversationSourceConsumer
from habitus.infrastructure.lifecycle import Quarantine

ISOLATED_CONSUMERS = frozenset({ConversationSourceConsumer.BEHAVIOR_SESSION})


class SourceQuarantine:
    def __init__(self, quarantine: Quarantine) -> None:
        if not isinstance(quarantine, Quarantine):
            raise TypeError("quarantine must be a Quarantine")
        self.quarantine = quarantine

    def isolates(self, consumer: ConversationSourceConsumer) -> bool:
        return ConversationSourceConsumer(consumer) in ISOLATED_CONSUMERS

    def isolate(
        self, envelope: ConversationSourceEnvelope, consumer: ConversationSourceConsumer, *, reason: str
    ) -> bool:
        if not self.isolates(consumer):
            raise ValueError(f"{ConversationSourceConsumer(consumer).value} sources are not quarantined")
        payload = encode_durable_record(
            envelope.to_dict(), max_bytes=self.quarantine.max_payload_bytes, label="quarantined source"
        )
        return self.quarantine.put(
            _item_id(envelope.source_id, consumer),
            payload,
            reason=reason,
            meta={"source_id": envelope.source_id, "consumer": ConversationSourceConsumer(consumer).value},
        )

    def contains(self, source_id: str, consumer: ConversationSourceConsumer) -> bool:
        return self.isolates(consumer) and self.quarantine.contains(_item_id(source_id, consumer))


def _item_id(source_id: str, consumer: ConversationSourceConsumer) -> str:
    return f"{source_id}/{ConversationSourceConsumer(consumer).value}"


__all__ = ["ISOLATED_CONSUMERS", "SourceQuarantine"]
