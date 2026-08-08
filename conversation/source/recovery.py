"""只补跑缺少终态 Consumer Receipt 的 Conversation Source。"""

from __future__ import annotations

from dataclasses import dataclass

from conversation.source.coordinator import ConversationSourceCoordinator, ConversationSourceDispatchResult
from conversation.source.model import ConversationSourceEnvelope
from conversation.source.receipt import ConversationSourceConsumer, ConversationSourceReceiptStore
from conversation.source.store import ConversationSourceStore


@dataclass(frozen=True)
class ConversationSourceRecoveryEntry:
    envelope: ConversationSourceEnvelope
    missing_consumers: tuple[ConversationSourceConsumer, ...]


class ConversationSourceRecovery:
    def __init__(
        self,
        sources: ConversationSourceStore,
        receipts: ConversationSourceReceiptStore,
        coordinator: ConversationSourceCoordinator,
    ) -> None:
        if coordinator.sources is not sources or coordinator.receipts is not receipts:
            raise ValueError("source recovery must share coordinator stores")
        self.sources = sources
        self.receipts = receipts
        self.coordinator = coordinator

    def pending(self) -> tuple[ConversationSourceRecoveryEntry, ...]:
        entries: list[ConversationSourceRecoveryEntry] = []
        for envelope in self.sources.list():
            missing = tuple(
                consumer
                for consumer in ConversationSourceConsumer
                if self.receipts.read(envelope.source_id, consumer) is None
            )
            if missing:
                entries.append(ConversationSourceRecoveryEntry(envelope, missing))
        return tuple(entries)

    async def recover_pending(self) -> tuple[ConversationSourceDispatchResult, ...]:
        results: list[ConversationSourceDispatchResult] = []
        for entry in self.pending():
            results.append(await self.coordinator.dispatch(entry.envelope))
        return tuple(results)
