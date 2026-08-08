"""按 Source/Consumer 独立恢复缺失的首次耐久 Outcome。"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from conversation.source.delivery import ConversationConsumerDelivery
from conversation.source.model import ConversationSourceEnvelope
from conversation.source.receipt import ConversationConsumerOutcome, ConversationSourceConsumer
from conversation.source.state import ConversationConsumerDeliveryState
from conversation.source.store import ConversationSourceStore


@dataclass(frozen=True)
class ConversationSourceRecoveryEntry:
    envelope: ConversationSourceEnvelope
    consumer: ConversationSourceConsumer
    state: ConversationConsumerDeliveryState


@dataclass(frozen=True)
class ConversationSourceRecoveryResult:
    envelope: ConversationSourceEnvelope
    consumer: ConversationSourceConsumer
    outcome: ConversationConsumerOutcome | None
    error: BaseException | None

    def __post_init__(self) -> None:
        if (self.outcome is None) == (self.error is None):
            raise ValueError("recovery result requires exactly one Outcome or error")


class ConversationSourceRecovery:
    """只处理 PENDING/OUTPUT_READY；终态保持，损坏状态 fail-closed。"""

    def __init__(
        self,
        sources: ConversationSourceStore,
        delivery: ConversationConsumerDelivery,
        *,
        batch_size: int,
    ) -> None:
        if delivery.sources is not sources:
            raise ValueError("source recovery and delivery must share the Source Store")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        self.sources = sources
        self.delivery = delivery
        self.batch_size = batch_size

    def pending(self) -> tuple[ConversationSourceRecoveryEntry, ...]:
        entries: list[ConversationSourceRecoveryEntry] = []
        for envelope in self.sources.list():
            for consumer in ConversationSourceConsumer:
                state = self.delivery.inspect(envelope, consumer)
                if state.state not in {
                    ConversationConsumerDeliveryState.COMMITTED,
                    ConversationConsumerDeliveryState.SKIPPED,
                }:
                    entries.append(ConversationSourceRecoveryEntry(envelope, consumer, state.state))
        return tuple(entries)

    async def recover_pending(self) -> tuple[ConversationSourceRecoveryResult, ...]:
        entries = self.pending()[: self.batch_size]
        behavior = tuple(
            entry for entry in entries if entry.consumer is ConversationSourceConsumer.BEHAVIOR_PROJECTION
        )
        memory = tuple(
            sorted(
                (entry for entry in entries if entry.consumer is ConversationSourceConsumer.MEMORY),
                key=lambda entry: (
                    entry.envelope.conversation_id,
                    entry.envelope.started_on,
                    entry.envelope.batch.start_sequence,
                    entry.envelope.batch.end_sequence,
                    entry.envelope.source_id,
                ),
            )
        )
        behavior_tasks = tuple(asyncio.create_task(self._recover(entry)) for entry in behavior)
        results: list[ConversationSourceRecoveryResult] = []
        for entry in memory:
            results.append(await self._recover(entry))
        if behavior_tasks:
            results.extend(await asyncio.gather(*behavior_tasks))
        return tuple(results)

    async def _recover(
        self, entry: ConversationSourceRecoveryEntry
    ) -> ConversationSourceRecoveryResult:
        if entry.state in {
            ConversationConsumerDeliveryState.BROKEN_OUTCOME,
            ConversationConsumerDeliveryState.CORRUPTED,
        }:
            state = self.delivery.inspect(entry.envelope, entry.consumer)
            assert state.error is not None
            return ConversationSourceRecoveryResult(
                entry.envelope,
                entry.consumer,
                None,
                state.error,
            )
        try:
            ensured = await self.delivery.ensure_outcome(entry.envelope, entry.consumer)
        except BaseException as exc:
            return ConversationSourceRecoveryResult(entry.envelope, entry.consumer, None, exc)
        return ConversationSourceRecoveryResult(
            entry.envelope,
            entry.consumer,
            ensured.outcome,
            None,
        )


__all__ = [
    "ConversationSourceRecovery",
    "ConversationSourceRecoveryEntry",
    "ConversationSourceRecoveryResult",
]
