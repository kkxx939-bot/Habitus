"""在统一执行栅栏内确保单个 Source Consumer 形成首次耐久 Outcome。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol

from conversation.source.fence import (
    ConversationConsumerExecutionFence,
    ConversationConsumerExecutionLease,
)
from conversation.source.model import ConversationSourceEnvelope, ConversationSourceError
from conversation.source.receipt import (
    ConversationConsumerOutcome,
    ConversationConsumerOutcomeState,
    ConversationConsumerOutcomeStore,
    ConversationConsumerRunDisposition,
    ConversationConsumerRunResult,
    ConversationSourceConsumer,
)
from conversation.source.state import (
    ConversationConsumerCorruptionError,
    ConversationConsumerDeliveryState,
    ConversationConsumerOutputStore,
    ConversationConsumerState,
    ConversationConsumerStateInspector,
)
from conversation.source.store import ConversationSourceStore


class ConversationMemoryPredecessorPendingError(ConversationSourceError):
    """同一 Conversation 的更早 Source 尚未形成 Memory Outcome。"""


class ConversationMemoryPredecessorBrokenError(ConversationSourceError):
    """同一 Conversation 的更早 Source 已损坏，后续 Memory 必须停止。"""


class ConversationDurableConsumer(Protocol):
    consumer: ConversationSourceConsumer
    processor_fingerprint: str

    @property
    def output_store(self) -> ConversationConsumerOutputStore: ...

    async def execute(
        self,
        envelope: ConversationSourceEnvelope,
        lease: ConversationConsumerExecutionLease,
    ) -> ConversationConsumerRunResult: ...


@dataclass(frozen=True)
class ConversationConsumerEnsureResult:
    outcome: ConversationConsumerOutcome
    runtime_result: object | None


class ConversationConsumerDelivery:
    """唯一实现锁内二次检查、Output 采用与首次 Outcome 提交的服务。"""

    def __init__(
        self,
        sources: ConversationSourceStore,
        outcomes: ConversationConsumerOutcomeStore,
        inspector: ConversationConsumerStateInspector,
        fence: ConversationConsumerExecutionFence,
        memory_consumer: ConversationDurableConsumer,
        behavior_projection_consumer: ConversationDurableConsumer,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if inspector.outcomes is not outcomes:
            raise ValueError("delivery and inspector must share the Outcome Store")
        if memory_consumer.consumer is not ConversationSourceConsumer.MEMORY:
            raise ValueError("memory_consumer has the wrong consumer identity")
        if behavior_projection_consumer.consumer is not ConversationSourceConsumer.BEHAVIOR_PROJECTION:
            raise ValueError("behavior_projection_consumer has the wrong consumer identity")
        self.sources = sources
        self.outcomes = outcomes
        self.inspector = inspector
        self.fence = fence
        self.memory_consumer = memory_consumer
        self.behavior_projection_consumer = behavior_projection_consumer
        self.clock = clock or (lambda: datetime.now(timezone.utc))

    async def ensure_outcome(
        self,
        source: ConversationSourceEnvelope,
        consumer: ConversationSourceConsumer,
    ) -> ConversationConsumerEnsureResult:
        implementation = self._consumer(consumer)
        state = self.inspector.inspect(
            source,
            implementation.consumer,
            implementation.processor_fingerprint,
            implementation.output_store,
        )
        terminal = self._terminal_result(state)
        if terminal is not None:
            return terminal
        self._raise_unrecoverable(state)
        if implementation.consumer is ConversationSourceConsumer.MEMORY:
            self._require_memory_predecessors(source)
        async with self.fence.acquire(source, implementation.consumer) as lease:
            locked_state = await lease.run_fenced(
                lambda: self.inspector.inspect(
                    source,
                    implementation.consumer,
                    implementation.processor_fingerprint,
                    implementation.output_store,
                )
            )
            terminal = self._terminal_result(locked_state)
            if terminal is not None:
                return terminal
            self._raise_unrecoverable(locked_state)
            if implementation.consumer is ConversationSourceConsumer.MEMORY:
                await lease.run_fenced(lambda: self._require_memory_predecessors(source))
            if locked_state.state is ConversationConsumerDeliveryState.OUTPUT_READY:
                outcome = await lease.run_fenced(
                    lambda: self._create_output_outcome(source, implementation, locked_state)
                )
                return ConversationConsumerEnsureResult(outcome, None)
            run = await implementation.execute(source, lease)
            lease.require_alive()
            outcome = await lease.run_fenced(
                lambda: self._commit_run(source, implementation, run)
            )
            return ConversationConsumerEnsureResult(outcome, run.runtime_result)

    def inspect(
        self,
        source: ConversationSourceEnvelope,
        consumer: ConversationSourceConsumer,
    ) -> ConversationConsumerState:
        implementation = self._consumer(consumer)
        return self.inspector.inspect(
            source,
            implementation.consumer,
            implementation.processor_fingerprint,
            implementation.output_store,
        )

    def restore_terminal(
        self,
        source: ConversationSourceEnvelope,
        consumer: ConversationSourceConsumer,
    ) -> object | None:
        implementation = self._consumer(consumer)
        state = self.inspector.require_terminal(
            source,
            implementation.consumer,
            implementation.processor_fingerprint,
            implementation.output_store,
        )
        if state.state is ConversationConsumerDeliveryState.SKIPPED:
            return None
        assert state.output is not None
        return implementation.output_store.restore(state.output)

    def _commit_run(
        self,
        source: ConversationSourceEnvelope,
        implementation: ConversationDurableConsumer,
        run: ConversationConsumerRunResult,
    ) -> ConversationConsumerOutcome:
        state = self.inspector.inspect(
            source,
            implementation.consumer,
            implementation.processor_fingerprint,
            implementation.output_store,
        )
        self._raise_unrecoverable(state)
        if run.disposition is ConversationConsumerRunDisposition.SKIPPED:
            if state.state is not ConversationConsumerDeliveryState.PENDING:
                raise ConversationConsumerCorruptionError("skipped run unexpectedly produced an output")
            candidate = ConversationConsumerOutcome.create(
                source_id=source.source_id,
                source_payload_digest=source.source_payload_digest,
                consumer=implementation.consumer,
                processor_fingerprint=implementation.processor_fingerprint,
                state=ConversationConsumerOutcomeState.SKIPPED,
                output_ref=None,
                skip_reason=run.skip_reason,
                completed_at=self.clock(),
            )
        else:
            if state.state is not ConversationConsumerDeliveryState.OUTPUT_READY or state.output is None:
                raise ConversationConsumerCorruptionError("consumer did not publish one valid output")
            output_ref = implementation.output_store.ref(state.output)
            if output_ref != run.output_ref:
                raise ConversationConsumerCorruptionError("run output reference differs from durable output")
            candidate = ConversationConsumerOutcome.create(
                source_id=source.source_id,
                source_payload_digest=source.source_payload_digest,
                consumer=implementation.consumer,
                processor_fingerprint=output_ref.processor_fingerprint,
                state=ConversationConsumerOutcomeState.COMMITTED,
                output_ref=output_ref,
                skip_reason=None,
                completed_at=self.clock(),
            )
        stored = self.outcomes.create_first(candidate)
        return self._require_created_terminal(source, implementation, stored)

    def _create_output_outcome(
        self,
        source: ConversationSourceEnvelope,
        implementation: ConversationDurableConsumer,
        state: ConversationConsumerState,
    ) -> ConversationConsumerOutcome:
        if state.state is not ConversationConsumerDeliveryState.OUTPUT_READY or state.output is None:
            raise ConversationSourceError("output adoption requires OUTPUT_READY")
        output_ref = implementation.output_store.ref(state.output)
        candidate = ConversationConsumerOutcome.create(
            source_id=source.source_id,
            source_payload_digest=source.source_payload_digest,
            consumer=implementation.consumer,
            processor_fingerprint=output_ref.processor_fingerprint,
            state=ConversationConsumerOutcomeState.COMMITTED,
            output_ref=output_ref,
            skip_reason=None,
            completed_at=self.clock(),
        )
        stored = self.outcomes.create_first(candidate)
        return self._require_created_terminal(source, implementation, stored)

    def _require_created_terminal(
        self,
        source: ConversationSourceEnvelope,
        implementation: ConversationDurableConsumer,
        stored: ConversationConsumerOutcome,
    ) -> ConversationConsumerOutcome:
        terminal = self.inspector.require_terminal(
            source,
            implementation.consumer,
            implementation.processor_fingerprint,
            implementation.output_store,
        )
        if terminal.outcome is None or terminal.outcome.outcome_record_digest != stored.outcome_record_digest:
            raise ConversationConsumerCorruptionError("first outcome was not durably read back")
        return terminal.outcome

    def _require_memory_predecessors(self, source: ConversationSourceEnvelope) -> None:
        current_key = self._source_order(source)
        for predecessor in self.sources.list():
            if predecessor.source_id == source.source_id:
                continue
            if (
                predecessor.conversation_id != source.conversation_id
                or predecessor.started_on != source.started_on
                or self._source_order(predecessor) >= current_key
            ):
                continue
            state = self.inspect(predecessor, ConversationSourceConsumer.MEMORY)
            if state.state is ConversationConsumerDeliveryState.COMMITTED:
                continue
            if state.state in {
                ConversationConsumerDeliveryState.BROKEN_OUTCOME,
                ConversationConsumerDeliveryState.CORRUPTED,
                ConversationConsumerDeliveryState.SKIPPED,
            }:
                raise ConversationMemoryPredecessorBrokenError(
                    f"earlier Memory Source {predecessor.source_id} is {state.state.value}"
                )
            raise ConversationMemoryPredecessorPendingError(
                f"earlier Memory Source {predecessor.source_id} has no terminal Outcome"
            )

    def _consumer(self, consumer: ConversationSourceConsumer) -> ConversationDurableConsumer:
        resolved = ConversationSourceConsumer(consumer)
        if resolved is ConversationSourceConsumer.MEMORY:
            return self.memory_consumer
        return self.behavior_projection_consumer

    @staticmethod
    def _source_order(source: ConversationSourceEnvelope) -> tuple[int, int, str]:
        return (source.batch.start_sequence, source.batch.end_sequence, source.source_id)

    @staticmethod
    def _terminal_result(
        state: ConversationConsumerState,
    ) -> ConversationConsumerEnsureResult | None:
        if state.state not in {
            ConversationConsumerDeliveryState.COMMITTED,
            ConversationConsumerDeliveryState.SKIPPED,
        }:
            return None
        assert state.outcome is not None
        return ConversationConsumerEnsureResult(state.outcome, None)

    @staticmethod
    def _raise_unrecoverable(state: ConversationConsumerState) -> None:
        if state.state in {
            ConversationConsumerDeliveryState.BROKEN_OUTCOME,
            ConversationConsumerDeliveryState.CORRUPTED,
        }:
            assert state.error is not None
            raise state.error


__all__ = [
    "ConversationConsumerDelivery",
    "ConversationConsumerEnsureResult",
    "ConversationDurableConsumer",
    "ConversationMemoryPredecessorBrokenError",
    "ConversationMemoryPredecessorPendingError",
]
