"""同一耐久 Source 上独立启动两个 Consumer。"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol

from conversation.source.model import ConversationSourceEnvelope
from conversation.source.receipt import (
    ConversationConsumerReceipt,
    ConversationConsumerReceiptState,
    ConversationSourceConsumer,
    ConversationSourceReceiptStore,
)
from conversation.source.store import ConversationSourceStore


@dataclass(frozen=True)
class ConversationConsumerExecution:
    """Consumer 输出与其终态回执所需的确定性身份。"""

    state: ConversationConsumerReceiptState
    result: object | None
    result_id: str | None
    result_digest: str | None

    def __post_init__(self) -> None:
        if not isinstance(self.state, ConversationConsumerReceiptState):
            raise TypeError("state must be ConversationConsumerReceiptState")
        if self.state is ConversationConsumerReceiptState.SUCCEEDED:
            if not isinstance(self.result_id, str) or not isinstance(self.result_digest, str):
                raise ValueError("successful execution requires result identity and digest")
        elif self.result is not None or self.result_id is not None or self.result_digest is not None:
            raise ValueError("skipped execution cannot contain a result")


class ConversationEnvelopeConsumer(Protocol):
    consumer: ConversationSourceConsumer

    async def consume(self, envelope: ConversationSourceEnvelope) -> ConversationConsumerExecution: ...

    async def completed(
        self,
        envelope: ConversationSourceEnvelope,
        receipt: ConversationConsumerReceipt,
    ) -> object | None: ...


@dataclass(frozen=True)
class ConversationConsumerCall:
    consumer: ConversationSourceConsumer
    run: Callable[[], Awaitable[ConversationConsumerExecution]]


@dataclass(frozen=True)
class ConversationConsumerDispatchOutcome:
    consumer: ConversationSourceConsumer
    execution: ConversationConsumerExecution | None
    error: BaseException | None

    def __post_init__(self) -> None:
        if (self.execution is None) == (self.error is None):
            raise ValueError("consumer dispatch outcome requires exactly one execution or error")


class ConversationConsumerDispatcher(Protocol):
    async def dispatch(
        self,
        calls: tuple[ConversationConsumerCall, ...],
    ) -> tuple[ConversationConsumerDispatchOutcome, ...]: ...


class AsyncioConversationConsumerDispatcher:
    """默认并发机制；Coordinator 只依赖可替换的 Dispatcher 契约。"""

    async def dispatch(
        self,
        calls: tuple[ConversationConsumerCall, ...],
    ) -> tuple[ConversationConsumerDispatchOutcome, ...]:
        async def invoke(call: ConversationConsumerCall) -> ConversationConsumerExecution:
            return await call.run()

        tasks: tuple[asyncio.Task[ConversationConsumerExecution], ...] = tuple(
            asyncio.create_task(invoke(call)) for call in calls
        )
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return tuple(
            ConversationConsumerDispatchOutcome(
                consumer=call.consumer,
                execution=None if isinstance(result, BaseException) else result,
                error=result if isinstance(result, BaseException) else None,
            )
            for call, result in zip(calls, results, strict=True)
        )


@dataclass(frozen=True)
class ConversationSourceDispatchResult:
    envelope: ConversationSourceEnvelope
    memory_result: object | None
    behavior_projection_result: object | None
    memory_receipt: ConversationConsumerReceipt | None
    behavior_projection_receipt: ConversationConsumerReceipt | None
    memory_error: BaseException | None
    behavior_projection_error: BaseException | None


class ConversationSourceCoordinator:
    """先耐久保存 Source，再从同一对象独立分发并立即保存各自回执。"""

    def __init__(
        self,
        sources: ConversationSourceStore,
        receipts: ConversationSourceReceiptStore,
        memory_consumer: ConversationEnvelopeConsumer,
        behavior_projection_consumer: ConversationEnvelopeConsumer,
        *,
        dispatcher: ConversationConsumerDispatcher | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if memory_consumer.consumer is not ConversationSourceConsumer.MEMORY:
            raise ValueError("memory_consumer has the wrong consumer identity")
        if behavior_projection_consumer.consumer is not ConversationSourceConsumer.BEHAVIOR_PROJECTION:
            raise ValueError("behavior_projection_consumer has the wrong consumer identity")
        self.sources = sources
        self.receipts = receipts
        self.memory_consumer = memory_consumer
        self.behavior_projection_consumer = behavior_projection_consumer
        self.dispatcher = dispatcher or AsyncioConversationConsumerDispatcher()
        self.clock = clock or (lambda: datetime.now(timezone.utc))

    async def dispatch(self, envelope: ConversationSourceEnvelope) -> ConversationSourceDispatchResult:
        durable = self.sources.put(envelope)
        consumers = (self.memory_consumer, self.behavior_projection_consumer)
        restored: dict[ConversationSourceConsumer, object | None] = {}
        receipts: dict[ConversationSourceConsumer, ConversationConsumerReceipt] = {}
        errors: dict[ConversationSourceConsumer, BaseException] = {}
        calls: list[ConversationConsumerCall] = []

        for consumer in consumers:
            receipt = self.receipts.read(durable.source_id, consumer.consumer)
            if receipt is not None:
                if receipt.source_digest != durable.content_digest:
                    raise ValueError("consumer receipt belongs to different source content")
                receipts[consumer.consumer] = receipt
            calls.append(self._consumer_call(consumer, durable, receipt))

        outcomes = await self.dispatcher.dispatch(tuple(calls))
        expected_consumers = tuple(call.consumer for call in calls)
        if tuple(outcome.consumer for outcome in outcomes) != expected_consumers:
            raise RuntimeError("consumer dispatcher returned incomplete or reordered outcomes")
        for outcome in outcomes:
            if outcome.error is not None:
                errors[outcome.consumer] = outcome.error
                continue
            assert outcome.execution is not None
            restored[outcome.consumer] = outcome.execution.result
            receipt = self.receipts.read(durable.source_id, outcome.consumer)
            if receipt is None:
                errors[outcome.consumer] = RuntimeError("consumer completed without a terminal receipt")
            else:
                receipts[outcome.consumer] = receipt

        return ConversationSourceDispatchResult(
            envelope=durable,
            memory_result=restored.get(ConversationSourceConsumer.MEMORY),
            behavior_projection_result=restored.get(ConversationSourceConsumer.BEHAVIOR_PROJECTION),
            memory_receipt=receipts.get(ConversationSourceConsumer.MEMORY),
            behavior_projection_receipt=receipts.get(ConversationSourceConsumer.BEHAVIOR_PROJECTION),
            memory_error=errors.get(ConversationSourceConsumer.MEMORY),
            behavior_projection_error=errors.get(ConversationSourceConsumer.BEHAVIOR_PROJECTION),
        )

    def _consumer_call(
        self,
        consumer: ConversationEnvelopeConsumer,
        envelope: ConversationSourceEnvelope,
        receipt: ConversationConsumerReceipt | None,
    ) -> ConversationConsumerCall:
        async def run() -> ConversationConsumerExecution:
            if receipt is None:
                return await self._consume_and_receipt(consumer, envelope)
            result = await consumer.completed(envelope, receipt)
            return ConversationConsumerExecution(
                state=receipt.state,
                result=result,
                result_id=receipt.result_id,
                result_digest=receipt.result_digest,
            )

        return ConversationConsumerCall(consumer=consumer.consumer, run=run)

    async def _consume_and_receipt(
        self,
        consumer: ConversationEnvelopeConsumer,
        envelope: ConversationSourceEnvelope,
    ) -> ConversationConsumerExecution:
        execution = await consumer.consume(envelope)
        receipt = ConversationConsumerReceipt.create(
            source_id=envelope.source_id,
            source_digest=envelope.content_digest,
            consumer=consumer.consumer,
            state=execution.state,
            result_id=execution.result_id,
            result_digest=execution.result_digest,
            completed_at=self.clock(),
        )
        self.receipts.put(receipt)
        return execution
