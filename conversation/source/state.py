"""Source、Outcome 与 Consumer Output 的联合耐久状态判定。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from conversation.source.model import ConversationSourceEnvelope, ConversationSourceError
from conversation.source.receipt import (
    ConsumerOutputRef,
    ConversationConsumerOutcome,
    ConversationConsumerOutcomeState,
    ConversationConsumerOutcomeStore,
    ConversationSourceConsumer,
)


class ConversationConsumerDeliveryState(str, Enum):
    PENDING = "pending"
    OUTPUT_READY = "output_ready"
    COMMITTED = "committed"
    SKIPPED = "skipped"
    BROKEN_OUTCOME = "broken_outcome"
    CORRUPTED = "corrupted"


class ConversationConsumerBrokenOutcomeError(ConversationSourceError):
    """Outcome 声明已提交，但引用的不可变 Output 已经缺失。"""


class ConversationConsumerCorruptionError(ConversationSourceError):
    """Source、Outcome 或 Output 的身份和摘要不再一致。"""


class ConversationConsumerOutputStore(Protocol):
    consumer: ConversationSourceConsumer

    def expected_output_id(
        self, source: ConversationSourceEnvelope, processor_fingerprint: str
    ) -> str: ...

    def read(self, source: ConversationSourceEnvelope, output_id: str) -> object | None: ...

    def list(self, source: ConversationSourceEnvelope) -> tuple[object, ...]: ...

    def ref(self, output: object) -> ConsumerOutputRef: ...

    def restore(self, output: object) -> object: ...

    def remove(self, source: ConversationSourceEnvelope, output_id: str) -> bool: ...


@dataclass(frozen=True)
class ConversationConsumerState:
    state: ConversationConsumerDeliveryState
    outcome: ConversationConsumerOutcome | None = None
    output: object | None = None
    error: ConversationSourceError | None = None

    def __post_init__(self) -> None:
        if self.state in {
            ConversationConsumerDeliveryState.BROKEN_OUTCOME,
            ConversationConsumerDeliveryState.CORRUPTED,
        }:
            if self.error is None:
                raise ValueError("broken or corrupted state requires an error")
        elif self.error is not None:
            raise ValueError("healthy delivery state cannot contain an error")


class ConversationConsumerStateInspector:
    """联合校验 Source、Outcome 与 Output；唯一拥有状态组合判定。"""

    def __init__(self, outcomes: ConversationConsumerOutcomeStore) -> None:
        if not isinstance(outcomes, ConversationConsumerOutcomeStore):
            raise TypeError("outcomes must be ConversationConsumerOutcomeStore")
        self.outcomes = outcomes

    def inspect(
        self,
        source: ConversationSourceEnvelope,
        consumer: ConversationSourceConsumer,
        processor_fingerprint: str,
        outputs: ConversationConsumerOutputStore,
    ) -> ConversationConsumerState:
        try:
            return self._inspect(source, consumer, processor_fingerprint, outputs)
        except ConversationConsumerBrokenOutcomeError as exc:
            return ConversationConsumerState(
                ConversationConsumerDeliveryState.BROKEN_OUTCOME, error=exc
            )
        except ConversationSourceError as exc:
            return ConversationConsumerState(
                ConversationConsumerDeliveryState.CORRUPTED,
                error=ConversationConsumerCorruptionError(str(exc)),
            )

    def require_terminal(
        self,
        source: ConversationSourceEnvelope,
        consumer: ConversationSourceConsumer,
        processor_fingerprint: str,
        outputs: ConversationConsumerOutputStore,
    ) -> ConversationConsumerState:
        state = self.inspect(source, consumer, processor_fingerprint, outputs)
        if state.state is ConversationConsumerDeliveryState.BROKEN_OUTCOME:
            assert state.error is not None
            raise ConversationConsumerBrokenOutcomeError(str(state.error))
        if state.state is ConversationConsumerDeliveryState.CORRUPTED:
            assert state.error is not None
            raise ConversationConsumerCorruptionError(str(state.error))
        if state.state not in {
            ConversationConsumerDeliveryState.COMMITTED,
            ConversationConsumerDeliveryState.SKIPPED,
        }:
            raise ConversationSourceError("consumer has not reached a durable terminal outcome")
        return state

    def _inspect(
        self,
        source: ConversationSourceEnvelope,
        consumer: ConversationSourceConsumer,
        processor_fingerprint: str,
        outputs: ConversationConsumerOutputStore,
    ) -> ConversationConsumerState:
        resolved_consumer = ConversationSourceConsumer(consumer)
        if outputs.consumer is not resolved_consumer:
            raise ConversationSourceError("output store belongs to another consumer")
        outcome = self.outcomes.read(source.source_id, resolved_consumer)
        current_output_id = outputs.expected_output_id(source, processor_fingerprint)
        current_output = outputs.read(source, current_output_id)
        all_outputs = outputs.list(source)
        if outcome is None:
            orphan_output = next(iter(all_outputs), None)
            if orphan_output is None:
                return ConversationConsumerState(ConversationConsumerDeliveryState.PENDING)
            if len(all_outputs) > 1:
                raise ConversationConsumerCorruptionError("multiple orphan outputs make first outcome ambiguous")
            # 当前 Processor 先按确定性 output_id 直读；仅在升级崩溃窗口下采用唯一旧 Output。
            output = current_output if current_output is not None else orphan_output
            outputs.ref(output)
            return ConversationConsumerState(
                ConversationConsumerDeliveryState.OUTPUT_READY, output=output
            )
        self._validate_outcome_source(source, resolved_consumer, outcome)
        if outcome.state is ConversationConsumerOutcomeState.SKIPPED:
            if all_outputs:
                raise ConversationConsumerCorruptionError("skipped outcome has a consumer output")
            return ConversationConsumerState(
                ConversationConsumerDeliveryState.SKIPPED, outcome=outcome
            )
        assert outcome.output_ref is not None
        output = outputs.read(source, outcome.output_ref.output_id)
        if output is None:
            raise ConversationConsumerBrokenOutcomeError("committed outcome references a missing output")
        actual_ref = outputs.ref(output)
        if actual_ref != outcome.output_ref:
            raise ConversationConsumerCorruptionError("outcome and output reference differ")
        if len(all_outputs) != 1 or outputs.ref(all_outputs[0]) != actual_ref:
            raise ConversationConsumerCorruptionError("committed outcome has unexpected sibling outputs")
        return ConversationConsumerState(
            ConversationConsumerDeliveryState.COMMITTED,
            outcome=outcome,
            output=output,
        )

    @staticmethod
    def _validate_outcome_source(
        source: ConversationSourceEnvelope,
        consumer: ConversationSourceConsumer,
        outcome: ConversationConsumerOutcome,
    ) -> None:
        if outcome.source_id != source.source_id or outcome.consumer is not consumer:
            raise ConversationConsumerCorruptionError("outcome belongs to another source or consumer")
        if outcome.source_payload_digest != source.source_payload_digest:
            raise ConversationConsumerCorruptionError("outcome source payload digest differs")
        # 已完成 Outcome 使用它自己的 Processor Fingerprint；当前版本升级不构成损坏。


__all__ = [
    "ConversationConsumerBrokenOutcomeError",
    "ConversationConsumerCorruptionError",
    "ConversationConsumerDeliveryState",
    "ConversationConsumerOutputStore",
    "ConversationConsumerState",
    "ConversationConsumerStateInspector",
]
