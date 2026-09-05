"""在统一执行栅栏内确保单个 Source Consumer 形成首次耐久 Outcome。"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from time import monotonic
from types import MappingProxyType
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
from foundation.observability import (
    NullObserver,
    ObservationEvent,
    ObservationStatus,
    Observer,
)

_OBSERVATION_CATEGORY = "conversation.source"
_OBSERVATION_OPERATION = "consumer_delivery"


class ConversationOrderedPredecessorPendingError(ConversationSourceError):
    """同一 Conversation 的更早 Source 尚未形成该 Consumer 的 Outcome。"""


class ConversationOrderedPredecessorBrokenError(ConversationSourceError):
    """同一 Conversation 的更早 Source 已损坏，该 Consumer 必须停止。"""


class ConversationDurableConsumer(Protocol):
    consumer: ConversationSourceConsumer
    processor_fingerprint: str

    # 是否要求同一 Conversation 内严格按 Source 顺序处理。声明为 True 时，交付前
    # 必须确认同会话内所有更早 Source 都已形成终态 Outcome，并额外持有一把会话级
    # 顺序锁。排序需求属于 Consumer 自身语义，由它自己声明；交付层与执行栅栏都
    # 不再按 Consumer 名字分支。
    ordered_within_conversation: bool

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
        consumers: Mapping[ConversationSourceConsumer, ConversationDurableConsumer],
        *,
        clock: Callable[[], datetime] | None = None,
        observer: Observer | None = None,
    ) -> None:
        if inspector.outcomes is not outcomes:
            raise ValueError("delivery and inspector must share the Outcome Store")
        if not isinstance(consumers, Mapping):
            raise TypeError("consumers must map each ConversationSourceConsumer to its implementation")
        registry = {ConversationSourceConsumer(key): value for key, value in consumers.items()}
        for key, implementation in registry.items():
            if implementation.consumer is not key:
                raise ValueError(f"consumer registered under {key.value} has the wrong consumer identity")
            if not isinstance(implementation.ordered_within_conversation, bool):
                raise TypeError(f"consumer {key.value} must declare ordered_within_conversation as a boolean")
        # 枚举本身就是"一个 Source 有哪些 Consumer"的定义，缺一个即为装配错误：
        # 未注册的 Consumer 不会报错，只会永远收不到交付。
        missing = sorted(item.value for item in ConversationSourceConsumer if item not in registry)
        if missing:
            raise ValueError(f"delivery is missing consumers: {missing}")
        self.sources = sources
        self.outcomes = outcomes
        self.inspector = inspector
        self.fence = fence
        self.consumers: Mapping[ConversationSourceConsumer, ConversationDurableConsumer] = MappingProxyType(registry)
        self.clock = clock or (lambda: datetime.now(UTC))
        if observer is not None and not callable(getattr(observer, "record", None)):
            raise TypeError("observer must implement record")
        self.observer: Observer = observer or NullObserver()

    async def ensure_outcome(
        self,
        source: ConversationSourceEnvelope,
        consumer: ConversationSourceConsumer,
    ) -> ConversationConsumerEnsureResult:
        """确保交付到达终态，并把结果记为一条观察事件。

        观察挂在这里而不是某个调用方，因为前台写入、启动恢复和显式恢复三条
        路径全都汇合于此；挂在调用方就必然漏掉没挂的那条（启动恢复曾经因此
        把 Behavior 的失败整个丢掉）。
        """

        started = monotonic()
        try:
            result = await self._ensure_outcome(source, consumer)
        except BaseException as error:
            self._observe(
                source,
                consumer,
                self._failure_status(error),
                started,
                {"error_type": type(error).__name__},
            )
            raise
        self._observe(
            source,
            consumer,
            ObservationStatus.SUCCESS,
            started,
            {"outcome_state": result.outcome.state.value},
        )
        return result

    async def _ensure_outcome(
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
        ordered = implementation.ordered_within_conversation
        if ordered:
            self._require_ordered_predecessors(source, implementation.consumer)
        # 这把锁把同一 (source, consumer) 的 inspect → execute → commit 串成一段。
        # 它挡住什么，取决于该 Consumer——不要按"没有它就一定写坏数据"来理解：
        #
        # - 本方法的两次耐久写本身就是幂等的：Output 只创建不覆盖且撞车时按内容
        #   摘要复用，Outcome 由第一份固定。因此对于 execute 是来源纯函数的
        #   Consumer（Behavior 投影就是），锁避免的是重复计算，而不是错误数据。
        # - 对 execute 带外部副作用的 Consumer 它仍然承重：Memory 会写 Conversation
        #   journal 并派发 Job，重复执行不是白算一遍那么简单。
        # - 对跨版本并发它始终承重：两个 processor fingerprint 会写出两个不同
        #   output_id 的文件，同一来源出现多个 Output 是不可自愈的损坏。单进程
        #   碰不到，但锁的 host 级作用域本来就是为多进程留的。
        # - `_commit_run` 中"run 的 output_ref 与耐久 Output 不一致即判损坏"这类
        #   断言以串行为前提；放开锁必须同时重新论证它们。
        async with self.fence.acquire(
            source,
            implementation.consumer,
            conversation_ordered=ordered,
        ) as lease:
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
            if ordered:
                await lease.run_fenced(
                    lambda: self._require_ordered_predecessors(source, implementation.consumer)
                )
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

    def _require_ordered_predecessors(
        self,
        source: ConversationSourceEnvelope,
        consumer: ConversationSourceConsumer,
    ) -> None:
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
            state = self.inspect(predecessor, consumer)
            if state.state is ConversationConsumerDeliveryState.COMMITTED:
                continue
            if state.state in {
                ConversationConsumerDeliveryState.BROKEN_OUTCOME,
                ConversationConsumerDeliveryState.CORRUPTED,
                ConversationConsumerDeliveryState.SKIPPED,
            }:
                raise ConversationOrderedPredecessorBrokenError(
                    f"earlier {consumer.value} Source {predecessor.source_id} is {state.state.value}"
                )
            raise ConversationOrderedPredecessorPendingError(
                f"earlier {consumer.value} Source {predecessor.source_id} has no terminal Outcome"
            )

    def _consumer(self, consumer: ConversationSourceConsumer) -> ConversationDurableConsumer:
        return self.consumers[ConversationSourceConsumer(consumer)]

    @staticmethod
    def _failure_status(error: BaseException) -> ObservationStatus:
        """区分"等前序完成"和"这条来源已经废了"。

        前者是有序 Consumer 的正常等待，重试即可推进；后者不会自愈，必须由
        人工处置，两者混成同一个状态会让真正的损坏淹没在排队噪音里。
        """

        if isinstance(error, ConversationOrderedPredecessorPendingError):
            return ObservationStatus.DEGRADED
        return ObservationStatus.FAILURE

    def _observe(
        self,
        source: ConversationSourceEnvelope,
        consumer: ConversationSourceConsumer,
        status: ObservationStatus,
        started: float,
        attributes: dict[str, str | int | float | bool],
    ) -> None:
        """记录一条有界事件；观测后端失败绝不影响交付结果。"""

        try:
            self.observer.record(
                ObservationEvent(
                    category=_OBSERVATION_CATEGORY,
                    operation=_OBSERVATION_OPERATION,
                    status=status,
                    duration_seconds=max(0.0, monotonic() - started),
                    attributes={
                        "consumer": ConversationSourceConsumer(consumer).value,
                        "source_id": source.source_id,
                        **attributes,
                    },
                )
            )
        except Exception:
            pass

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
    "ConversationOrderedPredecessorBrokenError",
    "ConversationOrderedPredecessorPendingError",
]
