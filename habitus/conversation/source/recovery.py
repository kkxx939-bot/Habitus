"""按 Source/Consumer 独立恢复缺失的首次耐久 Outcome。

恢复由生命周期周期驱动（26 号清单 L9.2），启动时也跑一遍。每次尝试的失败记进重试账：可重试的按退避推迟，次数用完或
确定性的放弃、等人工修复之后 ``forgive``。一批只取"现在该试"的——损坏、退避中、已放弃的都不占批，所以更早的坏条目
挡不住更新的待恢复项。"等前序"不是失败，是有序 Consumer 的正常排队：不记账，下一轮再看。

放弃之后：能放弃的 Consumer（Behavior）按自己的方式落终态——还没写进树的轮写成没读懂空白（23 号第一节第 6 条），不按失败
种类再分，账判放弃就写；不能放弃的（Memory）停在那里等人工修复。
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Collection, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from habitus.conversation.source.address import SourceAddress
from habitus.conversation.source.delivery import (
    ConversationConsumerDelivery,
    ConversationOrderedPredecessorPendingError,
)
from habitus.conversation.source.model import ConversationSourceEnvelope, ConversationSourceError
from habitus.conversation.source.receipt import ConversationConsumerOutcome, ConversationSourceConsumer
from habitus.conversation.source.salvage import SourceSalvage
from habitus.conversation.source.state import ConversationConsumerDeliveryState
from habitus.conversation.source.store import ConversationSourceStore
from habitus.infrastructure.lifecycle import FailureKind, RetryDecision, RetryLedger

_TERMINAL = frozenset({ConversationConsumerDeliveryState.COMMITTED, ConversationConsumerDeliveryState.SKIPPED})
_BROKEN = frozenset({ConversationConsumerDeliveryState.BROKEN_OUTCOME, ConversationConsumerDeliveryState.CORRUPTED})


class ConversationSourceRecoveryDeferredError(ConversationSourceError):
    """这一份上次恢复失败了，还在退避，或者已经放弃、等人工修复；这一轮没有试。"""


@dataclass(frozen=True)
class ConversationSourceRecoveryEntry:
    envelope: ConversationSourceEnvelope
    consumer: ConversationSourceConsumer
    state: ConversationConsumerDeliveryState

    @property
    def key(self) -> str:
        return recovery_key(self.envelope.source_id, self.consumer)


@dataclass(frozen=True)
class ConversationSourceRecoveryResult:
    envelope: ConversationSourceEnvelope
    consumer: ConversationSourceConsumer
    outcome: ConversationConsumerOutcome | None
    error: BaseException | None

    def __post_init__(self) -> None:
        if (self.outcome is None) == (self.error is None):
            raise ValueError("recovery result requires exactly one Outcome or error")


@dataclass(frozen=True)
class ConversationSourceRecoveryPlan:
    """一轮该怎么处置每个没终态的 Source/Consumer。``attempt`` 按恢复顺序排好、最多一批；``salvaged`` 是这一轮
    刚补救过的（Behavior 的账坏了：没写过的轮落「没读懂」、能重试的进每晚重试的堆，从此按终态算）；``broken`` 是坏了又
    不补救的（Memory，等人工修复）。"""

    attempt: tuple[ConversationSourceRecoveryEntry, ...]
    broken: tuple[ConversationSourceRecoveryEntry, ...]
    deferred: tuple[ConversationSourceRecoveryEntry, ...]
    salvaged: tuple[ConversationSourceRecoveryEntry, ...] = ()
    #: 信封读不出来（损坏）的源：没法恢复，也不能让它拖垮这一轮，记下等人工处置（L10.9）。
    corrupt: tuple[str, ...] = ()


def recovery_key(source_id: str, consumer: ConversationSourceConsumer) -> str:
    return f"{source_id}/{ConversationSourceConsumer(consumer).value}"


class ConversationSourceRecovery:
    """只处理 PENDING/OUTPUT_READY；终态保持，损坏状态 fail-closed。"""

    def __init__(
        self,
        sources: ConversationSourceStore,
        delivery: ConversationConsumerDelivery,
        *,
        batch_size: int,
        retries: RetryLedger,
        salvage: SourceSalvage,
        orphan_grace: timedelta,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if delivery.sources is not sources:
            raise ValueError("source recovery and delivery must share the Source Store")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if not isinstance(retries, RetryLedger):
            raise TypeError("retries must be a RetryLedger")
        if not isinstance(salvage, SourceSalvage):
            raise TypeError("salvage must be a SourceSalvage")
        if not isinstance(orphan_grace, timedelta) or orphan_grace <= timedelta(0):
            raise ValueError("orphan_grace must be a positive timedelta")
        self.sources = sources
        self.delivery = delivery
        self.batch_size = batch_size
        self.retries = retries
        self.salvage = salvage
        self.orphan_grace = orphan_grace
        self._clock = clock if clock is not None else lambda: datetime.now(UTC)

    def pending(self) -> tuple[ConversationSourceRecoveryEntry, ...]:
        """只看"还没终态"的标记，不列全部信封。顺手清三种残留标记：

        - 两个消费者都已终态（行为侧的账坏了、已补救的算终态）：标记本该删掉，惰性地在这里删；
        - 信封已经不在、地址水位盖住了它：释放删到一半留下的；
        - 信封不在、水位也没盖住、标记已超过宽限期：put 写到一半崩溃、再没重投的孤儿。它的索引标记一起删，
          不然同地址更早的源永远"还在记录"，有序 Consumer 与释放都会卡在它前面。

        还在宽限期内的不动：那可能是正在进行的 put，同一份源重投时会补上信封。
        """

        return self._scan()[0]

    def _scan(self) -> tuple[tuple[ConversationSourceRecoveryEntry, ...], tuple[str, ...]]:
        ledger = self.delivery.ledger
        entries: list[ConversationSourceRecoveryEntry] = []
        corrupt: list[str] = []
        alive: set[str] = set()
        for source_id, address in ledger.pending.listing():
            try:
                envelope = self.sources.read(source_id)
            except ConversationSourceError:
                corrupt.append(source_id)  # 一份坏信封只挡它自己：启动与周期恢复照常处理别的
                continue
            if envelope is None:
                if self._released(source_id, address) or self._expired_orphan(source_id, address):
                    ledger.pending.clear(source_id)
                continue
            waiting = [
                ConversationSourceRecoveryEntry(envelope, consumer, state.state)
                for consumer in ConversationSourceConsumer
                if not self.salvage.contains(source_id, consumer)
                and (state := self.delivery.inspect(envelope, consumer)).state not in _TERMINAL
            ]
            if waiting:
                entries.extend(waiting)
                alive.update(entry.key for entry in waiting)
            else:
                ledger.pending.clear(source_id)
        self.retries.reap(alive.__contains__)
        ordered = tuple(sorted(entries, key=lambda entry: (entry.envelope.recorded_at, entry.envelope.source_id)))
        return ordered, tuple(corrupt)

    def plan(self, *, busy: Collection[tuple[str, ConversationSourceConsumer]] = ()) -> ConversationSourceRecoveryPlan:
        """``busy`` 是正在交付中的 Source/Consumer（协调器里还在跑的），这一轮不碰。"""

        attempt: list[ConversationSourceRecoveryEntry] = []
        broken: list[ConversationSourceRecoveryEntry] = []
        deferred: list[ConversationSourceRecoveryEntry] = []
        salvaged: list[ConversationSourceRecoveryEntry] = []
        pending, corrupt = self._scan()
        held = self._held_memory_addresses(pending, busy)
        for entry in pending:
            if (entry.envelope.source_id, entry.consumer) in busy or _waits_behind(entry, held):
                continue
            if entry.state in _BROKEN and self.salvage.salvages(entry.consumer):
                self.salvage.salvage(entry.envelope, entry.consumer)
                salvaged.append(entry)
            elif entry.state in _BROKEN:
                broken.append(entry)
            elif self._abandoning(entry):
                if len(attempt) < self.batch_size:
                    attempt.append(entry)  # 放弃了、还没落成没读懂（例如放弃那一刻进程退出了）：这一轮补上
            elif self.retries.blocked(entry.key):
                deferred.append(entry)
            elif len(attempt) < self.batch_size:
                attempt.append(entry)
        return ConversationSourceRecoveryPlan(
            _in_recovery_order(attempt), tuple(broken), tuple(deferred), tuple(salvaged), corrupt
        )

    def _held_memory_addresses(
        self,
        pending: tuple[ConversationSourceRecoveryEntry, ...],
        busy: Collection[tuple[str, ConversationSourceConsumer]],
    ) -> dict[SourceAddress, int]:
        """有序的 Memory 在每个地址上被挡在哪个序号：那一份坏了、在退避、或正在交付，同地址后面的只会报"等前序"，
        不该占这一批（L10.8；原先它们每轮占满批次，别的地址的新源永远进不来）。"""

        held: dict[SourceAddress, int] = {}
        for entry in pending:
            if entry.consumer is not ConversationSourceConsumer.MEMORY:
                continue
            blocked = (
                entry.state in _BROKEN
                or (entry.envelope.source_id, entry.consumer) in busy
                or self.retries.blocked(entry.key)
            )
            if blocked:
                address = SourceAddress.of(entry.envelope)
                start = entry.envelope.batch.start_sequence
                held[address] = min(start, held.get(address, start))
        return held

    async def recover_pending(self) -> tuple[ConversationSourceRecoveryResult, ...]:
        """一轮恢复并等它做完：损坏的与退避中的照样各出一条带错误的结果（"现在还卡着几条"要看得见），只是不试。"""

        plan = await asyncio.to_thread(self.plan)
        results = [self._broken(entry) for entry in plan.broken]
        results.extend(self._deferred(entry) for entry in plan.deferred)
        results.extend(await self.recover_all(plan.attempt))
        return tuple(results)

    async def recover_all(
        self, entries: Iterable[ConversationSourceRecoveryEntry]
    ) -> tuple[ConversationSourceRecoveryResult, ...]:
        """Memory 按会话顺序一份接一份（有序 Consumer），Behavior 各自并发。"""

        ordered = _in_recovery_order(entries)
        behavior_tasks = tuple(
            asyncio.create_task(self.recover(entry))
            for entry in ordered
            if entry.consumer is ConversationSourceConsumer.BEHAVIOR_SESSION
        )
        results: list[ConversationSourceRecoveryResult] = []
        for entry in ordered:
            if entry.consumer is ConversationSourceConsumer.MEMORY:
                results.append(await self.recover(entry))
        if behavior_tasks:
            results.extend(await asyncio.gather(*behavior_tasks))
        return tuple(results)

    async def recover(self, entry: ConversationSourceRecoveryEntry) -> ConversationSourceRecoveryResult:
        """试一次，把结果记进重试账；账判放弃、Consumer 又能放弃，就当场放弃（写没读懂、落终态）。"""

        if self._abandoning(entry):
            return await self._abandon(entry)
        try:
            ensured = await self.delivery.ensure_outcome(entry.envelope, entry.consumer)
        except BaseException as exc:
            if isinstance(exc, Exception):
                decision = await asyncio.to_thread(self._settle, entry, exc)
                if decision is not None and decision.give_up and self.delivery.abandons(entry.consumer):
                    return await self._abandon(entry)
            return ConversationSourceRecoveryResult(entry.envelope, entry.consumer, None, exc)
        await asyncio.to_thread(self._settle, entry, None)
        return ConversationSourceRecoveryResult(entry.envelope, entry.consumer, ensured.outcome, None)

    async def _abandon(self, entry: ConversationSourceRecoveryEntry) -> ConversationSourceRecoveryResult:
        """放弃没成（本地读写出错）就照旧挂着放弃，下一轮再补；成了就清账。"""

        found = self.retries.entry(entry.key)
        reason = "recovery given up" if found is None else f"recovery given up after {found.attempts} attempts"
        try:
            ensured = await self.delivery.ensure_abandoned(entry.envelope, entry.consumer, reason=reason)
        except Exception as exc:
            return ConversationSourceRecoveryResult(entry.envelope, entry.consumer, None, exc)
        await asyncio.to_thread(self.retries.clear, entry.key)
        return ConversationSourceRecoveryResult(entry.envelope, entry.consumer, ensured.outcome, None)

    def _abandoning(self, entry: ConversationSourceRecoveryEntry) -> bool:
        found = self.retries.entry(entry.key)
        return found is not None and found.kind is FailureKind.DETERMINISTIC and self.delivery.abandons(entry.consumer)

    def forgive(self, source_id: str, consumer: ConversationSourceConsumer) -> bool:
        """人工修复之后：清掉这一份的重试账，下一轮照常试。"""

        return self.retries.clear(recovery_key(source_id, consumer))

    def _settle(self, entry: ConversationSourceRecoveryEntry, error: Exception | None) -> RetryDecision | None:
        if error is None:
            self.retries.clear(entry.key)
        elif not isinstance(error, ConversationOrderedPredecessorPendingError):
            return self.retries.record_failure(entry.key, error)
        return None

    def _broken(self, entry: ConversationSourceRecoveryEntry) -> ConversationSourceRecoveryResult:
        state = self.delivery.inspect(entry.envelope, entry.consumer)
        assert state.error is not None
        return ConversationSourceRecoveryResult(entry.envelope, entry.consumer, None, state.error)

    def _deferred(self, entry: ConversationSourceRecoveryEntry) -> ConversationSourceRecoveryResult:
        found = self.retries.entry(entry.key)
        reason = "deferred" if found is None else f"{found.kind.value} after {found.attempts} attempts: {found.message}"
        error = ConversationSourceRecoveryDeferredError(f"{entry.consumer.value} recovery {reason}")
        return ConversationSourceRecoveryResult(entry.envelope, entry.consumer, None, error)

    def _released(self, source_id: str, address: SourceAddress) -> bool:
        """标记总是先建索引再建"还没终态"，释放总是先删索引再删"还没终态"：所以索引标记不在，或者在但已被
        地址水位盖住，都说明这份源已经释放过。"""

        ledger = self.delivery.ledger
        markers = [marker for marker in ledger.index.markers(address) if marker.source_id == source_id]
        return not markers or all(ledger.released(address, marker.end_sequence) for marker in markers)

    def _expired_orphan(self, source_id: str, address: SourceAddress) -> bool:
        """超过宽限期的孤儿：先删索引标记再交回调用方删"还没终态"——删到一半崩溃，下一轮按"索引不在"认作已释放。"""

        ledger = self.delivery.ledger
        marked = ledger.pending.marked_at(source_id)
        if marked is None or self._clock() - marked < self.orphan_grace:
            return False
        for marker in ledger.index.markers(address):
            if marker.source_id == source_id:
                ledger.index.remove(marker)
        return True


def _waits_behind(entry: ConversationSourceRecoveryEntry, held: dict[SourceAddress, int]) -> bool:
    if entry.consumer is not ConversationSourceConsumer.MEMORY:
        return False
    blocker = held.get(SourceAddress.of(entry.envelope))
    return blocker is not None and entry.envelope.batch.start_sequence > blocker


def _in_recovery_order(
    entries: Iterable[ConversationSourceRecoveryEntry],
) -> tuple[ConversationSourceRecoveryEntry, ...]:
    """Memory 按会话、起始日、序号排（有序 Consumer 必须先前后后）；Behavior 按记录时刻。"""

    def order(entry: ConversationSourceRecoveryEntry) -> tuple[object, ...]:
        envelope = entry.envelope
        if entry.consumer is ConversationSourceConsumer.MEMORY:
            return (
                0,
                envelope.conversation_id,
                envelope.started_on,
                envelope.batch.start_sequence,
                envelope.batch.end_sequence,
                envelope.source_id,
            )
        return (1, envelope.recorded_at, envelope.source_id)

    return tuple(sorted(entries, key=order))


__all__ = [
    "ConversationSourceRecovery",
    "ConversationSourceRecoveryDeferredError",
    "ConversationSourceRecoveryEntry",
    "ConversationSourceRecoveryPlan",
    "ConversationSourceRecoveryResult",
    "recovery_key",
]
