"""释放会话源：信封、两份 Outcome、两种输出、两把源锁行、两样标记一起退出（裁定 60，26 号清单 2.1）。

门槛，三条同时满足：
1. 同一地址后面有一份 start_sequence 更大、Memory 已 COMMITTED 的交付——插件同会话严格顺序投递，后一份
   已提交，前一份就不会再被重投；
2. 这份源两个消费者都已终态（COMMITTED / SKIPPED；行为侧的账坏了、已补救的算终态，``salvage``）；
3. 同一地址更早的源都已释放——所以水位只会连续推进。
每个地址最后一份永远满足不了第 1 条，随会话退休一起删（memory 侧，待定）。

删的顺序：持有两个消费者的执行锁 → 写地址水位（fence 住全局 lease 与两把锁）→ 删信封 → 删 Outcome → 删输出 →
放锁、删两把锁行 → 删"已补救"标记 → 删"还没终态"标记 → 最后删索引标记。任何一步崩溃，下一轮从索引标记续删：水位已盖住、且信封
已不在或两边已终态，就直接往下删，不再看第 1、3 条；信封还在又没终态的是序号回退后的新源，等它终态。
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

from habitus.conversation.source.fence import consumer_lock_key
from habitus.conversation.source.index import SourceMarker
from habitus.conversation.source.ledger import ConversationSourceLedger
from habitus.conversation.source.receipt import (
    ConversationConsumerOutcomeState,
    ConversationConsumerOutcomeStore,
    ConversationSourceConsumer,
)
from habitus.conversation.source.salvage import SourceSalvage
from habitus.conversation.source.state import ConversationConsumerOutputStore
from habitus.conversation.source.store import ConversationSourceStore
from habitus.infrastructure.lifecycle import LifecycleContext
from habitus.infrastructure.store.contracts import LockStore, LockToken

_TERMINAL = frozenset({ConversationConsumerOutcomeState.COMMITTED, ConversationConsumerOutcomeState.SKIPPED})


class ConversationSourceReleaser:
    def __init__(
        self,
        sources: ConversationSourceStore,
        outcomes: ConversationConsumerOutcomeStore,
        ledger: ConversationSourceLedger,
        outputs: Mapping[ConversationSourceConsumer, ConversationConsumerOutputStore],
        lock_store: LockStore,
        *,
        lock_ttl_seconds: int,
        salvage: SourceSalvage | None = None,
    ) -> None:
        missing = sorted(item.value for item in ConversationSourceConsumer if item not in outputs)
        if missing:
            raise ValueError(f"source releaser is missing output stores: {missing}")
        if isinstance(lock_ttl_seconds, bool) or not isinstance(lock_ttl_seconds, int) or lock_ttl_seconds <= 0:
            raise ValueError("lock_ttl_seconds must be a positive integer")
        self.sources = sources
        self.outcomes = outcomes
        self.ledger = ledger
        self.outputs: Mapping[ConversationSourceConsumer, ConversationConsumerOutputStore] = MappingProxyType(
            dict(outputs)
        )
        self.lock_store = lock_store
        self.lock_ttl_seconds = lock_ttl_seconds
        if salvage is not None and not isinstance(salvage, SourceSalvage):
            raise TypeError("salvage must be a SourceSalvage")
        self.salvage = salvage

    def releasable(self, directory_name: str) -> tuple[SourceMarker, ...]:
        """这个地址里现在可以放的源，按顺序；遇到第一份放不了的就停（第 3 条）。"""

        markers = self.ledger.index.markers_in(directory_name)
        if not markers:
            return ()
        address = markers[0].address
        through = self.ledger.watermarks.released_through(address)
        found: list[SourceMarker] = []
        for position, marker in enumerate(markers):
            if through is not None and marker.end_sequence <= through:
                # 水位已盖住：多半是上一轮删到一半，续删。但也可能是序号回退后进来的新源（插件状态重建、从头数），
                # 它还没被处理，等它到终态再放；不挡后面的源，它们的序号在水位之上、照常判门槛
                if self.sources.read(marker.source_id) is None or self._terminal(marker):
                    found.append(marker)
                continue
            later = markers[position + 1 :]
            if not any(self._memory_committed(item) for item in later) or not self._terminal(marker):
                break
            found.append(marker)
        return tuple(found)

    def release_address(self, directory_name: str, ctx: LifecycleContext) -> int:
        """按顺序放这个地址里能放的；有一份正在交付（锁拿不到）就停在那里，下一轮再来。"""

        released = 0
        for marker in self.releasable(directory_name):
            if not self.release(marker, ctx):
                break
            released += 1
        return released

    def release(self, marker: SourceMarker, ctx: LifecycleContext) -> bool:
        tokens = self._acquire(marker.source_id)
        if tokens is None:
            return False
        try:
            ctx.assert_owned()
            with self.lock_store.fenced((ctx.fence_token(), *tokens), ttl_seconds=self.lock_ttl_seconds):
                self.ledger.watermarks.advance(marker.address, marker.end_sequence)
            ctx.assert_owned()
            self.sources.discard(marker.source_id)
            for consumer in ConversationSourceConsumer:
                self.outcomes.discard(marker.source_id, consumer)
            for consumer in ConversationSourceConsumer:
                self.outputs[consumer].discard_all(marker.source_id)
        finally:
            for token in tokens:
                self.lock_store.release(token)
        for consumer in ConversationSourceConsumer:
            self.lock_store.discard_idle(consumer_lock_key(marker.source_id, consumer))
        if self.salvage is not None:
            self.salvage.discard(marker.source_id)
        ctx.assert_owned()
        self.ledger.pending.clear(marker.source_id)
        self.ledger.index.remove(marker)
        return True

    def settle_pending(self) -> int:
        """两个消费者都已终态的源，把"还没终态"标记清掉（每个地址最后那份要等到会话退休才释放，标记不能一直挂着）。"""

        settled = 0
        for source_id, _address in self.ledger.pending.listing():
            if all(
                self._salvaged(source_id, consumer)
                or ((outcome := self.outcomes.read(source_id, consumer)) is not None and outcome.state in _TERMINAL)
                for consumer in ConversationSourceConsumer
            ):
                settled += int(self.ledger.pending.clear(source_id))
        return settled

    def _acquire(self, source_id: str) -> tuple[LockToken, ...] | None:
        """两个消费者的执行锁都拿到才放：拿不到说明有交付正在跑这份源。"""

        acquired: list[LockToken] = []
        try:
            for consumer in ConversationSourceConsumer:
                acquired.append(self.lock_store.acquire(consumer_lock_key(source_id, consumer), self.lock_ttl_seconds))
        except TimeoutError:
            for token in acquired:
                self.lock_store.release(token)
            return None
        return tuple(acquired)

    def _memory_committed(self, marker: SourceMarker) -> bool:
        outcome = self.outcomes.read(marker.source_id, ConversationSourceConsumer.MEMORY)
        return outcome is not None and outcome.state is ConversationConsumerOutcomeState.COMMITTED

    def _salvaged(self, source_id: str, consumer: ConversationSourceConsumer) -> bool:
        return self.salvage is not None and self.salvage.contains(source_id, consumer)

    def _terminal(self, marker: SourceMarker) -> bool:
        if self.sources.read(marker.source_id) is None:
            return False  # put 还没写完
        for consumer in ConversationSourceConsumer:
            if self._salvaged(marker.source_id, consumer):
                continue
            outcome = self.outcomes.read(marker.source_id, consumer)
            if outcome is None or outcome.state not in _TERMINAL:
                return False
        return True


__all__ = ["ConversationSourceReleaser"]
