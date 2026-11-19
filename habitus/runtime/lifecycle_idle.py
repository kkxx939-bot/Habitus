"""闲置封存（裁定 107）：挂在 LifecycleWorker 上、每一拍都跑的参与者。

会话有没封存的内容、最后一条消息之后过了 ``idle_commit_seconds``，就当这场对话到此为止：先由系统如实收尾（没有结果的
工具调用补终态结果、没有最终回复的轮补一条说明中断的最终回复，``memory/conversation/interruption.py``），走现有的追加
主链写进 live；再走现有的 flush 把 live 全部封存，封出来的是普通 history 段，照样"先建任务、再发布、再激活"，按完整的轮
抽取记忆。封完当场试着放这个地址的会话源（最后一份靠"已经全部封存"这一条放，见 ``conversation/source/release.py``）。

这是一座桥，所以住在组合根：它把会话日志、Memory 的封存入口、会话源的顺序锁与释放接在一起。和这个地址正在跑的 Memory
交付不并发：持同一把有序 Consumer 顺序锁，拿不到就是有交付在跑，这一拍跳过它（不算失败），下一拍再看。

每拍从内存游标往后看 ``idle_scan_batch_conversations`` 个会话，到头了从头再来（游标不落盘：重启从头轮一遍不出错）。
一个会话封存失败只记进这一拍的报告，不挡别的会话；它仍闲置、仍有内容，下一拍自然重试。
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from contextlib import ExitStack
from datetime import datetime, timedelta

from habitus.config import ConversationLifecycleConfig
from habitus.conversation.source.address import SourceAddress
from habitus.conversation.source.fence import conversation_order_lock_key
from habitus.conversation.source.receipt import ConversationSourceConsumer
from habitus.conversation.source.release import ConversationSourceReleaser
from habitus.infrastructure.lifecycle import LifecycleContext, ParticipantReport
from habitus.infrastructure.store.contracts import PathLock
from habitus.memory.conversation import (
    ConversationAddress,
    ConversationMessageJournal,
    ConversationSemanticBoundaryScorer,
)
from habitus.memory.conversation.interruption import interruption_messages
from habitus.memory.workflow import ConversationMemoryEnqueuer
from habitus.pre.conversation import ConversationBatch


class ConversationIdleSealParticipant:
    name = "conversation-idle-seal"

    def __init__(
        self,
        enqueuer: ConversationMemoryEnqueuer,
        boundary_scorer: ConversationSemanticBoundaryScorer,
        releaser: ConversationSourceReleaser,
        path_lock: PathLock,
        config: ConversationLifecycleConfig,
        *,
        lock_ttl_seconds: int,
        wake_memory: Callable[[], None],
    ) -> None:
        self.enqueuer = enqueuer
        self.journal = enqueuer.conversations
        self.boundary_scorer = boundary_scorer
        self.releaser = releaser
        self.path_lock = path_lock
        self.config = config
        self.lock_ttl_seconds = lock_ttl_seconds
        self.wake_memory = wake_memory
        self._cursor: ConversationAddress | None = None

    async def maintain(self, ctx: LifecycleContext) -> ParticipantReport:
        addresses = await asyncio.to_thread(self.journal.list_addresses)
        batch = self._next_batch(addresses)
        idle_since = ctx.now - timedelta(seconds=self.config.idle_commit_seconds)
        sealed = 0
        failed: dict[str, int] = {}
        for address in batch:
            await asyncio.to_thread(ctx.assert_owned)
            try:
                sealed += await self._seal(address, idle_since, ctx)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - 一个会话封存失败只挡它自己，下一拍再来
                await asyncio.to_thread(ctx.assert_owned)  # 丢了全局 lease 的照旧中止这一拍
                failed[type(exc).__name__] = failed.get(type(exc).__name__, 0) + 1
        await asyncio.to_thread(ctx.assert_owned)
        if sealed:
            self.wake_memory()
        signals = tuple(f"{count} idle conversations failed to seal: {name}" for name, count in sorted(failed.items()))
        return ParticipantReport(released=sealed, signals=signals)

    async def _seal(self, address: ConversationAddress, idle_since: datetime, ctx: LifecycleContext) -> int:
        """够闲置就收尾、全部封存，返回封出的段数。"""

        live = await asyncio.to_thread(self.journal.read_live, address)
        if live is None or live.messages[-1].occurred_at > idle_since:
            return 0
        hints = await self.boundary_scorer.score(live)
        key = conversation_order_lock_key(
            address.conversation_id, address.started_on, ConversationSourceConsumer.MEMORY
        )

        def seal_locked() -> int:
            with ExitStack() as stack:
                try:
                    stack.enter_context(self.path_lock.acquire(key, ttl_seconds=self.lock_ttl_seconds))
                except TimeoutError:
                    return 0  # 这个地址的 Memory 交付正在跑：下一拍再看
                ctx.assert_owned()
                current = self.journal.read_live(address)  # 拿锁之前可能刚来了新消息：在锁里再看一眼
                if current is None or current.messages[-1].occurred_at > idle_since:
                    return 0
                closing = interruption_messages(current, at=ctx.now)
                if closing:
                    self.enqueuer.append(address, ConversationBatch(address.conversation_id, closing))
                jobs, _plan = self.enqueuer.enqueue_ready_segments(
                    address, after_turn=False, flush=True, boundary_hints=hints
                )
            if jobs:
                ctx.assert_owned()
                self.releaser.release_address(SourceAddress(address.conversation_id, address.started_on).digest, ctx)
            return len(jobs)

        return await asyncio.to_thread(seal_locked)

    def _next_batch(self, addresses: tuple[ConversationAddress, ...]) -> tuple[ConversationAddress, ...]:
        keys = [(item.started_on, item.conversation_id) for item in addresses]
        cursor = None if self._cursor is None else (self._cursor.started_on, self._cursor.conversation_id)
        after = [item for item, key in zip(addresses, keys, strict=True) if cursor is None or key > cursor]
        batch = tuple((after or list(addresses))[: self.config.idle_scan_batch_conversations])
        full = len(batch) == self.config.idle_scan_batch_conversations
        self._cursor = batch[-1] if batch and full else None
        return batch


def fully_sealed(journal: ConversationMessageJournal) -> Callable[[SourceAddress], bool]:
    """会话源释放的"已经全部封存"一条：这个地址的会话日志 live 里没有没封存的消息。"""

    return lambda address: journal.read_live(ConversationAddress(address.conversation_id, address.started_on)) is None


__all__ = ["ConversationIdleSealParticipant", "fully_sealed"]
