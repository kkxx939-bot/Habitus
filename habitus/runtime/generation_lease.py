"""跨进程租约（实时化方案第 9 节；阶段 2 第二小步「三把锁」）：拿、续、放，外加一把进程内的锁。三把锁都是它的实例：

- **预测根租约**（预测根派生）：预测 worker 每槽持一次，在它里面先看一眼换代（要就做今天这一代）、再做预测拍。账本只有一个
  写者；两个进程挂同一个根时同一天只换代一次（另加完成标记）。
- **语义树写锁**（语义树根派生，沿用原换代租约的键；``runtime.scene_lock``）：换代的语义树一步整步持有，模型队列写语义树的
  每一项在它里面、调模型的那几段除外。
- **队列租约**（语义树根派生、另一个键）：模型队列整趟持有（含等模型），同一时刻只有一个进程在做队列的活；换代不拿它。
- 每把都带一把进程内的 asyncio 锁：同一进程的两个持有者不靠 SQLite 超时互等。

拿锁的先后固定：队列租约（只有队列拿）→ 预测根租约 → sweep 锁 → 语义树写锁；任何一处都不反着拿。

租约在独立线程续（同步文件、SQLite 操作卡住事件循环时也不丢租约）：一次续不上（SQLite busy 之类）不取消，租约到期之前别人拿
不走它，下一拍再续；只有下一拍之前租约就要到期、还没续上，才取消持有它的那一段，抛 ``LeaseLostError``。
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
import weakref
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from habitus.infrastructure.store.contracts import LockStore, LockToken

logger = logging.getLogger("habitus.generation")


class LeaseBusyError(TimeoutError):
    """租约被另一个持有者拿着、等到时限也没拿到。"""


class LeaseLostError(RuntimeError):
    """续不上租约：持有它的那一段已取消，租约过期后别人能拿。"""


class Lease:
    """一把跨进程租约（``LockStore``，按 ``key``）加一把进程内的锁。同一个实例在同一进程的几个 worker 之间共用。"""

    def __init__(
        self,
        lock_store: LockStore,
        key: str,
        *,
        ttl_seconds: float,
        heartbeat_seconds: float,
        poll_seconds: float = 1.0,
    ) -> None:
        if not key:
            raise ValueError("lease key must be non-empty text")
        if not 0 < heartbeat_seconds < ttl_seconds:
            raise ValueError("heartbeat_seconds must be positive and below ttl_seconds")
        if poll_seconds <= 0:
            raise ValueError("poll_seconds must be positive")
        self.lock_store = lock_store
        self.key = key
        self.ttl_seconds = float(ttl_seconds)
        self.heartbeat_seconds = float(heartbeat_seconds)
        self.poll_seconds = float(poll_seconds)
        # asyncio 锁绑在事件循环上：按循环各建一把（测试里每次 ``asyncio.run`` 都是新循环）
        self._local: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Lock] = weakref.WeakKeyDictionary()
        self._guard = threading.Lock()

    def _local_lock(self) -> asyncio.Lock:
        loop = asyncio.get_running_loop()
        with self._guard:
            lock = self._local.get(loop)
            if lock is None:
                lock = self._local[loop] = asyncio.Lock()
            return lock

    @asynccontextmanager
    async def held(self, *, wait_seconds: float | None = None) -> AsyncIterator[LockToken]:
        """持有这把租约做一段事。``wait_seconds`` 是等多久（不给就等一个租期：持有者崩了，租约过期后就拿得到）；等不到抛
        ``LeaseBusyError``。持有期间续不上、租约要过期，取消这一段并抛 ``LeaseLostError``。"""

        deadline = time.monotonic() + (self.ttl_seconds if wait_seconds is None else max(0.0, wait_seconds))
        local = self._local_lock()
        if local.locked() and deadline <= time.monotonic():
            raise LeaseBusyError(f"lease {self.key} is held inside this process")
        try:
            await asyncio.wait_for(local.acquire(), timeout=max(deadline - time.monotonic(), 0.001))
        except TimeoutError as exc:
            raise LeaseBusyError(f"lease {self.key} is held inside this process") from exc
        try:
            token = await self._acquire(deadline)
            task = asyncio.current_task()
            assert task is not None
            stop = threading.Event()
            lost = threading.Event()
            loop = asyncio.get_running_loop()
            beat = asyncio.ensure_future(asyncio.to_thread(self._heartbeat, token, task, stop, lost, loop))
            try:
                yield token
            except asyncio.CancelledError:
                if lost.is_set():
                    task.uncancel()
                    raise LeaseLostError(f"lease {self.key} could not be renewed") from None
                raise
            finally:
                stop.set()
                await asyncio.gather(beat, return_exceptions=True)
                try:
                    await asyncio.to_thread(self.lock_store.release, token)
                except Exception:  # noqa: BLE001 - 放不掉照样活着：过期之后别人能拿
                    logger.warning("lease release failed key=%s", self.key)
            if lost.is_set():
                raise LeaseLostError(f"lease {self.key} could not be renewed")
        finally:
            local.release()

    async def _acquire(self, deadline: float) -> LockToken:
        while True:
            try:
                return await asyncio.to_thread(self.lock_store.acquire, self.key, int(self.ttl_seconds))
            except TimeoutError as exc:
                if time.monotonic() >= deadline:
                    raise LeaseBusyError(f"lease {self.key} is held by another holder") from exc
            await asyncio.sleep(min(self.poll_seconds, max(0.0, deadline - time.monotonic())))

    def _heartbeat(
        self,
        token: LockToken,
        task: asyncio.Task[object],
        stop: threading.Event,
        lost: threading.Event,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        ttl = int(self.ttl_seconds)
        beat = self.heartbeat_seconds
        expires = time.monotonic() + ttl  # 刚拿到租约
        while not stop.wait(timeout=beat):
            attempted = time.monotonic()
            try:
                self.lock_store.renew(token, ttl)
                expires = attempted + ttl
            except Exception:
                if time.monotonic() + beat < expires:
                    logger.warning("lease renewal failed; retrying before it expires key=%s", self.key)
                    continue
                lost.set()
                loop.call_soon_threadsafe(task.cancel)
                return


__all__ = ["Lease", "LeaseBusyError", "LeaseLostError"]
