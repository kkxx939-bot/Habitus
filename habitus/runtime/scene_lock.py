"""语义树写锁（实时化阶段 2；语义树根派生，沿用原换代租约的键）：换代读语义树时读到的是一份停住的样子。

- **换代的语义树一步**整步持有（``held``）：同步词表、判关系、写关系表、读完成章都在它里面。
- **模型队列写语义树的那几类活**（同步、写概念、映射、候选条件与先验）每一项在它里面做（``writing``）：命中、完成章、概念、
  先验缓存、提议簿、同步到的版本都落在锁内。**锁内不调模型**：这几类活的模型触点走 ``unlocked`` 包过的那一个结构化客户端，
  调模型的那一段先把锁放开、答复回来再拿回来接着写——模型多慢，换代的语义树一步都拿得到锁。

拿锁的先后固定（``runtime.generation_lease``）：队列租约 → 预测根租约 → sweep 锁 → 语义树写锁。它总是最里面的一把：持有它
的那一段不再去拿别的锁。

同一个实例在同一进程的换代与模型队列之间共用（``Lease`` 自带进程内那把锁）；另一个进程的队列靠跨进程的那一把挡住。
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import Any, TypeVar, cast

from habitus.model_client import ChatCallContext, ChatRequest, StructuredChatClient, StructuredResponse
from habitus.runtime.generation_lease import Lease, LeaseBusyError

T = TypeVar("T")


class SceneWriteLock:
    def __init__(self, lease: Lease) -> None:
        self.lease = lease
        #: 模型队列此刻持着锁的那一项：（哪个任务, 那一次持有）。调模型时放开、回来再拿，都只认这个任务。
        self._owner: asyncio.Task[object] | None = None
        self._holding: AbstractAsyncContextManager[object] | None = None

    def held(self) -> AbstractAsyncContextManager[object]:
        """换代的语义树一步：整步持有。等一个租期还拿不到（另一个进程的队列正写着）抛 ``LeaseBusyError``。"""

        return self.lease.held()

    @property
    def writing_now(self) -> bool:
        """模型队列的某一项此刻持着锁（测试与排查用）。"""

        return self._holding is not None

    @asynccontextmanager
    async def writing(self) -> AsyncIterator[None]:
        """模型队列的一项：整项在锁内，中间调模型的那几段除外（``asking``）。换代的语义树一步正做着就等它做完。"""

        if self._owner is not None:
            raise RuntimeError("the scene write lock is already held by a queue item")
        self._owner = cast("asyncio.Task[object] | None", asyncio.current_task())
        try:
            await self._take()
            try:
                yield
            except BaseException as error:
                # 这一项抛了（含续不上租约时的取消）：把它交回那一次持有——续不上的取消在那里换成 ``LeaseLostError``
                await self._drop(error)
                raise
            await self._drop()
        finally:
            self._owner = None

    @asynccontextmanager
    async def asking(self) -> AsyncIterator[None]:
        """调模型的那一段：持着锁的那一项先放开、答复回来（或抛了）再拿回来；别的调用方（没在锁里）原样过。"""

        if self._owner is None or self._owner is not asyncio.current_task() or self._holding is None:
            yield
            return
        await self._drop()
        try:
            yield
        finally:
            await self._take()

    async def _take(self) -> None:
        while True:
            holding = self.lease.held()
            try:
                await holding.__aenter__()
            except LeaseBusyError:
                continue  # 换代的语义树一步、或另一个进程的队列正持着：接着等（停机取消照常打断）
            self._holding = holding
            return

    async def _drop(self, error: BaseException | None = None) -> None:
        holding, self._holding = self._holding, None
        if holding is None:
            return
        if error is None:
            await holding.__aexit__(None, None, None)
        else:
            await holding.__aexit__(type(error), error, error.__traceback__)


class _UnlockedChat:
    """语义树各模型触点用的结构化客户端：调模型的那一段不持语义树写锁，别的原样转给里面那一个。"""

    def __init__(self, inner: StructuredChatClient, lock: SceneWriteLock) -> None:
        self._inner = inner
        self._lock = lock

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    async def complete_json_async(
        self,
        request: ChatRequest | str,
        *,
        schema: Mapping[str, object],
        name: str = "structured_response",
        validator: Callable[[object], T] | None = None,
        context: ChatCallContext | None = None,
    ) -> StructuredResponse[T | object]:
        async with self._lock.asking():
            return await self._inner.complete_json_async(
                request, schema=schema, name=name, validator=validator, context=context
            )


def unlocked(chat: StructuredChatClient, lock: SceneWriteLock) -> StructuredChatClient:
    """包一层：经它调模型的那一段不在语义树写锁之内。"""

    return cast(StructuredChatClient, _UnlockedChat(chat, lock))


__all__ = ["SceneWriteLock", "unlocked"]
