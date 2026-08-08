"""可跨异步模型调用持有并续租的耐久处理锁。"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from infrastructure.store.contracts import PathLock


class RenewableProcessingLock:
    def __init__(
        self,
        path_lock: PathLock,
        *,
        renewal_interval_seconds: float = 60.0,
        ttl_seconds: int = 300,
        wait_timeout_seconds: float = 30.0,
    ) -> None:
        if renewal_interval_seconds <= 0:
            raise ValueError("renewal_interval_seconds must be positive")
        self.path_lock = path_lock
        self.renewal_interval_seconds = renewal_interval_seconds
        self.ttl_seconds = ttl_seconds
        self.wait_timeout_seconds = wait_timeout_seconds

    @asynccontextmanager
    async def acquire(self, processing_identity: str) -> AsyncIterator[object]:
        context = self.path_lock.acquire(
            "behavior-processing:" + processing_identity,
            ttl_seconds=self.ttl_seconds,
            wait_timeout_seconds=self.wait_timeout_seconds,
            retry_delay_seconds=0.02,
        )
        guard = await asyncio.to_thread(context.__enter__)
        owner_task = asyncio.current_task()
        if owner_task is None:
            await asyncio.to_thread(context.__exit__, None, None, None)
            raise RuntimeError("processing lock requires an active asyncio Task")
        renewal_error: BaseException | None = None

        async def renew() -> None:
            nonlocal renewal_error
            try:
                while True:
                    await asyncio.sleep(self.renewal_interval_seconds)
                    await asyncio.to_thread(guard.checkpoint)
            except asyncio.CancelledError:
                raise
            except BaseException as exc:
                renewal_error = exc
                owner_task.cancel()

        renewal = asyncio.create_task(renew())
        body_error: BaseException | None = None
        try:
            yield guard
            if renewal_error is not None:
                raise RuntimeError("processing lease renewal failed") from renewal_error
        except asyncio.CancelledError as exc:
            if renewal_error is not None:
                body_error = RuntimeError("processing lease renewal failed")
                raise body_error from renewal_error
            body_error = exc
            raise
        except BaseException as exc:
            body_error = exc
            raise
        finally:
            renewal.cancel()
            try:
                await renewal
            except BaseException as exc:
                if not isinstance(exc, asyncio.CancelledError) and renewal_error is None:
                    renewal_error = exc
            try:
                await asyncio.to_thread(context.__exit__, None, None, None)
            except Exception as exc:
                if body_error is None:
                    raise
                raise body_error from exc
            if body_error is None and renewal_error is not None:
                raise RuntimeError("processing lease renewal failed") from renewal_error


__all__ = ["RenewableProcessingLock"]
