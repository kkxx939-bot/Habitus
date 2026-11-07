"""单独一轮超过请求体上限时的分块上传：本地服务按投递号暂存各块，收齐、核对摘要后拼回原 remember 请求体。

插件把一批 remember 请求体（序列化后的字节）无损切成几块、按顺序上传，每块带同一个投递号、块序号、总块数、
整批的字节数与 SHA-256。这里只负责暂存与拼回；拼回的请求体走与 ``POST /memory/remember`` 完全相同的处理路径，
幂等、序号与 after_turn 语义都不变（裁定 72 第 2 条；一轮在语义上不拆，裁定 69）。

暂存在内存里：插件 outbox 在最后一块被确认前一直留着这一项，任何失败（包括本地服务重启丢了暂存）都从第 0 块整批重送，
所以暂存不需要活过重启；落盘反而要在本地多存一份会话原文、多一套跨重启的清理。暂存有总字节上限与过期时间，不会无限堆。
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import hashlib
import time
from collections.abc import Callable
from dataclasses import dataclass, field


class RememberPartsRestartError(RuntimeError):
    """这一块接不上已暂存的部分（本地服务重启丢了暂存、过期被清、或块序号跳了）：插件要从第 0 块重送。"""


class RememberPartsCapacityError(RuntimeError):
    """暂存已被别的上传占满：稍后重送。"""


class RememberPartsTooLargeError(RuntimeError):
    """整批请求体本身就超过暂存上限：重送也收不下。"""


class RememberPartsIntegrityError(ValueError):
    """块的内容与它声明的整批字节数或摘要对不上。"""


@dataclass(frozen=True)
class RememberPart:
    delivery_id: str
    part_index: int
    part_count: int
    body_bytes: int
    body_sha256: str
    data: str


@dataclass
class _Upload:
    part_count: int
    body_bytes: int
    body_sha256: str
    updated_at: float
    chunks: list[bytes] = field(default_factory=list)
    staged_bytes: int = 0

    def append(self, chunk: bytes) -> None:
        self.chunks.append(chunk)
        self.staged_bytes += len(chunk)

    def matches(self, part: RememberPart) -> bool:
        return (self.part_count, self.body_bytes, self.body_sha256) == (
            part.part_count,
            part.body_bytes,
            part.body_sha256,
        )


class RememberPartsStaging:
    """按投递号暂存分块上传，收齐且摘要相符时交回整批请求体。"""

    def __init__(
        self,
        *,
        max_bytes: int,
        ttl_seconds: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_bytes <= 0:
            raise ValueError("max_bytes must be a positive integer")
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        self.max_bytes = max_bytes
        self.ttl_seconds = float(ttl_seconds)
        self._clock = clock
        self._uploads: dict[str, _Upload] = {}
        # 各上传已暂存字节数之和，随收块与放掉暂存增减，收一块不必把全部暂存重新累加一遍
        self._staged_bytes = 0
        self._lock = asyncio.Lock()

    @property
    def staged_bytes(self) -> int:
        return self._staged_bytes

    @property
    def pending_deliveries(self) -> tuple[str, ...]:
        return tuple(sorted(self._uploads))

    async def accept(self, part: RememberPart) -> bytes | None:
        """收下一块；收齐时返回拼好的请求体并放掉暂存，没收齐返回 ``None``。"""

        chunk = _decode(part)
        async with self._lock:
            now = self._clock()
            self._expire(now)
            if part.body_bytes > self.max_bytes:
                self._discard(part.delivery_id)
                raise RememberPartsTooLargeError(
                    f"the assembled request of {part.body_bytes} bytes exceeds the "
                    f"{self.max_bytes}-byte remember_parts_max_bytes limit"
                )
            upload = self._next_upload(part, now)
            if upload.staged_bytes + len(chunk) > part.body_bytes:
                self._discard(part.delivery_id)
                raise RememberPartsIntegrityError("remember parts carry more bytes than the declared body_bytes")
            if self.staged_bytes + len(chunk) > self.max_bytes:
                self._discard(part.delivery_id)
                raise RememberPartsCapacityError("remember parts staging is full")
            upload.append(chunk)
            self._staged_bytes += len(chunk)
            upload.updated_at = now
            self._uploads[part.delivery_id] = upload
            if len(upload.chunks) < upload.part_count:
                return None
            self._discard(part.delivery_id)
        body = b"".join(upload.chunks)
        if len(body) != upload.body_bytes or hashlib.sha256(body).hexdigest() != upload.body_sha256:
            raise RememberPartsIntegrityError("assembled remember parts do not match body_bytes and body_sha256")
        return body

    def _next_upload(self, part: RememberPart, now: float) -> _Upload:
        # 第 0 块总是重新开始（重送从头来，旧的暂存作废）；其余块必须正好接在已暂存的后面
        if part.part_index == 0:
            self._discard(part.delivery_id)
            return _Upload(part.part_count, part.body_bytes, part.body_sha256, now)
        upload = self._uploads.get(part.delivery_id)
        if upload is None or not upload.matches(part) or len(upload.chunks) != part.part_index:
            self._discard(part.delivery_id)
            raise RememberPartsRestartError(
                f"part {part.part_index} of delivery {part.delivery_id} does not follow the staged parts; "
                "upload again from part 0"
            )
        return upload

    def _expire(self, now: float) -> None:
        expired = [key for key, upload in self._uploads.items() if now - upload.updated_at > self.ttl_seconds]
        for key in expired:
            self._discard(key)

    def _discard(self, delivery_id: str) -> None:
        upload = self._uploads.pop(delivery_id, None)
        if upload is not None:
            self._staged_bytes -= upload.staged_bytes


def _decode(part: RememberPart) -> bytes:
    try:
        chunk = base64.b64decode(part.data, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise RememberPartsIntegrityError("remember part data must be standard base64") from exc
    if not chunk:
        raise RememberPartsIntegrityError("remember part data must not be empty")
    return chunk


__all__ = [
    "RememberPart",
    "RememberPartsCapacityError",
    "RememberPartsIntegrityError",
    "RememberPartsRestartError",
    "RememberPartsStaging",
    "RememberPartsTooLargeError",
]
