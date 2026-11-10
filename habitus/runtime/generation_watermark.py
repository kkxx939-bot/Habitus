"""预测拍的水位（实时化方案第 5、6 节；缺陷 1）：一个文件 ``<预测根>/<支>/tick_watermark.json``，原子替换。

- ``ticked_at``：上一拍跑到哪一刻（每拍跑完推进，词表迁移中整拍跳过也推进——那一拍系统在看）。下一拍离它超过容差，中间那段就是
  没在看的时段（写成会话一支的「未观测」）；
- ``unwatched_until``：最后一段写成「未观测」的空白的结尾。换代要等水位过了 max(昨天日终, 它) 再留几槽余量（用户 11-10）；
  "上次看到哪一刻"取 ``ticked_at`` 与它里面晚的那个（写了空白、这一拍又没跑完时，空白之后到下一拍之间系统是在看的）；
- ``formed_through`` 与 ``formed_generation``：上一次建预期看到了哪一刻、用的是哪一天那一代的表。这一拍从哪一刻起看：今天这一代
  第一次建预期时从零点起（补建，裁定 94 第 2 条），否则从 ``formed_through`` 起——取代原来只在内存里的 ``now - 一拍``。

进程不留这几样的内存副本：重启后照盘上的接着看。
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta
from datetime import time as clock_time
from pathlib import Path

from habitus.infrastructure.store.filesystem import (
    DurablePathIntegrityError,
    atomic_replace_bytes,
    ensure_real_directory,
    read_regular_bytes,
)

_SCHEMA = "tick_watermark_v1"
_MAX_BYTES = 64 * 1024
_KEYS = frozenset({"schema", "ticked_at", "unwatched_until", "formed_through", "formed_generation"})
WATERMARK_FILENAME = "tick_watermark.json"


class TickWatermarkError(ValueError):
    """水位文件坏了或读不了：明确报出，不当成"从没跑过"（那样会漏写没在看的时段）。"""


@dataclass(frozen=True)
class TickWatermark:
    ticked_at: datetime | None = None
    unwatched_until: datetime | None = None
    formed_through: datetime | None = None
    formed_generation: date | None = None

    def __post_init__(self) -> None:
        for value in (self.ticked_at, self.unwatched_until, self.formed_through):
            if value is not None and value.utcoffset() is None:
                raise TickWatermarkError("tick watermark times must be timezone-aware")

    @property
    def watched_until(self) -> datetime | None:
        """系统确知在看到哪一刻：``ticked_at`` 与最后一段空白的结尾里晚的那个；都没有是 None（第一次启动）。"""

        known = [value for value in (self.ticked_at, self.unwatched_until) if value is not None]
        return max(known) if known else None

    def ticked(self, at: datetime) -> TickWatermark:
        return replace(self, ticked_at=at)

    def unwatched(self, until: datetime) -> TickWatermark:
        return replace(self, unwatched_until=until)

    def formed(self, through: datetime, generation: date) -> TickWatermark:
        return replace(self, formed_through=through, formed_generation=generation)


class TickWatermarkStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve(strict=False)
        self.path = self.root / WATERMARK_FILENAME

    def read(self) -> TickWatermark:
        """盘上的水位；还没有文件是空水位（第一次启动），坏了抛 ``TickWatermarkError``。"""

        try:
            raw = read_regular_bytes(self.path, artifact_root=self.root, max_bytes=_MAX_BYTES)
        except FileNotFoundError:
            return TickWatermark()
        except (OSError, DurablePathIntegrityError) as exc:
            raise TickWatermarkError(f"tick watermark is unreadable: {type(exc).__name__}") from exc
        return decode_watermark(raw)

    def write(self, watermark: TickWatermark) -> None:
        payload = encode_watermark(watermark)
        try:
            ensure_real_directory(self.root, artifact_root=self.root)
            atomic_replace_bytes(self.path, payload, artifact_root=self.root)
        except DurablePathIntegrityError as exc:
            raise TickWatermarkError(f"tick watermark cannot be written: {exc}") from exc


class TickTrack:
    """预测拍怎么用水位（``runtime.foresight_tick`` 每拍调）：读、写没在看的时段、推进、这一拍从哪一刻起看。

    ``slot`` 是一拍（一个槽）多长，``tolerance_slots`` 是容差；``unwatched`` 是会话一支写「未观测」空判断的口（没接就只记水位）。"""

    def __init__(
        self,
        store: TickWatermarkStore,
        *,
        slot: timedelta,
        tolerance_slots: int,
        unwatched: Callable[[datetime, datetime], object] | None = None,
    ) -> None:
        if slot <= timedelta(0) or tolerance_slots < 1:
            raise ValueError("a tick track needs a positive slot and tolerance")
        self.store = store
        self.slot = slot
        self.tolerance_slots = int(tolerance_slots)
        self.unwatched = unwatched

    async def begin(self, now: datetime) -> tuple[TickWatermark, tuple[datetime, datetime] | None]:
        """这一拍开头：读水位；离系统确知在看的那一刻超过容差，就把这一段写成「未观测」、落盘它的结尾。写不成往上抛
        （这一拍报出、水位不推）。第一次启动（盘上没有水位）不写（不知道之前在不在跑）。"""

        mark = await asyncio.to_thread(self.store.read)
        since = mark.watched_until
        if since is None or now - since <= self.slot * self.tolerance_slots:
            return mark, None
        if self.unwatched is not None:
            await asyncio.to_thread(self.unwatched, since, now)
        mark = mark.unwatched(now)
        await self.write(mark)
        return mark, (since, now)

    async def write(self, mark: TickWatermark) -> None:
        await asyncio.to_thread(self.store.write, mark)

    @staticmethod
    def looked_since(now: datetime, mark: TickWatermark, generation: date) -> datetime:
        """这一拍从哪一刻起看：今天这一代第一次建预期从零点起（补建，裁定 94 第 2 条），之后从上一次建预期看到的那一刻起
        （落盘的 ``formed_through``，中间关机、合盖几个小时也接得上）。"""

        midnight = datetime.combine(now.date(), clock_time(0), tzinfo=now.tzinfo)
        if mark.formed_generation != generation or mark.formed_through is None:
            return midnight
        return max(midnight, mark.formed_through)


def encode_watermark(watermark: TickWatermark) -> bytes:
    def stamp(value: datetime | date | None) -> str | None:
        return None if value is None else value.isoformat()

    raw = {
        "schema": _SCHEMA,
        "ticked_at": stamp(watermark.ticked_at),
        "unwatched_until": stamp(watermark.unwatched_until),
        "formed_through": stamp(watermark.formed_through),
        "formed_generation": stamp(watermark.formed_generation),
    }
    return json.dumps(raw, sort_keys=True, indent=2).encode("utf-8")


def decode_watermark(raw: bytes) -> TickWatermark:
    try:
        data = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_keys)
        if not isinstance(data, dict) or set(data) != _KEYS or data["schema"] != _SCHEMA:
            raise TickWatermarkError("tick watermark has unexpected fields or schema")

        def moment(key: str) -> datetime | None:
            value = data[key]
            if value is None:
                return None
            if not isinstance(value, str):
                raise TickWatermarkError(f"tick watermark {key} must be a timestamp")
            return datetime.fromisoformat(value)

        generation = data["formed_generation"]
        if generation is not None and not isinstance(generation, str):
            raise TickWatermarkError("tick watermark formed_generation must be a date")
        return TickWatermark(
            ticked_at=moment("ticked_at"),
            unwatched_until=moment("unwatched_until"),
            formed_through=moment("formed_through"),
            formed_generation=None if generation is None else date.fromisoformat(generation),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        if isinstance(exc, TickWatermarkError):
            raise
        raise TickWatermarkError(f"tick watermark is corrupt: {exc}") from exc


def _unique_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """JSON 对象里同一个键写了两次：不收（F7）。"""

    found: dict[str, object] = {}
    for key, value in pairs:
        if key in found:
            raise TickWatermarkError(f"tick watermark repeats the key {key!r}")
        found[key] = value
    return found


__all__ = [
    "WATERMARK_FILENAME",
    "TickWatermark",
    "TickWatermarkError",
    "TickTrack",
    "TickWatermarkStore",
    "decode_watermark",
    "encode_watermark",
]
