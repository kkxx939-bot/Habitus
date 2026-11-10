"""单元历史的落盘（26 号清单 4.9）：每刻一份记录、记过的日子、每晚一份快照。住在组合根：``foresight`` 不许 import
infrastructure，形状与编解码在 ``foresight.history``，这里只管目录与字节；读账本、重标、退出在 ``runtime.foresight_history``。

    <root>/unit_moments/<lane>/<预期身份>.json        一条结完的预期在预测轴上的结果（原始强度）
    <root>/unit_moments/<lane>/_recorded.json        记过的日子
    <root>/unit_history/<lane>/<YYYY-MM-DD>.json      这一晚重标过的快照

坏文件按预测层各 store 的同一个总则（``runtime.foresight_files``）：一份读不出来（解析不了、旧 schema）的每刻记录跳过、计入观测
（``take_unreadable``），快照照写、压实照常——它那一刻从读数里少了，不让整晚停；"记过的日子"读不出来就当没记过，结完的日子
整天重记（同一条预期的记录整份替换，幂等）。
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date
from pathlib import Path
from typing import Any

from habitus.config.foresight import ForesightStoreConfig
from habitus.foresight import ForesightError
from habitus.foresight.history import (
    MomentRecord,
    UnitHistory,
    decode_history,
    decode_record,
    encode_history,
    encode_record,
)
from habitus.foundation.ids import canonical_path_identity
from habitus.runtime.foresight_files import JsonFiles
from habitus.runtime.foresight_nightly_files import NightlyFiles

HISTORY_SEGMENT = "unit_history"
MOMENT_SEGMENT = "unit_moments"
_RECORDED = "_recorded.json"


class HistoryStore:
    def __init__(
        self, root: str | Path, *, retained_nights: int = 3, limits: ForesightStoreConfig | None = None
    ) -> None:
        limits = limits if limits is not None else ForesightStoreConfig()
        self._files = NightlyFiles(
            root,
            segment=HISTORY_SEGMENT,
            label="unit history",
            max_bytes=limits.unit_history_max_bytes,
            retained_nights=retained_nights,
            max_entries=limits.nightly_max_entries,
        )
        self.root = self._files.root
        self._moments = self.root / MOMENT_SEGMENT
        self._records = JsonFiles(
            self.root,
            label="unit moment",
            max_bytes=limits.unit_moment_max_record_bytes,
            max_entries=limits.unit_moment_max_records,
            max_unreadable=limits.unreadable_max_paths,
        )

    # ── 快照 ────────────────────────────────────────────────────────────────

    def write(self, history: UnitHistory) -> Path:
        return self._files.write(history.lane, history.night, encode_history(history).encode("utf-8"))

    def latest_night(self, lane: str) -> date | None:
        return self._files.latest_night(lane)

    def read(self, lane: str, night: date) -> UnitHistory:
        return decode_history(self._files.read(lane, night), lane=lane, night=night)

    def stamp(self, lane: str, night: date) -> tuple[int, int, int] | None:
        """这一晚快照文件此刻的样子（inode、修改时刻、大小）：同一晚被重写后不一样，读侧的缓存据它失效；没有文件是 None。"""

        try:
            info = (self._files.lane_directory(lane) / f"{night.isoformat()}.json").stat()
        except FileNotFoundError:
            return None
        return info.st_ino, info.st_mtime_ns, info.st_size

    def latest(self, lane: str) -> UnitHistory | None:
        """盘上最晚一晚读得出来的快照（最晚那一份坏了往前退一晚，坏的计入观测）；没有是 None。"""

        return self._files.latest(lane, lambda text, night: decode_history(text, lane=lane, night=night))

    # ── 每刻的记录 ──────────────────────────────────────────────────────────

    def put_record(self, lane: str, item: MomentRecord) -> None:
        """同一条预期的记录整份替换（重结之后再结完，结果可能变了）。"""

        self._records.replace(self._lane(lane) / f"{_name(item.expectation_id)}.json", encode_record(item))

    def records(self, lane: str) -> tuple[MomentRecord, ...]:
        """这条 lane 的全部每刻记录；读不出来的跳过、计入观测。"""

        found = [item for _stem, item in self._records.decoded(self._lane(lane), decode_record)]
        return tuple(sorted(found, key=lambda item: (item.formed_at, item.expectation_id)))

    def discard_records(self, lane: str, expectation_ids: Iterable[str]) -> int:
        directory = self._lane(lane)
        return sum(1 for identity in expectation_ids if self._records.remove(directory / f"{_name(identity)}.json"))

    def recorded_days(self, lane: str) -> frozenset[date]:
        """记过的日子；读不出来当没记过（计入观测）。"""

        found = self._records.read(self._lane(lane) / _RECORDED, _recorded_days)
        return frozenset() if found is None else found

    def set_recorded_days(self, lane: str, days: Iterable[date]) -> None:
        self._records.replace(self._lane(lane) / _RECORDED, {"days": sorted(day.isoformat() for day in days)})

    def take_unreadable(self) -> tuple[str, ...]:
        """上次 ``take`` 以来跳过的坏文件（路径），取走即清。"""

        return (*self._records.take_unreadable(), *self._files.take_unreadable())

    def _lane(self, lane: str) -> Path:
        try:
            name = canonical_path_identity(lane, "unit moment lane")
        except ValueError as exc:
            raise ForesightError(str(exc)) from exc
        if name != lane:
            raise ForesightError(f"lane {lane!r} is not a canonical directory name")
        return self._moments / lane


def _recorded_days(raw: dict[str, Any]) -> frozenset[date]:
    days = raw.get("days")
    try:
        if not isinstance(days, list):
            raise TypeError("days must be a list")
        return frozenset(date.fromisoformat(str(day)) for day in days)
    except (TypeError, ValueError) as exc:
        raise ForesightError(f"recorded days are malformed: {exc}") from exc


def _name(expectation_id: str) -> str:
    try:
        name = canonical_path_identity(expectation_id, "expectation id")
    except ValueError as exc:
        raise ForesightError(str(exc)) from exc
    if name != expectation_id:
        raise ForesightError(f"expectation id {expectation_id!r} is not a canonical file name")
    return name


__all__ = ["HISTORY_SEGMENT", "MOMENT_SEGMENT", "HistoryStore"]
