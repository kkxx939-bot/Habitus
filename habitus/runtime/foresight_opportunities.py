"""机会表的落盘：一条 lane 一晚一份规范 JSON，整张重写，只留最近几晚（目录与字节的规矩见 ``runtime.foresight_nightly_files``）。

    <root>/opportunities/<lane>/<YYYY-MM-DD>.json

住在组合根：``foresight`` 不许 import infrastructure，形状与编解码在 ``foresight.opportunities``。
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

from habitus.foresight.opportunities import OpportunityTable, decode_table, encode_table
from habitus.runtime.foresight_nightly_files import NightlyFiles

OPPORTUNITIES_SEGMENT = "opportunities"
MAX_TABLE_BYTES = 16 * 1024 * 1024


class OpportunityStore:
    def __init__(self, root: str | Path, *, retained_nights: int = 3) -> None:
        self._files = NightlyFiles(
            root,
            segment=OPPORTUNITIES_SEGMENT,
            label="opportunity table",
            max_bytes=MAX_TABLE_BYTES,
            retained_nights=retained_nights,
        )
        self.root = self._files.root
        self.directory = self._files.directory
        self.retained_nights = retained_nights

    def write(self, table: OpportunityTable) -> Path:
        return self._files.write(table.lane, table.night, encode_table(table).encode("utf-8"))

    def lanes(self) -> tuple[str, ...]:
        return self._files.lanes()

    def nights(self, lane: str) -> tuple[date, ...]:
        return self._files.nights(lane)

    def latest_night(self, lane: str) -> date | None:
        return self._files.latest_night(lane)

    def read(self, lane: str, night: date) -> OpportunityTable:
        return decode_table(self._files.read(lane, night), lane=lane, night=night)

    def latest(self, lane: str) -> OpportunityTable | None:
        """盘上最晚一晚的表；没有是 None。"""

        night = self.latest_night(lane)
        return None if night is None else self.read(lane, night)


__all__ = ["MAX_TABLE_BYTES", "OPPORTUNITIES_SEGMENT", "OpportunityStore"]
