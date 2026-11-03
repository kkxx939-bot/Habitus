"""机会表的落盘：一条 lane 一晚一份规范 JSON，整张重写，只留最近几晚（目录与字节的规矩见 ``runtime.foresight_nightly_files``）。

    <root>/opportunities/<lane>/<YYYY-MM-DD>.json

住在组合根：``foresight`` 不许 import infrastructure，形状与编解码在 ``foresight.opportunities``。
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

from habitus.config.foresight import ForesightStoreConfig
from habitus.foresight.opportunities import OpportunityTable, decode_table, encode_table
from habitus.runtime.foresight_nightly_files import NightlyFiles

OPPORTUNITIES_SEGMENT = "opportunities"


class OpportunityStore:
    def __init__(
        self, root: str | Path, *, retained_nights: int = 3, limits: ForesightStoreConfig | None = None
    ) -> None:
        limits = limits if limits is not None else ForesightStoreConfig()
        self._files = NightlyFiles(
            root,
            segment=OPPORTUNITIES_SEGMENT,
            label="opportunity table",
            max_bytes=limits.opportunity_table_max_bytes,
            retained_nights=retained_nights,
            max_entries=limits.nightly_max_entries,
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
        """盘上最晚一晚读得出来的表（最晚那一份坏了往前退一晚，坏的计入观测）；没有是 None。"""

        return self._files.latest(lane, lambda text, night: decode_table(text, lane=lane, night=night))

    def take_unreadable(self) -> tuple[str, ...]:
        """上次 ``take`` 以来跳过的坏表（路径），取走即清。"""

        return self._files.take_unreadable()


__all__ = ["OPPORTUNITIES_SEGMENT", "OpportunityStore"]
