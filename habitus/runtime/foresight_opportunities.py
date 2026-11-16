"""机会表的落盘：一条 lane 一晚一份规范 JSON，整张重写，只留最近几晚（目录与字节的规矩见 ``runtime.foresight_nightly_files``）。

    <root>/opportunities/<lane>/<YYYY-MM-DD>.json

住在组合根：``foresight`` 不许 import infrastructure，形状与编解码在 ``foresight.opportunities``。
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date
from pathlib import Path

from habitus.config.foresight import ForesightStoreConfig
from habitus.foresight.errors import ForesightError
from habitus.foresight.opportunities import OpportunityTable, decode_table, encode_table
from habitus.runtime.foresight_nightly_files import NightlyFiles
from habitus.runtime.generation_marks import GenerationMarkError

OPPORTUNITIES_SEGMENT = "opportunities"


class OpportunityStore:
    def __init__(
        self,
        root: str | Path,
        *,
        retained_nights: int = 3,
        limits: ForesightStoreConfig | None = None,
        finished: Callable[[date], bool | None] | None = None,
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
        #: 某一晚那一代做完了没有（换代的完成标记）：删旧表时留下最晚那张做完了的——连着几天表写出了、完成标记没成时，
        #: 预测拍还有一张做完了的表可用（用户 2026-11-16）。没接就只按晚数删。
        self.finished = finished

    def write(self, table: OpportunityTable) -> Path:
        return self._files.write(
            table.lane,
            table.night,
            encode_table(table).encode("utf-8"),
            keep=None if self.finished is None else self._finished,
        )

    def _finished(self, night: date) -> bool:
        """删旧表时这一晚算不算做完了：与预测拍取表同一个判法（``usable_tables``），标记坏了的不算。"""

        assert self.finished is not None
        try:
            return self.finished(night) is True
        except GenerationMarkError:
            return False

    def lanes(self) -> tuple[str, ...]:
        return self._files.lanes()

    def nights(self, lane: str) -> tuple[date, ...]:
        return self._files.nights(lane)

    def latest_night(self, lane: str) -> date | None:
        return self._files.latest_night(lane)

    def read(self, lane: str, night: date) -> OpportunityTable:
        """读一晚的表；坏了明确报错、点出是哪个文件。"""

        try:
            return decode_table(self._files.read(lane, night), lane=lane, night=night)
        except ForesightError as exc:
            raise type(exc)(f"{self._files.lane_directory(lane) / f'{night.isoformat()}.json'}: {exc}") from exc

    def latest(self, lane: str, *, accept: Callable[[date], bool] | None = None) -> OpportunityTable | None:
        """盘上最晚一晚读得出来的表（最晚那一份坏了往前退一晚，坏的计入观测）；给了 ``accept`` 就只看它认的那几晚
        （预测拍建预期只认那一代做完了的）；没有是 None。"""

        return self._files.latest(lane, lambda text, night: decode_table(text, lane=lane, night=night), accept=accept)

    def take_unreadable(self) -> tuple[str, ...]:
        """上次 ``take`` 以来跳过的坏表（路径），取走即清。"""

        return self._files.take_unreadable()


__all__ = ["OPPORTUNITIES_SEGMENT", "OpportunityStore"]
