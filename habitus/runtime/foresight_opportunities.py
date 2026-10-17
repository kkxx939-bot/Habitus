"""机会表的落盘：一条 lane 一晚一份规范 JSON，整张重写，只留最近几晚。

    <root>/opportunities/<lane>/<YYYY-MM-DD>.json

住在组合根：``foresight`` 不许 import infrastructure，形状与编解码在 ``foresight.opportunities``，这里只管目录与字节。
它是派生物（每晚从同一份序列重算），不是账本：同一晚重跑整文件原子替换、回读比对；比盘上最晚一晚更早的晚不写
（派生树按固定先后处理已封口的历史，不往回改）；写完把更早的晚删到只剩 ``retained_nights`` 晚。
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

from habitus.foresight import ForesightError
from habitus.foresight.opportunities import OpportunityTable, decode_table, encode_table
from habitus.foundation.ids import canonical_path_identity
from habitus.infrastructure.store.filesystem import (
    DurableDirectoryEntry,
    DurablePathIntegrityError,
    atomic_replace_bytes,
    atomic_temporary_destination,
    durable_unlink,
    ensure_real_directory,
    list_real_directory,
    read_regular_bytes,
    real_directory_exists,
    regular_file_exists,
)

OPPORTUNITIES_SEGMENT = "opportunities"
MAX_TABLE_BYTES = 16 * 1024 * 1024
_MAX_ENTRIES = 10_000


class OpportunityStore:
    def __init__(self, root: str | Path, *, retained_nights: int = 3) -> None:
        if isinstance(retained_nights, bool) or not isinstance(retained_nights, int) or retained_nights <= 0:
            raise ForesightError("retained_nights must be a positive integer")
        self.root = Path(root).expanduser().resolve(strict=False)
        self.directory = self.root / OPPORTUNITIES_SEGMENT
        self.retained_nights = retained_nights

    # ── 写 ──────────────────────────────────────────────────────────────────

    def write(self, table: OpportunityTable) -> Path:
        latest = self.latest_night(table.lane)
        if latest is not None and table.night < latest:
            raise ForesightError(
                f"night {table.night} comes before the stored night {latest}; history is not rewritten"
            )
        directory = self._lane_directory(table.lane)
        payload = encode_table(table).encode("utf-8")
        if len(payload) > MAX_TABLE_BYTES:
            raise ForesightError("opportunity table exceeds its byte budget")
        path = directory / f"{table.night.isoformat()}.json"
        try:
            ensure_real_directory(directory, artifact_root=self.root)
            atomic_replace_bytes(path, payload, artifact_root=self.root)
        except DurablePathIntegrityError as exc:
            raise ForesightError(f"opportunity table cannot be written: {exc}") from exc
        if self._read_bytes(path) != payload:
            raise ForesightError("opportunity table failed read-back verification")
        self._prune(table.lane)
        return path

    # ── 读 ──────────────────────────────────────────────────────────────────

    def lanes(self) -> tuple[str, ...]:
        if not self._is_directory(self.directory):
            return ()
        return tuple(sorted(entry.name for entry in self._entries(self.directory) if entry.is_dir()))

    def nights(self, lane: str) -> tuple[date, ...]:
        directory = self._lane_directory(lane)
        if not self._is_directory(directory):
            return ()
        found: list[date] = []
        for entry in self._entries(directory):
            if (
                not entry.is_file()
                or entry.name.startswith(".")
                or atomic_temporary_destination(entry.name) is not None
            ):
                continue
            if not entry.name.endswith(".json"):
                raise ForesightError(f"unexpected entry in the opportunity directory: {entry.name!r}")
            try:
                found.append(date.fromisoformat(entry.name[:-5]))
            except ValueError as exc:
                raise ForesightError(f"opportunity table {entry.name!r} is not named by its night") from exc
        return tuple(sorted(found))

    def latest_night(self, lane: str) -> date | None:
        nights = self.nights(lane)
        return nights[-1] if nights else None

    def read(self, lane: str, night: date) -> OpportunityTable:
        payload = self._read_bytes(self._lane_directory(lane) / f"{night.isoformat()}.json")
        if payload is None:
            raise ForesightError(f"opportunity table of lane {lane!r} for night {night} does not exist")
        try:
            return decode_table(payload.decode("utf-8"), lane=lane, night=night)
        except UnicodeDecodeError as exc:
            raise ForesightError(f"opportunity table of lane {lane!r} for night {night} is corrupt: {exc}") from exc

    def latest(self, lane: str) -> OpportunityTable | None:
        """盘上最晚一晚的表；没有是 None。"""

        night = self.latest_night(lane)
        return None if night is None else self.read(lane, night)

    # ── 内部 ────────────────────────────────────────────────────────────────

    def _prune(self, lane: str) -> None:
        nights = self.nights(lane)
        for night in nights[: max(0, len(nights) - self.retained_nights)]:
            try:
                durable_unlink(self._lane_directory(lane) / f"{night.isoformat()}.json", artifact_root=self.root)
            except DurablePathIntegrityError as exc:
                raise ForesightError(f"old opportunity table cannot be removed: {exc}") from exc

    def _lane_directory(self, lane: str) -> Path:
        try:
            name = canonical_path_identity(lane, "opportunity lane")
        except ValueError as exc:
            raise ForesightError(str(exc)) from exc
        if name != lane:
            raise ForesightError(f"lane {lane!r} is not a canonical directory name")
        return self.directory / lane

    def _is_directory(self, directory: Path) -> bool:
        try:
            return real_directory_exists(directory, artifact_root=self.root)
        except DurablePathIntegrityError as exc:
            raise ForesightError(f"opportunity directory cannot be inspected: {exc}") from exc

    def _entries(self, directory: Path) -> tuple[DurableDirectoryEntry, ...]:
        try:
            return tuple(list_real_directory(directory, artifact_root=self.root, max_entries=_MAX_ENTRIES))
        except DurablePathIntegrityError as exc:
            raise ForesightError(f"opportunity directory cannot be listed: {exc}") from exc

    def _read_bytes(self, path: Path) -> bytes | None:
        try:
            if not regular_file_exists(path, artifact_root=self.root):
                return None
            return read_regular_bytes(path, artifact_root=self.root, max_bytes=MAX_TABLE_BYTES)
        except FileNotFoundError:
            return None
        except DurablePathIntegrityError as exc:
            raise ForesightError(f"opportunity table cannot be read: {exc}") from exc


__all__ = ["MAX_TABLE_BYTES", "OPPORTUNITIES_SEGMENT", "OpportunityStore"]
