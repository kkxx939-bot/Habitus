"""预测层每晚的派生物落盘：一条 lane 一晚一份规范 JSON，整份重写，只留最近几晚。机会表与单元历史共用这一份。

    <root>/<segment>/<lane>/<YYYY-MM-DD>.json

派生物每晚从同一份序列重算，不是账本：同一晚重跑整文件原子替换、回读比对；比盘上最晚一晚更早的晚不写（派生树按固定先后处理
已封口的历史，不往回改）；写完把更早的晚删到只剩 ``retained_nights`` 晚。住在组合根：``foresight`` 不许 import infrastructure，
形状与编解码在 ``foresight`` 里，这里只管目录与字节。

目录里只认 ``<YYYY-MM-DD>.json``：别的名字（编辑器备份、``copy``、写到一半的临时文件）不是哪一晚，跳过不报。读最晚一晚时
那一份读不出来（解析不了、形状或版本不对）就往前退一晚（留的那几代备用就是为这个，26 号清单 6.12），坏的那一份计入观测
（``take_unreadable``）。
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from datetime import date
from pathlib import Path

from habitus.foresight import ForesightError
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


class NightlyFiles:
    """``label`` 只进报错（"opportunity table"、"unit history"）。"""

    def __init__(
        self, root: str | Path, *, segment: str, label: str, max_bytes: int, retained_nights: int, max_entries: int
    ) -> None:
        if isinstance(retained_nights, bool) or not isinstance(retained_nights, int) or retained_nights <= 0:
            raise ForesightError("retained_nights must be a positive integer")
        if isinstance(max_entries, bool) or not isinstance(max_entries, int) or max_entries <= 0:
            raise ForesightError("max_entries must be a positive integer")
        self.root = Path(root).expanduser().resolve(strict=False)
        self.directory = self.root / segment
        self.label = label
        self.max_bytes = max_bytes
        self.retained_nights = retained_nights
        self.max_entries = max_entries
        self._unreadable: list[str] = []
        self._lock = threading.Lock()

    def write(self, lane: str, night: date, payload: bytes) -> Path:
        latest = self.latest_night(lane)
        if latest is not None and night < latest:
            raise ForesightError(f"night {night} comes before the stored night {latest}; history is not rewritten")
        if len(payload) > self.max_bytes:
            raise ForesightError(f"{self.label} exceeds its byte budget")
        directory = self.lane_directory(lane)
        path = directory / f"{night.isoformat()}.json"
        try:
            ensure_real_directory(directory, artifact_root=self.root)
            atomic_replace_bytes(path, payload, artifact_root=self.root)
        except DurablePathIntegrityError as exc:
            raise ForesightError(f"{self.label} cannot be written: {exc}") from exc
        if self._read_bytes(path) != payload:
            raise ForesightError(f"{self.label} failed read-back verification")
        self._prune(lane)
        return path

    def lanes(self) -> tuple[str, ...]:
        if not self._is_directory(self.directory):
            return ()
        return tuple(sorted(entry.name for entry in self._entries(self.directory) if entry.is_dir()))

    def nights(self, lane: str) -> tuple[date, ...]:
        directory = self.lane_directory(lane)
        if not self._is_directory(directory):
            return ()
        found: list[date] = []
        for entry in self._entries(directory):
            if (
                not entry.is_file()
                or entry.name.startswith(".")
                or atomic_temporary_destination(entry.name) is not None
                or not entry.name.endswith(".json")
            ):
                continue
            try:
                night = date.fromisoformat(entry.name[:-5])
            except ValueError:
                continue  # 不是按晚起名的（备份、副本）：不是哪一晚
            if entry.name == f"{night.isoformat()}.json":
                found.append(night)
        return tuple(sorted(found))

    def latest_night(self, lane: str) -> date | None:
        nights = self.nights(lane)
        return nights[-1] if nights else None

    def latest[T](self, lane: str, decode: Callable[[str, date], T]) -> T | None:
        """最晚一晚读得出来的那一份（``decode`` 把一晚的文本解码，坏的抛 ``ForesightError``）：最晚那一份坏了就往前退一晚，
        坏的计入观测；一晚都读不出来是 None。"""

        for night in reversed(self.nights(lane)):
            try:
                return decode(self.read(lane, night), night)
            except ForesightError:
                with self._lock:
                    path = str(self.lane_directory(lane) / f"{night.isoformat()}.json")
                    if path not in self._unreadable:
                        self._unreadable.append(path)
        return None

    def take_unreadable(self) -> tuple[str, ...]:
        """上次 ``take`` 以来读最晚一晚时跳过的坏文件（路径），取走即清。"""

        with self._lock:
            found = tuple(self._unreadable)
            self._unreadable.clear()
        return found

    def read(self, lane: str, night: date) -> str:
        payload = self._read_bytes(self.lane_directory(lane) / f"{night.isoformat()}.json")
        if payload is None:
            raise ForesightError(f"{self.label} of lane {lane!r} for night {night} does not exist")
        try:
            return payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ForesightError(f"{self.label} of lane {lane!r} for night {night} is corrupt: {exc}") from exc

    def lane_directory(self, lane: str) -> Path:
        try:
            name = canonical_path_identity(lane, f"{self.label} lane")
        except ValueError as exc:
            raise ForesightError(str(exc)) from exc
        if name != lane:
            raise ForesightError(f"lane {lane!r} is not a canonical directory name")
        return self.directory / lane

    def _prune(self, lane: str) -> None:
        nights = self.nights(lane)
        for night in nights[: max(0, len(nights) - self.retained_nights)]:
            try:
                durable_unlink(self.lane_directory(lane) / f"{night.isoformat()}.json", artifact_root=self.root)
            except DurablePathIntegrityError as exc:
                raise ForesightError(f"old {self.label} cannot be removed: {exc}") from exc

    def _is_directory(self, directory: Path) -> bool:
        try:
            return real_directory_exists(directory, artifact_root=self.root)
        except DurablePathIntegrityError as exc:
            raise ForesightError(f"{self.label} directory cannot be inspected: {exc}") from exc

    def _entries(self, directory: Path) -> tuple[DurableDirectoryEntry, ...]:
        try:
            return tuple(list_real_directory(directory, artifact_root=self.root, max_entries=self.max_entries))
        except DurablePathIntegrityError as exc:
            raise ForesightError(f"{self.label} directory cannot be listed: {exc}") from exc

    def _read_bytes(self, path: Path) -> bytes | None:
        try:
            if not regular_file_exists(path, artifact_root=self.root):
                return None
            return read_regular_bytes(path, artifact_root=self.root, max_bytes=self.max_bytes)
        except FileNotFoundError:
            return None
        except DurablePathIntegrityError as exc:
            raise ForesightError(f"{self.label} cannot be read: {exc}") from exc


__all__ = ["NightlyFiles"]
