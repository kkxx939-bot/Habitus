"""单元历史的落盘（26 号清单 4.9）：每刻一份记录、记过的日子、每晚一份快照。住在组合根：``foresight`` 不许 import
infrastructure，形状与编解码在 ``foresight.history``，这里只管目录与字节；读账本、重标、退出在 ``runtime.foresight_history``。

    <root>/unit_moments/<lane>/<预期身份>.json        一条结完的预期在预测轴上的结果（原始强度）
    <root>/unit_moments/<lane>/_recorded.json        记过的日子
    <root>/unit_history/<lane>/<YYYY-MM-DD>.json      这一晚重标过的快照
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from datetime import date
from pathlib import Path

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
from habitus.infrastructure.store.filesystem import (
    DurablePathIntegrityError,
    atomic_replace_bytes,
    durable_unlink,
    ensure_real_directory,
    list_real_directory,
    read_regular_bytes,
    real_directory_exists,
)
from habitus.runtime.foresight_nightly_files import NightlyFiles

HISTORY_SEGMENT = "unit_history"
MOMENT_SEGMENT = "unit_moments"
MAX_HISTORY_BYTES = 32 * 1024 * 1024
_MAX_RECORD_BYTES = 64 * 1024
_MAX_RECORDS = 200_000
_RECORDED = "_recorded.json"


class HistoryStore:
    def __init__(self, root: str | Path, *, retained_nights: int = 3) -> None:
        self._files = NightlyFiles(
            root,
            segment=HISTORY_SEGMENT,
            label="unit history",
            max_bytes=MAX_HISTORY_BYTES,
            retained_nights=retained_nights,
        )
        self.root = self._files.root
        self._moments = self.root / MOMENT_SEGMENT

    # ── 快照 ────────────────────────────────────────────────────────────────

    def write(self, history: UnitHistory) -> Path:
        return self._files.write(history.lane, history.night, encode_history(history).encode("utf-8"))

    def latest_night(self, lane: str) -> date | None:
        return self._files.latest_night(lane)

    def read(self, lane: str, night: date) -> UnitHistory:
        return decode_history(self._files.read(lane, night), lane=lane, night=night)

    def latest(self, lane: str) -> UnitHistory | None:
        night = self.latest_night(lane)
        return None if night is None else self.read(lane, night)

    # ── 每刻的记录 ──────────────────────────────────────────────────────────

    def put_record(self, lane: str, item: MomentRecord) -> None:
        """同一条预期的记录整份替换（重结之后再结完，结果可能变了）。"""

        self._replace(self._lane(lane) / f"{_name(item.expectation_id)}.json", encode_record(item))

    def records(self, lane: str) -> tuple[MomentRecord, ...]:
        directory = self._lane(lane)
        found = []
        for name in self._names(directory):
            if name == _RECORDED:
                continue
            found.append(decode_record(self._load(directory / name)))
        return tuple(sorted(found, key=lambda item: (item.formed_at, item.expectation_id)))

    def discard_records(self, lane: str, expectation_ids: Iterable[str]) -> int:
        directory = self._lane(lane)
        removed = 0
        for identity in expectation_ids:
            try:
                removed += durable_unlink(directory / f"{_name(identity)}.json", artifact_root=self.root)
            except DurablePathIntegrityError as exc:
                raise ForesightError(f"unit moment record cannot be removed: {exc}") from exc
        return removed

    def recorded_days(self, lane: str) -> frozenset[date]:
        path = self._lane(lane) / _RECORDED
        if _RECORDED not in self._names(self._lane(lane)):
            return frozenset()
        days = self._load(path).get("days")
        if not isinstance(days, list):
            raise ForesightError(f"recorded days of lane {lane!r} are malformed")
        try:
            return frozenset(date.fromisoformat(str(day)) for day in days)
        except (TypeError, ValueError) as exc:
            raise ForesightError(f"recorded days of lane {lane!r} are malformed: {exc}") from exc

    def set_recorded_days(self, lane: str, days: Iterable[date]) -> None:
        self._replace(self._lane(lane) / _RECORDED, {"days": sorted(day.isoformat() for day in days)})

    def _lane(self, lane: str) -> Path:
        try:
            name = canonical_path_identity(lane, "unit moment lane")
        except ValueError as exc:
            raise ForesightError(str(exc)) from exc
        if name != lane:
            raise ForesightError(f"lane {lane!r} is not a canonical directory name")
        return self._moments / lane

    def _names(self, directory: Path) -> tuple[str, ...]:
        try:
            if not real_directory_exists(directory, artifact_root=self.root):
                return ()
            entries = list_real_directory(directory, artifact_root=self.root, max_entries=_MAX_RECORDS)
        except DurablePathIntegrityError as exc:
            raise ForesightError(f"unit moment directory cannot be listed: {exc}") from exc
        return tuple(
            sorted(e.name for e in entries if e.is_file() and e.name.endswith(".json") and not e.name.startswith("."))
        )

    def _replace(self, path: Path, payload: Mapping[str, object]) -> None:
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        if len(encoded) > _MAX_RECORD_BYTES:
            raise ForesightError(f"unit moment file {path.name} exceeds its byte budget")
        try:
            ensure_real_directory(path.parent, artifact_root=self.root)
            atomic_replace_bytes(path, encoded, artifact_root=self.root)
        except DurablePathIntegrityError as exc:
            raise ForesightError(f"unit moment file cannot be written: {exc}") from exc

    def _load(self, path: Path) -> dict[str, object]:
        try:
            raw = json.loads(read_regular_bytes(path, artifact_root=self.root, max_bytes=_MAX_RECORD_BYTES))
        except (DurablePathIntegrityError, ValueError) as exc:
            raise ForesightError(f"unit moment file {path.name} cannot be read: {exc}") from exc
        if not isinstance(raw, dict):
            raise ForesightError(f"unit moment file {path.name} is not an object")
        return raw


def _name(expectation_id: str) -> str:
    try:
        name = canonical_path_identity(expectation_id, "expectation id")
    except ValueError as exc:
        raise ForesightError(str(exc)) from exc
    if name != expectation_id:
        raise ForesightError(f"expectation id {expectation_id!r} is not a canonical file name")
    return name


__all__ = ["HISTORY_SEGMENT", "MOMENT_SEGMENT", "HistoryStore"]
