"""``scene/occurrences/`` 这一支的存储：与行为树同构，``occurrences/YYYY/MM/DD/<行为叶名>.md``。

一条命中记录一份、写完不改（重做按地址覆写）。**完成标记 ``.done.json`` 按天**：日目录在第一条记录落盘
前就存在（原子写自己会建父目录），所以"这天做完没"不能看目录、只能看标记。标记里记：

- 当天应有几条（回读对不上就是中间掉了东西）；
- 映射口径（换了提示词/模型/概念集要全量重做，靠它认出来）；
- 当天**命中过哪些概念**（结算窗时先读一个几 KB 的标记就能整天跳过，不用逐条解码）；
- 未决几条（这天有几条没判成，夜批据此决定要不要重跑）。

一天没有任何 occurrence 也要有标记（记录数 0），否则夜批会永远重扫它。

标记纪律的另一半：**重做之前先撤**。``retain_only`` 删记录的同一步把标记也撤掉——删了记录还留着
"这天做完了"，崩溃后留下的就不是"没做完"（可自愈），而是"做完了但内容不对"（无人察觉）。
``complete_day`` 除了核条数还核每条的口径，不让两代口径混着盖章。
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import ClassVar

from habitus.behavior.model import BehaviorAddress, BehaviorKind, is_ascii_digits
from habitus.behavior.uri import BehaviorURI
from habitus.foundation.integrity import canonical_json
from habitus.scene.codec import SceneRecordError
from habitus.scene.occurrences.document import decode, encode
from habitus.scene.occurrences.model import ConceptHits
from habitus.scene.storage import SceneStore

OCCURRENCES_SEGMENT = "occurrences"
DONE_FILENAME = ".done.json"
MAX_RECORD_BYTES = 64 * 1024
MAX_MARKER_BYTES = 256 * 1024
#: 时刻窗换算成本地日目录时两侧各多展开的天数：记录按 occurrence 自己的本地日归目录，窗的起止带的可能是
#: 另一个偏移，差一天以内。
_WINDOW_DAY_MARGIN = 1


class ConceptHitStoreError(ValueError):
    """命中存储的路径逃逸、损坏记录，或与标记不符的一天。"""


@dataclass(frozen=True)
class DayMarker:
    """一天的完成标记。"""

    day: date
    records: int
    unresolved: int
    concepts: frozenset[str]
    mapper: str
    completed_at: datetime


class ConceptHitStore(SceneStore):
    error_type: ClassVar[type[ValueError]] = ConceptHitStoreError

    def __init__(self, scene_root: str | Path, **options: int) -> None:
        super().__init__(scene_root, **options)
        self.directory = self._inside(Path(OCCURRENCES_SEGMENT))

    # ── 写 ──────────────────────────────────────────────────────────────────

    def write(self, record: ConceptHits) -> Path:
        if not isinstance(record, ConceptHits):
            raise TypeError("record must be ConceptHits")
        path = self.path_for(record.address)
        self._atomic_write(path, encode(record).encode("utf-8"), maximum=MAX_RECORD_BYTES)
        return path

    def complete_day(self, day: date, *, records: int, completed_at: datetime, mapper: str) -> DayMarker:
        """把这一天标记为映射完成。**最后一步**：全部记录落盘之后才调。核条数、核口径、汇总命中概念。"""

        if isinstance(records, bool) or not isinstance(records, int) or records < 0:
            raise ValueError("records must be a non-negative integer")
        if not isinstance(completed_at, datetime) or completed_at.utcoffset() is None:
            raise ValueError("completed_at must be a timezone-aware datetime")
        if not isinstance(mapper, str) or not mapper.strip():
            raise ValueError("mapper must be non-empty text")
        found = self.read_day(day)
        if len(found) != records:
            raise ConceptHitStoreError(f"day claims {records} concept hit records but {len(found)} are readable")
        foreign = sorted({item.mapper for item in found} - {mapper})
        if foreign:
            raise ConceptHitStoreError(f"day mixes records of another mapper: {foreign}")
        directory = self._day_path(day)
        # 提交前顺手清掉崩溃遗留的临时文件与 .DS_Store 这类噪音：它们计入目录条目上限，没人清就一直堆着。
        self._discard_noise(directory, keep=frozenset({DONE_FILENAME}))
        concepts = frozenset().union(*(set(item.graded_hits) | set(item.graded_situations) for item in found)) if found else frozenset()
        marker = DayMarker(
            day=day,
            records=records,
            unresolved=sum(1 for item in found if item.unresolved),
            concepts=concepts,
            mapper=mapper,
            completed_at=completed_at.astimezone(UTC),
        )
        payload = canonical_json(
            {
                "day": day.isoformat(),
                "records": marker.records,
                "unresolved": marker.unresolved,
                "concepts": sorted(marker.concepts),
                "mapper": marker.mapper,
                "completed_at": marker.completed_at,
            }
        ).encode("utf-8")
        self._atomic_write(directory / DONE_FILENAME, payload, maximum=MAX_MARKER_BYTES)
        return marker

    def retain_only(self, day: date, keep: frozenset[str]) -> tuple[str, ...]:
        """重做一天前先清：撤掉完成标记，删掉不在 ``keep``（叶名集合）里的记录和噪音文件。

        删记录与撤标记必须同一步——留着标记等于宣称一个已经被改过的目录"做完了"。
        """

        directory = self._day_path(day)
        self.discard_completion(day)
        records, _noise = self._content_files(directory)
        removed: list[str] = []
        for entry in records:
            if entry.name.endswith(".md") and entry.name[:-3] in keep:
                continue
            self._discard(directory / entry.name)
            removed.append(entry.name.removesuffix(".md"))
        self._discard_noise(directory)
        return tuple(sorted(removed))

    def discard_completion(self, day: date) -> bool:
        marker = self._day_path(day) / DONE_FILENAME
        if not self._file_exists(marker):
            return False
        self._discard(marker)
        return True

    # ── 读 ──────────────────────────────────────────────────────────────────

    def read(self, address: BehaviorAddress) -> ConceptHits:
        path = self.path_for(address)
        payload = self._read_bytes(path, MAX_RECORD_BYTES)
        if payload is None:
            raise ConceptHitStoreError("concept hits record does not exist")
        return self._decode(payload, address)

    def exists(self, address: BehaviorAddress) -> bool:
        return self._file_exists(self.path_for(address))

    def read_day(self, day: date) -> tuple[ConceptHits, ...]:
        """这一天全部命中记录，按 occurrence 的时刻升序（不按文件名：不同偏移下两者会分叉）。

        点文件与原子写临时文件是噪音，跳过；别的非记录文件报错。
        """

        directory = self._day_path(day)
        records: list[ConceptHits] = []
        entries, _noise = self._content_files(directory)
        for entry in entries:
            if not entry.name.endswith(".md"):
                raise ConceptHitStoreError(f"a concept hits day may contain only Markdown records: {entry.name!r}")
            leaf = entry.name[:-3]
            try:
                address = BehaviorAddress.from_identity(BehaviorKind.OCCURRENCE, day, leaf)
            except (TypeError, ValueError) as exc:
                raise ConceptHitStoreError("a concept hits record has a non-canonical leaf name") from exc
            if address.identity_name != leaf:
                raise ConceptHitStoreError(f"concept hits file {entry.name!r} does not carry a canonical leaf name")
            records.append(self.read(address))
        return tuple(sorted(records, key=lambda item: (item.started_at.astimezone(UTC), item.occurrence_uri)))

    def read_window(self, start: datetime, end: datetime) -> tuple[ConceptHits, ...]:
        """``started_at`` 落在 [start, end) 的全部记录，按时刻升序。

        把"时刻窗 ↔ 本地日目录"的换算钉在这一层：两侧各多展开一天，再按时刻过滤，账本不用自己拼日子。
        """

        for label, value in (("start", start), ("end", end)):
            if not isinstance(value, datetime) or value.utcoffset() is None:
                raise ValueError(f"{label} must be a timezone-aware datetime")
        if end <= start:
            return ()
        first = start.date() - timedelta(days=_WINDOW_DAY_MARGIN)
        last = end.date() + timedelta(days=_WINDOW_DAY_MARGIN)
        found: list[ConceptHits] = []
        day = first
        while day <= last:
            found.extend(item for item in self.read_day(day) if start <= item.started_at < end)
            day += timedelta(days=1)
        return tuple(sorted(found, key=lambda item: (item.started_at.astimezone(UTC), item.occurrence_uri)))

    def read_marker(self, day: date) -> DayMarker | None:
        payload = self._read_bytes(self._day_path(day) / DONE_FILENAME, MAX_MARKER_BYTES)
        if payload is None:
            return None
        return self._marker(day, payload)

    def days_done(self, *, mapper: str | None = None) -> frozenset[date]:
        """映射**完成**的日期，只认标记。给了 ``mapper`` 就只算那个口径做的。"""

        days: set[date] = set()
        for day in self._day_directories():
            marker = self.read_marker(day)
            if marker is None or (mapper is not None and marker.mapper != mapper):
                continue
            days.add(day)
        return frozenset(days)

    def path_for(self, address: BehaviorAddress) -> Path:
        if not isinstance(address, BehaviorAddress):
            raise TypeError("address must be a BehaviorAddress")
        if address.kind is not BehaviorKind.OCCURRENCE:
            raise ConceptHitStoreError("concept hits are recorded for occurrences only")
        return self._day_path(address.occurred_on) / f"{address.identity_name}.md"

    # ── 内部 ────────────────────────────────────────────────────────────────

    def _day_path(self, day: date) -> Path:
        if isinstance(day, datetime) or not isinstance(day, date):
            raise TypeError("day must be a date")
        return self._inside(Path(OCCURRENCES_SEGMENT, f"{day.year:04d}", f"{day.month:02d}", f"{day.day:02d}"))

    def _day_directories(self) -> Sequence[date]:
        days: list[date] = []
        for year in self._directories(self.directory):
            if not _is_digits(year.name, 4):
                continue
            for month in self._directories(year):
                if not _is_digits(month.name, 2):
                    continue
                for day in self._directories(month):
                    if not _is_digits(day.name, 2):
                        continue
                    try:
                        days.append(date(int(year.name), int(month.name), int(day.name)))
                    except ValueError:
                        continue
        return sorted(days)

    def _marker(self, day: date, payload: bytes) -> DayMarker:
        try:
            raw = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ConceptHitStoreError(f"completion marker for {day} is corrupt") from exc
        try:
            if not isinstance(raw, dict) or raw["day"] != day.isoformat():
                raise ConceptHitStoreError(f"completion marker for {day} names another day")
            return DayMarker(
                day=day,
                records=int(raw["records"]),
                unresolved=int(raw["unresolved"]),
                concepts=frozenset(str(item) for item in raw["concepts"]),
                mapper=str(raw["mapper"]),
                completed_at=datetime.fromisoformat(str(raw["completed_at"])),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ConceptHitStoreError(f"completion marker for {day} is malformed") from exc

    def _decode(self, payload: bytes, address: BehaviorAddress) -> ConceptHits:
        try:
            return decode(payload.decode("utf-8"), expected_uri=str(BehaviorURI.from_address(address)))
        except (UnicodeDecodeError, SceneRecordError) as exc:
            raise ConceptHitStoreError(f"concept hits record {address.identity_name!r} is corrupt: {exc}") from exc


def _is_digits(value: str, width: int) -> bool:
    return len(value) == width and is_ascii_digits(value)


__all__ = [
    "DONE_FILENAME",
    "MAX_RECORD_BYTES",
    "OCCURRENCES_SEGMENT",
    "ConceptHitStore",
    "ConceptHitStoreError",
    "DayMarker",
]
