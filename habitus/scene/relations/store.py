"""``scene/relations/`` 这一支的存储（语义树新方案 ``13`` ④）：

```
relations/<lane>/<YYYY-MM-DD>.json     第 N 晚（截止日 N）这条 lane 全部关系的读数与状态，规范 JSON
relations/<lane>/transitions.jsonl     状态迁移日志：一行一条，只追加
```

一晚新增约两个文件，代替"每次前因一本账"（冒烟里 27 条 occurrence 生出 470 个账文件）。不依赖本地数据库表（插件交付）。

- **按固定先后、不往回改**：只能写比盘上最晚一晚更晚的一晚，或者重写最晚那一晚（同一晚重跑）；写更早的一晚拒绝
  （派生树按固定先后处理已封口的历史，不做"输入变了就重建"）。
- **迁移日志只追加**：重写同一晚时，那一晚已经记过的迁移与这一次相同就不再写；不同（输入变了）就把这一晚的新迁移
  整批追加、带上第几次写（``revision``），读的人取每晚最后一次。已写下的行永远不改。
- 重写一晚整个文件原子替换、回读比对（``SceneStore``）。
"""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path
from typing import ClassVar

from habitus.foundation.ids import canonical_path_identity
from habitus.foundation.integrity import canonical_json
from habitus.scene.relations.document import (
    RelationRecordError,
    decode_night,
    decode_transition,
    encode_night,
    encode_transition,
)
from habitus.scene.relations.state import LaneState, Transition
from habitus.scene.storage import SceneStore

RELATIONS_SEGMENT = "relations"
TRANSITIONS_FILENAME = "transitions.jsonl"
MAX_NIGHT_BYTES = 16 * 1024 * 1024
MAX_LOG_BYTES = 64 * 1024 * 1024


class RelationStoreError(ValueError):
    """关系表的路径逃逸、损坏记录，或违反先后顺序的写入。"""


class RelationStore(SceneStore):
    error_type: ClassVar[type[ValueError]] = RelationStoreError

    def __init__(self, scene_root: str | Path, **options: int) -> None:
        super().__init__(scene_root, **options)
        self.directory = self._inside(Path(RELATIONS_SEGMENT))

    # ── 写 ──────────────────────────────────────────────────────────────────

    def write(self, state: LaneState, transitions: tuple[Transition, ...]) -> Path:
        """写第 ``state.night`` 晚：先追加迁移、再写这一晚的状态文件（状态文件在，就说明这一晚完整了）。"""

        latest = self.latest_night(state.lane)
        if latest is not None and state.night < latest:
            raise RelationStoreError(
                f"night {state.night} comes before the stored night {latest}; history is not rewritten"
            )
        if any(item.night != state.night or item.key.lane != state.lane for item in transitions):
            raise RelationStoreError("transitions must belong to the night and lane being written")
        directory = self._lane_directory(state.lane)
        self._ensure_directory(directory)
        payload = encode_night(state).encode("utf-8")
        # 每批迁移带上它那一晚状态文件的摘要：先追加迁移、再写状态文件，中途被杀时日志里会多一批"没有状态文件"或"和状态文件
        # 对不上"的迁移；读的时候只认摘要对得上的那一批（第四轮评审 E14）
        self._append_transitions(state.lane, state.night, transitions, _digest(payload))
        path = directory / f"{state.night.isoformat()}.json"
        self._atomic_write(path, payload, maximum=MAX_NIGHT_BYTES)
        return path

    # ── 读 ──────────────────────────────────────────────────────────────────

    def lanes(self) -> tuple[str, ...]:
        return tuple(sorted(path.name for path in self._directories(self.directory)))

    def nights(self, lane: str) -> tuple[date, ...]:
        records, _noise = self._content_files(self._lane_directory(lane))
        found: list[date] = []
        for entry in records:
            if entry.name == TRANSITIONS_FILENAME:
                continue
            if not entry.name.endswith(".json"):
                raise RelationStoreError(f"unexpected entry in the relation directory: {entry.name!r}")
            try:
                found.append(date.fromisoformat(entry.name[:-5]))
            except ValueError as exc:
                raise RelationStoreError(f"relation night file {entry.name!r} is not named by its date") from exc
        return tuple(sorted(found))

    def latest_night(self, lane: str, *, before: date | None = None) -> date | None:
        nights = [night for night in self.nights(lane) if before is None or night < before]
        return nights[-1] if nights else None

    def stamp(self, lane: str, night: date) -> tuple[int, int] | None:
        """那一晚状态文件的（修改时间纳秒, 字节数）：读侧缓存的指纹——同一晚被重写（重跑最晚那一晚）也看得出来。没有这一晚是 None。"""

        path = self._lane_directory(lane) / f"{night.isoformat()}.json"
        try:
            info = path.stat()
        except FileNotFoundError:
            return None
        return info.st_mtime_ns, info.st_size

    def read(self, lane: str, night: date) -> LaneState:
        payload = self._read_bytes(self._lane_directory(lane) / f"{night.isoformat()}.json", MAX_NIGHT_BYTES)
        if payload is None:
            raise RelationStoreError(f"relation night {night} of lane {lane!r} does not exist")
        try:
            return decode_night(payload.decode("utf-8"), lane=lane, night=night)
        except (UnicodeDecodeError, RelationRecordError) as exc:
            raise RelationStoreError(f"relation night {night} of lane {lane!r} is corrupt: {exc}") from exc

    def previous(self, lane: str, night: date) -> LaneState | None:
        """第 ``night`` 晚折叠时要接的那一晚：盘上早于它的最近一晚（重跑最晚那一晚时，接的是它之前那一晚）。"""

        earlier = self.latest_night(lane, before=night)
        return None if earlier is None else self.read(lane, earlier)

    def transitions(self, lane: str) -> tuple[Transition, ...]:
        """每晚取与盘上状态文件对得上的最后一批迁移，按晚排；没有状态文件的那一晚（写到一半被杀）不算。"""

        batches: dict[tuple[date, int], tuple[str, list[Transition]]] = {}
        for revision, night, item, state in self._log(lane):
            batch = batches.setdefault((night, revision), (state, []))
            if item is not None:
                batch[1].append(item)
        stored = {night: self._state_digest(lane, night) for night, _revision in batches}
        chosen: dict[date, list[Transition]] = {}
        for (night, _revision), (state, items) in sorted(batches.items()):
            if stored[night] is not None and state == stored[night]:
                chosen[night] = items
        return tuple(item for night in sorted(chosen) for item in chosen[night])

    # ── 内部 ────────────────────────────────────────────────────────────────

    def _lane_directory(self, lane: str) -> Path:
        try:
            name = canonical_path_identity(lane, "relation lane")
        except ValueError as exc:
            raise RelationStoreError(str(exc)) from exc
        if name != lane:
            raise RelationStoreError(f"lane {lane!r} is not a canonical directory name")
        return self._inside(Path(RELATIONS_SEGMENT, lane))

    def _state_digest(self, lane: str, night: date) -> str | None:
        payload = self._read_bytes(self._lane_directory(lane) / f"{night.isoformat()}.json", MAX_NIGHT_BYTES)
        return None if payload is None else _digest(payload)

    def _log(self, lane: str) -> list[tuple[int, date, Transition | None, str]]:
        """（第几次写, 哪一晚, 迁移——空标记是 None, 那一批对应的状态文件摘要）。"""

        payload = self._read_bytes(self._lane_directory(lane) / TRANSITIONS_FILENAME, MAX_LOG_BYTES)
        if payload is None:
            return []
        rows: list[tuple[int, date, Transition | None, str]] = []
        for number, line in enumerate(payload.decode("utf-8").splitlines(), start=1):
            try:
                envelope = json.loads(line)
                revision = int(envelope["revision"])
                state = str(envelope["state"])
                if "empty_night" in envelope:
                    night = date.fromisoformat(envelope["empty_night"])
                    expected = _empty_line(revision, night, state)
                    rows.append((revision, night, None, state))
                else:
                    item = decode_transition(canonical_json(envelope["transition"]), lane=lane)
                    expected = _line(revision, item, state)
                    rows.append((revision, item.night, item, state))
            except (KeyError, TypeError, ValueError) as exc:
                raise RelationStoreError(f"transition log line {number} of lane {lane!r} is corrupt: {exc}") from exc
            if expected != line:
                raise RelationStoreError(f"transition log line {number} of lane {lane!r} is not in canonical form")
        return rows

    def _append_transitions(self, lane: str, night: date, transitions: tuple[Transition, ...], state: str) -> None:
        rows = self._log(lane)
        revisions = [revision for revision, when, _item, _state in rows if when == night]
        if revisions:
            last = max(revisions)
            recorded = [
                (item, digest)
                for revision, when, item, digest in rows
                if when == night and revision == last and item is not None
            ]
            last_state = next(digest for revision, when, _item, digest in rows if when == night and revision == last)
            if [item for item, _digest in recorded] == list(transitions) and last_state == state:
                return  # 同一晚重跑、结果相同：不重复记
            revision = last + 1
        elif not transitions:
            return
        else:
            revision = 1
        path = self._lane_directory(lane) / TRANSITIONS_FILENAME
        existing = self._read_bytes(path, MAX_LOG_BYTES) or b""
        # 重跑之后这一晚没有迁移了：记一行空标记，读的人才知道最后一次是"没有"
        lines = "".join(_line(revision, item, state) + "\n" for item in transitions) or (
            _empty_line(revision, night, state) + "\n"
        )
        self._atomic_write(path, existing + lines.encode("utf-8"), maximum=MAX_LOG_BYTES)


def _line(revision: int, item: Transition, state: str) -> str:
    return canonical_json({"revision": revision, "state": state, "transition": json.loads(encode_transition(item))})


def _empty_line(revision: int, night: date, state: str) -> str:
    return canonical_json({"revision": revision, "state": state, "empty_night": night.isoformat()})


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()[:16]


__all__ = ["RELATIONS_SEGMENT", "TRANSITIONS_FILENAME", "RelationStore", "RelationStoreError"]
