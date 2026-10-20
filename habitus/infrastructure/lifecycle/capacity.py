"""容量水位快照：各参与者在自己那一轮流式算好量，落盘；健康检查只读快照，不再自己全量扫。

落盘而不是放内存：多进程时，持有租约做维护的进程和回答健康检查的进程可能不是同一个。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from habitus.foundation.ids import require_safe_path_segment
from habitus.foundation.integrity import canonical_json
from habitus.infrastructure.store.filesystem import (
    atomic_replace_bytes,
    list_real_directory,
    read_regular_bytes,
)

_SNAPSHOT_SUFFIX = ".json"
_MAX_SNAPSHOT_BYTES = 1024 * 1024
_MAX_PARTICIPANTS = 256


@dataclass(frozen=True)
class CapacityReading:
    """一个量：``limit`` 为 None 时只报数，不算占比。"""

    name: str
    count: int
    limit: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("capacity reading name must be non-empty text")
        if isinstance(self.count, bool) or not isinstance(self.count, int) or self.count < 0:
            raise ValueError("capacity count must be a non-negative integer")
        if self.limit is not None and (
            isinstance(self.limit, bool) or not isinstance(self.limit, int) or self.limit <= 0
        ):
            raise ValueError("capacity limit must be a positive integer")

    @property
    def share(self) -> float | None:
        """占上限的比例；没有上限时为 None。"""

        return None if self.limit is None else self.count / self.limit


@dataclass(frozen=True)
class CapacitySnapshot:
    participant: str
    at: datetime
    readings: tuple[CapacityReading, ...]

    def __post_init__(self) -> None:
        require_safe_path_segment(self.participant, "participant")
        if not isinstance(self.at, datetime) or self.at.utcoffset() is None:
            raise ValueError("snapshot time must be timezone-aware")
        names = [reading.name for reading in self.readings]
        if len(names) != len(set(names)):
            raise ValueError("capacity readings must have distinct names")


class CapacitySnapshotStore:
    """每个参与者一份快照文件，后写覆盖先写。"""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve(strict=False)
        self.directory = self.root / "lifecycle" / "capacity"

    def publish(self, snapshot: CapacitySnapshot) -> None:
        payload = {
            "participant": snapshot.participant,
            "at": snapshot.at.isoformat(),
            "readings": [{"name": item.name, "count": item.count, "limit": item.limit} for item in snapshot.readings],
        }
        path = self.directory / f"{snapshot.participant}{_SNAPSHOT_SUFFIX}"
        atomic_replace_bytes(path, canonical_json(payload).encode("utf-8"), artifact_root=self.root)

    def read_all(self) -> tuple[CapacitySnapshot, ...]:
        entries = list_real_directory(self.directory, artifact_root=self.root, max_entries=_MAX_PARTICIPANTS)
        snapshots = []
        for entry in entries:
            if not entry.is_file() or not entry.name.endswith(_SNAPSHOT_SUFFIX):
                continue
            encoded = read_regular_bytes(
                self.directory / entry.name, artifact_root=self.root, max_bytes=_MAX_SNAPSHOT_BYTES
            )
            snapshots.append(_decode(json.loads(encoded.decode("utf-8"))))
        return tuple(snapshots)


def _decode(payload: dict[str, object]) -> CapacitySnapshot:
    readings = payload.get("readings")
    if not isinstance(readings, list):
        raise ValueError("capacity snapshot readings must be a list")
    return CapacitySnapshot(
        participant=str(payload["participant"]),
        at=datetime.fromisoformat(str(payload["at"])),
        readings=tuple(
            CapacityReading(name=str(item["name"]), count=item["count"], limit=item["limit"]) for item in readings
        ),
    )


__all__ = ["CapacityReading", "CapacitySnapshot", "CapacitySnapshotStore"]
