"""待刷新的日目录集合：与检查点分开耐久。

刷新失败不再扣住检查点（那会让一次 LLM 抖动后的 merge 撞出永久冲突），而是把日子记在这里，每轮 sweep 逐日重试、
成功一天清一天。
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import date
from pathlib import Path

from habitus.behavior.reduction.errors import BehaviorReductionError
from habitus.behavior.reduction.refresh import RefreshRetries
from habitus.behavior.tree.dirty_days import DirtyDayLog
from habitus.infrastructure.store.contracts.path_lock import LeaseGuard
from habitus.infrastructure.store.filesystem import atomic_replace_bytes, read_regular_bytes

_REFRESH_PENDING_NAME = "refresh_pending.json"
_MAX_REFRESH_PENDING_BYTES = 1_048_576


class RefreshPendingLog:
    """``{ledger.root}/refresh_pending.json`` 的读写；``touch`` 同时记脏日增量、清这些天的重试账。"""

    def __init__(self, root: Path, *, dirty_days: DirtyDayLog | None, refresh_retries: RefreshRetries | None) -> None:
        self.root = root
        self.path = root / _REFRESH_PENDING_NAME
        self.dirty_days = dirty_days
        self.refresh_retries = refresh_retries

    def days(self) -> set[date]:
        try:
            encoded = read_regular_bytes(self.path, artifact_root=self.root, max_bytes=_MAX_REFRESH_PENDING_BYTES)
        except FileNotFoundError:
            return set()
        try:
            raw = json.loads(encoded.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise BehaviorReductionError("reduction refresh_pending file is not decodable") from exc
        if not isinstance(raw, list) or any(not isinstance(item, str) for item in raw):
            raise BehaviorReductionError("reduction refresh_pending file shape is invalid")
        try:
            return {date.fromisoformat(item) for item in raw}
        except ValueError as exc:
            raise BehaviorReductionError("reduction refresh_pending file holds a non-ISO date") from exc

    def write(self, days: set[date], guard: LeaseGuard) -> None:
        with guard.fenced():
            if not days:
                self.path.unlink(missing_ok=True)
                return
            encoded = json.dumps(sorted(day.isoformat() for day in days)).encode("utf-8")
            atomic_replace_bytes(self.path, encoded, artifact_root=self.root)

    def touch(self, days: Iterable[date], guard: LeaseGuard) -> None:
        """这些天的内容刚变过：记进待刷新集合，并清掉它们的重试账（放弃的是旧内容，新内容重新给机会）。"""

        fresh = set(days)
        if not fresh:
            return
        if self.dirty_days is not None:
            self.dirty_days.record(fresh)
        self.write(self.days() | fresh, guard)
        if self.refresh_retries is not None:
            self.refresh_retries.touched(fresh)
