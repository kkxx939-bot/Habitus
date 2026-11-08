"""定稿日账：派生层（语义树、预测层的结算）只处理这里记为 closed 的日子；open 是树变了但还没定稿的日子。

哪天能定稿由纯函数 ``closure.plan_closure`` 定，这里只读写账本文件、取那天各条记录的本地偏移。
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import date, datetime, timedelta
from pathlib import Path

from habitus.behavior.model import BehaviorKind
from habitus.behavior.reduction.closure import DayClosureState, plan_closure
from habitus.behavior.reduction.errors import BehaviorReductionError
from habitus.behavior.tree import BehaviorTree
from habitus.infrastructure.store.contracts.path_lock import LeaseGuard
from habitus.infrastructure.store.filesystem import atomic_replace_bytes, read_regular_bytes

_DAY_CLOSURE_NAME = "day_closure.json"
_MAX_DAY_CLOSURE_BYTES = 4_194_304


class DayClosureLedger:
    """``{ledger.root}/day_closure.json`` 的读写与一轮的定稿记账。"""

    def __init__(self, root: Path, *, tree: BehaviorTree, now: Callable[[], datetime]) -> None:
        self.root = root
        self.path = root / _DAY_CLOSURE_NAME
        self.tree = tree
        self._now = now

    def closed_days(self) -> tuple[date, ...]:
        """已定稿的本地日历日（升序）：那一天的链都已落树、且封口视界已过那天的本地结束。"""

        return tuple(sorted(self.state().closed))

    def close(
        self, days: set[date], guard: LeaseGuard, *, unsealed_days: frozenset[date], horizon: datetime | None
    ) -> tuple[int, int]:
        """按 ``closure.plan_closure`` 定这一轮能定稿的日子（含没有记录的空白日），记账。"""

        state = self.state()
        plan = plan_closure(state, days, horizon=horizon, unsealed_days=unsealed_days, offsets_of=self._offsets_of)
        stamp = self._now().isoformat(timespec="seconds")
        state.closed.update(dict.fromkeys(plan.closed_now, stamp))
        state.open = set(plan.still_open)
        state.last_offset = plan.last_offset
        self._write(state, guard)
        return len(plan.closed_now), len(plan.still_open)

    def _offsets_of(self, day: date) -> tuple[timedelta, ...]:
        """那一天各条记录（行为与空白）的本地偏移。"""

        offsets = []
        for kind in (BehaviorKind.OCCURRENCE, BehaviorKind.GAP):
            for document in self.tree.read_day(kind, day):
                offset = document.address.started_at.utcoffset()
                if offset is not None:
                    offsets.append(offset)
        return tuple(offsets)

    def state(self) -> DayClosureState:
        try:
            encoded = read_regular_bytes(self.path, artifact_root=self.root, max_bytes=_MAX_DAY_CLOSURE_BYTES)
        except FileNotFoundError:
            return DayClosureState()
        try:
            raw = json.loads(encoded.decode("utf-8"))
            offset = raw["last_offset_seconds"]
            return DayClosureState(
                open={date.fromisoformat(item) for item in raw["open"]},
                closed={date.fromisoformat(key): str(value) for key, value in raw["closed"].items()},
                last_offset=None if offset is None else timedelta(seconds=int(offset)),
            )
        except (UnicodeDecodeError, ValueError, TypeError, KeyError) as exc:
            raise BehaviorReductionError("reduction day_closure file is not decodable") from exc

    def _write(self, state: DayClosureState, guard: LeaseGuard) -> None:
        body = {
            "open": sorted(day.isoformat() for day in state.open),
            "closed": {day.isoformat(): value for day, value in sorted(state.closed.items())},
            "last_offset_seconds": None if state.last_offset is None else int(state.last_offset.total_seconds()),
        }
        with guard.fenced():
            atomic_replace_bytes(self.path, json.dumps(body, sort_keys=True).encode("utf-8"), artifact_root=self.root)
