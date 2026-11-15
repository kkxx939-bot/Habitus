"""词表状态多进程测试用的一个进程（``test_kinds_multiprocess.py`` 起它）。

用法：``python kinds_mp_worker.py <root> <mode> <worker_id> <count>``；开跑前等 ``<root>/go`` 出现（几个进程同时起跑），做完写
``<root>/<worker_id>.done``。几个进程共用 ``<root>/tree`` 这棵树的词表状态与 ``<root>/locks.sqlite3`` 这个锁库，和生产里两个 Runtime
开在同一个根上一样。

- ``grow``：像会话 lane 选词那样，一轮建一个节点（``grow.settle``），名字带 ``worker_id``，一共 ``count`` 个；上级轮着用前 ``count``
  个第 1 层初始词（每个上级的候选位置不会满，不挤）。
- ``count``：像换代数词表那样，每一次 ``count_active`` 都让一个新的初始第 2 层词凑够两个日子转在用，一共 ``count`` 次。
"""

from __future__ import annotations

import sys
import time
from collections.abc import Callable
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

CST = timezone(timedelta(hours=8))


def main(root: Path, mode: str, worker_id: str, count: int) -> None:
    from habitus.behavior.kinds.active import count_active
    from habitus.behavior.kinds.config import BehaviorKindConfig
    from habitus.behavior.kinds.grow import Proposal, settle
    from habitus.behavior.kinds.ids import Lane
    from habitus.behavior.kinds.model import Vocabulary
    from habitus.behavior.kinds.store import BehaviorKindStore
    from habitus.behavior.kinds.words import initial_words
    from habitus.infrastructure.store.sqlite import SQLiteLockStore

    config = BehaviorKindConfig()
    store = BehaviorKindStore(
        root / "tree",
        lane=Lane.SESSION,
        lock_store=SQLiteLockStore(root / "locks.sqlite3"),
        lock_ttl_seconds=30,
        config=config,
    )
    words = initial_words(Lane.SESSION).words
    parents = [word.id for word in words if word.level == 1]
    seconds = [word.id for word in words if word.level == 2]
    while not (root / "go").exists():
        time.sleep(0.01)

    def grow(index: int) -> Callable[[Vocabulary], Vocabulary]:
        at = datetime(2026, 4, 13, 9, tzinfo=CST) + timedelta(seconds=index)
        proposal = Proposal(f"{worker_id}{index:02d}号", f"{worker_id} 写的第 {index} 个")
        return lambda vocabulary: settle(vocabulary, parents[index], (proposal,), at=at, config=config)[0]

    def recount(index: int) -> Callable[[Vocabulary], Vocabulary]:
        days = (date(2026, 4, 13), date(2026, 4, 14))
        picks = [(str(word_id), day) for word_id in seconds[: index + 1] for day in days]
        today = date(2026, 4, 15)
        return lambda vocabulary: (
            count_active(vocabulary, picks, today=today, min_days=2, cap=1000, retire_days=20).vocabulary
        )

    for index in range(count):
        store.update(grow(index) if mode == "grow" else recount(index))
    (root / f"{worker_id}.done").write_text("done", encoding="utf-8")


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2], sys.argv[3], int(sys.argv[4]))
