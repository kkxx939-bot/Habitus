"""行为树哪些日子变过（26 号清单 6.7）：归约每次改树都把改到的日子记一笔增量，按天缓存的读侧（``series.reader``）据此只重读
变过的日子。一条 lane 一份，放在那一支的归约根下：

    <lane 根>/reduction/dirty_days.jsonl     一行 {"seq": 序号, "days": [...]}，只追加

读侧记着自己读到第几号（消费水位），读完把它及以前的截掉（唯一的消费者）。记的是**增量**：归约里"待刷新集合"是合并后的
集合、刷新成功还会去掉日子，拿它当变化记录会漏掉"变过又刷完"的那些天。放在行为树这一层而不是归约包里：读侧（series）
只认得行为树，不该为了读这份日志把归约（连同模型客户端）一起 import 进来。
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterable
from datetime import date
from pathlib import Path

from habitus.infrastructure.store.filesystem import (
    DurablePathIntegrityError,
    atomic_replace_bytes,
    ensure_real_directory,
    read_regular_bytes,
)

DIRTY_DAYS_FILENAME = "dirty_days.jsonl"
_MAX_BYTES = 16 * 1024 * 1024
# 记（归约）与截（读侧）都是"读整份、改、整份写回"，分在两个线程里：同一份文件的两次改写必须串起来，否则截断会把
# 刚记下的那一号一起抹掉（L10.3）。按路径共用一把，两个实例指向同一份文件也串得住。
_LOCKS: dict[Path, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()


def _lock_for(path: Path) -> threading.Lock:
    with _LOCKS_GUARD:
        return _LOCKS.setdefault(path, threading.Lock())


class DirtyDayLogError(ValueError):
    pass


class DirtyDayLog:
    def __init__(self, reduction_root: str | Path) -> None:
        self.root = Path(reduction_root).expanduser().resolve(strict=False)
        self.path = self.root / DIRTY_DAYS_FILENAME
        self._lock = _lock_for(self.path)

    def record(self, days: Iterable[date]) -> int:
        """记一笔：这些天刚变过。返回这一笔的序号（没有日子时不记，返回当前最后一号）。"""

        found = sorted({day for day in days})
        with self._lock:
            entries = self._entries()
            last = entries[-1][0] if entries else 0
            if not found:
                return last
            entries.append((last + 1, tuple(found)))
            self._write(entries)
            return last + 1

    def last(self) -> int:
        entries = self._entries()
        return entries[-1][0] if entries else 0

    def after(self, seq: int) -> tuple[int, frozenset[date]]:
        """序号 ``seq`` 之后记下的全部日子，与读到的最后一号。"""

        entries = [(number, days) for number, days in self._entries() if number > seq]
        last = entries[-1][0] if entries else seq
        return last, frozenset(day for _number, days in entries for day in days)

    def truncate_through(self, seq: int) -> None:
        """``seq`` 及以前的记录截掉（读侧已经读过）；序号照旧往后接着数。"""

        with self._lock:
            entries = self._entries()
            kept = [(number, days) for number, days in entries if number > seq]
            if len(kept) == len(entries):
                return
            if not kept and entries:
                # 留最后一号的空记录：序号不能从头再数，否则读侧的水位会跳过之后新记的
                kept = [(entries[-1][0], ())]
            self._write(kept)

    def _entries(self) -> list[tuple[int, tuple[date, ...]]]:
        try:
            encoded = read_regular_bytes(self.path, artifact_root=self.root, max_bytes=_MAX_BYTES)
        except FileNotFoundError:
            return []
        except DurablePathIntegrityError as exc:
            raise DirtyDayLogError(f"dirty-day log cannot be read: {exc}") from exc
        found: list[tuple[int, tuple[date, ...]]] = []
        for number, line in enumerate(encoded.decode("utf-8").splitlines(), start=1):
            try:
                raw = json.loads(line)
                found.append((int(raw["seq"]), tuple(date.fromisoformat(str(day)) for day in raw["days"])))
            except (KeyError, TypeError, ValueError) as exc:
                raise DirtyDayLogError(f"dirty-day log line {number} is invalid: {exc}") from exc
        return found

    def _write(self, entries: list[tuple[int, tuple[date, ...]]]) -> None:
        payload = "".join(
            json.dumps({"seq": number, "days": [day.isoformat() for day in days]}, separators=(",", ":")) + "\n"
            for number, days in entries
        ).encode("utf-8")
        try:
            ensure_real_directory(self.root, artifact_root=self.root)
            atomic_replace_bytes(self.path, payload, artifact_root=self.root)
        except DurablePathIntegrityError as exc:
            raise DirtyDayLogError(f"dirty-day log cannot be written: {exc}") from exc


__all__ = ["DIRTY_DAYS_FILENAME", "DirtyDayLog", "DirtyDayLogError"]
