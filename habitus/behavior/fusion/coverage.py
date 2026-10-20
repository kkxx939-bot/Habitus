"""融合覆盖索引：哪些观测已经被判过——按日分目录，每条记录按自己的释放条件删。

## 为什么不用回执目录

"这段观测处理过没有"以前靠全量枚举回执目录回答，归约封口每轮都读一遍。回执只增不减，
一周 2,689 份、一年十几万份，越过 ``max_receipt_files`` 即枚举永久失效（2026-08 真实数据审计）。而
这个问题真正需要的信息只有观测 id 与"何时判的"，且只在**同一个单位可能再送来**的时间跨度内有意义。

所以覆盖信息单独成索引：``fusion/coverage/YYYY-MM-DD/<receipt_id>.json``，一份回执一个小文件，
按 ``judged_at`` 的 UTC 日期分目录。读取看全部记录（见 ``covered_observation_ids``：读侧开窗会和删除侧的窗口
形成死锁）；删除由 ``expire`` 按每条记录的释放条件做，删空的日目录一起删。窗口进 Config（默认 7 天）。

## 会话 lane 的记录跟着会话源走（26 号清单 6.1）

会话 lane 的记录带着这一轮所属的会话源（``source_refs``）：那份源的 Behavior 到终态之前不删——做到一半的源隔 10 天才恢复，
前几轮照样认得出写过，不会重新描述一遍。到终态之后才删，排在会话源释放写水位（``conversation.source``）之后；"到终态"
由组合根注入（``settled``，behavior 不 import conversation）。没带源的（物理 lane、旧记录）照旧按窗口过期。

## 读的缓存（26 号清单 6.2）

记录只创建、不改写：按文件路径缓存解码好的内容，删的时候一起淘汰。

## 与释放的顺序

观测在链发布到树之后释放（``behavior/reduction``）。封口视界靠"未被覆盖的最早观测"定，所以
**必须先有覆盖记录、再删观测**——否则旧观测被误判为没判过、视界永远拖在过去。本索引在两条 lane
落盘回执的同一步写入（各自的 ``recorder``），早于任何释放。
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from habitus.behavior.fusion.errors import BehaviorFusionError
from habitus.behavior.fusion.receipt import BehaviorFusionReceipt
from habitus.infrastructure.store.filesystem import (
    DurablePathIntegrityError,
    ImmutableArtifactConflictError,
    atomic_create_bytes,
    durable_unlink,
    list_real_directory,
    read_regular_bytes,
)

COVERAGE_SCHEMA_VERSION = "behavior_fusion_coverage_v1"
_DAY_DIRECTORY = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_RECORD_FILE = re.compile(r"^(?P<receipt_id>[0-9a-f]{64})\.json$")
_MAX_RECORD_BYTES = 1_048_576
_MAX_ENTRIES = 100_000


@dataclass(frozen=True)
class _Record:
    observation_ids: frozenset[str]
    source_refs: tuple[str, ...]


class BehaviorCoverageIndex:
    """按日分区、按窗口过期的"已融合观测"索引。``settled(source_id)``：那份会话源的 Behavior 到终态了没有。"""

    def __init__(
        self,
        behavior_root: str | Path,
        *,
        window_days: int = 7,
        settled: Callable[[str], bool] | None = None,
    ) -> None:
        if isinstance(window_days, bool) or not isinstance(window_days, int) or window_days < 1:
            raise ValueError("coverage window_days must be a positive integer")
        if settled is not None and not callable(settled):
            raise TypeError("settled must be callable or None")
        self.root = Path(behavior_root).expanduser().resolve(strict=False)
        self.coverage_root = self.root / "fusion" / "coverage"
        self.window_days = window_days
        self.settled = settled
        self._cache: dict[Path, _Record] = {}

    # ── 写 ──────────────────────────────────────────────────────────────────────────

    def record(self, receipt: BehaviorFusionReceipt, *, sources: tuple[str, ...] = ()) -> None:
        """记下这份回执覆盖的全部观测（会话 lane 连同这一轮所属的会话源）；同身份同内容重复写入幂等。"""

        if not isinstance(receipt, BehaviorFusionReceipt):
            raise TypeError("receipt must be BehaviorFusionReceipt")
        payload: dict[str, object] = {
            "schema_version": COVERAGE_SCHEMA_VERSION,
            "receipt_id": receipt.receipt_id,
            "judged_at": receipt.judged_at.astimezone(UTC).isoformat(timespec="microseconds"),
            "observation_ids": sorted(set(receipt.observation_ids)),
        }
        if sources:
            payload["source_refs"] = sorted(set(sources))
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        path = self._path(receipt.judged_at, receipt.receipt_id)
        try:
            atomic_create_bytes(path, encoded, artifact_root=self.root)
        except ImmutableArtifactConflictError as exc:
            existing = read_regular_bytes(path, artifact_root=self.root, max_bytes=_MAX_RECORD_BYTES)
            if existing != encoded:
                raise BehaviorFusionError(
                    f"coverage record for receipt {receipt.receipt_id} conflicts with an existing record"
                ) from exc

    # ── 读 ──────────────────────────────────────────────────────────────────────────

    def covered_observation_ids(self, now: datetime | None = None) -> frozenset[str]:
        """盘上**全部**覆盖记录所覆盖的观测身份。

        读取侧不再设窗口：删除权完全归 ``expire(retain=…)``（它只删"观测已释放"的记录）。两侧都设
        窗口会形成死锁——记录留得下却读不回，于是交付永远不满足释放条件、覆盖永远读不到，封口前沿
        钉死、已判过的交付被当成没判过重做（审计 NEW-1 复现）。``expire`` 之后盘上剩下的窗口外记录数 =
        仍在存储里的交付数，量很小，全量读的代价可以忽略。``now`` 仅为兼容保留，不参与判断。
        """

        covered: set[str] = set()
        for day_path in self._day_directories(None):
            for entry in self._entries(day_path):
                if entry.is_file() and _RECORD_FILE.fullmatch(entry.name) is not None:
                    covered.update(self._strict(day_path / entry.name).observation_ids)
        return frozenset(covered)

    # ── 过期 ─────────────────────────────────────────────────────────────────────────

    def expire(self, now: datetime, *, retain: frozenset[str] | set[str] = frozenset()) -> int:
        """删掉再也用不上的覆盖记录；返回删掉的记录数。

        - 带会话源的：那几份源的 Behavior 都到终态了才删（不看窗口）；
        - 不带的（物理 lane、旧记录）：窗口之前的才删。
        ``retain`` 是**仍在观测存储里**的观测身份：引用到它们的记录一律不删——交付还在，它的"已融合"就还得答得出来，
        否则会被当成没判过重做、产出第二套判断，并把封口视界钉在过去（2026-08 真实数据审计）。
        """

        start = self._window_start(now)
        removed = 0
        if not self.coverage_root.is_dir():
            return 0
        keep = set(retain)
        for day_path in self._day_directories(None):
            old = date.fromisoformat(day_path.name) < start
            for item in self._entries(day_path):
                if not (item.is_file() and _RECORD_FILE.fullmatch(item.name)):
                    continue
                path = day_path / item.name
                record = self._lenient(path)
                if record is not None and keep and record.observation_ids & keep:
                    continue
                if not self._released(record, old=old):
                    continue
                if durable_unlink(path, artifact_root=self.root):
                    removed += 1
                self._cache.pop(path, None)
            if old or not self._entries(day_path):
                try:
                    os.rmdir(day_path)
                except OSError:
                    # 目录里还有记录，或不属于本索引的东西（云同步副本之类）：留着，不影响正确性。
                    pass
        return removed

    # ── 内部 ─────────────────────────────────────────────────────────────────────────

    def _released(self, record: _Record | None, *, old: bool) -> bool:
        """解不开的记录当作"什么都没覆盖"、按窗口照常删（覆盖索引是可重建的派生物，损坏应当自愈）。"""

        if record is None or not record.source_refs:
            return old
        return self.settled is not None and all(self.settled(source) for source in record.source_refs)

    def _strict(self, path: Path) -> _Record:
        """读侧：读不出来就硬拒——那里读错会把已融合误判成未融合。"""

        cached = self._cache.get(path)
        if cached is not None:
            return cached
        encoded = read_regular_bytes(path, artifact_root=self.root, max_bytes=_MAX_RECORD_BYTES)
        try:
            payload = json.loads(encoded.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise BehaviorFusionError(f"coverage record is not decodable: {path.name}") from exc
        ids = payload.get("observation_ids")
        if not isinstance(ids, list) or any(not isinstance(item, str) for item in ids):
            raise BehaviorFusionError(f"coverage record has invalid observation_ids: {path.name}")
        sources = payload.get("source_refs", [])
        if not isinstance(sources, list) or any(not isinstance(item, str) for item in sources):
            raise BehaviorFusionError(f"coverage record has invalid source_refs: {path.name}")
        record = _Record(frozenset(ids), tuple(sources))
        self._cache[path] = record
        return record

    def _lenient(self, path: Path) -> _Record | None:
        try:
            return self._strict(path)
        except (BehaviorFusionError, OSError, DurablePathIntegrityError):
            return None

    def _window_start(self, now: datetime) -> date:
        if not isinstance(now, datetime) or now.utcoffset() is None:
            raise TypeError("now must be a timezone-aware datetime")
        return (now.astimezone(UTC) - timedelta(days=self.window_days)).date()

    def _path(self, judged_at: datetime, receipt_id: str) -> Path:
        if _RECORD_FILE.fullmatch(f"{receipt_id}.json") is None:
            raise BehaviorFusionError("receipt_id must be lowercase SHA-256 text")
        day = judged_at.astimezone(UTC).date().isoformat()
        return self.coverage_root / day / f"{receipt_id}.json"

    def _day_directories(self, start: date | None) -> list[Path]:
        """日目录；``start`` 为 None 时不设下界（读取侧全量）。"""

        if not self.coverage_root.is_dir():
            return []
        days: list[Path] = []
        for entry in self._entries(self.coverage_root):
            if not (entry.is_dir() and _DAY_DIRECTORY.fullmatch(entry.name)):
                continue
            if start is not None and date.fromisoformat(entry.name) < start:
                continue
            days.append(self.coverage_root / entry.name)
        return sorted(days)

    def _entries(self, directory: Path) -> list:
        try:
            return list(list_real_directory(directory, artifact_root=self.root, max_entries=_MAX_ENTRIES))
        except FileNotFoundError:
            return []
        except DurablePathIntegrityError as exc:
            raise BehaviorFusionError("coverage directory is invalid or exceeds its bound") from exc


__all__ = ["COVERAGE_SCHEMA_VERSION", "BehaviorCoverageIndex"]
