"""融合覆盖索引：哪些观测已经被融合过——有窗口、按日整块过期。

## 为什么不用回执目录

"这段观测处理过没有"以前靠全量枚举回执目录回答，入队扫描与归约封口每轮都读一遍。回执只增不减，
一周 2,689 份、一年十几万份，越过 ``max_receipt_files`` 即枚举永久失效（BHV-REALDATA-001）。而
这个问题真正需要的信息只有观测 id 与"何时融合"，且只在**上游可能补发**的时间跨度内有意义——
云侧 agent 重启后补发几小时前的观测是常态，补发跨度之外的观测不会再来，记着也没用。

所以覆盖信息单独成索引：``fusion/coverage/YYYY-MM-DD/<receipt_id>.json``，一份回执一个小文件，
按 ``judged_at`` 的 UTC 日期分目录。读取只看窗口内的日目录；过期是删整个日目录——不堆积，
删除也是整块操作。窗口等于上游最大补发跨度（进 Config，契约未定前取 7 天）。

## 与释放的顺序

观测在链发布到树之后释放（``behavior/reduction``）。封口视界靠"未被覆盖的最早观测"定，所以
**必须先有覆盖记录、再删观测**——否则旧观测被误判为未融合、视界永远拖在过去。本索引在融合
落盘回执的同一步写入（``BehaviorFusionRunner._persist``），早于任何释放。
"""

from __future__ import annotations

import json
import os
import re
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from behavior.fusion.errors import BehaviorFusionError
from behavior.fusion.receipt import BehaviorFusionReceipt
from infrastructure.store.filesystem import (
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


class BehaviorCoverageIndex:
    """按日分区、按窗口过期的"已融合观测"索引。"""

    def __init__(self, behavior_root: str | Path, *, window_days: int = 7) -> None:
        if isinstance(window_days, bool) or not isinstance(window_days, int) or window_days < 1:
            raise ValueError("coverage window_days must be a positive integer")
        self.root = Path(behavior_root).expanduser().resolve(strict=False)
        self.coverage_root = self.root / "fusion" / "coverage"
        self.window_days = window_days

    # ── 写 ──────────────────────────────────────────────────────────────────────────

    def record(self, receipt: BehaviorFusionReceipt) -> None:
        """记下这份回执覆盖的全部观测；同身份同内容重复写入幂等。"""

        if not isinstance(receipt, BehaviorFusionReceipt):
            raise TypeError("receipt must be BehaviorFusionReceipt")
        payload = {
            "schema_version": COVERAGE_SCHEMA_VERSION,
            "receipt_id": receipt.receipt_id,
            "judged_at": receipt.judged_at.astimezone(UTC).isoformat(timespec="microseconds"),
            "observation_ids": sorted(set(receipt.observation_ids)),
        }
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
        钉死、已融合的交付被重新入队（审计 NEW-1 复现）。``expire`` 之后盘上剩下的窗口外记录数 =
        仍在存储里的交付数，量很小，全量读的代价可以忽略。``now`` 仅为兼容保留，不参与判断。
        """

        covered: set[str] = set()
        for day_path in self._day_directories(None):
            for entry in self._entries(day_path):
                match = _RECORD_FILE.fullmatch(entry.name)
                if not entry.is_file() or match is None:
                    continue
                encoded = read_regular_bytes(
                    day_path / entry.name, artifact_root=self.root, max_bytes=_MAX_RECORD_BYTES
                )
                try:
                    payload = json.loads(encoded.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise BehaviorFusionError(f"coverage record is not decodable: {entry.name}") from exc
                ids = payload.get("observation_ids")
                if not isinstance(ids, list) or any(not isinstance(item, str) for item in ids):
                    raise BehaviorFusionError(f"coverage record has invalid observation_ids: {entry.name}")
                covered.update(ids)
        return frozenset(covered)

    # ── 过期 ─────────────────────────────────────────────────────────────────────────

    def expire(self, now: datetime, *, retain: frozenset[str] | set[str] = frozenset()) -> int:
        """删掉窗口之前的覆盖记录；返回删掉的记录数。

        ``retain`` 是**仍在观测存储里**的观测身份：引用到它们的记录不删——交付还在，它的"已融合"
        就还得答得出来，否则窗口一过它会被当成未融合重新入队、重跑模型、产出第二套判断，并把封口
        视界钉在过去（BHV-REALDATA-001 审计）。释放（``_release_unreferenced``）之后下一轮再过期。
        """

        start = self._window_start(now)
        removed = 0
        if not self.coverage_root.is_dir():
            return 0
        keep = set(retain)
        for entry in self._entries(self.coverage_root):
            if not entry.is_dir() or _DAY_DIRECTORY.fullmatch(entry.name) is None:
                continue
            if date.fromisoformat(entry.name) >= start:
                continue
            day_path = self.coverage_root / entry.name
            for item in self._entries(day_path):
                if not (item.is_file() and _RECORD_FILE.fullmatch(item.name)):
                    continue
                if keep and self._references(day_path / item.name, keep):
                    continue
                if durable_unlink(day_path / item.name, artifact_root=self.root):
                    removed += 1
            try:
                os.rmdir(day_path)
            except OSError:
                # 目录里还有不属于本索引的东西（云同步副本之类）：留着，不影响正确性。
                pass
        return removed

    # ── 内部 ─────────────────────────────────────────────────────────────────────────

    def _references(self, path: Path, ids: set[str]) -> bool:
        """这份记录是否覆盖了仍在存储里的观测。

        解不开的记录当作"什么都没覆盖"→ 照常被删掉（覆盖索引是可重建的派生物，历史记录损坏
        应当自愈）。硬拒留给 ``covered_observation_ids``：那里读不出来会把已融合误判成未融合。
        """

        try:
            encoded = read_regular_bytes(path, artifact_root=self.root, max_bytes=_MAX_RECORD_BYTES)
            payload = json.loads(encoded.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError, BehaviorFusionError, OSError):
            return False
        listed = payload.get("observation_ids")
        return isinstance(listed, list) and any(item in ids for item in listed)

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
