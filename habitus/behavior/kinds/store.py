"""词表的文件存储（插件交付：全是文件，不建表）。树根下四个文件：

- ``kinds.md``            当前清单：给人看的列表 + 版本号元数据；由变更日志推出来，是视图不是真相；
- ``kinds.changes.jsonl`` 变更日志：一行一版，只追加；当前词表 = 依次应用每一版（真相）；
- ``kinds.pending.json``  待定池；
- ``kinds.migration.json`` 迁移计划：只在"重打树"期间存在，内容就是将要追加的那一版；崩溃后按它续做；
- ``kinds.jobs.json``    两件定时活的跨次状态：上次每晚新增、上次定期拆改的时刻，最近几次提过的合并（合并证据）。

写入都带期望版本（CAS）：变更日志按"当前最后一版"比对，待定池按自己的修订号比对。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from habitus.behavior.kinds.changes import VersionRecord, apply, replay
from habitus.behavior.kinds.codec import (
    BehaviorKindCodecError,
    job_state_from,
    job_state_payload,
    pending_from,
    pending_payload,
    record_from,
    record_payload,
)
from habitus.behavior.kinds.config import BehaviorKindConfig
from habitus.behavior.kinds.model import BehaviorKindError, Vocabulary
from habitus.behavior.kinds.pending import PendingPool
from habitus.behavior.kinds.render import render_catalog
from habitus.behavior.kinds.schedule import JobState
from habitus.behavior.model import KINDS_REGISTRY_FILENAME
from habitus.infrastructure.store.filesystem import (
    DurablePathIntegrityError,
    atomic_replace_bytes,
    durable_unlink,
    read_regular_bytes,
)

KINDS_SCHEMA_VERSION = "behavior_kinds_v6"
CHANGES_FILENAME = "kinds.changes.jsonl"
PENDING_FILENAME = "kinds.pending.json"
MIGRATION_FILENAME = "kinds.migration.json"
JOBS_FILENAME = "kinds.jobs.json"


class BehaviorKindStoreError(ValueError):
    pass


class BehaviorKindConflictError(RuntimeError):
    pass


@dataclass(frozen=True)
class PendingSnapshot:
    pool: PendingPool
    revision: int


class BehaviorKindStore:
    def __init__(self, behavior_root: str | Path, *, config: BehaviorKindConfig | None = None) -> None:
        resolved = config or BehaviorKindConfig()
        if not isinstance(resolved, BehaviorKindConfig):
            raise TypeError("config must be BehaviorKindConfig")
        self.root = Path(behavior_root).expanduser().resolve(strict=False)
        self.config = resolved
        self.catalog_path = self.root / KINDS_REGISTRY_FILENAME
        self.changes_path = self.root / CHANGES_FILENAME
        self.pending_path = self.root / PENDING_FILENAME
        self.migration_path = self.root / MIGRATION_FILENAME
        self.jobs_path = self.root / JOBS_FILENAME

    # ── 变更日志与当前词表 ─────────────────────────────────────────────────────────

    def records(self) -> tuple[VersionRecord, ...]:
        text = self._read_text(self.changes_path)
        if text is None:
            return ()
        records: list[VersionRecord] = []
        # 只按 "\n" 切：JSON 已把记录内部的换行转义，``splitlines`` 还会在 U+2028 这类字符上切，把一条记录切成两半。
        for number, line in enumerate((item for item in text.split("\n") if item), start=1):
            try:
                records.append(record_from(json.loads(line)))
            except (json.JSONDecodeError, BehaviorKindCodecError) as exc:
                raise BehaviorKindStoreError(f"change log line {number} is invalid: {exc}") from exc
        return tuple(records)

    def read(self) -> Vocabulary:
        try:
            return replay(self.records())
        except BehaviorKindError as exc:
            raise BehaviorKindStoreError(f"change log does not replay: {exc}") from exc

    def append(self, record: VersionRecord, *, expected_version: int) -> Vocabulary:
        """追加一版；当前最后一版不是 ``expected_version`` 就拒绝。随后重写清单视图。"""

        current = self.read()
        if current.version != expected_version:
            raise BehaviorKindConflictError(
                f"vocabulary version changed: expected {expected_version}, found {current.version}"
            )
        try:
            updated = apply(current, record)
        except BehaviorKindError as exc:
            raise BehaviorKindStoreError(f"version {record.version} does not apply: {exc}") from exc
        existing = self._read_text(self.changes_path) or ""
        line = json.dumps(record_payload(record), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        self._write(self.changes_path, f"{existing}{line}\n")
        self._write(self.catalog_path, render_catalog(updated, schema_version=KINDS_SCHEMA_VERSION))
        return updated

    def catalog_in_sync(self) -> bool:
        """清单视图是不是当前词表推出来的样子（监控用；崩在两次写之间时会是 False，下一次追加即修复）。"""

        expected = render_catalog(self.read(), schema_version=KINDS_SCHEMA_VERSION)
        return self._read_text(self.catalog_path) == expected

    # ── 待定池 ───────────────────────────────────────────────────────────────────

    def read_pending(self) -> PendingSnapshot:
        text = self._read_text(self.pending_path)
        if text is None:
            return PendingSnapshot(PendingPool(), 0)
        try:
            payload = json.loads(text)
            if not isinstance(payload, dict) or set(payload) != {"revision", "entries"}:
                raise BehaviorKindStoreError("pending pool shape is invalid")
            entries = [pending_from(item) for item in payload["entries"]]
            revision = payload["revision"]
        except (json.JSONDecodeError, BehaviorKindCodecError, TypeError) as exc:
            raise BehaviorKindStoreError(f"pending pool is invalid: {exc}") from exc
        if isinstance(revision, bool) or not isinstance(revision, int) or revision <= 0:
            raise BehaviorKindStoreError("pending pool revision must be positive")
        return PendingSnapshot(PendingPool({entry.occurrence: entry for entry in entries}), revision)

    def replace_pending(self, pool: PendingPool, *, expected_revision: int) -> PendingSnapshot:
        if len(pool.entries) > self.config.max_pending:
            raise BehaviorKindStoreError(f"pending pool exceeds its bound of {self.config.max_pending} entries")
        current = self.read_pending()
        if current.revision != expected_revision:
            raise BehaviorKindConflictError(
                f"pending pool revision changed: expected {expected_revision}, found {current.revision}"
            )
        snapshot = PendingSnapshot(pool, expected_revision + 1)
        payload = {
            "revision": snapshot.revision,
            "entries": [pending_payload(entry) for entry in pool.entries.values()],
        }
        self._write(self.pending_path, json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=1) + "\n")
        return snapshot

    # ── 迁移计划 ─────────────────────────────────────────────────────────────────

    def read_migration(self) -> VersionRecord | None:
        text = self._read_text(self.migration_path)
        if text is None:
            return None
        try:
            return record_from(json.loads(text))
        except (json.JSONDecodeError, BehaviorKindCodecError) as exc:
            raise BehaviorKindStoreError(f"migration plan is invalid: {exc}") from exc

    def write_migration(self, record: VersionRecord) -> None:
        """写下将要追加的那一版；已有一份没做完的计划时拒绝（先续做完它）。"""

        if self.read_migration() is not None:
            raise BehaviorKindConflictError("an unfinished migration plan exists; resume it first")
        current = self.read()
        if record.version != current.version + 1:
            raise BehaviorKindConflictError("migration plan must be the next version")
        try:
            apply(current, record)  # 先试应用：不可应用的计划不许落盘（否则先改树、后在追加日志时永久失败）
        except BehaviorKindError as exc:
            raise BehaviorKindStoreError(f"migration plan does not apply: {exc}") from exc
        payload = json.dumps(record_payload(record), ensure_ascii=False, sort_keys=True, indent=1)
        self._write(self.migration_path, payload + "\n")

    def clear_migration(self) -> None:
        try:
            durable_unlink(self.migration_path, artifact_root=self.root)
        except DurablePathIntegrityError as exc:
            raise BehaviorKindStoreError("migration plan cannot be removed safely") from exc

    # ── 定时状态 ─────────────────────────────────────────────────────────────────

    def read_jobs(self) -> JobState:
        text = self._read_text(self.jobs_path)
        if text is None:
            return JobState()
        try:
            return job_state_from(json.loads(text))
        except (json.JSONDecodeError, BehaviorKindCodecError) as exc:
            raise BehaviorKindStoreError(f"job state is invalid: {exc}") from exc

    def replace_jobs(self, state: JobState) -> None:
        payload = json.dumps(job_state_payload(state), ensure_ascii=False, sort_keys=True, indent=1)
        self._write(self.jobs_path, payload + "\n")

    # ── 文件 ─────────────────────────────────────────────────────────────────────

    def _read_text(self, path: Path) -> str | None:
        try:
            encoded = read_regular_bytes(path, artifact_root=self.root, max_bytes=self.config.max_encoded_bytes)
        except FileNotFoundError:
            return None
        except DurablePathIntegrityError as exc:
            raise BehaviorKindStoreError(f"{path.name} cannot be read safely") from exc
        try:
            return encoded.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise BehaviorKindStoreError(f"{path.name} is not valid UTF-8") from exc

    def _write(self, path: Path, text: str) -> None:
        encoded = text.encode("utf-8")
        if len(encoded) > self.config.max_encoded_bytes:
            raise BehaviorKindStoreError(f"{path.name} exceeds its encoded byte bound")
        try:
            atomic_replace_bytes(path, encoded, artifact_root=self.root)
        except DurablePathIntegrityError as exc:
            raise BehaviorKindStoreError(f"{path.name} cannot be written safely") from exc


__all__ = [
    "CHANGES_FILENAME",
    "KINDS_SCHEMA_VERSION",
    "MIGRATION_FILENAME",
    "JOBS_FILENAME",
    "PENDING_FILENAME",
    "BehaviorKindConflictError",
    "BehaviorKindStore",
    "BehaviorKindStoreError",
    "PendingSnapshot",
]
