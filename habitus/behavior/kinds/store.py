"""词表的文件存储（插件交付：全是文件，不建表）。树根下两个文件：

- ``kinds.state.json``  在用状态：哪些词在用、从哪一天起，和修订号（在用集合每变一次加一）——真相；
- ``kinds.md``          给人看的清单：由初始词表 + 在用状态渲染出来，是视图不是真相。

初始词随代码发布、只读（``words.py``），不在这里。没有变更日志、版本回放、快照截断：没有拆分合并以后，日志只剩"谁转了在用"，
一个状态文件就够（词表方案 2026-11-14）。写入带期望修订号（CAS）。

旧词表（变更日志、快照、待定池、迁移计划、拆改状态）不迁移、不兼容：盘上还有这些文件时读就明确报错并点出文件。
"""

from __future__ import annotations

import json
import threading
from collections.abc import Mapping
from datetime import date
from pathlib import Path

from habitus.behavior.kinds.config import BehaviorKindConfig
from habitus.behavior.kinds.ids import ClassId, KindIdError, Lane
from habitus.behavior.kinds.model import BehaviorKindError, Vocabulary
from habitus.behavior.kinds.render import render_catalog
from habitus.behavior.kinds.words import initial_words
from habitus.behavior.model import KINDS_REGISTRY_FILENAME
from habitus.infrastructure.store.filesystem import (
    DurablePathIntegrityError,
    atomic_replace_bytes,
    read_regular_bytes,
)

STATE_SCHEMA_VERSION = "behavior_kinds_state_v1"
STATE_FILENAME = "kinds.state.json"
#: 旧词表留在盘上的文件：在就报错，不悄悄忽略（不迁移）。
LEGACY_FILENAMES = (
    "kinds.changes.jsonl",
    "kinds.snapshot.json",
    "kinds.pending.json",
    "kinds.migration.json",
    "kinds.jobs.json",
    "kinds.revision.json",
)
_STATE_KEYS = frozenset({"schema_version", "lane", "revision", "active"})
_ACTIVE_KEYS = frozenset({"id", "since"})

# 同一份状态的"读、改、写回"可能分在不同线程里（换代、运维入口）：按路径共用一把锁串起来。
_LOCKS: dict[Path, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()


def _lock(path: Path) -> threading.Lock:
    with _LOCKS_GUARD:
        return _LOCKS.setdefault(path, threading.Lock())


class BehaviorKindStoreError(ValueError):
    pass


class BehaviorKindConflictError(RuntimeError):
    pass


class BehaviorKindStore:
    def __init__(self, behavior_root: str | Path, *, lane: Lane, config: BehaviorKindConfig | None = None) -> None:
        self.root = Path(behavior_root).expanduser().resolve(strict=False)
        self.lane = Lane(lane)
        self.config = config or BehaviorKindConfig()
        self.state_path = self.root / STATE_FILENAME
        self.catalog_path = self.root / KINDS_REGISTRY_FILENAME
        self._lock = _lock(self.state_path)

    def check(self) -> None:
        """盘上有旧词表的文件就报错（启动时与每次读时都查）。"""

        found = [name for name in LEGACY_FILENAMES if (self.root / name).exists()]
        if found:
            raise BehaviorKindStoreError(
                f"old vocabulary files are not supported (no migration); remove them: "
                f"{', '.join(str(self.root / name) for name in found)}"
            )

    def read(self) -> Vocabulary:
        """初始词表 + 盘上的在用状态。没有状态文件就是全新的盘面：在用的词 0 个，修订号 0。"""

        self.check()
        words = initial_words(self.lane)
        text = self._read_text(self.state_path)
        if text is None:
            return words
        revision, active = self._decode(text)
        try:
            return Vocabulary(self.lane, words.words, active, revision)
        except BehaviorKindError as exc:
            raise BehaviorKindStoreError(f"{self.state_path}: {exc}") from exc

    def replace(self, vocabulary: Vocabulary, *, expected_revision: int) -> Vocabulary:
        """把在用状态换成 ``vocabulary`` 的；盘上的修订号不是 ``expected_revision`` 就是别人先改了（CAS）。"""

        if vocabulary.lane is not self.lane:
            raise BehaviorKindStoreError("vocabulary belongs to another lane")
        with self._lock:
            current = self.read()
            if current.revision != expected_revision:
                raise BehaviorKindConflictError(
                    f"vocabulary state revision changed: expected {expected_revision}, found {current.revision}"
                )
            if vocabulary.revision == current.revision:
                return current
            self._write(self.state_path, self._encode(vocabulary))
            # 清单视图跟着状态写：视图写失败不影响真相，下一次写状态时再写
            self._write(self.catalog_path, render_catalog(vocabulary, schema_version=STATE_SCHEMA_VERSION))
            return vocabulary

    def _encode(self, vocabulary: Vocabulary) -> str:
        record = {
            "schema_version": STATE_SCHEMA_VERSION,
            "lane": vocabulary.lane.value,
            "revision": vocabulary.revision,
            "active": [
                {"id": str(word_id), "since": since.isoformat()} for word_id, since in vocabulary.active.items()
            ],
        }
        return json.dumps(record, ensure_ascii=False, sort_keys=True, indent=1) + "\n"

    def _decode(self, text: str) -> tuple[int, dict[ClassId, date]]:
        """严格读：截断、未知字段、错的 lane、坏编号、重复都明确报错并点出文件。"""

        try:
            raw = json.loads(text)
            if not isinstance(raw, Mapping) or set(raw) != _STATE_KEYS:
                raise BehaviorKindStoreError(f"keys must be {sorted(_STATE_KEYS)}")
            if raw["schema_version"] != STATE_SCHEMA_VERSION:
                raise BehaviorKindStoreError(f"schema_version must be {STATE_SCHEMA_VERSION}")
            if raw["lane"] != self.lane.value:
                raise BehaviorKindStoreError(f"lane must be {self.lane.value}")
            revision = raw["revision"]
            if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0:
                raise BehaviorKindStoreError("revision must be a non-negative integer")
            if not isinstance(raw["active"], list):
                raise BehaviorKindStoreError("active must be a list")
            active: dict[ClassId, date] = {}
            for item in raw["active"]:
                if not isinstance(item, Mapping) or set(item) != _ACTIVE_KEYS:
                    raise BehaviorKindStoreError(f"active entries must have keys {sorted(_ACTIVE_KEYS)}")
                word_id = ClassId.parse(item["id"])
                if word_id in active:
                    raise BehaviorKindStoreError(f"active word repeats: {word_id}")
                since = item["since"]
                if not isinstance(since, str):
                    raise BehaviorKindStoreError(f"active word {word_id} since must be an ISO date text")
                active[word_id] = date.fromisoformat(since)
        except (json.JSONDecodeError, KindIdError, ValueError) as exc:
            raise BehaviorKindStoreError(f"{self.state_path}: vocabulary state is invalid: {exc}") from exc
        return revision, active

    # ── 文件 ─────────────────────────────────────────────────────────────────────

    def _read_text(self, path: Path) -> str | None:
        try:
            encoded = read_regular_bytes(path, artifact_root=self.root, max_bytes=self.config.max_encoded_bytes)
        except FileNotFoundError:
            return None
        except DurablePathIntegrityError as exc:
            raise BehaviorKindStoreError(f"{path} cannot be read safely") from exc
        try:
            return encoded.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise BehaviorKindStoreError(f"{path} is not valid UTF-8") from exc

    def _write(self, path: Path, text: str) -> None:
        encoded = text.encode("utf-8")
        if len(encoded) > self.config.max_encoded_bytes:
            raise BehaviorKindStoreError(f"{path.name} exceeds its encoded byte bound")
        try:
            atomic_replace_bytes(path, encoded, artifact_root=self.root)
        except DurablePathIntegrityError as exc:
            raise BehaviorKindStoreError(f"{path.name} cannot be written safely") from exc


__all__ = [
    "LEGACY_FILENAMES",
    "STATE_FILENAME",
    "STATE_SCHEMA_VERSION",
    "BehaviorKindConflictError",
    "BehaviorKindStore",
    "BehaviorKindStoreError",
]
