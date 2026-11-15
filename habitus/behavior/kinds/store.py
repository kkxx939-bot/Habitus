"""词表的文件存储（插件交付：全是文件，不建表）。树根下两个文件：

- ``kinds.state.json``  词表状态——真相：这个用户自己长的、还活着的节点，"已删"的记号，在用的词与从哪一天起，退出过在用的词
  哪天退出的，在用状态的修订号（在用集合每变一次加一，下游记的"词表版本"），和一个每次写都加一的写入序号；
- ``kinds.md``          给人看的清单：由词表渲染出来，是视图不是真相。

初始词随代码发布、只读（``words.py``），不在这里。写的人有两个：会话 lane 选词时（长节点、记被选中的时刻），换代里数在用的词，
可能在不同的线程、不同的进程里。所以写入是"读、改、写回"整段放在一个跨进程的租约里（``PathLock`` + 行为根共用的锁库，与回执、
行为树写入同一套），租约有界地等；写回前再比一次写入序号，是第二道防线（租约中途过期、被别人接手时不盖掉别人的修改）。
原子替换只保证读不到半个文件。持着这个租约时不拿别的锁、不调模型：``update`` 的修改函数是纯计算（长节点、数词表）。

旧形状（二甲的 ``behavior_kinds_state_v1``）与更旧的词表文件不迁移、不兼容：读到就明确报错并点出文件。
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from datetime import date, datetime
from pathlib import Path

from habitus.behavior.kinds.config import BehaviorKindConfig
from habitus.behavior.kinds.ids import ClassId, KindIdError, Lane
from habitus.behavior.kinds.model import BehaviorKindError, DeletedWord, Vocabulary, Word, WordOrigin
from habitus.behavior.kinds.render import render_catalog
from habitus.behavior.kinds.words import initial_words
from habitus.behavior.model import KINDS_REGISTRY_FILENAME
from habitus.infrastructure.store.contracts.lock import LockStore
from habitus.infrastructure.store.contracts.path_lock import PathLock
from habitus.infrastructure.store.filesystem import (
    DurablePathIntegrityError,
    atomic_replace_bytes,
    read_regular_bytes,
)

STATE_SCHEMA_VERSION = "behavior_kinds_state_v2"
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
_STATE_KEYS = frozenset({"schema_version", "lane", "revision", "sequence", "active", "exited", "grown", "deleted"})
_ACTIVE_KEYS = frozenset({"id", "since"})
_EXITED_KEYS = frozenset({"id", "on"})
_GROWN_KEYS = frozenset({"id", "level", "name", "parent", "definition", "created_at", "last_picked", "status"})
_DELETED_KEYS = frozenset({"id", "parent", "deleted_on"})
_CANDIDATE, _ACTIVE = "candidate", "active"


class BehaviorKindStoreError(ValueError):
    pass


class BehaviorKindConflictError(RuntimeError):
    pass


class BehaviorKindStore:
    """``lock_store`` 是行为根共用的锁库；``lock_ttl_seconds`` 是写入租约的时长（行为树写锁同一个配置
    ``behavior.write_lock_ttl_seconds``），等别人放手最多也等这么久：别人持有的租约到点就过期，等一个租约时长是有界的上限。"""

    def __init__(
        self,
        behavior_root: str | Path,
        *,
        lane: Lane,
        lock_store: LockStore,
        lock_ttl_seconds: int,
        config: BehaviorKindConfig | None = None,
    ) -> None:
        self.root = Path(behavior_root).expanduser().resolve(strict=False)
        self.lane = Lane(lane)
        self.config = config or BehaviorKindConfig()
        self.state_path = self.root / STATE_FILENAME
        self.catalog_path = self.root / KINDS_REGISTRY_FILENAME
        self.lock_store = lock_store
        self.lock_ttl_seconds = lock_ttl_seconds
        self._path_lock = PathLock(lock_store)
        self._lock_key = "behavior-kinds-state:" + hashlib.sha256(str(self.state_path).encode("utf-8")).hexdigest()[:24]

    def check(self) -> None:
        """盘上有旧词表的文件就报错（启动时与每次读时都查）。"""

        found = [name for name in LEGACY_FILENAMES if (self.root / name).exists()]
        if found:
            raise BehaviorKindStoreError(
                f"old vocabulary files are not supported (no migration); remove them: "
                f"{', '.join(str(self.root / name) for name in found)}"
            )

    def read(self) -> Vocabulary:
        """此刻的词表。没有状态文件就是全新的盘面：只有初始词，在用的词 0 个，修订号 0。"""

        return self._load()[0]

    def update(self, change: Callable[[Vocabulary], Vocabulary]) -> Vocabulary:
        """读、改、写回，整段持写入租约：``change`` 拿此刻的词表给出改完的一份（没有要改的原样返回同一份，不写）。
        等不到租约照 ``PathLock`` 抛 ``TimeoutError``（还没准备好，不是错误）；写回前盘上的写入序号变了抛
        ``BehaviorKindConflictError``——租约在手时不该发生，发生了说明租约中途过期、被别人接手，宁可这一次不写。"""

        with self._path_lock.acquire(
            self._lock_key, ttl_seconds=self.lock_ttl_seconds, wait_timeout_seconds=float(self.lock_ttl_seconds)
        ) as guard:
            current, sequence = self._load()
            changed = change(current)
            if changed is not current:
                if changed.lane is not self.lane:
                    raise BehaviorKindStoreError("vocabulary belongs to another lane")
                with self._path_lock.fenced((guard,)):
                    if self._load()[1] != sequence:
                        raise BehaviorKindConflictError(
                            f"{self.state_path}: vocabulary state changed while the lease was held"
                        )
                    self._write(self.state_path, self._encode(changed, sequence + 1))
                    # 清单视图跟着状态写：视图写失败不影响真相，下一次写状态时再写
                    self._write(self.catalog_path, render_catalog(changed, schema_version=STATE_SCHEMA_VERSION))
        # 租约放了：空闲的这一行删掉（别人正持有的不动），锁库不跟着写入次数长
        self.lock_store.discard_idle(self._lock_key)
        return changed

    def _load(self) -> tuple[Vocabulary, int]:
        self.check()
        words = initial_words(self.lane)
        text = self._read_text(self.state_path)
        if text is None:
            return words, 0
        try:
            return self._decode(text, words)
        except (BehaviorKindError, KindIdError, ValueError, TypeError) as exc:
            raise BehaviorKindStoreError(f"{self.state_path}: vocabulary state is invalid: {exc}") from exc

    def _encode(self, vocabulary: Vocabulary, sequence: int) -> str:
        record = {
            "schema_version": STATE_SCHEMA_VERSION,
            "lane": vocabulary.lane.value,
            "revision": vocabulary.revision,
            "sequence": sequence,
            "active": [
                {"id": str(word_id), "since": since.isoformat()} for word_id, since in vocabulary.active.items()
            ],
            "exited": [{"id": str(word_id), "on": on.isoformat()} for word_id, on in vocabulary.exited.items()],
            "grown": [
                {
                    "id": str(word.id),
                    "level": word.level,
                    "name": word.name,
                    "parent": str(word.parent),
                    "definition": word.definition,
                    "created_at": None if word.created_at is None else word.created_at.isoformat(),
                    "last_picked": None if word.last_picked is None else word.last_picked.isoformat(),
                    "status": _ACTIVE if word.id in vocabulary.active else _CANDIDATE,
                }
                for word in vocabulary.words
                if word.grown
            ],
            "deleted": [
                {"id": str(mark.id), "parent": str(mark.parent), "deleted_on": mark.deleted_on.isoformat()}
                for mark in vocabulary.deleted
            ],
        }
        return json.dumps(record, ensure_ascii=False, sort_keys=True, indent=1) + "\n"

    def _decode(self, text: str, initial: Vocabulary) -> tuple[Vocabulary, int]:
        """严格读：截断、未知字段、旧版本、错的 lane、坏编号、重复、上级不在、层数对不上、和活着的节点重名、编号和初始词撞、
        在用的词已删都明确报错（调用方点出文件）。"""

        try:
            raw = json.loads(text)
        except json.JSONDecodeError as exc:
            raise BehaviorKindStoreError(f"not valid JSON: {exc}") from exc
        if not isinstance(raw, Mapping):
            raise BehaviorKindStoreError("vocabulary state must be a JSON object")
        if raw.get("schema_version") != STATE_SCHEMA_VERSION:
            raise BehaviorKindStoreError(
                f"schema_version must be {STATE_SCHEMA_VERSION}, found {raw.get('schema_version')!r} (no migration)"
            )
        if set(raw) != _STATE_KEYS:
            raise BehaviorKindStoreError(f"keys must be {sorted(_STATE_KEYS)}")
        if raw["lane"] != self.lane.value:
            raise BehaviorKindStoreError(f"lane must be {self.lane.value}")
        revision, sequence = _count(raw["revision"], "revision"), _count(raw["sequence"], "sequence")
        active = {
            ClassId.parse(item["id"]): _day(item["since"], "since") for item in _entries(raw, "active", _ACTIVE_KEYS)
        }
        exited = {ClassId.parse(item["id"]): _day(item["on"], "on") for item in _entries(raw, "exited", _EXITED_KEYS)}
        if len(active) != len(raw["active"]) or len(exited) != len(raw["exited"]):
            raise BehaviorKindStoreError("active or exited words repeat")
        grown = self._grown(_entries(raw, "grown", _GROWN_KEYS), initial, active)
        deleted = [
            DeletedWord(
                ClassId.parse(item["id"]), ClassId.parse(item["parent"]), _day(item["deleted_on"], "deleted_on")
            )
            for item in _entries(raw, "deleted", _DELETED_KEYS)
        ]
        vocabulary = Vocabulary(self.lane, (*initial.words, *grown), active, revision, deleted, exited)
        return vocabulary, sequence

    def _grown(
        self, entries: list[Mapping[str, object]], initial: Vocabulary, active: Mapping[ClassId, date]
    ) -> list[Word]:
        """自己长的节点（按编号：上级总比下级先写出来）；领域跟上级。"""

        grown: list[Word] = []
        domains = {word.id: word.domain for word in initial.words}
        for item in sorted(entries, key=lambda item: ClassId.parse(item["id"])):
            word_id, parent = ClassId.parse(item["id"]), ClassId.parse(item["parent"])
            level = item["level"]
            if isinstance(level, bool) or not isinstance(level, int) or not 2 <= level <= self.config.max_level:
                raise BehaviorKindStoreError(
                    f"grown word {word_id} level must be between 2 and {self.config.max_level}"
                )
            if len(_string(item["name"], "name")) > self.config.max_name_chars:
                raise BehaviorKindStoreError(f"grown word {word_id} name is longer than {self.config.max_name_chars}")
            if parent not in domains:
                raise BehaviorKindStoreError(
                    f"grown word {word_id} points to a parent that is not a live word: {parent}"
                )
            status = item["status"]
            if status not in (_CANDIDATE, _ACTIVE) or (status == _ACTIVE) != (word_id in active):
                raise BehaviorKindStoreError(f"grown word {word_id} status does not match the active words")
            word = Word(
                id=word_id,
                level=level,
                name=_string(item["name"], "name"),
                parent=parent,
                definition=_string(item["definition"], "definition"),
                domain=domains[parent],
                origin=WordOrigin.GROWN,
                created_at=_moment(item["created_at"], "created_at"),
                last_picked=None if item["last_picked"] is None else _moment(item["last_picked"], "last_picked"),
            )
            domains[word_id] = word.domain
            grown.append(word)
        return grown

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


def _entries(raw: Mapping[str, object], key: str, keys: frozenset[str]) -> list[Mapping[str, object]]:
    items = raw[key]
    if not isinstance(items, list):
        raise BehaviorKindStoreError(f"{key} must be a list")
    for item in items:
        if not isinstance(item, Mapping) or set(item) != keys:
            raise BehaviorKindStoreError(f"{key} entries must have keys {sorted(keys)}")
    return items


def _count(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise BehaviorKindStoreError(f"{label} must be a non-negative integer")
    return value


def _string(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise BehaviorKindStoreError(f"{label} must be text")
    return value


def _day(value: object, label: str) -> date:
    if not isinstance(value, str):
        raise BehaviorKindStoreError(f"{label} must be an ISO date text")
    return date.fromisoformat(value)


def _moment(value: object, label: str) -> datetime:
    if not isinstance(value, str):
        raise BehaviorKindStoreError(f"{label} must be an ISO datetime text")
    return datetime.fromisoformat(value)


__all__ = [
    "LEGACY_FILENAMES",
    "STATE_FILENAME",
    "STATE_SCHEMA_VERSION",
    "BehaviorKindConflictError",
    "BehaviorKindStore",
    "BehaviorKindStoreError",
]
