"""词表变更日志的"快照 + 分段"（26 号清单 6.5）：截断之后读当前词表不再从第 1 版重放。

- ``kinds.snapshot.json``：截断那一版的整份词表；
- ``kinds.changes.jsonl``：快照之后的每一版，只追加（真相仍是"快照 + 依次应用"）。

**截断**：调用方给一个下限版本（还要读 ``changes_since(下限)`` 的最老那一个——语义树各 lane 关系状态记的词表版本、概念集
的同步进度、拆改冷却期内最老的一版，由组合根算好；behavior 不认识语义树）。先写那一版的快照、再把日志改写成只剩它之后的；
中途崩溃时日志里还有快照及以前的版本，读的时候一律跳过。请求比快照更早的变更直接报错，不返回不完整的结果。

读写都经存储注入的两个函数（字节上限、路径安全在存储那一层，这里只管内容）。
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from pathlib import Path

from habitus.behavior.kinds.changes import VersionRecord, apply
from habitus.behavior.kinds.codec import (
    BehaviorKindCodecError,
    record_from,
    record_payload,
    vocabulary_from,
    vocabulary_payload,
)
from habitus.behavior.kinds.model import BehaviorKindError, Vocabulary

SNAPSHOT_FILENAME = "kinds.snapshot.json"


class VocabularyJournalError(ValueError):
    pass


class VocabularyJournal:
    def __init__(
        self,
        changes_path: Path,
        snapshot_path: Path,
        *,
        read_text: Callable[[Path], str | None],
        write_text: Callable[[Path, str], None],
    ) -> None:
        self.changes_path = changes_path
        self.snapshot_path = snapshot_path
        self._read_text = read_text
        self._write_text = write_text

    def base(self) -> Vocabulary:
        """快照那一版的词表；没截断过是空词表（第 0 版）。"""

        text = self._read_text(self.snapshot_path)
        if text is None:
            return Vocabulary()
        try:
            return vocabulary_from(json.loads(text))
        except (json.JSONDecodeError, BehaviorKindCodecError) as exc:
            raise VocabularyJournalError(f"vocabulary snapshot is invalid: {exc}") from exc

    def records(self) -> tuple[VersionRecord, ...]:
        """快照之后的每一版（截断写到一半留下的旧版本跳过）。"""

        return self._after(self.base().version)

    def current(self) -> Vocabulary:
        base = self.base()
        try:
            return replay_onto(base, self._after(base.version))
        except BehaviorKindError as exc:
            raise VocabularyJournalError(f"change log does not replay: {exc}") from exc

    def snapshot_at(self, version: int) -> Vocabulary:
        """第 ``version`` 版的整份词表；早于快照或晚于当前都报错。"""

        base = self.base()
        records = self._after(base.version)
        latest = records[-1].version if records else base.version
        if not base.version <= version <= latest:
            raise VocabularyJournalError(f"version {version} lies outside the journal [{base.version}, {latest}]")
        return replay_onto(base, (record for record in records if record.version <= version))

    def segments(self) -> tuple[int, int]:
        """（快照那一版, 最后一版）：日志里只存这一段。"""

        base = self.base()
        records = self._after(base.version)
        return base.version, (records[-1].version if records else base.version)

    def changes_since(self, version: int) -> tuple[VersionRecord, ...]:
        base = self.base()
        if version < base.version:
            raise VocabularyJournalError(
                f"changes since version {version} were truncated; the journal starts after version {base.version}"
            )
        return tuple(record for record in self._after(base.version) if record.version > version)

    def append(self, record: VersionRecord) -> None:
        existing = self._read_text(self.changes_path) or ""
        line = json.dumps(record_payload(record), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        self._write_text(self.changes_path, f"{existing}{line}\n")

    def truncate_before(self, version: int) -> bool:
        """把 ``version`` 及以前的版本折进快照；``version`` 不比现在的快照新就什么也不做。返回做没做。"""

        base = self.base()
        if version <= base.version:
            return False
        snapshot = self.snapshot_at(version)
        self._write_text(
            self.snapshot_path, json.dumps(vocabulary_payload(snapshot), ensure_ascii=False, sort_keys=True) + "\n"
        )
        kept = self._after(version)
        lines = "".join(
            json.dumps(record_payload(record), ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
            for record in kept
        )
        self._write_text(self.changes_path, lines)
        return True

    def _after(self, version: int) -> tuple[VersionRecord, ...]:
        text = self._read_text(self.changes_path)
        if text is None:
            return ()
        found: list[VersionRecord] = []
        # 只按 "\n" 切：JSON 已把记录内部的换行转义，``splitlines`` 还会在 U+2028 这类字符上切，把一条记录切成两半。
        for number, line in enumerate((item for item in text.split("\n") if item), start=1):
            try:
                record = record_from(json.loads(line))
            except (json.JSONDecodeError, BehaviorKindCodecError) as exc:
                raise VocabularyJournalError(f"change log line {number} is invalid: {exc}") from exc
            if record.version > version:
                found.append(record)
        return tuple(found)


def replay_onto(base: Vocabulary, records: Iterable[VersionRecord]) -> Vocabulary:
    vocabulary = base
    for record in records:
        vocabulary = apply(vocabulary, record)
    return vocabulary


__all__ = ["SNAPSHOT_FILENAME", "VocabularyJournal", "VocabularyJournalError", "replay_onto"]
