"""词表变更日志的"快照 + 分段"（26 号清单 6.5）：截断之后当前词表一样、读当前不再从第 1 版重放；截断下限那一版之后的变更
仍然完整；更早的请求报错；截断写到一半崩了也不会读错。"""

from __future__ import annotations

from pathlib import Path

import pytest

from habitus.behavior.kinds.changes import AddClass, ChangeReason, ReviseClass, VersionRecord, merge_operations
from habitus.behavior.kinds.journal import VocabularyJournalError
from habitus.behavior.kinds.reader import VocabularyReader
from habitus.behavior.kinds.store import BehaviorKindStore, BehaviorKindStoreError
from tests.unit.behavior.kinds_fixtures import EDIT, NOW, RESEARCH, edit_class, named_classes, research_class

RUN = named_classes(("编辑", "调研", "跑步"))[2]


def seeded(tmp_path: Path) -> BehaviorKindStore:
    store = BehaviorKindStore(tmp_path)
    store.append(VersionRecord(1, NOW, ChangeReason.NIGHTLY, (AddClass(edit_class()),)), expected_version=0)
    store.append(VersionRecord(2, NOW, ChangeReason.NIGHTLY, (AddClass(research_class()),)), expected_version=1)
    revised = edit_class().__class__(**{**edit_class().__dict__, "criterion": "改了判据：只算改代码"})
    store.append(VersionRecord(3, NOW, ChangeReason.REVISION, (ReviseClass(revised),)), expected_version=2)
    store.append(VersionRecord(4, NOW, ChangeReason.REVISION, merge_operations(RESEARCH, EDIT)), expected_version=3)
    return store


def test_truncation_keeps_the_vocabulary_and_everything_after_the_floor(tmp_path: Path) -> None:
    store = seeded(tmp_path)
    before = store.read()
    assert store.truncate_before(2)
    assert store.read() == before
    assert store.journal.segments() == (2, 4)
    assert [record.version for record in store.records()] == [3, 4]
    reader = VocabularyReader(store)
    assert [record.version for record in reader.changes_since(2)] == [3, 4]  # 下限那一版之后完整
    with pytest.raises(BehaviorKindStoreError, match="truncated"):
        reader.changes_since(1)
    assert store.journal.snapshot_at(3).classes[EDIT].criterion == "改了判据：只算改代码"
    with pytest.raises(VocabularyJournalError, match="outside"):
        store.journal.snapshot_at(1)
    assert not store.truncate_before(2)  # 不比现在的快照新：不动
    # 截断之后照常追加
    store.append(VersionRecord(5, NOW, ChangeReason.NIGHTLY, (AddClass(RUN),)), expected_version=4)
    assert store.read().version == 5 and store.catalog_in_sync()


def test_a_crash_between_the_snapshot_and_the_shortened_log_reads_the_same(tmp_path: Path) -> None:
    store = seeded(tmp_path)
    before = store.read()
    full_log = store.changes_path.read_text()
    store.truncate_before(3)
    store.changes_path.write_text(full_log)  # 快照写了、日志还没改写就崩了
    assert store.read() == before
    assert [record.version for record in store.records()] == [4]


def test_truncation_waits_while_a_migration_is_unfinished(tmp_path: Path) -> None:
    store = seeded(tmp_path)
    store.write_migration(VersionRecord(5, NOW, ChangeReason.REVISION, (AddClass(RUN),)))
    assert not store.truncate_before(3)
    assert store.journal.segments() == (0, 4)
