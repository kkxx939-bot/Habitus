"""词表变更日志的"快照 + 分段"（26 号清单 6.5）：截断之后当前词表一样、读当前不再从第 1 版重放；截断下限那一版之后的变更
仍然完整；更早的请求报错；截断写到一半崩了也不会读错。"""

from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path

import pytest

from habitus.behavior.kinds.changes import AddClass, ChangeReason, ReviseClass, VersionRecord, merge_operations
from habitus.behavior.kinds.codec import class_payload, pending_payload, record_payload
from habitus.behavior.kinds.ids import Lane
from habitus.behavior.kinds.journal import VocabularyJournalError
from habitus.behavior.kinds.pending import PendingEntry
from habitus.behavior.kinds.reader import VocabularyReader
from habitus.behavior.kinds.store import BehaviorKindStore, BehaviorKindStoreError
from tests.unit.behavior.kinds_fixtures import (
    EDIT,
    NOW,
    RESEARCH,
    catalog_in_sync,
    edit_class,
    named_classes,
    research_class,
)

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
    assert store.read().version == 5 and catalog_in_sync(store)


def test_a_crash_between_the_snapshot_and_the_shortened_log_reads_the_same(tmp_path: Path) -> None:
    store = seeded(tmp_path)
    before = store.read()
    full_log = store.changes_path.read_bytes()
    store.truncate_before(3)
    store.changes_path.write_bytes(full_log)  # 快照写了、日志还没改写就崩了
    assert store.read() == before
    assert [record.version for record in store.records()] == [4]


def test_truncation_waits_while_a_migration_is_unfinished(tmp_path: Path) -> None:
    store = seeded(tmp_path)
    store.write_migration(VersionRecord(5, NOW, ChangeReason.REVISION, (AddClass(RUN),)))
    assert not store.truncate_before(3)
    assert store.journal.segments() == (0, 4)


def _class_payload_with(field: str, bad: object) -> dict[str, object]:
    payload = class_payload(RUN)
    if field == "examples":
        payload["examples"] = ["跑了五公里", bad]
    elif field == "exclusion text":
        payload["excludes"] = [{"text": bad, "goes_to": None}]
    else:
        payload[field] = bad
    return payload


@pytest.mark.parametrize(
    ("tamper", "message"),
    [
        ({"version": True}, "version must be a positive integer"),
        ({"version": "5"}, "version must be a positive integer"),
        ({"moves": [[1, "a", "b"]]}, "move occurrence must be non-empty text"),
        ({"moves": [["behavior://x", None, "b"]]}, "move source must be non-empty text"),
        ({"moves": [["behavior://x", "a", 3]]}, "move target must be non-empty text"),
        *(
            (
                {"operations": [{"op": "add", "class": _class_payload_with(field, 1)}]},
                f"behavior class is invalid: {label} must be non-empty text without surrounding whitespace",
            )
            for field, label in (
                ("name", "class name"),
                ("criterion", "class criterion"),
                ("reminder", "class reminder"),
                ("examples", "class example"),
                ("exclusion text", "exclusion text"),
            )
        ),
    ],
)
def test_a_migration_plan_with_wrong_types_is_refused_by_the_codec(
    tmp_path: Path, tamper: dict[str, object], message: str
) -> None:
    """W16：迁移计划文件里类型错的字段在 ``record_from``/``class_from`` 构造之前就被拒，报错与值对象那条检查同一句。"""

    store = seeded(tmp_path)
    payload = {**record_payload(VersionRecord(5, NOW, ChangeReason.REVISION, (AddClass(RUN),))), **tamper}
    store.migration_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(
        BehaviorKindStoreError, match=re.escape(f"migration plan is invalid: version record is invalid: {message}")
    ):
        store.read_migration()


@pytest.mark.parametrize("bad", [True, "4", None])
def test_a_snapshot_with_a_non_integer_version_is_refused_by_the_codec(tmp_path: Path, bad: object) -> None:
    store = seeded(tmp_path)
    assert store.truncate_before(2)
    payload = json.loads(store.journal.snapshot_path.read_text(encoding="utf-8"))
    payload["version"] = bad
    store.journal.snapshot_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(BehaviorKindStoreError, match="vocabulary version must be a non-negative integer"):
        store.read()


@pytest.mark.parametrize(
    ("field", "bad", "message"),
    [
        ("occurrence", 1, "pending occurrence must be non-empty text"),
        ("proposed", None, "pending proposed must be non-empty text"),
        ("content", ["内容"], "pending content must be non-empty text"),
        ("rechecked", "no", "pending rechecked must be a bool"),
        ("rechecked", 0, "pending rechecked must be a bool"),
    ],
)
def test_a_pending_pool_with_wrong_types_is_refused_by_the_codec(
    tmp_path: Path, field: str, bad: object, message: str
) -> None:
    store = BehaviorKindStore(tmp_path)
    day = date(2026, 3, 1)
    entry = pending_payload(PendingEntry("behavior://p1", Lane.SESSION, day, "跑步", "早上跑了五公里", entered_on=day))
    store.pending_path.parent.mkdir(parents=True, exist_ok=True)
    store.pending_path.write_text(
        json.dumps({"revision": 1, "entries": [{**entry, field: bad}]}, ensure_ascii=False), encoding="utf-8"
    )
    with pytest.raises(BehaviorKindStoreError, match=re.escape(f"pending entry is invalid: {message}")):
        store.read_pending()


@pytest.mark.parametrize("bad", [True, "1", 1.0, 0])
def test_a_pending_pool_with_a_non_positive_integer_revision_is_refused(tmp_path: Path, bad: object) -> None:
    store = BehaviorKindStore(tmp_path)
    store.pending_path.parent.mkdir(parents=True, exist_ok=True)
    store.pending_path.write_text(json.dumps({"revision": bad, "entries": []}), encoding="utf-8")
    with pytest.raises(BehaviorKindStoreError, match="pending pool revision must be positive"):
        store.read_pending()
