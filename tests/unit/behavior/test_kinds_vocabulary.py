"""词表的身份、自洽校验、变更日志重放与文件存储。"""

from __future__ import annotations

from dataclasses import replace
from datetime import date
from pathlib import Path

import pytest

from habitus.behavior.kinds.changes import (
    AddClass,
    Branch,
    ChangeReason,
    Move,
    RetireClass,
    ReviseClass,
    VersionRecord,
    apply,
    merge_operations,
    replay,
    split_operations,
)
from habitus.behavior.kinds.config import BehaviorKindConfig
from habitus.behavior.kinds.ids import (
    ClassId,
    KindIdError,
    Lane,
    is_marker,
    lane_of_token,
    not_event_token,
    pending_token,
)
from habitus.behavior.kinds.model import BehaviorClass, BehaviorKindError, ClassOrigin, Exclusion, Vocabulary
from habitus.behavior.kinds.pending import PendingEntry
from habitus.behavior.kinds.store import BehaviorKindConflictError, BehaviorKindStore, BehaviorKindStoreError
from tests.unit.behavior.kinds_fixtures import EDIT, NOW, RESEARCH, edit_class, research_class


def bootstrap() -> VersionRecord:
    return VersionRecord(1, NOW, ChangeReason.NIGHTLY, (AddClass(edit_class()), AddClass(research_class())))


def new_class(number: int, name: str, *, lane: Lane = Lane.SESSION) -> BehaviorClass:
    return BehaviorClass(ClassId(lane, number), name, f"{name}的判据", f"该{name}了", origin=ClassOrigin.PROMOTED)


# ── 编号与占位标记 ─────────────────────────────────────────────────────────────────


def test_ids_are_path_safe_and_round_trip() -> None:
    assert str(EDIT) == "s-k0001" and ClassId.parse("p-k0123") == ClassId(Lane.PHYSICAL, 123)
    assert str(ClassId(Lane.SESSION, 12345)) == "s-k12345"
    for bad in ("s:k0001", "x-k0001", "s-k1", "s-k0000", 7):
        with pytest.raises(KindIdError):
            ClassId.parse(bad)


def test_markers_belong_to_a_lane_and_are_not_classes() -> None:
    assert pending_token(Lane.SESSION) == "s-待定" and not_event_token(Lane.PHYSICAL) == "p-非事件"
    assert is_marker("s-待定") and is_marker("p-非事件") and not is_marker("s-k0001") and not is_marker(None)
    assert lane_of_token("p-待定") is Lane.PHYSICAL and lane_of_token("s-k0002") is Lane.SESSION
    with pytest.raises(KindIdError):
        lane_of_token("修改代码")
    with pytest.raises(KindIdError):
        lane_of_token(3)


# ── 自洽校验 ─────────────────────────────────────────────────────────────────────


def test_class_rejects_empty_text_self_exclusion_and_repeated_examples() -> None:
    with pytest.raises(BehaviorKindError):
        replace(edit_class(), name=" ")
    with pytest.raises(BehaviorKindError):
        replace(edit_class(), excludes=(Exclusion("自己", EDIT),))
    with pytest.raises(BehaviorKindError):
        replace(edit_class(), examples=("a", "a"))
    with pytest.raises(BehaviorKindError):
        Exclusion("x", "s-k0001")  # type: ignore[arg-type]


def test_vocabulary_rejects_duplicate_active_names_and_cross_lane_pointers() -> None:
    twin = replace(research_class(), name="修改代码", excludes=())
    with pytest.raises(BehaviorKindError):
        Vocabulary(1, {EDIT: edit_class(), RESEARCH: twin})
    eat = new_class(1, "吃饭", lane=Lane.PHYSICAL)
    pointing = replace(research_class(), excludes=(Exclusion("吃饭", eat.id),))
    with pytest.raises(BehaviorKindError):
        Vocabulary(1, {EDIT: edit_class(), RESEARCH: pointing, eat.id: eat})
    with pytest.raises(BehaviorKindError):
        Vocabulary(1, {EDIT: edit_class(), eat.id: eat}, branches={EDIT: (eat.id,)})


def test_same_name_is_allowed_across_lanes_and_for_retired_classes() -> None:
    physical = new_class(1, "修改代码", lane=Lane.PHYSICAL)
    retired = replace(new_class(3, "修改代码"), status="retired")
    vocabulary = Vocabulary(1, {EDIT: edit_class(), physical.id: physical, retired.id: retired})
    assert [item.id for item in vocabulary.active(Lane.SESSION)] == [EDIT]
    assert vocabulary.next_id(Lane.SESSION) == ClassId(Lane.SESSION, 4)
    assert vocabulary.next_id(Lane.PHYSICAL) == ClassId(Lane.PHYSICAL, 2)


# ── 变更日志 ─────────────────────────────────────────────────────────────────────


def test_replay_follows_merges_and_splits_for_descendants() -> None:
    count = new_class(3, "统计代码规模")
    narrowed = replace(edit_class(), criterion="只算动手改代码")
    split_out = new_class(4, "清理")
    vocabulary = replay(
        [
            bootstrap(),
            VersionRecord(2, NOW, ChangeReason.NIGHTLY, (AddClass(count),)),
            VersionRecord(3, NOW, ChangeReason.REVISION, merge_operations(count.id, RESEARCH)),
            VersionRecord(4, NOW, ChangeReason.REVISION, split_operations(narrowed, split_out)),
        ]
    )
    assert vocabulary.version == 4
    assert vocabulary.descendants(count.id) == (RESEARCH,)
    assert vocabulary.descendants(EDIT) == (EDIT, split_out.id)
    assert vocabulary.get(EDIT).criterion == "只算动手改代码"
    assert vocabulary.descendants(ClassId(Lane.SESSION, 99)) == ()


def test_apply_refuses_gaps_duplicates_and_changes_to_retired_classes() -> None:
    first = apply(Vocabulary(), bootstrap())
    with pytest.raises(BehaviorKindError):
        apply(first, replace(bootstrap(), version=3))
    with pytest.raises(BehaviorKindError):
        apply(first, VersionRecord(2, NOW, ChangeReason.NIGHTLY, (AddClass(edit_class()),)))
    retired = apply(first, VersionRecord(2, NOW, ChangeReason.REVISION, (RetireClass(EDIT),)))
    for operation in (RetireClass(EDIT), ReviseClass(edit_class())):
        with pytest.raises(BehaviorKindError):
            apply(retired, VersionRecord(3, NOW, ChangeReason.REVISION, (operation,)))
    with pytest.raises(BehaviorKindError):
        apply(first, VersionRecord(2, NOW, ChangeReason.REVISION, (Branch(EDIT, RESEARCH), Branch(EDIT, RESEARCH))))
    with pytest.raises(BehaviorKindError):
        apply(first, VersionRecord(2, NOW, ChangeReason.REVISION, (Branch(EDIT, ClassId(Lane.SESSION, 9)),)))


def test_version_record_requires_a_change_and_unique_moves() -> None:
    with pytest.raises(BehaviorKindError):
        VersionRecord(1, NOW, ChangeReason.NIGHTLY, ())
    with pytest.raises(BehaviorKindError):
        VersionRecord(
            1, NOW, ChangeReason.NIGHTLY, (), (Move("u", "s-待定", "s-k0001"), Move("u", "s-待定", "s-k0002"))
        )
    with pytest.raises(BehaviorKindError):
        Move("u", "s-k0001", "s-k0001")
    with pytest.raises(BehaviorKindError, match="neither a class id nor a marker"):
        Move("u", "为Tagent添加ReAct支持", "s-k0001")  # 旧口径的 token 不再有迁移路径（裁定 18 第 7 条）


# ── 存储 ─────────────────────────────────────────────────────────────────────────


def test_store_appends_with_cas_and_replays_the_log(tmp_path: Path) -> None:
    store = BehaviorKindStore(tmp_path)
    assert store.read() == Vocabulary() and store.records() == ()
    moved = VersionRecord(
        1, NOW, ChangeReason.NIGHTLY, bootstrap().operations, (Move("behavior://x", "s-待定", "s-k0001"),)
    )
    store.append(moved, expected_version=0)
    with pytest.raises(BehaviorKindConflictError):
        store.append(replace(moved, version=2), expected_version=0)
    reopened = BehaviorKindStore(tmp_path)
    assert reopened.read().version == 1 and reopened.records()[0].moves == moved.moves
    assert reopened.catalog_in_sync()
    catalog = (tmp_path / "kinds.md").read_text(encoding="utf-8")
    assert "修改代码**〔s-k0001〕" in catalog and "不含：动手改代码（→修改代码）" in catalog


def test_store_refuses_a_record_that_does_not_apply(tmp_path: Path) -> None:
    store = BehaviorKindStore(tmp_path)
    store.append(bootstrap(), expected_version=0)
    with pytest.raises(BehaviorKindStoreError):
        store.append(VersionRecord(2, NOW, ChangeReason.NIGHTLY, (AddClass(edit_class()),)), expected_version=1)


def test_store_reports_a_corrupt_log_and_a_stale_catalog(tmp_path: Path) -> None:
    store = BehaviorKindStore(tmp_path)
    store.append(bootstrap(), expected_version=0)
    (tmp_path / "kinds.md").write_text("手改过", encoding="utf-8")
    assert not store.catalog_in_sync()
    (tmp_path / "kinds.changes.jsonl").write_text("{坏}\n", encoding="utf-8")
    with pytest.raises(BehaviorKindStoreError):
        store.read()


def test_pending_pool_is_idempotent_bounded_and_cas(tmp_path: Path) -> None:
    store = BehaviorKindStore(tmp_path, config=BehaviorKindConfig(max_pending=1))
    entry = PendingEntry("behavior://a", Lane.SESSION, date(2026, 10, 1), "统计代码规模", "数行数")
    pool = store.read_pending().pool.with_entries([entry, entry])
    snapshot = store.replace_pending(pool, expected_revision=0)
    assert snapshot.revision == 1 and store.read_pending().pool.lane(Lane.SESSION) == (entry,)
    with pytest.raises(BehaviorKindConflictError):
        store.replace_pending(pool, expected_revision=0)
    other = replace(entry, occurrence="behavior://b")
    with pytest.raises(BehaviorKindStoreError):
        store.replace_pending(pool.with_entries([other]), expected_revision=1)
    assert store.read_pending().pool.without(["behavior://a"]).entries == {}


def test_migration_plan_is_single_next_version_and_clearable(tmp_path: Path) -> None:
    store = BehaviorKindStore(tmp_path)
    plan = VersionRecord(
        1, NOW, ChangeReason.NIGHTLY, bootstrap().operations, (Move("behavior://x", "s-待定", "s-k0001"),)
    )
    with pytest.raises(BehaviorKindConflictError):
        store.write_migration(replace(plan, version=2))
    store.write_migration(plan)
    assert store.read_migration() == plan
    with pytest.raises(BehaviorKindConflictError):
        store.write_migration(plan)
    store.clear_migration()
    assert store.read_migration() is None
    store.clear_migration()


# ── 定时状态 ─────────────────────────────────────────────────────────────


def test_job_state_round_trips_through_the_store(tmp_path: Path) -> None:
    from habitus.behavior.kinds.schedule import JobState

    store = BehaviorKindStore(tmp_path)
    assert store.read_jobs() == JobState()
    state = JobState().after_nightly(NOW).after_revision(NOW, "2026-09-30", frozenset({("s-k0003", "s-k0001")}), keep=2)
    store.replace_jobs(state)
    assert store.read_jobs() == state
    (tmp_path / "kinds.jobs.json").write_text(
        '{"nightly_at": 3, "revision_at": null, "merge_history": []}', encoding="utf-8"
    )
    with pytest.raises(BehaviorKindStoreError):
        store.read_jobs()


def test_branches_must_point_to_active_classes_and_never_form_a_cycle() -> None:
    first = apply(Vocabulary(), bootstrap())
    merged = apply(first, VersionRecord(2, NOW, ChangeReason.REVISION, merge_operations(RESEARCH, EDIT)))
    with pytest.raises(BehaviorKindError, match="active"):
        apply(merged, VersionRecord(3, NOW, ChangeReason.REVISION, (Branch(EDIT, RESEARCH),)))
    third = new_class(3, "清理")
    chained = apply(first, VersionRecord(2, NOW, ChangeReason.REVISION, (AddClass(third), Branch(EDIT, third.id))))
    with pytest.raises(BehaviorKindError, match="cycle"):
        apply(chained, VersionRecord(3, NOW, ChangeReason.REVISION, (Branch(third.id, EDIT),)))


def test_nightly_does_not_run_twice_on_the_dst_fall_back_day() -> None:
    from datetime import datetime
    from zoneinfo import ZoneInfo

    from habitus.behavior.kinds.schedule import nightly_due

    berlin = ZoneInfo("Europe/Berlin")
    ran = datetime(2026, 10, 25, 2, 30, tzinfo=berlin)
    assert not nightly_due(datetime(2026, 10, 25, 2, 15, tzinfo=berlin, fold=1), ran, hour=2)
    assert nightly_due(datetime(2026, 10, 26, 2, 5, tzinfo=berlin), ran, hour=2)


@pytest.mark.parametrize("separator", [" ", " ", "\u0085", "\n", "\x00"])
def test_class_text_rejects_line_breaking_and_unprintable_characters(separator: str) -> None:
    """类名、判据多半是模型写的：混进"换行"字符的条目进了变更日志，按行读就切断记录（第二轮评审 C5）。"""

    with pytest.raises(BehaviorKindError):
        replace(edit_class(), name=f"修改{separator}代码")
    with pytest.raises(BehaviorKindError):
        replace(edit_class(), examples=(f"改{separator}了",))


def test_change_log_reads_records_whose_free_text_holds_a_unicode_line_separator(tmp_path) -> None:
    """版本备注不过条目校验；里面的 U+2028 原样写进日志后，读日志仍按 "\\n" 切、一条不断。"""

    from datetime import UTC, datetime

    from habitus.behavior.kinds.changes import AddClass, ChangeReason, VersionRecord
    from habitus.behavior.kinds.store import BehaviorKindStore

    store = BehaviorKindStore(tmp_path)
    store.append(
        VersionRecord(1, datetime(2026, 10, 7, tzinfo=UTC), ChangeReason.NIGHTLY, (AddClass(edit_class()),), note="甲 乙"),
        expected_version=0,
    )
    (record,) = store.records()
    assert record.note == "甲 乙" and store.read().version == 1
