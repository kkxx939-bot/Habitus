"""词表的"下一版"落到树上：迁移只改仍是源 token 的条目、日志只记生效的、崩溃后按计划续做；起步重归旧 token；
每晚新增把待定条目转成新类并清池；认作已有类的交白天归类重判，归进类的落到树上，其余在池里记下"已重判"。"""

from __future__ import annotations

import asyncio
from datetime import date, datetime
from pathlib import Path
from typing import Any

from habitus.behavior import BehaviorDocumentWriter
from habitus.behavior.kinds.changes import AddClass, ChangeReason, Move, VersionRecord
from habitus.behavior.kinds.ids import Lane
from habitus.behavior.kinds.pending import PendingEntry
from habitus.behavior.model import BehaviorKind
from habitus.behavior.tree import BehaviorTree
from habitus.behavior.uri import BehaviorURI
from habitus.infrastructure.store.locks import ProcessLocalLockStore
from tests.unit.behavior.kinds_fixtures import (
    NOW,
    NameMatchingProvider,
    ScriptedProvider,
    kind_components,
    named_classes,
)
from tests.unit.behavior.tree_payloads import local, occurrence_payload


def publish(tree: BehaviorTree, *, name: str, token: str, minute: int, day: int = 16) -> str:
    started = local(19, minute).replace(day=day)
    payload = occurrence_payload(
        name=name,
        kind_token=token,
        occurred_on=started.date().isoformat(),
        started_at=started.isoformat(timespec="microseconds"),
        last_observed_at=started.replace(second=30).isoformat(timespec="microseconds"),
        onset_available_at=started.replace(second=5).isoformat(timespec="microseconds"),
        basis=(),
    )
    document = BehaviorDocumentWriter(tree, ProcessLocalLockStore(), clock=lambda: local(23, 0)).publish(
        BehaviorKind.OCCURRENCE, payload, links=()
    )
    return str(BehaviorURI.from_address(document.address))


def setup(tmp_path: Path, provider: Any = None, names: tuple[str, ...] = ("修改代码", "调研")):  # type: ignore[no-untyped-def]
    tree = BehaviorTree(tmp_path / "tree")
    stamping, jobs, store = kind_components(
        tree, ProcessLocalLockStore(), names=names, provider=provider or NameMatchingProvider()
    )
    return tree, stamping, jobs, store


def token_of(tree: BehaviorTree, uri: str) -> str:
    return str(tree.read(BehaviorURI.parse(uri).to_address()).fields["kind_token"])


def test_migration_moves_only_matching_tokens_and_logs_what_took_effect(tmp_path: Path) -> None:
    tree, _, jobs, store = setup(tmp_path)
    moved = publish(tree, name="数行数", token="s-待定", minute=1)
    changed_elsewhere = publish(tree, name="又数行数", token="s-k0001", minute=2)
    new = named_classes(("修改代码", "调研", "统计代码规模"))[2]
    record = VersionRecord(
        2,
        NOW,
        ChangeReason.NIGHTLY,
        (AddClass(new),),
        (Move(moved, "s-待定", "s-k0003"), Move(changed_elsewhere, "s-待定", "s-k0003")),
    )
    outcome = jobs.migrator.apply(record, checkpoint=lambda: None)
    assert token_of(tree, moved) == "s-k0003" and token_of(tree, changed_elsewhere) == "s-k0001"
    assert outcome.applied == 1 and outcome.skipped == 1 and outcome.days == frozenset({date(2026, 8, 16)})
    assert store.records()[-1].moves == (Move(moved, "s-待定", "s-k0003"),)
    assert store.read_migration() is None and store.read().get(new.id).name == "统计代码规模"


def test_an_interrupted_migration_is_finished_from_its_plan(tmp_path: Path) -> None:
    tree, _, jobs, store = setup(tmp_path)
    uri = publish(tree, name="数行数", token="s-k0002", minute=1)
    record = VersionRecord(2, NOW, ChangeReason.REVISION, (), (Move(uri, "s-k0002", "s-k0001"),))
    store.write_migration(record)  # 崩在写完计划之后
    outcome = jobs.migrator.resume(checkpoint=lambda: None)
    assert outcome is not None and outcome.applied == 1 and token_of(tree, uri) == "s-k0001"
    assert store.read().version == 2 and store.read_migration() is None
    assert jobs.migrator.resume(checkpoint=lambda: None) is None


def test_a_plan_whose_version_is_already_logged_only_cleans_up(tmp_path: Path) -> None:
    tree, _, jobs, store = setup(tmp_path)
    uri = publish(tree, name="数行数", token="s-k0001", minute=1)
    record = VersionRecord(2, NOW, ChangeReason.REVISION, (), (Move(uri, "s-k0002", "s-k0001"),))
    store.write_migration(record)
    store.append(record, expected_version=1)  # 崩在写完日志之后
    pending = PendingEntry(uri, Lane.SESSION, date(2026, 8, 16), "x", "y")
    store.replace_pending(store.read_pending().pool.with_entries([pending]), expected_revision=0)
    outcome = jobs.migrator.resume(checkpoint=lambda: None)
    assert outcome is not None and store.read_migration() is None and store.read_pending().pool.entries == {}


def test_nightly_growth_turns_recurring_pending_entries_into_a_class(tmp_path: Path) -> None:
    uris_days = [(f"数行数{i}", 1 + i, 16 + i) for i in range(3)]
    body = {
        "groups": [
            {
                "members": ["P1", "P2", "P3"],
                "name": "统计代码规模",
                "criterion": "数代码行数、统计代码规模",
                "reminder": "该统计代码规模了",
                "excludes": [],
                "examples": ["数行数0"],
                "existing": None,
            }
        ]
    }
    tree, _, jobs, store = setup(tmp_path, provider=ScriptedProvider([body]))
    uris = [publish(tree, name=name, token="s-待定", minute=minute, day=day) for name, minute, day in uris_days]
    entries = [
        PendingEntry(uri, Lane.SESSION, date(2026, 8, day), name, name)
        for uri, (name, _, day) in zip(uris, uris_days, strict=True)
    ]
    store.replace_pending(store.read_pending().pool.with_entries(entries), expected_revision=0)
    outcome = asyncio.run(jobs.grow(datetime(2026, 8, 20, 3, 0, tzinfo=NOW.tzinfo), checkpoint=lambda: None))
    assert {token_of(tree, uri) for uri in uris} == {"s-k0003"}
    assert store.read().get(named_classes(("a", "b", "c"))[2].id).name == "统计代码规模"
    assert store.read_pending().pool.entries == {} and store.read_jobs().nightly_at is not None
    assert len(outcome.days) == 3


def test_nightly_recheck_moves_what_the_classifier_assigns_and_marks_the_rest(tmp_path: Path) -> None:
    nightly = {
        "groups": [
            {
                "members": ["P1", "P2"],
                "name": "修改代码",
                "criterion": "改代码",
                "reminder": "该改代码了",
                "excludes": [],
                "examples": [],
                "existing": "C1",
            }
        ]
    }
    recheck = {
        "items": [
            {"record": "R1", "reason": "就是改代码", "choice": "C1", "proposed": None},
            {"record": "R2", "reason": "不是改代码", "choice": "都不是", "proposed": "重命名项目"},
        ]
    }
    tree, _, jobs, store = setup(tmp_path, provider=ScriptedProvider([nightly, recheck]))
    edit = publish(tree, name="改一行代码", token="s-待定", minute=1)
    rename = publish(tree, name="重命名项目", token="s-待定", minute=2)
    entries = [PendingEntry(uri, Lane.SESSION, date(2026, 8, 16), name, name) for uri, name in ((edit, "改代码"), (rename, "改名"))]
    store.replace_pending(store.read_pending().pool.with_entries(entries), expected_revision=0)
    outcome = asyncio.run(jobs.grow(NOW, checkpoint=lambda: None))
    assert outcome.model_calls == 2
    assert token_of(tree, edit) == "s-k0001" and token_of(tree, rename) == "s-待定"
    pool = store.read_pending().pool.entries
    assert set(pool) == {rename} and pool[rename].rechecked  # 标记落盘（编解码往返）：下一晚不再问
    assert store.read().version == 2 and outcome.days == frozenset({date(2026, 8, 16)})


def test_revision_through_the_jobs_records_state_even_without_changes(tmp_path: Path) -> None:
    tree, _, jobs, store = setup(tmp_path, provider=ScriptedProvider([{"proposals": []}]))
    publish(tree, name="修改代码", token="s-k0001", minute=1)
    outcome = asyncio.run(jobs.revise(NOW, checkpoint=lambda: None))
    assert outcome.model_calls == 1 and store.read_jobs().revision_at == NOW and store.read().version == 1


def test_due_jobs_run_once_per_schedule(tmp_path: Path) -> None:
    tree, _, jobs, store = setup(tmp_path, provider=ScriptedProvider([{"proposals": []}]))
    first = asyncio.run(jobs.run_due(NOW, checkpoint=lambda: None))
    state = store.read_jobs()
    assert state.nightly_at == NOW and state.revision_at == NOW
    assert first.model_calls == 1  # 空池的每晚新增不调模型；本周的拆改提议调一次
    again = asyncio.run(jobs.run_due(NOW, checkpoint=lambda: None))
    assert again == type(again)()  # 同一天、同一周期都不再跑


# ── 评审修复（`05-评审/汇总.md` 甲类）──────────────────────────────────────────────


def test_scheduled_state_is_saved_before_the_migration_runs(tmp_path: Path) -> None:
    body = {
        "groups": [
            {
                "members": ["P1", "P2", "P3"],
                "name": "统计代码规模",
                "criterion": "数行数",
                "reminder": "该统计了",
                "excludes": [],
                "examples": [],
                "existing": None,
            }
        ]
    }
    tree, _, jobs, store = setup(tmp_path, provider=ScriptedProvider([body]))
    uris = [publish(tree, name=f"数行数{i}", token="s-待定", minute=i, day=16 + i) for i in range(3)]
    entries = [PendingEntry(uri, Lane.SESSION, date(2026, 8, 16 + i), "统计", "数行数") for i, uri in enumerate(uris)]
    store.replace_pending(store.read_pending().pool.with_entries(entries), expected_revision=0)

    def crash(*, checkpoint):  # type: ignore[no-untyped-def]
        raise RuntimeError("lease lost")

    jobs.migrator.resume = crash  # type: ignore[method-assign]
    try:
        asyncio.run(jobs.grow(NOW, checkpoint=lambda: None))
    except RuntimeError:
        pass
    assert store.read_jobs().nightly_at == NOW  # 状态已存：下一轮不会再跑一遍每晚新增
    assert store.read_migration() is not None  # 计划在：下一轮续做


def test_resuming_an_already_logged_plan_still_reports_the_changed_days(tmp_path: Path) -> None:
    tree, _, jobs, store = setup(tmp_path)
    uri = publish(tree, name="数行数", token="s-k0001", minute=1)
    record = VersionRecord(2, NOW, ChangeReason.REVISION, (), (Move(uri, "s-k0002", "s-k0001"),))
    store.write_migration(record)
    store.append(record, expected_version=1)
    outcome = jobs.migrator.resume(checkpoint=lambda: None)
    assert outcome is not None and outcome.days == frozenset({date(2026, 8, 16)})


def test_a_migration_plan_that_does_not_apply_is_refused_before_touching_the_tree(tmp_path: Path) -> None:
    from habitus.behavior.kinds.changes import RetireClass
    from habitus.behavior.kinds.ids import ClassId as _ClassId
    from habitus.behavior.kinds.store import BehaviorKindStoreError

    _, _, _, store = setup(tmp_path)
    broken = VersionRecord(2, NOW, ChangeReason.REVISION, (RetireClass(_ClassId(Lane.SESSION, 99)),))
    try:
        store.write_migration(broken)
        raise AssertionError("expected refusal")
    except BehaviorKindStoreError:
        pass
    assert store.read_migration() is None


def test_a_full_pending_pool_refuses_the_overflow_instead_of_raising(tmp_path: Path) -> None:
    from habitus.behavior.kinds.config import BehaviorKindConfig

    tree = BehaviorTree(tmp_path / "tree")
    stamping, _, store = kind_components(
        tree,
        ProcessLocalLockStore(),
        names=("修改代码",),
        provider=NameMatchingProvider(),
        config=BehaviorKindConfig(max_pending=1),
    )
    entries = [PendingEntry(f"behavior://x{i}", Lane.SESSION, date(2026, 8, 16), "新事", "内容") for i in range(3)]
    admission = stamping.record_pending(entries)
    assert admission.added == 1 and len(admission.refused) == 2
    assert any("kind_pending_pool_full 2" in note for note in admission.signals())
    assert len(store.read_pending().pool.entries) == 1


def test_classification_renews_the_lease_between_chunks(tmp_path: Path) -> None:
    from habitus.behavior.kinds.classify import ClassifyRequest, OccurrenceContent
    from habitus.behavior.kinds.config import BehaviorKindConfig

    tree = BehaviorTree(tmp_path / "tree")
    stamping, _, store = kind_components(
        tree,
        ProcessLocalLockStore(),
        names=("修改代码",),
        provider=NameMatchingProvider(),
        config=BehaviorKindConfig(batch_size=2),
    )
    requests = [ClassifyRequest(f"k{i}", Lane.SESSION, OccurrenceContent(name="修改代码")) for i in range(5)]
    beats: list[int] = []
    result = asyncio.run(stamping.classify_requests(requests, store.read(), checkpoint=lambda: beats.append(1)))
    assert len(result.verdicts) == 5 and len(beats) == 3  # 5 条、每块 2 条 → 3 块，块间各续一次


def test_due_jobs_use_the_configured_zone_not_the_process_zone(tmp_path: Path) -> None:
    """同一时刻、同一份上次运行记录：东八区还没到每晚 02:00，西八区已经过了——两种时区结论相反，读进程时区的实现过不了两条。"""

    from datetime import timedelta, timezone

    last = datetime(2026, 10, 5, 18, 30, tzinfo=NOW.tzinfo)  # 东八区 10-06 02:30 / 西八区 10-05 10:30
    instant = datetime(2026, 10, 6, 17, 30, tzinfo=NOW.tzinfo)  # 东八区 10-07 01:30 / 西八区 10-06 09:30
    for hours, expect_ran in ((8, False), (-8, True)):
        tree = BehaviorTree(tmp_path / f"tree{hours}")
        _, jobs, store = kind_components(
            tree, ProcessLocalLockStore(), names=(), provider=ScriptedProvider([{"groups": []}])
        )
        jobs.zone = timezone(timedelta(hours=hours))
        store.replace_jobs(store.read_jobs().after_nightly(last))
        asyncio.run(jobs.run_due(instant, checkpoint=lambda: None))
        assert (store.read_jobs().nightly_at == instant) is expect_ran, hours
