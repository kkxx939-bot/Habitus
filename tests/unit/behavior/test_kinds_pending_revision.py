"""定期拆改只出计划、下一次换代才改树（实时化阶段 2 · C+D「生效时点」，立项第 3 条；预期 CD-11）。

拆改出了计划：计划连同拆改状态存成一份待生效（``kinds.revision.json``），不算"迁移中"（``read_migration`` 不认它，预测拍与
结算照常）；有待生效的拆改时这一天不再长新类；换代最前那一步（``VocabularyRunner.resume`` → ``apply_revision``）交给迁移器：
写迁移计划 → 写拆改状态 → 执行迁移，再撤掉待生效那一份。拆改的判法换成替身（这里只看生效时点）。
"""

from __future__ import annotations

import asyncio
from datetime import timedelta
from pathlib import Path

from habitus.behavior.kinds.changes import AddClass, ChangeReason, Move, VersionRecord
from habitus.behavior.kinds.ids import Lane
from habitus.behavior.kinds.pending import PendingEntry
from habitus.behavior.kinds.revision.reviser import RevisionResult
from habitus.behavior.kinds.schedule import JobState
from tests.unit.behavior.kinds_fixtures import NOW, named_classes
from tests.unit.behavior.test_reduction_kinds import publish, setup, token_of

LATER = NOW + timedelta(hours=1)


def planned(tmp_path: Path):  # type: ignore[no-untyped-def]
    """一条「修改代码」的记录；拆改（替身）提出：新类「统计代码规模」、把它挪过去。"""

    tree, _, jobs, store = setup(tmp_path)
    uri = publish(tree, name="数行数", token="s-k0001", minute=1)
    new = named_classes(("修改代码", "调研", "统计代码规模"))[2]
    record = VersionRecord(2, NOW, ChangeReason.REVISION, (AddClass(new),), (Move(uri, "s-k0001", "s-k0003"),))
    state = JobState(revision_at=NOW)
    asked: list[str] = []

    async def revise(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        asked.append("revise")
        return RevisionResult(record=record, state=state, model_calls=1, signals=())

    jobs.reviser.revise = revise  # type: ignore[method-assign]
    return tree, jobs, store, uri, record, asked


def test_cd_11_a_revision_plan_waits_for_the_next_generation_and_is_not_a_migration(tmp_path: Path) -> None:
    """CD-11 前半：白天拆改出了计划——树、词表版本、拆改状态都没动；计划连同拆改状态存成待生效；``read_migration`` 认不出它
    （不是迁移中：预测拍与结算靠它判）。撤回即失败：拆改出了计划当场交迁移器，树上的编号当场就改了。"""

    tree, jobs, store, uri, record, _asked = planned(tmp_path)
    outcome = asyncio.run(jobs.revise(NOW, checkpoint=lambda: None))
    assert outcome.model_calls == 1 and any("kind_revision_pending v2" in signal for signal in outcome.signals)
    assert token_of(tree, uri) == "s-k0001" and store.read().version == 1
    assert store.read_migration() is None and store.read_jobs().revision_at is None
    assert store.read_revision() == (record, JobState(revision_at=NOW))


def test_cd_11_while_a_plan_waits_no_new_class_grows_and_the_period_is_not_revised_again(tmp_path: Path) -> None:
    """CD-11：有待生效的拆改时，这一天不长新类、也不再拆改——``due`` 说不欠，``nightly`` 什么都不做（计划是对着这一版词表出的）。"""

    tree, jobs, store, _uri, _record, asked = planned(tmp_path)
    night = NOW.astimezone(jobs.zone).date()
    asyncio.run(jobs.revise(NOW, checkpoint=lambda: None))
    fresh = publish(tree, name="打包发布", token="s-待定", minute=5)
    entry = PendingEntry(fresh, Lane.SESSION, night, "打包", "打包发布")
    store.replace_pending(store.read_pending().pool.with_entries([entry]), expected_revision=0)
    assert jobs.due(night) is False
    outcome = asyncio.run(jobs.nightly(LATER, night=night, checkpoint=lambda: None))
    assert outcome.model_calls == 0 and asked == ["revise"] and store.read().version == 1
    assert set(store.read_pending().pool.entries) == {fresh}  # 待定池原样留着


def test_cd_11_the_generation_hands_the_plan_to_the_migrator_first(tmp_path: Path) -> None:
    """CD-11 后半：换代最前那一步（``apply_revision``）把它交给迁移器——树上改了编号、词表到第 2 版、拆改状态记上、待生效那一份
    撤掉；之后不再有待生效、也不是迁移中，这一天又能长新类。"""

    tree, jobs, store, uri, _record, _asked = planned(tmp_path)
    asyncio.run(jobs.revise(NOW, checkpoint=lambda: None))
    outcome = jobs.apply_revision(lambda: None)
    assert token_of(tree, uri) == "s-k0003" and store.read().version == 2
    assert store.read_jobs().revision_at == NOW and store.read_revision() is None and store.read_migration() is None
    assert outcome.days
    assert jobs.apply_revision(lambda: None).days == frozenset()  # 没有待生效的：空操作


def test_cd_11_a_plan_overtaken_by_another_change_is_dropped_but_the_period_counts_as_revised(tmp_path: Path) -> None:
    """计划出了之后词表又动过（不再是下一版）：作废、留信号，拆改状态照记——不每天重做这次拆改；树不动。"""

    tree, jobs, store, uri, _record, _asked = planned(tmp_path)
    asyncio.run(jobs.revise(NOW, checkpoint=lambda: None))
    other = named_classes(("修改代码", "调研", "别的类"))[2]
    store.append(VersionRecord(2, NOW, ChangeReason.NIGHTLY, (AddClass(other),)), expected_version=1)
    jobs.apply_revision(lambda: None)
    assert token_of(tree, uri) == "s-k0001" and store.read().version == 2
    assert store.read_jobs().revision_at == NOW and store.read_revision() is None


def test_cd_11_the_runner_reports_the_pending_plan_and_its_resume_applies_it(tmp_path: Path) -> None:
    """换代那一步走的路径（``VocabularyRunner.revision_pending`` / ``resume``，持 sweep 锁）：有待生效时报出来，``resume``
    之后树上改了编号、不再待生效。"""

    from tests.unit.behavior.test_reduction_runner import Harness

    harness = Harness(tmp_path)
    jobs, store = harness.vocabulary_jobs, harness.kind_store
    uri = publish(harness.tree, name="洗手", token="s-k0001", minute=1)
    new = named_classes(("洗手", "洗脸"))[1]
    record = VersionRecord(2, NOW, ChangeReason.REVISION, (AddClass(new),), (Move(uri, "s-k0001", "s-k0002"),))

    async def revise(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        return RevisionResult(record=record, state=JobState(revision_at=NOW), model_calls=1, signals=())

    jobs.reviser.revise = revise  # type: ignore[method-assign]
    asyncio.run(jobs.revise(NOW, checkpoint=lambda: None))
    vocabulary = harness.runner.vocabulary
    assert vocabulary.revision_pending() and token_of(harness.tree, uri) == "s-k0001"
    asyncio.run(vocabulary.resume())
    assert not vocabulary.revision_pending() and token_of(harness.tree, uri) == "s-k0002"
    assert store.read().version == 2 and store.read_migration() is None
