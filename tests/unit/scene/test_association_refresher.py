"""关联的夜批编排：顺序、幂等、检查点、失败记账、L1/L0/由来，以及完成标记的把关。"""

from __future__ import annotations

import asyncio
from datetime import date, timedelta

import pytest

from habitus.prediction.model import SlotKey
from habitus.scene.association.model import AssociationAssembly, AssociationDraft
from habitus.scene.association.progress import AssociationProgressError, TaskKey
from habitus.scene.association.refresher import (
    AssociationBusyError,
    AssociationRefreshConfig,
    AssociationRefresher,
)
from habitus.scene.association.service import AssociationLimitError
from habitus.scene.backlog import AssociationTask, CauseFacts
from habitus.scene.model import RegularityLevel
from habitus.scene.regularity.overview import Overview, rebuild
from tests.unit.scene.association_ground import FRIDAY, NEXT_FRIDAY, Ground, at


def weekly(tmp_path, *, weeks: int = 2) -> Ground:
    """连着几个周五打球，每次之前都跟朋友通个电话。"""

    ground = Ground(tmp_path)
    for week in range(weeks):
        day = FRIDAY + timedelta(days=7 * week)
        ground.record(day, "吃早饭", 8, 0)
        ground.record(day, "和朋友通电话", 18, 0, kind="通话")
        ground.record(day, "打球", 19, 0, kind="打球")
    return ground


def test_a_candidates_days_are_associated_in_time_order(tmp_path) -> None:
    """候选内部必须升序：规律级是增量叠加的，乱序会让情境的演化顺序错乱。"""

    ground = weekly(tmp_path)

    report = ground.run(only="打球")

    assert [payload.day for payload in ground.associator.payloads] == [FRIDAY, NEXT_FRIDAY]
    assert report.associated == ("打球/2026-09-04", "打球/2026-09-11")
    assert report.model_calls == 2


def test_a_completed_day_never_comes_back(tmp_path) -> None:
    """差集天然幂等：完成标记写下之后，那一天不再进待办。"""

    ground = weekly(tmp_path)
    ground.run(only="打球")
    calls = len(ground.associator.payloads)

    second = ground.run(only="打球")

    assert len(ground.associator.payloads) == calls
    assert second.associated == () and second.model_calls == 0


def test_the_records_land_with_their_context_and_the_overview_accumulates(tmp_path) -> None:
    ground = weekly(tmp_path)

    ground.run(only="打球")

    assert ground.regularity_tree.days_for("打球") == frozenset({FRIDAY, NEXT_FRIDAY})
    (record,) = ground.regularity_tree.read_day("打球", NEXT_FRIDAY)
    assert record.context == "2026-09-11 第 1 次打球"
    overview = Overview.read(ground.regularity_tree, "打球")
    assert [(item.text, item.days) for item in overview.situations] == [("打球的第一种情形", (FRIDAY, NEXT_FRIDAY))]
    assert ground.regularity_tree.read_layer("打球", RegularityLevel.ABSTRACT) == "打球的第一种情形"


def test_the_origin_is_written_once_and_never_rewritten(tmp_path) -> None:
    """由来与 L0 的更新节奏不同：L0 随情境分布变，由来定了就定了。"""

    ground = weekly(tmp_path)

    ground.run(only="打球")

    assert Overview.read(ground.regularity_tree, "打球").origin == "2026-09-04 第 1 次打球"


def test_a_new_situation_only_opens_when_the_model_says_it_is_new(tmp_path) -> None:
    """归进已有的一种就只多记一个日期，L1 的文字一个字都不动。"""

    ground = weekly(tmp_path, weeks=3)
    ground.run(only="打球")

    overview = Overview.read(ground.regularity_tree, "打球")

    assert len(overview.situations) == 1
    assert overview.situations[0].days == (FRIDAY, NEXT_FRIDAY, FRIDAY + timedelta(days=14))


def test_the_overview_can_be_rebuilt_from_the_records(tmp_path) -> None:
    """增量追加是为了不每夜重扫全部历史；全量重建作为修复口留着。"""

    ground = weekly(tmp_path)
    ground.run(only="打球")

    assert (
        rebuild(ground.regularity_tree, "打球").situations == Overview.read(ground.regularity_tree, "打球").situations
    )


def test_a_candidate_that_did_not_occur_that_day_is_skipped_without_a_call(tmp_path) -> None:
    ground = weekly(tmp_path)
    task = ground.tasks()[0]
    missing = type(task)(kind_token="从没发生过的行为", day=task.day, slots=task.slots)

    report = asyncio.run(ground.refresher.refresh((missing,), causes=CauseFacts(ground.tree())))

    assert report.skipped == ("从没发生过的行为/2026-09-04",) and report.model_calls == 0


def test_an_unanswered_target_leaves_the_day_open(tmp_path) -> None:
    """有目标没拿到草稿，这一天就不写完成标记——下一轮还会再来。"""

    ground = Ground(tmp_path)
    ground.record(FRIDAY, "打球", 10, 0, kind="打球")
    ground.record(FRIDAY, "打球", 19, 0, kind="打球")

    def answer(payload):  # type: ignore[no-untyped-def]
        first = payload.targets[0]
        return AssociationAssembly(
            drafts=(AssociationDraft(occurrence_no=first, context="只说了第一次", cites=(first,), new_situation="x"),),
            unanswered=(payload.targets[1],),
        )

    ground.associator.answer = answer
    report = ground.run(only="打球")

    assert ground.regularity_tree.days_for("打球") == frozenset()
    assert any("day left open" in note for note in report.signals)
    assert len(ground.regularity_tree.read_day("打球", FRIDAY)) == 1


def test_the_call_budget_defers_the_rest_of_the_run(tmp_path) -> None:
    ground = weekly(tmp_path, weeks=3)
    ground.refresher.config = AssociationRefreshConfig(max_model_calls_per_run=2)

    report = ground.run(only="打球")

    assert len(report.associated) == 2 and report.deferred == ("打球/2026-09-18",)
    assert any("budget exhausted" in note for note in report.signals)


def test_a_checkpoint_replays_without_a_second_model_call(tmp_path) -> None:
    """检查点存的是已经解析完的记录，所以重放是纯 I/O——顺带绕过情境重新编号这个坑。"""

    ground = weekly(tmp_path)
    task = next(task for task in ground.tasks() if task.kind_token == "打球" and task.day == FRIDAY)
    ground.regularity_tree.write = _explode  # type: ignore[method-assign]
    failed = asyncio.run(ground.refresher.refresh((task,), causes=CauseFacts(ground.tree())))
    assert failed.failed == ("打球/2026-09-04",)
    calls = len(ground.associator.payloads)
    del ground.regularity_tree.write

    report = asyncio.run(ground.refresher.refresh((task,), causes=CauseFacts(ground.tree())))

    assert len(ground.associator.payloads) == calls
    assert any("checkpoint replayed" in note for note in report.signals)
    assert report.associated == ("打球/2026-09-04",)


def _explode(*_args, **_kwargs):  # type: ignore[no-untyped-def]
    raise RuntimeError("disk on fire")


def test_the_same_input_failing_three_times_blocks_it_and_frees_the_quota(tmp_path) -> None:
    """一件永久失败的任务不许把那个候选的配额永久吃掉。"""

    ground = weekly(tmp_path)
    ground.associator.explode = RuntimeError("model down")

    for _ in range(3):
        report = ground.run(only="打球")

    assert report.blocked == ("打球/2026-09-04", "打球/2026-09-11")
    assert ground.refresher.progress.days_for("打球") == frozenset({FRIDAY, NEXT_FRIDAY})
    # 被挡住的不再进待办——一件永久失败的任务不会每轮都排在最前面。
    assert [task.day for task in ground.tasks() if task.kind_token == "打球"] == []


def test_a_success_wipes_the_failure_count_so_a_flaky_day_does_not_creep_toward_blocked(tmp_path) -> None:
    """记的是"同一输入连续失败几次"。不清掉的话，几周里各抖一次也会把它攒到封锁线上。"""

    ground = weekly(tmp_path)
    task = next(task for task in ground.tasks() if task.kind_token == "打球" and task.day == FRIDAY)
    ground.associator.explode = RuntimeError("flaky")
    for _ in range(2):
        asyncio.run(ground.refresher.refresh((task,), causes=CauseFacts(ground.tree())))
    ground.associator.explode = None

    asyncio.run(ground.refresher.refresh((task,), causes=CauseFacts(ground.tree())))

    assert ground.refresher.progress.failures() == {}


def test_a_deterministic_failure_blocks_on_the_first_try(tmp_path) -> None:
    """超限不是抖动，重试没有意义。"""

    ground = weekly(tmp_path)
    ground.associator.explode = AssociationLimitError("too many targets")

    report = ground.run(only="打球")

    assert report.blocked == ("打球/2026-09-04", "打球/2026-09-11")
    assert all("deterministic failure" in note or "failed" in note for note in report.signals if "blocked" in note)


def test_a_blocked_task_that_is_handed_over_again_is_refused_until_its_input_changes(tmp_path) -> None:
    """封锁按**输入指纹**记：待办不再给它，但运维正门递过来时，输入没变仍然拦，变了就放行。"""

    ground = weekly(tmp_path)
    ground.associator.explode = AssociationLimitError("too many targets")
    ground.run(only="打球")
    ground.associator.explode = None
    task = next(task for task in _all_tasks(ground) if task.day == FRIDAY and task.kind_token == "打球")

    still = asyncio.run(ground.refresher.refresh((task,), causes=CauseFacts(ground.tree())))
    assert still.blocked == ("打球/2026-09-04",)

    ground.record(FRIDAY, "喝水", 18, 30)  # 当天的流变了 → 指纹变了 → 自动解封
    again = asyncio.run(ground.refresher.refresh((task,), causes=CauseFacts(ground.tree())))
    assert again.associated == ("打球/2026-09-04",)


def _all_tasks(ground: Ground):  # type: ignore[no-untyped-def]
    from habitus.scene.backlog import backlog

    return backlog(ground.tree(), ground.regularity_tree, per_candidate=10, limit=50)


def test_two_runs_cannot_overlap(tmp_path) -> None:
    ground = weekly(tmp_path)

    async def both() -> None:
        held = ground.refresher._path_lock.acquire(ground.refresher._lock_key, ttl_seconds=60)
        with held:
            with pytest.raises(AssociationBusyError):
                await ground.refresher.refresh((), causes=CauseFacts(ground.tree()))

    asyncio.run(both())


def test_the_refresher_refuses_a_progress_root_inside_the_tree(tmp_path) -> None:
    ground = Ground(tmp_path)
    with pytest.raises(ValueError, match="must not overlap"):
        AssociationRefresher(
            behavior_tree=ground.behavior_tree,
            regularity_tree=ground.regularity_tree,
            associator=ground.associator,
            progress_root=ground.regularity_tree.root / "inside",
            lock_store=ground.lock_store,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("behavior_tree", object(), "BehaviorTree"),
        ("regularity_tree", object(), "RegularityTree"),
        ("associator", object(), "associate"),
        ("lock_store", object(), "LockStore"),
    ],
)
def test_the_refresher_checks_what_it_is_given(tmp_path, field: str, value: object, message: str) -> None:
    ground = Ground(tmp_path)
    parts = {
        "behavior_tree": ground.behavior_tree,
        "regularity_tree": ground.regularity_tree,
        "associator": ground.associator,
        "lock_store": ground.lock_store,
    }
    parts[field] = value
    with pytest.raises(TypeError, match=message):
        AssociationRefresher(progress_root=tmp_path / "progress", **parts)  # type: ignore[arg-type]


def test_a_task_key_round_trips_and_refuses_junk() -> None:
    key = TaskKey("Gym", date(2026, 9, 4))
    assert TaskKey.parse(key.identity) == TaskKey("gym", date(2026, 9, 4))
    for junk in ("no-slash", "a/b/c", "gym/not-a-date", 7):
        with pytest.raises(AssociationProgressError):
            TaskKey.parse(junk)


def test_the_day_facts_come_out_deterministic(tmp_path) -> None:
    ground = weekly(tmp_path)
    ground.gap(FRIDAY, 13, 14)

    ground.run(only="打球")

    facts = ground.associator.payloads[0].facts
    assert facts.weekday == 4 and facts.month == 9
    assert facts.observed_gaps == ((at(FRIDAY, 13), at(FRIDAY, 14)),)


# ── 指纹与重放（三方审查指出这一整块零覆盖） ─────────────────────────────────


def test_a_changed_day_stream_invalidates_the_checkpoint(tmp_path) -> None:
    """指纹存在的全部理由就是这条：输入变了，上次那份产物说的是另一件事。"""

    ground = weekly(tmp_path)
    task = next(task for task in ground.tasks() if task.kind_token == "打球" and task.day == FRIDAY)
    ground.regularity_tree.write = _explode  # type: ignore[method-assign]
    asyncio.run(ground.refresher.refresh((task,), causes=CauseFacts(ground.tree())))
    del ground.regularity_tree.write
    calls = len(ground.associator.payloads)
    ground.record(FRIDAY, "喝水", 18, 30)

    asyncio.run(ground.refresher.refresh((task,), causes=CauseFacts(ground.tree())))

    assert len(ground.associator.payloads) == calls + 1  # 重新问了一次，没有拿旧产物重放


def test_an_observation_gap_added_later_invalidates_the_checkpoint(tmp_path) -> None:
    """观测空白直接渲染进提示词。不摘进指纹的话，补一个洞之后重放会拿一份"那天一直在看"的答复。"""

    ground = weekly(tmp_path)
    task = next(task for task in ground.tasks() if task.kind_token == "打球" and task.day == FRIDAY)
    ground.regularity_tree.write = _explode  # type: ignore[method-assign]
    asyncio.run(ground.refresher.refresh((task,), causes=CauseFacts(ground.tree())))
    del ground.regularity_tree.write
    calls = len(ground.associator.payloads)
    ground.gap(FRIDAY, 13, 14)

    asyncio.run(ground.refresher.refresh((task,), causes=CauseFacts(ground.tree())))

    assert len(ground.associator.payloads) == calls + 1
    assert ground.associator.payloads[-1].facts.observed_gaps


def test_the_target_set_is_part_of_the_fingerprint(tmp_path) -> None:
    ground = weekly(tmp_path)
    task = next(task for task in ground.tasks() if task.kind_token == "打球" and task.day == FRIDAY)
    ground.regularity_tree.write = _explode  # type: ignore[method-assign]
    asyncio.run(ground.refresher.refresh((task,), causes=CauseFacts(ground.tree())))
    del ground.regularity_tree.write
    calls = len(ground.associator.payloads)
    ground.record(FRIDAY, "打球", 21, 0, kind="打球")  # 同一天多了一次目标

    asyncio.run(ground.refresher.refresh((task,), causes=CauseFacts(ground.tree())))

    assert len(ground.associator.payloads) == calls + 1


# ── 答不全不再永久循环 ───────────────────────────────────────────────────────


def _answers_only_the_first(payload):  # type: ignore[no-untyped-def]
    first = payload.targets[0]
    return AssociationAssembly(
        drafts=(AssociationDraft(occurrence_no=first, context="只说了第一次", cites=(first,), new_situation="x"),),
        unanswered=tuple(payload.targets[1:]),
    )


def test_a_day_left_open_is_its_own_outcome_and_stops_after_the_attempt_limit(tmp_path) -> None:
    """报成"已关联"的话，这一天既不 done 也不 blocked，每轮重来、每轮白烧一次调用。"""

    ground = Ground(tmp_path)
    ground.record(FRIDAY, "打球", 10, 0, kind="打球")
    ground.record(FRIDAY, "打球", 19, 0, kind="打球")
    ground.associator.answer = _answers_only_the_first

    first = ground.run(only="打球")
    assert first.open == ("打球/2026-09-04",) and first.associated == ()

    second = ground.run(only="打球")
    third = ground.run(only="打球")

    assert third.blocked == ("打球/2026-09-04",)
    assert second.model_calls == 0  # 检查点留着，重来那次连调用也省掉
    assert [task.day for task in ground.tasks() if task.kind_token == "打球"] == []


def test_an_open_day_keeps_its_checkpoint(tmp_path) -> None:
    ground = Ground(tmp_path)
    ground.record(FRIDAY, "打球", 10, 0, kind="打球")
    ground.record(FRIDAY, "打球", 19, 0, kind="打球")
    ground.associator.answer = _answers_only_the_first

    ground.run(only="打球")

    assert ground.refresher.progress.failures()


# ── 重做一天：孤儿记录要清掉 ─────────────────────────────────────────────────


def test_redoing_a_day_discards_the_records_that_are_no_longer_part_of_it(tmp_path) -> None:
    """``write`` 按地址覆写。本轮比上轮少时，多出来的会让完成标记的条数核对永远过不去。"""

    ground = Ground(tmp_path)
    ground.record(FRIDAY, "打球", 10, 0, kind="打球")
    ground.record(FRIDAY, "打球", 19, 0, kind="打球")
    ground.run(only="打球")
    assert len(ground.regularity_tree.read_day("打球", FRIDAY)) == 2

    ground.associator.answer = _answers_only_the_first
    task = AssociationTask(kind_token="打球", day=FRIDAY, slots=(SlotKey(weekday=4, slot=40),))
    report = asyncio.run(ground.refresher.refresh((task,), causes=CauseFacts(ground.tree()), force=True))

    assert len(ground.regularity_tree.read_day("打球", FRIDAY)) == 1
    assert any("records_discarded" in note for note in report.signals)


# ── 已完成的一天不再重问 ─────────────────────────────────────────────────────


def test_handing_the_same_task_over_twice_only_costs_one_call(tmp_path) -> None:
    ground = weekly(tmp_path)
    task = next(task for task in ground.tasks() if task.kind_token == "打球" and task.day == FRIDAY)

    report = asyncio.run(ground.refresher.refresh((task, task), causes=CauseFacts(ground.tree())))

    assert report.model_calls == 1
    assert report.associated == ("打球/2026-09-04",) and report.skipped == ("打球/2026-09-04",)


def test_changing_the_prompt_version_puts_finished_days_back_on_the_list(tmp_path) -> None:
    """换提示词版本要全量重做。完成标记不记版本的话，一天也不会重做。"""

    ground = weekly(tmp_path)
    ground.run(only="打球")
    assert [task.day for task in ground.tasks() if task.kind_token == "打球"] == []

    ground.associator.version = "scripted_association_v2+schema000000000000"
    ground.refresher.progress.version = ground.associator.version

    assert [task.day for task in ground.tasks() if task.kind_token == "打球"] == [FRIDAY, NEXT_FRIDAY]


def test_blocking_is_scoped_to_the_version_that_could_not_answer(tmp_path) -> None:
    ground = weekly(tmp_path)
    ground.associator.explode = AssociationLimitError("too many targets")
    ground.run(only="打球")
    assert ground.refresher.progress.days_for("打球") == frozenset({FRIDAY, NEXT_FRIDAY})

    ground.refresher.progress.version = "scripted_association_v2+schema000000000000"

    assert ground.refresher.progress.days_for("打球") == frozenset()


def test_a_kind_with_capital_letters_is_still_found_on_its_day(tmp_path) -> None:
    """键按规范身份（casefold）比，但读树要用人写法：拿 ``gym`` 去比 occurrence 上的 ``Gym`` 一条都对不上，
    这个候选就会每晚被判成"那天没发生"，永远关联不上（2026-09-16 在 54 天真实数据上实测：vLLM、Tagent 全军覆没）。"""

    ground = Ground(tmp_path)
    ground.record(FRIDAY, "去 Gym 练腿", 19, kind="Gym")
    ground.record(FRIDAY + timedelta(days=7), "去 Gym 练背", 19, kind="Gym")
    tasks = [task for task in ground.tasks() if task.kind_token == "Gym"]
    assert tasks, "the backlog hands the token out in its human form"

    report = asyncio.run(ground.refresher.refresh(tuple(tasks), causes=CauseFacts(ground.tree())))

    assert report.skipped == () and report.model_calls == len(tasks)
    assert set(report.associated) == {f"gym/{task.day.isoformat()}" for task in tasks}
    assert ground.regularity_tree.days_for("Gym", version=ground.associator.version) == {task.day for task in tasks}


def test_a_candidate_spelled_differently_is_the_same_task(tmp_path) -> None:
    """人写法不参与相等——否则失败次数永远停在 1，``clear_failure`` 永远 pop 不中。"""

    assert TaskKey("Gym", FRIDAY) == TaskKey("gym", FRIDAY)
    assert TaskKey("Gym", FRIDAY).identity == "gym/2026-09-04"


# ── 物化：编号还原与两类边 ───────────────────────────────────────────────────


def test_the_records_carry_both_edge_kinds_and_the_premise_fields(tmp_path) -> None:
    """前因落成 results_from 指向那条行为，兑现落成 needs 指向建立前提的行为，权威是字段。"""

    ground = Ground(tmp_path)
    ground.record(FRIDAY, "和朋友通电话", 18, 0, kind="通话")
    ground.record(FRIDAY, "打球", 19, 0, kind="打球")
    ground.record(NEXT_FRIDAY, "打球", 19, 0, kind="打球")

    ground.run(only="打球")

    (first,) = ground.regularity_tree.read_day("打球", FRIDAY)
    assert [link.link_type.value for link in first.links] == ["results_from"]
    assert first.left == (("为下一次打球做好了准备", "打球"),)
    (second,) = ground.regularity_tree.read_day("打球", NEXT_FRIDAY)
    assert second.consumed == ((first.occurrence_uri, "为下一次打球做好了准备"),)
    assert sorted(link.link_type.value for link in second.links) == ["needs", "results_from"]


# ── 指纹逐字段（只改一样，其余不动） ─────────────────────────────────────────


def _digest(ground: Ground, **overrides: object) -> str:
    from tests.unit.scene.association_payloads import association_input

    return ground.refresher._source_digest(association_input(**overrides))


def test_the_fingerprint_covers_each_thing_the_prompt_shows(tmp_path) -> None:
    """逐字段各改一样：漏掉哪一样，重放就会拿一份"模型没见过这个变化"的旧答复。"""

    from habitus.scene.association.model import DayFacts
    from tests.unit.scene.association_payloads import association_input, instant, occurrence

    ground = Ground(tmp_path)
    base = _digest(ground)
    rows = association_input().occurrences
    two = (*rows, occurrence(4, 21, "打球", kind="打球"))
    gap = ((instant(13), instant(14)),)

    # 目标集合：行完全一样，只是这次问的是哪几条不同
    both = _digest(ground, occurrences=two, targets=(3, 4))
    assert _digest(ground, occurrences=two, targets=(4,)) != both
    # 日型：其余一切相同，只有这一天被标成调休
    workday = _digest(ground, facts=DayFacts(weekday=4, month=9, day_note="工作日", observed_gaps=gap))
    assert _digest(ground, facts=DayFacts(weekday=4, month=9, day_note="调休", observed_gaps=gap)) != workday
    # 观测空白：其余一切相同，只是那天多了一个洞
    assert _digest(ground, facts=DayFacts(weekday=4, month=9, day_note="工作日")) != workday
    # summary：模型看得见的那句话变了
    changed = (*rows[:2], type(rows[2])(**{**vars(rows[2]), "summary": "换了一句话"}))
    assert _digest(ground, occurrences=changed) != base
    # goal
    goal = (*rows[:2], type(rows[2])(**{**vars(rows[2]), "goal": "换个目标"}))
    assert _digest(ground, occurrences=goal) != base
    # 同一份输入两次算出来的指纹必须相同
    assert _digest(ground) == base
    assert instant(19).tzinfo is not None


def test_the_fingerprint_ignores_what_grows_with_our_own_progress(tmp_path) -> None:
    """情境与未兑现前提每关联一次就变。摘进去的话，检查点这条保护永远不生效。"""

    from habitus.scene.association.model import SituationRow

    ground = Ground(tmp_path)
    base = _digest(ground)

    assert _digest(ground, situations=()) == base
    assert _digest(ground, situations=(SituationRow(no=1, text="另一种情形", days=(FRIDAY,)),)) == base
    assert _digest(ground, pending=()) == base
    assert _digest(ground, origin="第一次是同事带着去的") == base


# ── 由来只写第一条 ───────────────────────────────────────────────────────────


def test_the_origin_comes_from_the_earliest_record_not_the_last_one(tmp_path) -> None:
    ground = Ground(tmp_path)
    ground.record(FRIDAY, "打球", 10, 0, kind="打球")
    ground.record(FRIDAY, "打球", 19, 0, kind="打球")

    ground.run(only="打球")

    records = ground.regularity_tree.read_day("打球", FRIDAY)
    assert len(records) == 2
    assert Overview.read(ground.regularity_tree, "打球").origin == records[0].context


# ── 两条前提来自同一条行为时，needs 边只要一条 ───────────────────────────────


def test_two_premises_from_one_behaviour_make_one_needs_edge(tmp_path) -> None:
    """边的粒度是行为，字段才是前提：同一条行为留下的两条都被用掉时，边重复没有意义。"""

    ground = Ground(tmp_path)
    ground.record(FRIDAY, "去超市买菜", 15, 0, kind="买菜")
    ground.record(NEXT_FRIDAY, "做晚饭", 19, 0, kind="做饭")

    def leaves_two(payload):  # type: ignore[no-untyped-def]
        target = payload.targets[0]
        return AssociationAssembly(
            drafts=(
                AssociationDraft(
                    occurrence_no=target,
                    context="一句话",
                    cites=(target,),
                    new_situation="一种情形",
                    consumed=tuple(row.no for row in payload.pending),
                    left=(("家里有菜", "做饭"), ("买到了灯泡", "做饭")) if payload.kind_token == "买菜" else (),
                ),
            )
        )

    ground.associator.answer = leaves_two
    ground.run(only="买菜")
    ground.run(only="做饭")

    (dinner,) = ground.regularity_tree.read_day("做饭", NEXT_FRIDAY)
    assert len(dinner.consumed) == 2
    assert [link.link_type.value for link in dinner.links] == ["needs"]


# ── 前因候选的排序与配额 ─────────────────────────────────────────────────────


def test_causes_are_shown_in_time_order_after_the_ranking_has_picked_them(tmp_path) -> None:
    ground = Ground(tmp_path)
    ground.record(FRIDAY, "打球", 19, 0, kind="打球")
    for hour, name in ((8, "吃早饭"), (12, "看手机"), (18, "和朋友通电话")):
        ground.record(FRIDAY + timedelta(days=3), name, hour, 0, kind=name)
    ground.record(NEXT_FRIDAY, "打球", 19, 0, kind="打球")

    ground.run(only="打球")

    causes = ground.associator.payloads[-1].causes
    assert [row.started_at for row in causes] == sorted(row.started_at for row in causes)


def test_one_frequent_action_cannot_eat_the_whole_cause_budget(tmp_path) -> None:
    """ "每次打球前都在看手机"这种又高频又高转移的动作会把所有格子吃掉，体检一条都进不来。"""

    ground = Ground(tmp_path)
    ground.record(FRIDAY, "打球", 19, 0, kind="打球")
    middle = FRIDAY + timedelta(days=3)
    for minute in range(0, 60, 3):
        ground.record(middle, "看手机", 12, minute, kind="看手机")
    ground.record(middle, "去体检", 9, 0, kind="体检")
    ground.record(NEXT_FRIDAY, "打球", 19, 0, kind="打球")

    ground.run(only="打球")

    actions = [row.name for row in ground.associator.payloads[-1].causes]
    assert actions.count("看手机") <= 2
    assert "去体检" in actions


def test_a_candidates_tasks_must_arrive_in_ascending_day_order(tmp_path) -> None:
    """乱序会让情境的演化顺序错乱，而"上一次"会指到未来。重放脚本够得着这条。"""

    ground = weekly(tmp_path)
    tasks = tuple(task for task in ground.tasks() if task.kind_token == "打球")

    with pytest.raises(ValueError, match="ascending day order"):
        asyncio.run(ground.refresher.refresh(tuple(reversed(tasks)), causes=CauseFacts(ground.tree())))


def test_the_progress_root_cannot_sit_inside_the_tree_either_way(tmp_path) -> None:
    ground = Ground(tmp_path)
    with pytest.raises(ValueError, match="must not overlap"):
        AssociationRefresher(
            behavior_tree=ground.behavior_tree,
            regularity_tree=ground.regularity_tree,
            associator=ground.associator,
            progress_root=ground.regularity_tree.root.parent,
            lock_store=ground.lock_store,
        )


def test_a_candidate_that_finishes_earlier_runs_first_so_its_premises_are_visible(tmp_path) -> None:
    """按候选最早的那一格排的话，"07:00 做早饭、20:00 做晚饭"会整批排在"09:00 买菜"前面，
    于是买菜留下的前提，当天晚上那次做饭一条都看不见。"""

    ground = Ground(tmp_path)
    ground.record(FRIDAY, "做早饭", 7, 0, kind="做饭")
    ground.record(FRIDAY, "去超市买菜", 9, 0, kind="买菜")
    ground.record(FRIDAY, "做晚饭", 20, 0, kind="做饭")

    def leaves_for_cooking(payload):  # type: ignore[no-untyped-def]
        target = payload.targets[0]
        return AssociationAssembly(
            drafts=(
                AssociationDraft(
                    occurrence_no=target,
                    context="一句话",
                    cites=(target,),
                    new_situation="一种情形",
                    left=(("家里有今晚要用的食材", "做饭"),) if payload.kind_token == "买菜" else (),
                ),
            )
            + tuple(
                AssociationDraft(occurrence_no=other, context="一句话", cites=(other,), new_situation="一种情形")
                for other in payload.targets[1:]
            )
        )

    ground.associator.answer = leaves_for_cooking
    ground.run()

    assert [payload.kind_token for payload in ground.associator.payloads] == ["买菜", "做饭"]
    assert [row.text for row in ground.associator.payloads[-1].pending] == ["家里有今晚要用的食材"]


# ── 重放的正门 ───────────────────────────────────────────────────────────────


def test_reset_puts_a_candidate_back_to_never_associated(tmp_path) -> None:
    """换版本按版本作废只解决一半：待办会回来，但 L1 只增不减，旧版本的情形会与新的混着累积。"""

    ground = weekly(tmp_path)
    ground.run(only="打球")
    assert ground.regularity_tree.days_for("打球") == frozenset({FRIDAY, NEXT_FRIDAY})
    assert Overview.read(ground.regularity_tree, "打球").situations

    assert ground.refresher.reset("打球") == ("打球",)

    assert ground.regularity_tree.days_for("打球") == frozenset()
    assert Overview.read(ground.regularity_tree, "打球").situations == ()
    assert not ground.regularity_tree.layer_exists("打球", RegularityLevel.ABSTRACT)
    assert [task.day for task in ground.tasks() if task.kind_token == "打球"] == [FRIDAY, NEXT_FRIDAY]
    # 记录本身不删：重做按地址覆写，多出来的由 retain_only 清。
    assert len(ground.regularity_tree.read_day("打球", FRIDAY)) == 1


def test_reset_lifts_a_block_and_clears_the_checkpoint(tmp_path) -> None:
    ground = weekly(tmp_path)
    ground.associator.explode = AssociationLimitError("too many targets")
    ground.run(only="打球")
    assert ground.refresher.progress.days_for("打球")

    ground.refresher.reset("打球")

    assert ground.refresher.progress.days_for("打球") == frozenset()
    assert ground.refresher.progress.failures() == {}


def test_reset_without_a_candidate_clears_every_one(tmp_path) -> None:
    ground = weekly(tmp_path)
    ground.run()

    cleared = ground.refresher.reset()

    assert "打球" in cleared and "通话" in cleared
    assert ground.regularity_tree.days_for("打球") == frozenset()


def test_a_replay_does_not_stack_the_old_situations_on_top_of_the_new_ones(tmp_path) -> None:
    ground = weekly(tmp_path)
    ground.run(only="打球")
    ground.refresher.reset("打球")
    ground.associator.version = "scripted_association_v2+schema000000000000"
    ground.refresher.progress.version = ground.associator.version

    ground.run(only="打球")

    assert len(Overview.read(ground.regularity_tree, "打球").situations) == 1
