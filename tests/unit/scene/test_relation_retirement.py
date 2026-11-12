"""关系行退休（实时化阶段 3；方案 ``19`` §8，用户 11-10「2 同意」、11-12「全部按建议」）。

规则（共享说明 3.3）：从上一代带进来、这一代没有迁移、仍是候选 / 失效 / 前向未复现的行，前因或后果任一端"久没来"就退休
（``temporal.absence.long_absence``：这次空白比历来没截尾的间隔都长，且不短于 H 个典型间隔；只数在看的日子、只数来过的日型；
样本不够不判）。日历情境当前因只判后果；成立的不退。退休行只留键与退休那一代，以后只用退休之后的数据重新发现。

这里一律 H = 8、最少 2 个间隔。造数见 ``retirement_fixtures``（前因 10:00、后果 16:00，链上从不相接，带着的候选停在"接着攒"）；
每条用例的造数前提（旧代码下这一行不迁移、两端手算的久没来判定）在 c55f7da 上核过。
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date
from pathlib import Path

import pytest

from habitus.scene.concepts import ConceptSet
from habitus.scene.relations import (
    LaneState,
    LaneTests,
    LaneTimeline,
    Relation,
    RelationConfig,
    RelationKey,
    Status,
    Transition,
    build_timelines,
    fold,
)
from habitus.scene.relations.longspan import MEASURE_VERSIONS
from habitus.scene.relations.retirement import Watching
from habitus.scene.relations.store import RelationStore
from habitus.series import EventSeries
from habitus.temporal.coverage import Coverage, coverage_of
from tests.unit.scene import retirement_fixtures as fx
from tests.unit.scene.relation_fixtures import CODE, CONCEPTS, DISCUSS, key, planted, workdays

C = fx.chain_key()
PLANTED = key(DISCUSS, CODE)


def retirement_config() -> RelationConfig:
    """H = 8（遗忘半衰期）、最少 2 个间隔（``foresight.strength.MIN_HISTORY``）。"""

    return RelationConfig(
        slot_minutes=fx.SLOT_MINUTES, transition_window_slots=3, absence_typical_multiple=8.0, absence_min_gaps=2
    )


def fold_with(
    previous: LaneState | None,
    events: EventSeries,
    line: LaneTimeline,
    concepts: ConceptSet,
    coverage: Coverage,
    day_type: Callable[[date], str | None] = fx.every_day,
) -> tuple[LaneState, tuple[Transition, ...]]:
    tests = LaneTests(line, concepts, retirement_config(), events.cutoff)
    watching = Watching(coverage=coverage, day_type=day_type, through=events.last_day)
    return fold(previous, tests, watching=watching)


def row(state: LaneState, relation: RelationKey) -> Relation | None:
    return next((item for item in state.relations if item.key == relation), None)


def moves(transitions: tuple[Transition, ...], relation: RelationKey) -> list[Transition]:
    return [item for item in transitions if item.key == relation]


def absent_cause(gap: int, previous: Relation) -> tuple[LaneState, tuple[Transition, ...], EventSeries]:
    """前因连续 21 天每天来、然后停 ``gap`` 天（覆盖全满、日型都一样），后果天天来。"""

    through = 20 + gap
    events, line, concepts = fx.build(cause_days=range(21), effect_days=range(through + 1), through=through)
    state, transitions = fold_with(
        fx.previous_state(through - 1, previous), events, line, concepts, fx.coverage(0, through)
    )
    return state, transitions, events


# ── 谁退、什么时候退 ───────────────────────────────────────────────────────────


def test_a_candidate_retires_once_its_antecedent_has_been_away_eight_typical_gaps() -> None:
    state, transitions, events = absent_cause(8, fx.candidate(C, discovered=18))
    retired = row(state, C)
    # 退休行只留键、状态、方向与退休那一代；读数一概不存
    assert retired == Relation(key=C, status=Status.RETIRED, upward=True, retired_on=events.cutoff)
    (moved,) = moves(transitions, C)
    assert (moved.before, moved.after, moved.night) == (Status.CANDIDATE, Status.RETIRED, events.cutoff)
    assert "前因" in moved.reason
    assert C not in state.carried() and retired in state.of(Status.RETIRED)

    # 下一代：前因还是没来。退休行不再带着读数、不再迁移，退休那一代不变
    through = 29
    later, line, concepts = fx.build(cause_days=range(21), effect_days=range(through + 1), through=through)
    following, again = fold_with(state, later, line, concepts, fx.coverage(0, through))
    kept = row(following, C)
    assert kept is not None and kept.status is Status.RETIRED and kept.retired_on == events.cutoff
    assert kept.tonight is None and kept.forward is None and kept.maintenance is None
    assert C not in following.carried()
    assert moves(again, C) == []


def test_a_candidate_whose_antecedent_has_been_away_only_seven_typical_gaps_stays() -> None:
    state, transitions, _events = absent_cause(7, fx.candidate(C, discovered=18))
    kept = row(state, C)
    assert kept is not None and kept.status is Status.CANDIDATE and kept.retired_on is None
    assert moves(transitions, C) == []


def test_a_candidate_retires_when_its_consequent_is_the_end_that_stopped() -> None:
    through = 28
    events, line, concepts = fx.build(cause_days=range(through + 1), effect_days=range(21), through=through)
    state, transitions = fold_with(
        fx.previous_state(through - 1, fx.candidate(C, discovered=20)), events, line, concepts, fx.coverage(0, through)
    )
    assert row(state, C) == Relation(key=C, status=Status.RETIRED, upward=True, retired_on=events.cutoff)
    (moved,) = moves(transitions, C)
    assert (moved.before, moved.after) == (Status.CANDIDATE, Status.RETIRED)
    assert "后果" in moved.reason


@pytest.mark.parametrize(
    ("previous", "status"),
    [(fx.rejected(C, on=10), Status.REJECTED), (fx.expired(C, on=12), Status.EXPIRED)],
    ids=["rejected", "expired"],
)
def test_dormant_rows_retire_too(previous: Relation, status: Status) -> None:
    state, transitions, events = absent_cause(8, previous)
    assert row(state, C) == Relation(key=C, status=Status.RETIRED, upward=True, retired_on=events.cutoff)
    (moved,) = moves(transitions, C)
    assert (moved.before, moved.after) == (status, Status.RETIRED)


def test_an_established_relation_does_not_retire_and_keeps_its_maintenance_check() -> None:
    state, transitions, _events = absent_cause(8, fx.established(C, on=18))
    kept = row(state, C)
    assert kept is not None and kept.status is Status.ESTABLISHED and kept.retired_on is None
    assert kept.maintenance is not None  # 维持检验照常量了这一段
    assert moves(transitions, C) == []


def test_a_row_that_moved_this_generation_is_judged_next_generation_only() -> None:
    """这一代刚从前向未复现回到候选（有迁移）的行，即使后果已经久没来，也到下一代才判。"""

    days = workdays(60)
    quiet = frozenset(days[-9:])  # 最后 9 个工作日不再修改代码
    whole = planted(60, linked=(30, 51))
    events = EventSeries(
        cutoff=whole.cutoff,
        records=tuple(item for item in whole.records if not (item.kind_token == CODE and item.day in quiet)),
    )
    line = build_timelines(events, CONCEPTS, {}, retirement_config())["session"]
    previous = LaneState(
        lane="session",
        night=events.last_day,
        family=0,
        relations=(Relation(key=PLANTED, status=Status.REJECTED, upward=True, rejected_on=days[29]),),
        measure_versions=MEASURE_VERSIONS,
    )
    state, transitions = fold_with(
        previous, events, line, CONCEPTS, coverage_of(events, "session", slot_minutes=fx.SLOT_MINUTES), _weekday_type
    )
    kept = row(state, PLANTED)
    assert kept is not None and kept.status is Status.CANDIDATE
    assert [(item.before, item.after) for item in moves(transitions, PLANTED)] == [(Status.REJECTED, Status.CANDIDATE)]


# ── 什么时候不判 ─────────────────────────────────────────────────────────────


def test_a_calendar_situation_antecedent_is_not_judged_only_its_consequent() -> None:
    """前因「节假日」前 21 天天天成立、之后 8 天都不成立（若判前因，它久没来）；后果天天在：不退。"""

    through = 28
    events, line, concepts = fx.build(
        cause_days=range(through + 1), effect_days=range(through + 1), through=through, holiday_days=range(21)
    )
    relation = fx.holiday_key()
    state, transitions = fold_with(
        fx.previous_state(through - 1, fx.candidate(relation, discovered=24)),
        events,
        line,
        concepts,
        fx.coverage(0, through),
    )
    kept = row(state, relation)
    assert kept is not None and kept.status is Status.CANDIDATE
    assert moves(transitions, relation) == []


def test_one_gap_is_not_enough_history_to_judge() -> None:
    """前因只来过 2 天（1 个间隔），之后 28 天没来：样本不够，不判。"""

    through = 29
    events, line, concepts = fx.build(cause_days=[0, 1], effect_days=range(through + 1), through=through)
    state, transitions = fold_with(
        fx.previous_state(through - 1, fx.candidate(C, discovered=1)), events, line, concepts, fx.coverage(0, through)
    )
    kept = row(state, C)
    assert kept is not None and kept.status is Status.CANDIDATE
    assert moves(transitions, C) == []


def test_days_nobody_was_watching_are_not_absence() -> None:
    """前因 21 天每天来，之后 14 天整天「未观测」（覆盖 0）：空白有效长度 0，不退。同一份数据按每天都在看算会退。"""

    through = 34
    events, line, concepts = fx.build(cause_days=range(21), effect_days=range(through + 1), through=through)
    state, transitions = fold_with(
        fx.previous_state(through - 1, fx.candidate(C, discovered=18)),
        events,
        line,
        concepts,
        fx.coverage(0, through, unwatched=range(21, through + 1)),
    )
    kept = row(state, C)
    assert kept is not None and kept.status is Status.CANDIDATE
    assert moves(transitions, C) == []


def test_only_the_day_types_it_came_on_count() -> None:
    """前因只在工作日来过（21 天），之后 10 天是节假日、它没来：只数工作日，空白 0，不退。不分日型数会是 10 天、会退。"""

    through = 30
    events, line, concepts = fx.build(cause_days=range(21), effect_days=range(through + 1), through=through)
    state, transitions = fold_with(
        fx.previous_state(through - 1, fx.candidate(C, discovered=18)),
        events,
        line,
        concepts,
        fx.coverage(0, through),
        fx.holidays_from(21),
    )
    kept = row(state, C)
    assert kept is not None and kept.status is Status.CANDIDATE
    assert moves(transitions, C) == []


# ── 退休之后 ─────────────────────────────────────────────────────────────────


def _planted_after_retirement(linked: tuple[int, int], retired_index: int) -> tuple[LaneState, EventSeries, date]:
    events = planted(60, linked=linked)
    retired_on = workdays(60)[retired_index]
    previous = LaneState(
        lane="session",
        night=events.last_day,
        family=0,
        relations=(Relation(key=PLANTED, status=Status.RETIRED, upward=True, retired_on=retired_on),),
        measure_versions=MEASURE_VERSIONS,
    )
    return previous, events, retired_on


def _fold_planted(previous: LaneState | None, events: EventSeries) -> tuple[LaneState, tuple[Transition, ...]]:
    line = build_timelines(events, CONCEPTS, {}, retirement_config())["session"]
    return fold_with(
        previous, events, line, CONCEPTS, coverage_of(events, "session", slot_minutes=fx.SLOT_MINUTES), _weekday_type
    )


def _weekday_type(moment: date) -> str | None:
    return fx.WORKDAY if moment.weekday() < 5 else "周末"


def test_a_retired_row_is_only_rediscovered_on_data_after_its_retirement() -> None:
    """关系只埋在前 25 个工作日：全部数据上它过第 1 道；退休那一代（第 31 个工作日）之后的数据上不过 → 仍是退休，不存读数。"""

    previous, events, retired_on = _planted_after_retirement((0, 25), 30)
    whole, _moved = _fold_planted(None, events)
    first = row(whole, PLANTED)
    assert first is not None and first.status is Status.CANDIDATE  # 用全部数据会过
    state, transitions = _fold_planted(previous, events)
    assert row(state, PLANTED) == Relation(key=PLANTED, status=Status.RETIRED, upward=True, retired_on=retired_on)
    assert moves(transitions, PLANTED) == []


def test_a_retired_row_comes_back_as_a_candidate_when_new_data_is_significant() -> None:
    previous, events, _retired_on = _planted_after_retirement((30, 60), 29)
    state, transitions = _fold_planted(previous, events)
    back = row(state, PLANTED)
    assert back is not None and back.status is Status.CANDIDATE
    assert back.discovered_on == events.cutoff and back.retired_on is None
    (moved,) = moves(transitions, PLANTED)
    assert (moved.before, moved.after) == (Status.RETIRED, Status.CANDIDATE)


def test_a_retired_row_is_dropped_when_its_concept_is_retired_by_the_vocabulary() -> None:
    through = 28
    retired_on = fx.day(20)
    events, line, concepts = fx.build(
        cause_days=range(21),
        effect_days=range(through + 1),
        through=through,
        concept_set=fx.concepts(cause_retired=True),
    )
    previous = fx.previous_state(
        through - 1, Relation(key=C, status=Status.RETIRED, upward=True, retired_on=retired_on)
    )
    state, _transitions = fold_with(previous, events, line, concepts, fx.coverage(0, through))
    assert row(state, C) is None


def test_a_retired_row_without_a_reading_is_kept_as_it_was() -> None:
    """前因这一段一次都没来（没读数）、概念还在：退休行原样留着。"""

    through = 28
    events, line, concepts = fx.build(cause_days=[], effect_days=range(through + 1), through=through)
    kept = Relation(key=C, status=Status.RETIRED, upward=True, retired_on=fx.day(10))
    state, transitions = fold_with(
        fx.previous_state(through - 1, kept), events, line, concepts, fx.coverage(0, through)
    )
    assert row(state, C) == kept
    assert moves(transitions, C) == []


# ── 重做 ─────────────────────────────────────────────────────────────────────


def test_folding_the_same_input_twice_gives_the_same_generation() -> None:
    first = absent_cause(8, fx.candidate(C, discovered=18))
    second = absent_cause(8, fx.candidate(C, discovered=18))
    assert first[0] == second[0] and first[1] == second[1]
    assert any(item.after is Status.RETIRED for item in first[1])


def test_rewriting_a_generation_with_a_retirement_does_not_duplicate_its_transition(tmp_path: Path) -> None:
    """中途被杀、同一代重做：状态文件相同，迁移日志不重复追加（``scene/relations/store.py`` 同一代重跑的规则）。"""

    state, transitions, _events = absent_cause(8, fx.candidate(C, discovered=18))
    store = RelationStore(tmp_path / "scene")
    store.write(state, transitions)
    night_file = tmp_path / "scene" / "relations" / fx.LANE / f"{state.night.isoformat()}.json"
    log_file = tmp_path / "scene" / "relations" / fx.LANE / "transitions.jsonl"
    written, logged = night_file.read_bytes(), log_file.read_bytes()
    again, again_moved, _ = absent_cause(8, fx.candidate(C, discovered=18))
    store.write(again, again_moved)
    assert night_file.read_bytes() == written and log_file.read_bytes() == logged
    assert store.read(fx.LANE, state.night) == state
    assert [item.after for item in store.transitions(fx.LANE) if item.key == C] == [Status.RETIRED]


def test_the_retirement_night_is_the_cutoff_of_the_generation() -> None:
    """``retired_on`` 是这一代的截止日（迁移的 ``night`` 同一天），不是前因最后一次来的那天。"""

    state, _transitions, events = absent_cause(9, fx.candidate(C, discovered=18))
    retired = row(state, C)
    assert retired is not None and retired.retired_on == events.cutoff
    assert events.cutoff == fx.day(30)  # 前因最后一次是第 20 天
