"""关系行退休：独立验证补测（实时化阶段 3 第 4 步；共享说明 3.3、3.4）。

基线没写的用例与对抗用例：词表拆改打回的行同一代不判退休、一条键一代只迁移一次、``RelationConfig`` 两个新字段的取值校验、
退休行文件被手改、退休行之后概念停用 / 依赖的条件不在 / 又被打回、日历情境与非日历情境当前因、空覆盖与 ``through`` 早于最后一次、
同一天既有被提醒的又有自然的记号。造数沿用 ``retirement_fixtures``（H = 8、最少 2 个间隔）。
"""

from __future__ import annotations

import json
import math
from collections import Counter
from datetime import date, timedelta
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
    Segment,
    Status,
    Transition,
    build_timelines,
    fold,
)
from habitus.scene.relations.config import RelationConfigError
from habitus.scene.relations.document import RelationRecordError, decode_night, encode_night
from habitus.scene.relations.retirement import Watching, absence, visits
from habitus.scene.relations.state import revert_touched
from habitus.scene.relations.store import RelationStore
from habitus.series import EventSeries
from habitus.temporal.absence import Visit
from habitus.temporal.coverage import Coverage
from tests.unit.scene import retirement_fixtures as fx
from tests.unit.scene.concept_fixtures import base, situation
from tests.unit.temporal.fixtures import at, record, series

C = fx.chain_key()


def config() -> RelationConfig:
    return RelationConfig(
        slot_minutes=fx.SLOT_MINUTES, transition_window_slots=3, absence_typical_multiple=8.0, absence_min_gaps=2
    )


def fold_at(
    previous: LaneState | None,
    *,
    cause_days: range | list[int],
    effect_days: range | list[int],
    through: int,
    concept_set: ConceptSet | None = None,
    coverage: Coverage | None = None,
    holiday_days: range | None = None,
    watch_through: date | None = None,
) -> tuple[LaneState, tuple[Transition, ...], EventSeries]:
    events, line, concepts = fx.build(
        cause_days=cause_days,
        effect_days=effect_days,
        through=through,
        concept_set=concept_set,
        holiday_days=holiday_days,
    )
    tests = LaneTests(line, concepts, config(), events.cutoff)
    watching = Watching(
        coverage=coverage if coverage is not None else fx.coverage(0, through),
        day_type=fx.every_day,
        through=watch_through if watch_through is not None else events.last_day,
    )
    state, transitions = fold(previous, tests, watching=watching)
    return state, transitions, events


def row(state: LaneState, relation: RelationKey) -> Relation | None:
    return next((item for item in state.relations if item.key == relation), None)


def once_per_key(transitions: tuple[Transition, ...]) -> None:
    """一条键一代只迁移一次。"""

    repeated = [key for key, count in Counter(item.key for item in transitions).items() if count > 1]
    assert repeated == []


# ── 被 revert_touched 打回的行：这一代不判，下一代判 ───────────────────────────


@pytest.mark.parametrize("stored", ["candidate", "established"])
def test_a_row_sent_back_by_the_vocabulary_is_not_retired_in_the_same_generation(stored: str) -> None:
    through = 28  # 前因第 0–20 天来、之后 8 天没来：按 3.5 是退休的那一侧
    relation = fx.candidate(C, discovered=18) if stored == "candidate" else fx.established(C, on=18)
    previous = fx.previous_state(through - 1, relation)
    cutoff = fx.day(through + 1)
    sent_back, reverted = revert_touched(previous, frozenset({fx.CAUSE}), cutoff)
    assert [(item.key, item.after) for item in reverted] == [(C, Status.CANDIDATE)]

    state, transitions, events = fold_at(
        sent_back, cause_days=range(21), effect_days=range(through + 1), through=through
    )
    assert events.cutoff == cutoff
    kept = row(state, C)
    assert kept is not None and kept.status is Status.CANDIDATE and kept.retired_on is None
    assert [item for item in transitions if item.key == C] == []
    once_per_key((*reverted, *transitions))

    # 下一代：前因仍没来（空白 9 天），这一回退
    later, again, later_events = fold_at(
        state, cause_days=range(21), effect_days=range(through + 2), through=through + 1
    )
    retired = row(later, C)
    assert retired == Relation(key=C, status=Status.RETIRED, upward=True, retired_on=later_events.cutoff)
    assert [(item.before, item.after) for item in again if item.key == C] == [(Status.CANDIDATE, Status.RETIRED)]
    once_per_key(again)


def test_a_carried_row_retiring_moves_once_per_generation() -> None:
    state, transitions, _events = fold_at(
        fx.previous_state(27, fx.candidate(C, discovered=18)), cause_days=range(21), effect_days=range(29), through=28
    )
    assert row(state, C) is not None and row(state, C).status is Status.RETIRED  # type: ignore[union-attr]
    once_per_key(transitions)


# ── RelationConfig 两个新字段 ────────────────────────────────────────────────


@pytest.mark.parametrize("bad", [0.0, -1.0, math.nan, math.inf, -math.inf])
def test_the_absence_multiple_must_be_positive_and_finite(bad: float) -> None:
    with pytest.raises(RelationConfigError, match="absence_typical_multiple"):
        RelationConfig(slot_minutes=15, transition_window_slots=3, absence_typical_multiple=bad, absence_min_gaps=2)


@pytest.mark.parametrize("bad", [0, -1])
def test_the_minimum_gap_count_must_be_at_least_one(bad: int) -> None:
    with pytest.raises(RelationConfigError, match="absence_min_gaps"):
        RelationConfig(slot_minutes=15, transition_window_slots=3, absence_typical_multiple=8.0, absence_min_gaps=bad)


def test_one_gap_is_an_allowed_minimum() -> None:
    built = RelationConfig(slot_minutes=15, transition_window_slots=3, absence_typical_multiple=0.5, absence_min_gaps=1)
    assert (built.absence_typical_multiple, built.absence_min_gaps) == (0.5, 1)


# ── 退休行文件被手改 ─────────────────────────────────────────────────────────


def _retired_file() -> tuple[str, LaneState]:
    state = fx.previous_state(20, Relation(key=C, status=Status.RETIRED, upward=True, retired_on=fx.day(15)))
    return encode_night(state), state


def _rewrite(text: str, change: dict[str, object] | None = None, drop: str | None = None) -> str:
    payload = json.loads(text)
    body = payload["relations"][0]
    if drop is not None:
        del body[drop]
    body.update(change or {})
    payload["relations"][0] = dict(sorted(body.items()))
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def test_a_retired_row_file_round_trips() -> None:
    text, state = _retired_file()
    assert decode_night(text, lane=fx.LANE, night=state.night) == state
    assert set(json.loads(text)["relations"][0]) == {"key", "status", "upward", "retired_on"}


@pytest.mark.parametrize(
    ("change", "drop"),
    [
        ({"tonight": None}, None),
        ({"forward": None}, None),
        ({"discovered_on": "2026-06-10"}, None),
        ({"misses": 0}, None),
        ({"retired_on": None}, None),
        (None, "retired_on"),
    ],
    ids=["tonight", "forward", "discovered_on", "misses", "retired_on-null", "retired_on-missing"],
)
def test_a_hand_edited_retired_row_is_refused(change: dict[str, object] | None, drop: str | None) -> None:
    text, state = _retired_file()
    with pytest.raises(RelationRecordError) as caught:
        decode_night(_rewrite(text, change, drop), lane=fx.LANE, night=state.night)
    assert "relation night is not valid" in str(caught.value)


def test_a_non_retired_row_carrying_a_retirement_day_is_refused() -> None:
    state = fx.previous_state(20, fx.candidate(C, discovered=10))
    text = encode_night(state)
    assert "retired_on" not in text
    with pytest.raises(RelationRecordError) as caught:
        decode_night(_rewrite(text, {"retired_on": "2026-06-15"}), lane=fx.LANE, night=state.night)
    assert "only a retired relation records a retirement day" in str(caught.value)


# ── 退休行之后的清理 ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("cause_days", [range(21), []], ids=["measured", "no-reading"])
def test_a_retired_row_goes_when_its_concept_is_retired_later(cause_days: range | list[int]) -> None:
    previous = fx.previous_state(27, Relation(key=C, status=Status.RETIRED, upward=True, retired_on=fx.day(20)))
    state, transitions, _events = fold_at(
        previous,
        cause_days=cause_days,
        effect_days=range(29),
        through=28,
        concept_set=fx.concepts(cause_retired=True),
    )
    assert row(state, C) is None
    assert all(item.key != C for item in transitions)
    assert all(item.key != C for item in state.relations)


def test_a_retired_row_goes_when_the_condition_it_depends_on_is_gone() -> None:
    conditioned = RelationKey(fx.LANE, fx.CAUSE, fx.EFFECT, Segment.CHAIN, "timeline:不存在的条件")
    previous = fx.previous_state(
        27, Relation(key=conditioned, status=Status.RETIRED, upward=True, retired_on=fx.day(20))
    )
    events, line, concepts = fx.build(cause_days=range(29), effect_days=range(29), through=28)
    tests = LaneTests(line, concepts, config(), events.cutoff)
    assert not tests.still_defined(conditioned)
    state, transitions = fold(previous, tests, watching=Watching(fx.coverage(0, 28), fx.every_day, events.last_day))
    assert row(state, conditioned) is None
    assert all(item.key != conditioned for item in transitions)


def test_a_retired_row_is_not_sent_back_by_the_vocabulary() -> None:
    retired = Relation(key=C, status=Status.RETIRED, upward=True, retired_on=fx.day(20))
    previous = fx.previous_state(27, retired)
    after, moved = revert_touched(previous, frozenset({fx.CAUSE, fx.EFFECT}), fx.day(29))
    assert after == previous and moved == ()


def test_rewriting_a_generation_with_a_sent_back_row_keeps_the_log_single(tmp_path: Path) -> None:
    """同一代重做（含打回 + 别的行退休）：状态文件相同、迁移日志不重复追加。"""

    other = fx.chain_key(fx.EFFECT, fx.CAUSE)

    def generation() -> tuple[LaneState, tuple[Transition, ...]]:
        previous = fx.previous_state(27, fx.candidate(C, discovered=18), fx.candidate(other, discovered=18))
        sent_back, reverted = revert_touched(previous, frozenset({"不相干"}), fx.day(29))
        state, transitions, _ = fold_at(sent_back, cause_days=range(21), effect_days=range(29), through=28)
        return state, (*reverted, *transitions)

    state, transitions = generation()
    assert {item.key for item in transitions if item.after is Status.RETIRED} == {C, other}
    store = RelationStore(tmp_path / "scene")
    store.write(state, transitions)
    lane_dir = tmp_path / "scene" / "relations" / fx.LANE
    written = {path.name: path.read_bytes() for path in lane_dir.iterdir()}
    store.write(*generation())
    assert {path.name: path.read_bytes() for path in lane_dir.iterdir()} == written
    assert Counter(item.key for item in store.transitions(fx.LANE)) == Counter({C: 1, other: 1})


# ── 两端 ─────────────────────────────────────────────────────────────────────


def test_a_calendar_antecedent_retires_on_its_consequent() -> None:
    """前因「节假日」天天成立；后果第 0–20 天来、之后 8 天没来：后果一端久没来 → 退。"""

    relation = fx.holiday_key()
    previous = fx.previous_state(27, fx.candidate(relation, discovered=24))
    state, transitions, events = fold_at(
        previous, cause_days=range(29), effect_days=range(21), through=28, holiday_days=range(29)
    )
    assert row(state, relation) == Relation(key=relation, status=Status.RETIRED, upward=True, retired_on=events.cutoff)
    (moved,) = [item for item in transitions if item.key == relation]
    assert "后果" in moved.reason


def test_a_non_calendar_situation_antecedent_retires_when_it_stops_holding() -> None:
    """非日历情境当前因：第 0–20 天成立、之后 8 天不成立（人在场，后果天天来）→ 前因久没来，退。"""

    name = "出差中"
    concept_set = ConceptSet((base(fx.CAUSE_TITLE), base(fx.EFFECT_TITLE), situation(name, "人在常住地之外过夜")))
    events, line, concepts = fx.build(cause_days=range(29), effect_days=range(29), through=28, concept_set=concept_set)
    line.situations[name] = {fx.day(index): index <= 20 for index in range(29)}
    relation = RelationKey(fx.LANE, name, fx.EFFECT, Segment.SITUATION)
    tests = LaneTests(line, concepts, config(), events.cutoff)
    previous = fx.previous_state(27, fx.candidate(relation, discovered=24))
    state, transitions = fold(previous, tests, watching=Watching(fx.coverage(0, 28), fx.every_day, events.last_day))
    assert row(state, relation) == Relation(key=relation, status=Status.RETIRED, upward=True, retired_on=events.cutoff)
    (moved,) = [item for item in transitions if item.key == relation]
    assert "前因" in moved.reason


# ── 退化输入 ─────────────────────────────────────────────────────────────────


def test_an_empty_coverage_neither_crashes_nor_retires() -> None:
    state, transitions, _events = fold_at(
        fx.previous_state(27, fx.candidate(C, discovered=18)),
        cause_days=range(21),
        effect_days=range(29),
        through=28,
        coverage=Coverage(slot_minutes=fx.SLOT_MINUTES, days={}),
    )
    assert row(state, C) is not None and row(state, C).status is Status.CANDIDATE  # type: ignore[union-attr]
    assert [item for item in transitions if item.key == C] == []


def test_a_watch_end_before_the_last_visit_neither_crashes_nor_retires() -> None:
    state, transitions, _events = fold_at(
        fx.previous_state(27, fx.candidate(C, discovered=18)),
        cause_days=range(21),
        effect_days=range(29),
        through=28,
        watch_through=fx.day(10),
    )
    assert row(state, C) is not None and row(state, C).status is Status.CANDIDATE  # type: ignore[union-attr]
    assert [item for item in transitions if item.key == C] == []


# ── 被提醒的记号 ─────────────────────────────────────────────────────────────


def _line_with(marks: list[tuple[int, bool]], through: int) -> LaneTimeline:
    events = series(
        [
            record(at(fx.day(index), 10 + number), fx.CAUSE, reminded=flag, number=number)
            for number, (index, flag) in enumerate(marks)
        ],
        cutoff=fx.day(through) + timedelta(days=1),
    )
    return build_timelines(events, fx.concepts(), {}, config())[fx.LANE]


def test_a_day_with_a_natural_mark_is_not_prompted_and_an_all_reminded_day_is() -> None:
    line = _line_with([(0, False), (1, True), (1, False), (2, True), (2, True)], through=3)
    assert visits(line, fx.CAUSE) == (
        Visit(fx.day(0), prompted=False),
        Visit(fx.day(1), prompted=False),
        Visit(fx.day(2), prompted=True),
    )


@pytest.mark.parametrize(("natural_too", "retires"), [(False, True), (True, False)])
def test_a_gap_ending_on_a_reminded_only_day_is_censored(natural_too: bool, retires: bool) -> None:
    """第 0–4、15–20 天来，第 15 天只有被提醒之后才做（那段 11 天的间隔截尾，不进"历来最长"）；最后一次第 20 天、空白 8 天。
    截尾时历来最长 1、典型 1 → 8 天久没来；第 15 天另有一次自然的 → 不截尾，历来最长 11 → 不算。"""

    marks = [(index, False) for index in (*range(5), *range(16, 21))] + [(15, True)]
    if natural_too:
        marks.append((15, False))
    line = _line_with(marks, through=28)
    watching = Watching(fx.coverage(0, 28), fx.every_day, fx.day(28))
    found = absence(visits(line, fx.CAUSE), watching, config())
    assert (found is not None) is retires
