"""段的量法版本：判据改了版本的那一段（带条件的同一条链单列 ``chain|moderated``），带着的行清掉；其余段原样（26 号清单 3.5）。"""

from __future__ import annotations

from datetime import date, timedelta

from habitus.scene.relations import LaneState, LaneTests, RelationKey, Status, build_timelines, fold
from habitus.scene.relations.conditions import Condition, Template
from habitus.scene.relations.longspan import CHAIN_MODERATED, MEASURE_VERSIONS
from habitus.scene.relations.spans import Segment
from habitus.scene.relations.state import Relation, _drop_stale_measures
from tests.unit.scene.relation_fixtures import CODE, CONCEPTS, CONFIG, DISCUSS, RESEARCH, key, planted, watching

NIGHT = date(2026, 10, 18)


def _relation(segment: Segment) -> Relation:
    return Relation(
        key=key(DISCUSS, CODE, segment) if segment is not Segment.CHAIN else key(DISCUSS, CODE),
        status=Status.CANDIDATE,
        upward=True,
        discovered_on=NIGHT,
    )


def test_rows_of_a_segment_with_an_older_measure_are_dropped() -> None:
    previous = LaneState(
        lane="session", night=NIGHT, family=2, relations=(_relation(Segment.RETURN), _relation(Segment.CHAIN))
    )
    upgraded = _drop_stale_measures(previous)
    assert [relation.key.segment for relation in upgraded.relations] == [Segment.CHAIN]
    assert upgraded.measure_versions[Segment.RETURN.value] == MEASURE_VERSIONS[Segment.RETURN.value]


def test_current_versions_keep_everything() -> None:
    previous = LaneState(
        lane="session",
        night=NIGHT,
        family=2,
        relations=(_relation(Segment.RETURN),),
        measure_versions=dict(MEASURE_VERSIONS),
    )
    assert _drop_stale_measures(previous) is previous


def test_a_v2_state_drops_established_moderated_chain_rows_and_keeps_plain_chain_rows() -> None:
    """裁定 95 第 4 条改了带条件的同一条链的量法（两道对照都显著才算），版本记在 ``chain|moderated``。上一晚是 V2 时代写出的
    状态（量法版本只有 ``return``）：已成立的"讨论方案 → 修改代码｜先调研过"是按只配第 1 道的旧判据成立的，fold 时清掉、
    在新判据下从数据重新发现；已成立的不带条件"讨论方案 → 修改代码"量法没变，照旧带着。输出的量法版本记上新键。"""

    series = planted(60, moderated=True)
    night = series.cutoff - timedelta(days=1)
    moderated = key(DISCUSS, CODE, condition=Condition(Template.TIMELINE, RESEARCH).text)
    plain = key(DISCUSS, CODE)

    def held(item: RelationKey) -> Relation:
        return Relation(
            key=item,
            status=Status.ESTABLISHED,
            upward=True,
            discovered_on=night - timedelta(days=20),
            established_on=night,
            forward_passed_on=night,
            maintained_through=night,
            maintenance_needed=40,
        )

    previous = LaneState(
        lane="session",
        night=night,
        family=2,
        relations=(held(moderated), held(plain)),
        measure_versions={Segment.RETURN.value: 2},
    )
    line = build_timelines(series, CONCEPTS, {}, CONFIG)["session"]
    tests = LaneTests(line, CONCEPTS, CONFIG, series.cutoff, established=(moderated, plain))
    state, _ = fold(previous, tests, watching=watching(series))
    rows = {item.key: item for item in state.relations}
    assert rows[plain].status is Status.ESTABLISHED  # 不带条件的同一条链量法没变：照旧带着
    # 带条件的那条旧行清掉了：今晚若又出现，也只能是新判据下当晚重新量出来的，不是带过来的成立
    assert moderated not in rows or (
        rows[moderated].established_on is None and rows[moderated].discovered_on == state.night
    )
    assert state.measure_versions[CHAIN_MODERATED] == 2
    assert state.measure_versions == MEASURE_VERSIONS
