"""关系迁移日志按大小轮转：只在换晚时轮，旧的留几份，状态文件的列举不受影响（26 号清单 2.5.3）。"""

from __future__ import annotations

from dataclasses import replace
from datetime import timedelta

import pytest

from habitus.scene.relations import LaneTests, Status, Transition, build_timelines, fold
from habitus.scene.relations import store as relation_store
from habitus.scene.relations.store import RelationStore
from tests.unit.scene.relation_fixtures import CODE, CONCEPTS, CONFIG, DISCUSS, key, planted, workdays


def _nights(count: int):
    series = planted(40, linked=(0, 25))
    previous = None
    found = []
    for day in workdays(40)[5 : 5 + count]:
        tonight = series.until(day + timedelta(days=1))
        line = build_timelines(tonight, CONCEPTS, {}, CONFIG)["session"]
        previous, transitions = fold(previous, LaneTests(line, CONCEPTS, CONFIG, tonight.cutoff))
        found.append((previous, transitions))
    return found


def _with_a_transition(state, transitions):
    if transitions:
        return transitions
    planted_key = key(DISCUSS, CODE)
    return (Transition(night=state.night, key=planted_key, before=None, after=Status.TESTED, reason="测试用的一条"),)


def test_old_logs_are_rotated_and_capped(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(relation_store, "ROTATE_AT_BYTES", 1)
    store = RelationStore(tmp_path / "scene", rotate_files=2)
    nights = _nights(6)
    for state, transitions in nights:
        store.write(state, _with_a_transition(state, transitions))

    directory = tmp_path / "scene" / "relations" / "session"
    archives = sorted(path.name for path in directory.iterdir() if path.name.startswith("transitions-"))
    assert len(archives) == 2
    assert store.nights("session") == tuple(state.night for state, _ in nights[-3:])  # 最晚一晚 + 两代备用
    assert {item.night for item in store.transitions("session")} == {nights[-1][0].night}


def test_a_rerun_of_the_same_night_stays_in_one_log(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(relation_store, "ROTATE_AT_BYTES", 1)
    store = RelationStore(tmp_path / "scene", rotate_files=2)
    ((state, transitions),) = _nights(1)
    store.write(state, _with_a_transition(state, transitions))
    rerun = replace(state, vocabulary=state.vocabulary + 1)
    store.write(rerun, _with_a_transition(rerun, transitions))
    directory = tmp_path / "scene" / "relations" / "session"
    assert not any(path.name.startswith("transitions-") for path in directory.iterdir())
