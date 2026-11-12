"""关系迁移日志按大小轮转：只在换晚时轮，旧的留几份，状态文件的列举不受影响（26 号清单 2.5.3）。"""

from __future__ import annotations

from dataclasses import replace
from datetime import timedelta

import pytest

from habitus.scene.relations import LaneTests, Status, Transition, build_timelines, fold
from habitus.scene.relations import store as relation_store
from habitus.scene.relations.store import RelationStore
from tests.unit.scene.relation_fixtures import CODE, CONCEPTS, CONFIG, DISCUSS, key, planted, watching, workdays


def _nights(count: int):
    series = planted(40, linked=(0, 25))
    previous = None
    found = []
    for day in workdays(40)[5 : 5 + count]:
        tonight = series.until(day + timedelta(days=1))
        line = build_timelines(tonight, CONCEPTS, {}, CONFIG)["session"]
        previous, transitions = fold(previous, LaneTests(line, CONCEPTS, CONFIG, tonight.cutoff), watching=watching(tonight))
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


@pytest.mark.parametrize("separator", [" ", " ", "\u0085"])
def test_a_log_line_with_a_unicode_line_separator_reads_back(tmp_path, separator: str) -> None:
    """日志是规范 JSON、不转义非 ASCII：迁移理由里的 U+2028 / U+2029 / U+0085 原样留在行内。读侧只按 LF 切行，
    下一晚照样接得上（以前 ``splitlines`` 把这一行切成两半，之后每晚都报日志坏了）。"""

    from datetime import date

    from habitus.scene.relations import LaneState

    store = RelationStore(tmp_path)
    first = date(2026, 10, 1)
    planted_key = key(DISCUSS, CODE)
    reason = f"理由{separator}第二段"
    store.write(
        LaneState(lane="session", night=first, family=0, relations=()),
        (Transition(night=first, key=planted_key, before=None, after=Status.CANDIDATE, reason=reason),),
    )
    for offset in (1, 2):
        night = first + timedelta(days=offset)
        store.write(LaneState(lane="session", night=night, family=0, relations=()), ())
    assert [item.reason for item in store.transitions("session")] == [reason]


def test_a_relation_key_and_a_proposed_value_refuse_unprintable_text() -> None:
    """键要写进关系表与日志的一行：不可打印的字符（换行、行分隔符）在构造键、编译模型提的条件时就拒。"""

    from habitus.scene.relations import RelationKey, Segment
    from habitus.scene.relations.conditions import ConditionError, compile_proposal

    for text in ("老 王", "老\n王", "老\u0085王"):
        with pytest.raises(ValueError, match="printable"):
            RelationKey("session", DISCUSS, CODE, Segment.CHAIN, condition=f"with={text}")
        with pytest.raises(ConditionError):
            compile_proposal({"template": "field", "field": "with", "value": text}, CONCEPTS, lane="session")
