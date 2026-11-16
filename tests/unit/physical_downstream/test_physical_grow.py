"""第二步丙二：物理 lane 选词也能写新节点（预期表 D3，场景 C18–C22；共享说明 3.4）。

提示词在物理那一份上加【新节点：new】一节和输出里的 ``new``，仍没有 ``none``、``continue``；模型写完以后怎么收、候选、转在用、
删除与已删换算全部和会话一支同一套、同一组配置，物理一支的"数在用的词"在物理一支的换代里做。词表状态看这一支的
``kinds.state.json``。改动前物理的输出只有 ``pick``，这几条在"物理一支建了节点"那一处失败（C19 是守门，改动前就成立）。
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Any

from habitus.runtime.scene_vocabulary import kind_resolution
from habitus.series.reader import SeriesReader
from tests.unit.physical_downstream.site import (
    MEDICINE,
    RUN,
    Grow,
    Named,
    Pick,
    World,
    at,
    day_delivery,
)
from tests.unit.word_grow.site import node

D13 = date(2026, 4, 13)


def _grown_named(world: World, lane: str, name: str) -> list[dict[str, Any]]:
    return [item for item in world.grown(lane).values() if item.get("name") == name]


def _reduce_day(world: World, day: date, behaviors: list[tuple[str, int, int, int]]) -> None:
    world.deliver(day_delivery(day, behaviors), now=at(day + timedelta(days=1), 1, 30))
    world.reduce()


def test_d3_the_physical_prompt_has_the_new_node_section_and_new_but_still_no_none_or_continue(
    tmp_path: Path,
) -> None:
    world = World(tmp_path, start=at(D13, 0))
    try:
        _reduce_day(world, D13, [("吃药", 8, 0, 10)])
        (call,) = world.router.physical_calls()
        assert "【新节点" in call.text
        properties = (call.schema or {}).get("properties", {})
        assert "new" in properties
        assert '"none"' not in call.text and '"continue"' not in call.text
        assert "none" not in str(properties.get("pick")) and "continue" not in str(properties.get("pick"))
        assert "continues" not in properties
    finally:
        world.close()


def test_c18_a_physical_pick_that_writes_first_builds_a_candidate_under_the_same_rules_as_the_session(
    tmp_path: Path,
) -> None:
    world = World(tmp_path, start=at(D13, 0))
    try:
        world.router.physical.extend(
            [
                Grow("吃药", node("吃降压药")),  # 建候选，记录挂在它上面
                Grow("吃药", node("吃降压药")),  # 同上级同名：复用
                Grow("吃药", node("吃很多种不同药")),  # 名字超过 6 字：不建，记录落在选的词上
                Grow("跑步", node("吃降压药")),  # 别处已有同名：不建
            ]
        )
        _reduce_day(world, D13, [("吃药", 8, 0, 10), ("吃药", 12, 0, 10), ("吃药", 20, 0, 10), ("跑步", 21, 0, 30)])
        built = _grown_named(world, "physical", "吃降压药")
        assert len(built) == 1, world.grown("physical")
        assert built[0].get("parent") == MEDICINE
        assert not _grown_named(world, "physical", "吃很多种不同药")
        assert built[0]["id"] not in world.in_use("physical")  # 候选不在用（在用另由换代数）
        tokens = [token for _, token in sorted(world.tokens("physical").items())]
        assert tokens == [built[0]["id"], built[0]["id"], MEDICINE, RUN]
    finally:
        world.close()


def test_c19_none_or_continue_in_a_physical_answer_is_invalid_and_asked_again(tmp_path: Path) -> None:
    """守门：改动前物理的输出就没有 ``none``、``continue``（第二步甲 C21）；加了 ``new`` 之后仍然如此。"""

    world = World(tmp_path, start=at(D13, 0))
    try:
        world.router.physical.extend([Pick("none"), Pick("continue"), Named("吃药")])
        _reduce_day(world, D13, [("吃药", 8, 0, 10)])
        assert len(world.router.physical_calls()) == 3
        assert list(world.tokens("physical").values()) == [MEDICINE]
    finally:
        world.close()


def test_c20_a_physical_candidate_picked_on_two_days_is_in_use_after_the_next_physical_generation(
    tmp_path: Path,
) -> None:
    world = World(tmp_path, start=at(D13, 0))
    try:
        world.router.physical.append(Grow("吃药", node("吃降压药")))
        world.physical_days(D13, [[("吃药", 8, 0, 10)]])
        built = _grown_named(world, "physical", "吃降压药")
        assert len(built) == 1, "the physical branch built no candidate"
        identity = built[0]["id"]
        world.router.physical.append(Named("吃降压药"))
        world.physical_days(D13 + timedelta(days=1), [[("吃药", 8, 0, 10)], []])
        assert identity in world.in_use("physical")
        assert (world.scene("physical") / "concepts" / f"{identity}.md").is_file()
    finally:
        world.close()


def _retire_soon(raw: dict[str, Any]) -> None:
    raw["behavior"]["kinds_retire_days"] = 3


def test_c21_a_grown_physical_node_not_picked_for_the_retire_days_is_deleted_and_read_as_its_parent(
    tmp_path: Path,
) -> None:
    world = World(tmp_path, start=at(D13, 0), edit=_retire_soon)
    try:
        world.router.physical.append(Grow("吃药", node("吃降压药")))
        world.physical_days(D13, [[("吃药", 8, 0, 10)]])
        built = _grown_named(world, "physical", "吃降压药")
        assert len(built) == 1, "the physical branch built no candidate"
        identity = built[0]["id"]
        world.physical_days(D13 + timedelta(days=1), [[("跑步", 18, 0, 30)]] * 5)
        assert identity in world.deleted("physical")
        assert identity not in world.grown("physical")
        reader = SeriesReader(world.physical.tree, (), name="probe", kinds=kind_resolution(world.physical.kind_store))
        series = reader.read(cutoff=D13 + timedelta(days=6))
        first = [record.kind_token for record in series.records if record.started_at.date() == D13]
        assert first == [MEDICINE]
    finally:
        world.close()


def test_c22_a_session_and_a_physical_node_with_the_same_name_are_both_built(tmp_path: Path) -> None:
    world = World(tmp_path, start=at(D13, 0))
    try:
        world.router.picks.append(Grow("写文档", node("晨间例行")))
        world.set_clock(at(D13, 9, 2))
        world.turn("conv", at(D13, 9), "帮我写一份晨间例行的说明")
        world.router.physical.append(Grow("吃药", node("晨间例行")))
        _reduce_day(world, D13, [("吃药", 8, 0, 10)])
        assert len(_grown_named(world, "session", "晨间例行")) == 1
        built = _grown_named(world, "physical", "晨间例行")
        assert len(built) == 1, world.grown("physical")
        assert built[0].get("parent") == MEDICINE
    finally:
        world.close()
