"""丙一"应当不变"的（预期表第 1 节 S1、S2、S4、S5、S6；S3 是一条命令：``tests/unit/word_grow``、``tests/unit/word_pick`` 不动就过）。

基线数据（``site.plant_baseline``）里没有一对上下级同时有记录：在用的写周报、写方案是兄弟，其余在用的都是没有下级的第 1 层词，
也没有不在用却有记录的类。逐项比的基线是改动前（``780e298``）的代码跑出来的：

- ``baseline_s1.json.gz``：概念集、复现、时间线上的记号、两代关系表、机会表、当天的预期与递送（S1）；
- ``baseline_s4.json``：兄弟之间每一行当晚的读数，上级写文档不在用 / 也在用（没有直接记录）两种（S4）；
- ``baseline_s5_priors.json``：第一代之后模型队列补问先验的每一次调用（输出名、消息、输出形状）（S5）。

重生成：``out/baseline/scripts/dump_baseline.py``（实验目录 ``2026-11-14-词表实现/第二步丙一-下游认上下级``）。
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from tests.unit.hierarchy_downstream.site import (
    BASELINE_NIGHTS,
    WEEKLY,
    WRITE_DOC,
    World,
    answering_world,
    at,
    conditioned_established,
    drain,
    dump,
    load,
    plant_baseline,
)

DAY = date(2026, 4, 13)


def _normal(value: object) -> object:
    return json.loads(dump(value))


# S1、S4 原来那两条（产物逐项相同、兄弟的行逐行在）收口时换成了阶段 C 按设计口径写的两条（test_verify_c.py：S1 除了预期上的
# 代码指纹 implementation 逐项相同；S4 兄弟的读数逐项相同、检验族变大时可以判成稀疏），见丙一报告。


def test_s2_the_ids_on_the_tree_and_the_hit_records_are_left_as_they_were(tmp_path: Path) -> None:
    world = World(tmp_path)
    world.use(WRITE_DOC, WEEKLY)
    uri = world.record(WEEKLY, at(DAY, 10))
    (path,) = [item for item in (tmp_path / "behavior" / "tree").rglob("*.md") if "occurrences" in item.as_posix()]
    original = path.read_bytes()
    world.generate(DAY.replace(day=14))
    assert path.read_bytes() == original
    (hit,) = world.scene.hits.read_day(DAY)
    assert (hit.occurrence_uri, hit.kind_token, hit.base_concept) == (uri, WEEKLY, WEEKLY)


def test_s5_the_prior_questions_and_their_touchpoint_are_as_before(tmp_path: Path) -> None:
    world, provider = answering_world(tmp_path)
    plant_baseline(world)
    world.generate(BASELINE_NIGHTS[0])
    drain(world, BASELINE_NIGHTS[0])
    priors = [item for item in provider.requests if item["name"] != "scene_relation_conditions"]
    assert _normal(priors) == load("baseline_s5_priors.json")


#: 改动前第二代关系表里出现过的全部条件（机械列举）与成立的带条件关系。
C30_CONDITIONS = [
    "",
    "concept:周一",
    "concept:周三",
    "concept:周二",
    "concept:周五",
    "concept:周四",
    "field:place=书房",
    "field:with=同事B",
    "timeline:s-k0019",
    "timeline:s-k0043",
    "timeline:s-k0046",
]
C30_ESTABLISHED = [
    "s-k0002->s-k0004|chain|field:place=书房",
    "s-k0002->s-k0004|chain|timeline:s-k0046",
]


def test_s6_c30_the_mechanical_conditions_and_the_established_conditioned_relations_are_as_before(
    tmp_path: Path,
) -> None:
    world = World(tmp_path)
    plant_baseline(world)
    for night in BASELINE_NIGHTS:
        world.generate(night)
    state = world.scene.relations.read("session", BASELINE_NIGHTS[1])
    assert sorted({relation.key.condition for relation in state.relations}) == C30_CONDITIONS
    assert conditioned_established(state) == C30_ESTABLISHED
