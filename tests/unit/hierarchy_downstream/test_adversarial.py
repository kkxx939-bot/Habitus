"""丙一对抗测试（阶段 C）：盘上的概念文件、机会表被改坏成"上下级"规则不允许的样子，读的时候要明确报错、点出是哪一份。

- 概念文件里行为概念管的第一个类不是它自己；
- 概念管的编号里混进别的 lane 的编号；
- 两个不相干（谁也不是谁的上级）的概念管同一个编号；
- 机会表里关系的类元组为空，或第一个不是那一头的节点自己。

概念文件是换代的语义树一步（``site.World.generate``）真写出来的，改的时候正文与末尾的结构注释一起改（保持规范形式，
只让"上下级"那一条不成立）；机会表经 ``OpportunityStore.write`` 写出、再按同一规范 JSON 改。
"""

from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from habitus.foresight import ForesightError
from habitus.foresight.opportunities import OpportunityTable, RelationOpportunities
from habitus.scene.concepts.store import ConceptStoreError
from tests.unit.hierarchy_downstream.site import QUERY, WEEKLY, WRITE_DOC, World, at

DAY = date(2026, 4, 13)
NIGHT = date(2026, 4, 14)


def _concept_world(tmp_path: Path) -> World:
    world = World(tmp_path)
    world.use(WRITE_DOC, WEEKLY, QUERY)
    world.record(WEEKLY, at(DAY, 10))
    world.record(QUERY, at(DAY, 11))
    world.generate(NIGHT)
    return world


def _rewrite_classes(world: World, identity: str, classes: list[str]) -> Path:
    """把盘上 ``identity`` 那份概念文件管的编号改成 ``classes``（正文"类："一行与结构注释一起改）。"""

    path = world.config.scene_root / "concepts" / f"{identity}.md"
    text = path.read_text(encoding="utf-8")
    text = re.sub(r"^类：.*$", "类：" + " / ".join(classes), text, count=1, flags=re.MULTILINE)
    text = re.sub(r'"classes":\[[^\]]*\]', '"classes":' + json.dumps(classes, separators=(",", ":")), text, count=1)
    path.write_text(text, encoding="utf-8")
    return path


def test_a_concept_file_whose_first_class_is_not_itself_is_refused_naming_it(tmp_path: Path) -> None:
    world = _concept_world(tmp_path)
    original = world.scene.concepts.read(WRITE_DOC).classes
    _rewrite_classes(world, WRITE_DOC, [WEEKLY, WRITE_DOC, *original[2:]])
    with pytest.raises(ConceptStoreError, match=rf"concept record '{WRITE_DOC}' is corrupt.*own class id") as raised:
        world.scene.concepts.read_all()
    print(raised.value)


def test_a_concept_managing_a_class_of_another_lane_is_refused_naming_it(tmp_path: Path) -> None:
    world = _concept_world(tmp_path)
    original = world.scene.concepts.read(WEEKLY).classes
    _rewrite_classes(world, WEEKLY, [*original, "p-k0001"])
    with pytest.raises(ConceptStoreError, match=rf"concept record '{WEEKLY}' is corrupt.*another lane") as raised:
        world.scene.concepts.read_all()
    print(raised.value)


def test_two_unrelated_concepts_managing_the_same_class_are_refused_naming_both(tmp_path: Path) -> None:
    world = _concept_world(tmp_path)
    original = world.scene.concepts.read(QUERY).classes
    _rewrite_classes(world, QUERY, [*original, WEEKLY])
    with pytest.raises(ConceptStoreError, match="neither is above the other") as raised:
        world.scene.concepts.read_all()
    message = str(raised.value)
    print(message)
    assert QUERY in message and WEEKLY in message
    # 两个概念的身份就是它们的文件名（concepts/<身份>.md）
    assert WRITE_DOC in message


def _table(night: date) -> OpportunityTable:
    relation = RelationOpportunities(
        key=f"session|{QUERY}|{WRITE_DOC}|chain|",
        antecedent=QUERY,
        consequent=WRITE_DOC,
        antecedent_label="查数据",
        consequent_label="写文档",
        antecedent_classes=(QUERY,),
        consequent_classes=(WRITE_DOC, WEEKLY),
        segment="chain",
        status="established",
        upward=True,
        items=(),
    )
    return OpportunityTable("session", night, night, 15, 3, (relation,), ())


def _tamper(path: Path, change: Any) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    change(payload["relations"][0])
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")), encoding="utf-8")


@pytest.mark.parametrize(
    ("label", "classes"),
    [("empty", []), ("child first", [WEEKLY, WRITE_DOC])],
)
def test_an_opportunity_table_whose_consequent_classes_do_not_start_with_the_consequent_is_refused(
    tmp_path: Path, label: str, classes: list[str]
) -> None:
    world = World(tmp_path)
    store = world.foresight.opportunities
    good = store.write(_table(date(2026, 4, 13)))
    path = store.write(_table(NIGHT))
    assert store.read("session", NIGHT).relations[0].consequent_classes == (WRITE_DOC, WEEKLY)

    def change(relation: dict[str, Any]) -> None:
        relation["consequent_classes"] = classes

    _tamper(path, change)
    with pytest.raises(ForesightError, match="consequent_classes must start with the consequent itself") as raised:
        store.read("session", NIGHT)
    print(label, raised.value)
    # 换代之后预测拍、结算读的是 ``latest``：坏的那一晚跳过（往前退一晚），路径计入观测
    latest = store.latest("session")
    assert latest is not None and latest.night == date(2026, 4, 13)
    assert store.take_unreadable() == (str(path),)
    assert good != path
