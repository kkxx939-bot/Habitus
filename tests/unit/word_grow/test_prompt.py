"""第二步乙：会话选词的提示词（预期表 D1、C17；共享说明 3.1）。旧代码上应当失败。

- D1：和实验 ``s2-v1`` 的提示词（``session_system_s2v1.txt``：实验 ``prompt.py`` 的 ``head``，词表是初始会话词表）比，只差
  定义写法那一句（"写成'选中的词那件事 + 一个限定'" → "只写一句，只说这一种是什么"）。
- C17：上一轮长出来的节点下一轮就在词表里：缩进一级（第 3 层两级），紧跟在它上级的初始下级后面，同一上级下按编号排。
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from tests.unit.word_grow.site import (
    HERE,
    LAST_DOC_CHILD,
    NAMES,
    SUMMARY,
    WEEKLY,
    WRITE_DOC,
    Grow,
    GrowSite,
    at,
    grown,
    line_of,
    node,
    pick_calls,
    system_prompt,
)
from tests.unit.word_pick.site import Word

D13 = date(2026, 4, 13)


@pytest.fixture
def site(tmp_path: Path):  # type: ignore[no-untyped-def]
    built = GrowSite(tmp_path)
    yield built
    built.close()


def _number(token: str) -> str:
    return str(int(token.split("-k", 1)[1]))


def test_d1_the_session_prompt_differs_from_the_experiment_s2_v1_only_in_how_a_definition_is_written(
    site: GrowSite,
) -> None:
    site.record("conv-d1", at(D13, 9), "帮我写一份本周周报", Word(WEEKLY))
    (call,) = pick_calls(site)
    actual = system_prompt(call).splitlines()
    expected = (HERE / "session_system_s2v1.txt").read_text(encoding="utf-8").splitlines()
    assert len(actual) == len(expected)
    differing = [(old, new) for old, new in zip(expected, actual, strict=True) if old != new]
    assert len(differing) == 1, differing
    ((old, new),) = differing
    assert old.startswith("- 名字和词表里的词同一个写法") and "一个限定" in old
    head = old[: old.index("时间不进名字。") + len("时间不进名字。")]
    assert new.startswith(head)
    assert "只说这一种是什么" in new and "一个限定" not in new
    assert call.schema is not None and set(call.schema["properties"]) == {"pick", "continues", "new"}


def test_c17_a_node_grown_last_turn_is_in_the_vocabulary_next_turn_after_its_parents_initial_children(
    site: GrowSite,
) -> None:
    site.record("conv-c17", at(D13, 9), "帮我写一份产品需求文档", Grow("写文档", node("写需求稿", "写产品需求文档")))
    site.record("conv-c17", at(D13, 10), "再写一份接口设计文档", Grow("写文档", node("写设计稿", "写技术设计文档")))
    block = site.probe(D13)
    lines = [word.name for word in block.words]
    first, second = line_of(block, "写需求稿"), line_of(block, "写设计稿")
    assert (first.id, second.id) == (_number(grown(1)), _number(grown(2)))
    assert (first.parent_id, first.parent_name) == (_number(WRITE_DOC), "写文档")
    assert first.definition == "写产品需求文档"
    weekly = line_of(block, NAMES[WEEKLY])
    assert first.indent == weekly.indent  # 缩进一级：和初始的第 2 层一样
    last_child = lines.index(NAMES[LAST_DOC_CHILD])
    assert lines[last_child + 1 : last_child + 3] == ["写需求稿", "写设计稿"]  # 初始下级后面，按编号
    assert block.levels()[first.id] == 2


def test_c17_a_third_level_node_is_indented_two_levels_right_after_its_parent(site: GrowSite) -> None:
    site.record("conv-c17b", at(D13, 9), "帮我写一份项目周报", Grow(NAMES[WEEKLY], node("写项目周报")))
    block = site.probe(D13)
    lines = [word.name for word in block.words]
    third = line_of(block, "写项目周报")
    assert (third.parent_id, third.parent_name) == (_number(WEEKLY), NAMES[WEEKLY])
    assert block.levels()[third.id] == 3
    assert third.indent > line_of(block, NAMES[WEEKLY]).indent
    weekly = lines.index(NAMES[WEEKLY])
    assert lines[weekly + 1] == "写项目周报" and lines[weekly + 2] == NAMES[SUMMARY]
