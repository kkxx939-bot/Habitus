"""第二步乙：模型写新节点、代码怎么收（预期表 C1–C19、D2 新节点那一半与旧形状、D3；共享说明 3.1、3.2）。

全部经组合根装出的会话 lane 走：一轮交 ``fusion.consume``（描述 + 选词），归约落树。看三处：行为树上记录的 ``kind_token``、
下一次选词提示词里的词表（``probe``）、词表状态文件（按值宽口径找条目）。编号按"这条 lane 用过的最大号加一"：初始 279 个，
第 n 个长出来的是 ``grown(n)``。旧代码上应当失败：会话答复带 ``new`` 不合格，重问用尽记「待定」。
"""

from __future__ import annotations

import asyncio
import json
from datetime import date
from pathlib import Path

import pytest

from tests.unit.word_grow.site import (
    NAMES,
    QUERY,
    WEEKLY,
    WRITE_DOC,
    Barrier,
    Grow,
    GrowSite,
    Hook,
    Literal,
    Named,
    at,
    entries,
    flat,
    grown,
    mentions_day,
    names,
    node,
)
from tests.unit.word_pick.site import NONE, Word

D13, D14, D15, D16 = (date(2026, 4, day) for day in (13, 14, 15, 16))
FIVE = ("写需求书", "写设计书", "写测试书", "写用户书", "写运维书")
SIXTH = "写培训书"


@pytest.fixture
def site(tmp_path: Path):  # type: ignore[no-untyped-def]
    built = GrowSite(tmp_path)
    yield built
    built.close()


def _entry(site: GrowSite, name: str) -> dict[str, object]:
    found = entries(site.state(), name)
    assert len(found) == 1, f"expected one state entry named {name!r}, found {found}"
    return found[0]


def _values(entry: dict[str, object]) -> list[object]:
    return flat(entry)


# ── 建节点 ────────────────────────────────────────────────────────────────


def test_c1_d2_a_first_node_under_a_first_level_word_is_built_and_carries_the_record(site: GrowSite) -> None:
    site.record("conv-c1", at(D13, 9), "帮我写一份产品需求文档", Grow("写文档", node("写需求书", "写产品需求文档")))
    assert site.token_at(at(D13, 9)) == grown(1)
    assert site.router.count("pick") == 1
    entry = _entry(site, "写需求书")
    values = _values(entry)
    assert grown(1) in values and WRITE_DOC in values  # 编号、上级
    assert 2 in values and "写产品需求文档" in values  # 层、定义
    assert mentions_day(entry, D13)  # 哪一刻写的（这一轮的时刻）
    assert "写需求书" in names(site.probe(D13))


def test_c2_first_and_second_are_built_and_the_record_hangs_on_the_second(site: GrowSite) -> None:
    site.record(
        "conv-c2",
        at(D13, 9),
        "帮我写一份接口说明里的错误码文档",
        Grow("写文档", node("写接口书"), node("写错误码")),
    )
    assert site.token_at(at(D13, 9)) == grown(2)
    second = _values(_entry(site, "写错误码"))
    assert grown(2) in second and grown(1) in second and 3 in second
    block = site.probe(D13)
    assert {"写接口书", "写错误码"} <= set(names(block))


def test_c3_a_first_node_under_a_second_level_word_is_the_third_level(site: GrowSite) -> None:
    site.record("conv-c3", at(D13, 9), "帮我写一份项目周报", Grow(NAMES[WEEKLY], node("写项目周报")))
    assert site.token_at(at(D13, 9)) == grown(1)
    values = _values(_entry(site, "写项目周报"))
    assert WEEKLY in values and 3 in values


def test_c4_a_second_that_would_be_the_fourth_level_is_not_built_and_the_record_stays_on_first(
    site: GrowSite,
) -> None:
    site.record(
        "conv-c4", at(D13, 9), "帮我写一份项目周报的风险部分", Grow(NAMES[WEEKLY], node("写项目周报"), node("写风险"))
    )
    assert site.token_at(at(D13, 9)) == grown(1)
    block = names(site.probe(D13))
    assert "写项目周报" in block and "写风险" not in block
    assert entries(site.state(), "写风险") == []


def test_c5_no_node_is_grown_under_a_third_level_node(site: GrowSite) -> None:
    site.record("conv-c5", at(D13, 9), "帮我写一份项目周报", Grow(NAMES[WEEKLY], node("写项目周报")))
    site.record("conv-c5", at(D13, 10), "再写一份项目周报的风险部分", Grow("写项目周报", node("写风险")))
    assert site.token_at(at(D13, 10)) == grown(1)
    assert "写风险" not in names(site.probe(D13))


def test_c6_a_seven_character_first_is_not_built_nor_its_second(site: GrowSite) -> None:
    site.record(
        "conv-c6", at(D13, 9), "帮我写一份产品需求文档", Grow("写文档", node("写产品需求文档"), node("写需求点"))
    )
    assert site.token_at(at(D13, 9)) == WRITE_DOC
    assert site.router.count("pick") == 1  # 不建不是不合格：不重问
    block = names(site.probe(D13))
    assert "写产品需求文档" not in block and "写需求点" not in block
    assert len(block) == 279


def test_c7_surrounding_whitespace_is_trimmed_before_counting_and_naming(site: GrowSite) -> None:
    site.record("conv-c7", at(D13, 9), "帮我写一份产品需求书", Grow("写文档", node("  写产品需求书 ")))
    assert site.token_at(at(D13, 9)) == grown(1)
    assert "写产品需求书" in names(site.probe(D13))
    assert len(entries(site.state(), "写产品需求书")) == 1


def test_c8_the_same_name_under_the_same_parent_reuses_the_node_and_updates_its_last_pick(site: GrowSite) -> None:
    site.record("conv-c8", at(D13, 9), "帮我写一份产品需求文档", Grow("写文档", node("写需求书")))
    assert not mentions_day(_entry(site, "写需求书"), D14)
    site.record("conv-c8", at(D14, 9), "再写一份需求文档", Grow("写文档", node("写需求书", "另一种说法")))
    assert site.token_at(at(D14, 9)) == grown(1)
    assert names(site.probe(D14)).count("写需求书") == 1
    assert mentions_day(_entry(site, "写需求书"), D14)  # 最后被选中的时刻更新了


def test_c8_with_a_second_the_reused_node_takes_the_new_child(site: GrowSite) -> None:
    site.record("conv-c8b", at(D13, 9), "帮我写一份产品需求文档", Grow("写文档", node("写需求书")))
    site.record(
        "conv-c8b", at(D13, 10), "写一份需求文档里的验收标准", Grow("写文档", node("写需求书"), node("写验收段"))
    )
    assert site.token_at(at(D13, 10)) == grown(2)
    assert grown(1) in _values(_entry(site, "写验收段"))  # 挂在复用的那个下面
    assert names(site.probe(D13)).count("写需求书") == 1


def test_c9_the_name_of_an_initial_child_reuses_the_initial_child(site: GrowSite) -> None:
    site.record("conv-c9", at(D13, 9), "帮我写一份本周周报", Grow("写文档", node(NAMES[WEEKLY])))
    assert site.token_at(at(D13, 9)) == WEEKLY
    assert len(site.probe(D13).words) == 279


def test_c10_a_name_alive_elsewhere_is_not_built_and_the_record_stays_on_the_picked_word(site: GrowSite) -> None:
    site.record("conv-c10", at(D13, 9), "帮我写一份查数据的文档", Grow("写文档", node(NAMES[QUERY])))
    assert site.token_at(at(D13, 9)) == WRITE_DOC
    block = site.probe(D13)
    assert names(block).count(NAMES[QUERY]) == 1 and len(block.words) == 279


def test_c11_only_a_second_is_invalid_and_asked_again(site: GrowSite) -> None:
    only_second = {"first": None, "second": {"name": "写项目周报", "definition": "写项目的周报"}}
    site.router.picks.append(Literal(Named(NAMES[WEEKLY]), only_second))
    site.record("conv-c11", at(D13, 9), "帮我写一份本周周报", Word(WEEKLY))
    assert site.router.count("pick") == 2
    assert site.token_at(at(D13, 9)) == WEEKLY
    assert "写项目周报" not in names(site.probe(D13))
    # 对照：只填 first 是合格的，一次就收（旧代码上任何带 new 的答复都不合格，上面两句在旧代码上也成立）
    calls = site.router.count("pick")
    site.record("conv-c11", at(D13, 10), "帮我写一份项目周报", Grow(NAMES[WEEKLY], node("写项目周报")))
    assert site.router.count("pick") == calls + 1
    assert site.token_at(at(D13, 10)) == grown(1)


def test_c12_new_with_a_fixed_choice_is_invalid_and_asked_again(site: GrowSite) -> None:
    stray = {"first": {"name": "写需求书", "definition": "写产品需求文档"}, "second": None}
    site.router.picks.append(Literal("none", stray))
    site.record("conv-c12", at(D13, 9), "谢谢，辛苦了", NONE)
    assert site.router.count("pick") == 2
    assert site.token_at(at(D13, 9)).endswith("非事件")
    assert "写需求书" not in names(site.probe(D13))
    # 对照：pick 是词编号时同样的 new 合格，一次就收
    calls = site.router.count("pick")
    site.record("conv-c12", at(D13, 10), "帮我写一份产品需求文档", Grow("写文档", node("写需求书")))
    assert site.router.count("pick") == calls + 1
    assert site.token_at(at(D13, 10)) == grown(1)


# ── 候选位置 ──────────────────────────────────────────────────────────────


def _five_candidates(site: GrowSite, day: date, *, with_children: bool = False) -> None:
    for index, name in enumerate(FIVE):
        second = node(f"{name[:3]}细") if with_children else None
        site.record(f"conv-five-{index}", at(day, 9, 10 * index), f"帮我{name}", Grow("写文档", node(name), second))


def test_c13_a_sixth_candidate_evicts_the_one_picked_longest_ago(site: GrowSite) -> None:
    _five_candidates(site, D13)
    site.record("conv-again", at(D13, 10), "再写一份需求书", Named(FIVE[0]))  # 第一个刚被选过
    site.record("conv-six", at(D13, 11), "帮我写一份培训材料", Grow("写文档", node(SIXTH)))
    assert site.token_at(at(D13, 11)) == grown(6)
    block = names(site.probe(D13))
    assert FIVE[1] not in block  # 最久没被选中的变成"已删"
    assert {FIVE[0], *FIVE[2:], SIXTH} <= set(block)
    assert grown(2) in json.dumps(site.state())  # 已删的记号还在状态文件里


def test_c14_a_candidate_with_a_live_child_is_skipped_and_the_next_oldest_goes(site: GrowSite) -> None:
    site.record("conv-a", at(D13, 9), "帮我写一份需求书", Grow("写文档", node(FIVE[0])))
    site.record("conv-b", at(D13, 9, 10), "写设计书里的接口部分", Grow("写文档", node(FIVE[1]), node("写接口段")))
    for index, name in enumerate(FIVE[2:], start=2):
        site.record(f"conv-{index}", at(D13, 9, 10 * index), f"帮我{name}", Grow("写文档", node(name)))
    site.record("conv-again", at(D13, 10), "再写一份需求书", Named(FIVE[0]))
    site.record("conv-six", at(D13, 11), "帮我写一份培训材料", Grow("写文档", node(SIXTH)))
    block = names(site.probe(D13))
    assert {FIVE[1], "写接口段", SIXTH} <= set(block)
    assert FIVE[2] not in block


def test_c15_when_every_candidate_has_a_live_child_nothing_is_built(site: GrowSite) -> None:
    _five_candidates(site, D13, with_children=True)
    site.record("conv-six", at(D13, 11), "帮我写一份培训材料", Grow("写文档", node(SIXTH)))
    assert site.token_at(at(D13, 11)) == WRITE_DOC
    block = names(site.probe(D13))
    assert SIXTH not in block and set(FIVE) <= set(block)


def test_c16_nodes_in_use_take_no_candidate_place_and_are_never_evicted(site: GrowSite) -> None:
    for name, hour in (("写周计划", 9), ("写月计划", 10)):
        site.record(f"conv-{name}", at(D13, hour), f"帮我{name}", Grow("写文档", node(name)))
        site.record(f"conv-{name}", at(D14, hour), f"再{name}", Named(name))
    site.generate(D15)
    in_use = {grown(1), grown(2)}
    assert in_use <= site.in_use()
    _five_candidates(site, D15)
    site.record("conv-six", at(D15, 11), "帮我写一份培训材料", Grow("写文档", node(SIXTH)))
    block = names(site.probe(D15))
    assert {"写周计划", "写月计划", SIXTH, *FIVE[1:]} <= set(block)
    assert FIVE[0] not in block  # 挤掉的是候选里最久没被选中的，不是在用的
    assert site.token_at(at(D15, 11)) == grown(8)


# ── 被杀、并发 ────────────────────────────────────────────────────────────


def test_c18_a_node_written_before_the_judgement_was_killed_stays_and_is_reused_on_the_redo(
    site: GrowSite, monkeypatch: pytest.MonkeyPatch
) -> None:
    site.router.picks.append(Grow("写文档", node("写需求书")))
    turn = site.turn("conv-c18", at(D13, 9), "帮我写一份产品需求文档")

    def killed(*_: object, **__: object) -> None:
        raise RuntimeError("进程被杀")

    with monkeypatch.context() as patched:
        patched.setattr(site.session.judgements, "put_payload", killed)
        with pytest.raises(RuntimeError):
            site.consume(turn)
    assert site.router.count("pick") == 1
    assert len(entries(site.state(), "写需求书")) == 1  # 选词那一刻就落盘了
    site.reduce()
    assert site.occurrences() == []  # 没有挂在它上面的记录

    site.router.picks.append(Grow("写文档", node("写需求书")))
    site.consume(turn)
    site.reduce()
    assert site.token_at(at(D13, 9)) == grown(1)  # 复用，不出第二个
    assert names(site.probe(D13)).count("写需求书") == 1


def test_c19_two_turns_growing_different_names_under_one_parent_at_once_both_land(site: GrowSite) -> None:
    site.router.barrier = Barrier()
    site.router.picks.extend([Grow("写文档", node("写需求书")), Grow("写文档", node("写设计书"))])
    first = site.turn("conv-x", at(D13, 9), "帮我写一份产品需求文档")
    second = site.turn("conv-y", at(D13, 9, 1), "帮我写一份技术设计文档")

    async def both() -> None:
        await asyncio.gather(site.session.fusion.consume((first,)), site.session.fusion.consume((second,)))

    asyncio.run(both())
    site.router.barrier = None
    site.reduce()
    tokens = {site.token_at(at(D13, 9)), site.token_at(at(D13, 9, 1))}
    assert tokens == {grown(1), grown(2)}
    assert len(entries(site.state(), "写需求书")) == 1 and len(entries(site.state(), "写设计书")) == 1


def test_c19b_a_generation_writing_the_state_while_the_model_answers_loses_neither_change(site: GrowSite) -> None:
    site.record("conv-q", at(D13, 9), "查一下上个月的订单数", Named(NAMES[QUERY]))
    site.record("conv-q", at(D14, 9), "再查一下退货数", Named(NAMES[QUERY]))
    site.record(
        "conv-g",
        at(D15, 9),
        "帮我写一份产品需求文档",
        Hook(site.generate_in_thread(D15), Grow("写文档", node("写需求书"))),
    )
    assert QUERY in site.in_use()
    state = json.dumps(site.state(), ensure_ascii=False)
    assert QUERY in state and "写需求书" in state  # 换代写的在用、选词写的节点都在
    assert site.token_at(at(D15, 9)) == grown(1)


# ── D2 旧形状、D3 词表版本 ────────────────────────────────────────────────


def test_d2_a_step_a_state_file_is_refused_and_names_the_file(site: GrowSite) -> None:
    old = {
        "schema_version": "behavior_kinds_state_v1",
        "lane": "session",
        "revision": 1,
        "active": [{"id": WEEKLY, "since": "2026-04-15"}],
    }
    site.state_path.parent.mkdir(parents=True, exist_ok=True)  # 词表文件在这一支树外的根下，这时可能还没建（裁定 112）
    site.state_path.write_text(json.dumps(old, ensure_ascii=False, sort_keys=True, indent=1) + "\n", encoding="utf-8")
    with pytest.raises(Exception) as caught:
        site.session.kind_store.read()
    assert str(site.state_path) in str(caught.value)


def test_d3_a_generation_that_only_saw_new_candidates_keeps_the_vocabulary_version(site: GrowSite) -> None:
    site.record("conv-d3", at(D13, 9), "帮我写一份本周周报", Word(WEEKLY))
    site.generate(D14)
    generation = site.runtime.components.pipelines[0].generation.generation
    before = generation.marks.read(D14)
    assert before is not None
    site.record("conv-d3", at(D14, 9), "帮我写一份产品需求文档", Grow("写文档", node("写需求书")))
    site.generate(D15)  # 这一代只看到一个新候选
    after = generation.marks.read(D15)
    assert after is not None and after.vocabulary_version == before.vocabulary_version
    assert "写需求书" in names(site.probe(D15))
    site.record("conv-d3", at(D15, 9), "再写一份需求书", Named("写需求书"))
    site.generate(D16)  # 两个不同日子：候选转在用，在用集合变了
    assert grown(1) in site.in_use()
    turned = generation.marks.read(D16)
    assert turned is not None and turned.vocabulary_version == before.vocabulary_version + 1
