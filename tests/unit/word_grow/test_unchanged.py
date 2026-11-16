"""第二步乙：应当不变的（预期表 S1、S2、S3、S5、S6）。旧代码（第二步甲）上应当全过。

会话选词的答复照请求的输出形状答：请求里有 ``new`` 才带 ``"new": null``（``GrowRouter``），所以同一份断言在改动前后都成立。
S4（描述这一轮、后台通知轮、选词失败走错误那条路、中断后接着写完不重问）由第二步甲的测试守着，见报告。
"""

from __future__ import annotations

import asyncio
import re
from datetime import date
from pathlib import Path

import pytest

from habitus.behavior.model import BehaviorKind
from tests.unit.word_grow.site import HERE, QUERY, WEEKLY, GrowSite, Named, at, pick_calls, system_prompt
from tests.unit.word_pick.site import MISSING, NONE, Continue, Raw, Word
from tests.unit.word_pick.test_physical_pick import delivery

D13, D14, D15 = date(2026, 4, 13), date(2026, 4, 14), date(2026, 4, 15)
WORD_ID = re.compile(r"^[a-z]-k\d{4,}$")


def marker(token: str) -> str:
    return token if WORD_ID.match(token) else token.split("-", 1)[1]


@pytest.fixture
def site(tmp_path: Path):  # type: ignore[no-untyped-def]
    built = GrowSite(tmp_path)
    yield built
    built.close()


def test_s1_a_word_with_new_null_lands_on_the_word_and_leaves_the_vocabulary_state_alone(site: GrowSite) -> None:
    before = site.state_bytes()
    site.record("conv-s1", at(D13, 9), "帮我写一份本周周报", Word(WEEKLY))
    assert site.token_at(at(D13, 9)) == WEEKLY
    assert site.router.count("pick") == 1
    assert site.state_bytes() == before  # 初始词被选中不记时刻、不建节点：状态文件不动


@pytest.mark.parametrize(
    ("answers", "expected", "calls"),
    [
        ([NONE], "非事件", 1),
        ([MISSING], "表外", 1),
        ([Raw("不是 JSON"), Raw("还是不是"), Raw("{}")], "待定", 3),
    ],
    ids=["none", "missing", "pending"],
)
def test_s2_fixed_choices_and_exhausted_retries_land_the_same_markers(
    site: GrowSite, answers: list[object], expected: str, calls: int
) -> None:
    site.router.picks.extend(answers[1:])
    site.record("conv-s2", at(D13, 10), "谢谢，辛苦了", answers[0])
    assert marker(site.token_at(at(D13, 10))) == expected
    assert site.router.count("pick") == calls


def test_s2_continue_lands_a_continuation(site: GrowSite) -> None:
    site.record("conv-s2c", at(D13, 9), "帮我写一份本周周报", Word(WEEKLY))
    site.record("conv-s2c", at(D13, 9, 30), "周报写得怎么样了", Continue(said="帮我写一份本周周报"))
    assert marker(site.token_at(at(D13, 9, 30))) == "续话"
    assert site.token_at(at(D13, 9)) == WEEKLY


def _physical_tokens(site: GrowSite) -> list[str]:
    tree = site.physical.tree
    return [str(tree.read(address).fields["kind_token"]) for address in tree.list_addresses(BehaviorKind.OCCURRENCE)]


def test_s3_the_physical_pick_prompt_is_the_old_one_plus_the_new_node_section(site: GrowSite) -> None:
    """丙二改了设计：物理 lane 也写新节点。提示词 = 二甲那一份 + 【新节点：new】一节（文字照会话那一份，"这一轮"说成"这一段"）
    + 输出里的 ``new``；别的一字不差。"""

    site.router.picks.append(Word("p-k0005"))
    asyncio.run(site.runtime.deliver_physical(delivery("换衣服")))
    site.reduce(site.physical)
    (call,) = pick_calls(site)
    text = system_prompt(call)
    start, end = text.index("【新节点：new】"), text.index("【词表】")
    section = text[start:end]
    assert "这一段的事是这个词里更具体的一种" in section and "这一轮" not in section
    old = (HERE / "physical_system_2a.txt").read_text(encoding="utf-8")
    restored = (
        (text[:start] + text[end:])
        .replace("在下面选一项，并可选地写一个新节点。", "在下面选一项。")
        .replace(
            ', "new": null | {"first": {"name": 名字, "definition": 一句定义}, '
            '"second": null | {"name": 名字, "definition": 一句定义}}}',
            "}",
        )
    )
    assert restored == old
    assert call.schema is not None and set(call.schema["properties"]) == {"pick", "new"}
    assert _physical_tokens(site) == ["p-k0005"]


def test_s3_a_physical_answer_carrying_new_null_is_valid(site: GrowSite) -> None:
    site.router.picks.append(Raw('{"pick": 5, "new": null}'))
    asyncio.run(site.runtime.deliver_physical(delivery("换衣服")))
    site.reduce(site.physical)
    assert site.router.count("pick") == 1
    assert _physical_tokens(site) == ["p-k0005"]


def test_s5_an_initial_word_is_in_use_after_two_different_days_and_not_after_one_day_twice(site: GrowSite) -> None:
    site.record("conv-a", at(D13, 9), "帮我写一份本周周报", Word(WEEKLY))
    site.record("conv-a", at(D14, 9), "这周的周报也帮我写一下", Word(WEEKLY))
    site.record("conv-b", at(D13, 10), "查一下上个月的订单数", Named("查数据"))
    site.record("conv-b", at(D13, 15), "再查一下退货数", Named("查数据"))
    site.generate(D15)
    assert site.in_use() == {WEEKLY}
    assert QUERY not in site.in_use()


def _strip_new_node_additions(text: str) -> str:
    """去掉第二步乙加回来的三处：开头那半句、【新节点：new】一节（连同它前面的空行）、输出里的 ``new``。"""

    lines = text.splitlines()
    lines[0] = lines[0].replace("，并可选地写一个新节点", "")
    if any(line.startswith("【新节点") for line in lines):
        start = next(index for index, line in enumerate(lines) if line.startswith("【新节点"))
        end = start
        while end < len(lines) and lines[end].strip():
            end += 1
        del lines[start - 1 : end]
    lines[-1] = re.sub(r', "new":.*$', "}", lines[-1])
    return "\n".join(lines)


def test_s6_without_grown_nodes_the_session_prompt_is_the_step_a_prompt_plus_only_the_new_node_parts(
    site: GrowSite,
) -> None:
    site.record("conv-s6", at(D13, 9), "帮我写一份本周周报", Word(WEEKLY))
    (call,) = pick_calls(site)
    expected = (HERE / "session_system_2a.txt").read_text(encoding="utf-8")
    assert _strip_new_node_additions(system_prompt(call)) == expected
