"""第二步甲：会话 lane 每轮在固定初始词表里选一个词（共享说明 3.1、3.2；预期表 D1、D2、C1–C13、C22、C23）。

全部经组合根装出的会话 lane 走：一轮交给 ``branches.session.fusion.consume``（描述 + 选词），再归约落树，看行为树上落了
什么 ``kind_token``、序列里有没有它。模型是确定性替身 ``Router``：描述按现行输出形状答，选词按设计的输出形状答。
旧代码上这些都应当失败：会话 lane 只描述、不选词；``kind_token`` 由归约里的白天归类给。
"""

from __future__ import annotations

import asyncio
import re
import time
from datetime import date, timedelta
from pathlib import Path

import pytest

from habitus.behavior.fusion.lanes.session import SessionMessageRole, TurnDisposition, TurnHandler
from habitus.behavior.model import BehaviorKind
from habitus.behavior.retry import RetryCause, RetryKind
from habitus.model_client import ModelAuthenticationError, ModelContentSafetyError
from habitus.series.model import SkipReason
from habitus.series.reader import read_series
from tests.unit.word_pick.site import (
    MISSING,
    NONE,
    SESSION_WORDS,
    Continue,
    Pick,
    Raw,
    Site,
    Word,
    at,
    record_lines,
    record_number,
    section,
    vocabulary_block,
)

DAY = date(2026, 4, 13)
NEXT = date(2026, 4, 15)
WORD_ID = re.compile(r"^[a-z]-k\d{4,}$")


def marker(token: str) -> str:
    """``kind_token`` 是词编号就原样给；是占位标记就给标记名（``s-续话`` → ``续话``）。"""

    return token if WORD_ID.match(token) else token.split("-", 1)[1]


def occurrences(site: Site) -> list[dict[str, object]]:
    tree = site.session.tree
    return [dict(tree.read(address).fields) for address in tree.list_addresses(BehaviorKind.OCCURRENCE)]


def only_occurrence(site: Site) -> dict[str, object]:
    (fields,) = occurrences(site)
    return fields


@pytest.fixture
def site(tmp_path: Path):  # type: ignore[no-untyped-def]
    built = Site(tmp_path)
    yield built
    built.close()


# ── D2 / C1：答一个词 ────────────────────────────────────────────────────────


def test_c1_a_picked_word_becomes_the_kind_token_and_the_record_is_classified(site: Site) -> None:
    site.record("conv-c1", at(DAY, 9), "帮我写一份本周周报", Word("s-k0002"))
    assert only_occurrence(site)["kind_token"] == "s-k0002"
    (record,) = read_series(site.session.tree, cutoff=NEXT).records
    assert (record.kind_token, record.classified) == ("s-k0002", True)
    assert (site.router.count("describe"), site.router.count("pick")) == (1, 1)


def test_d2_the_session_lane_picks_with_the_turn_and_the_reduction_asks_no_model(site: Site) -> None:
    """选词在 lane 里紧跟描述之后做、结果随判断存；归约对会话的判断不调模型，``kind_token`` 来自判断上存的选词结果。"""

    site.router.picks.append(Word("s-k0003"))
    turn = site.turn("conv-d2", at(DAY, 9, 20), "把这个季度的工作写个总结")
    (outcome,) = site.consume(turn)
    assert outcome.disposition is TurnDisposition.RECORDED
    assert (site.router.count("describe"), site.router.count("pick")) == (1, 1)
    before = len(site.router.calls)
    site.router.pick_default = Word("s-k0005")  # 归约里要是再问一次，就会落成另一个词
    site.reduce()
    asked = [call.kind for call in site.router.calls[before:]]
    assert "pick" not in asked and "classify" not in asked
    assert only_occurrence(site)["kind_token"] == "s-k0003"


# ── C2：none ────────────────────────────────────────────────────────────────


def test_c2_none_lands_a_not_event_record_outside_the_series(site: Site) -> None:
    site.record("conv-c2", at(DAY, 10), "谢谢，辛苦了", NONE)
    assert site.router.count("pick") == 1  # 「非事件」由选词答 none 定，不是归约里另问一次
    assert marker(str(only_occurrence(site)["kind_token"])) == "非事件"
    series = read_series(site.session.tree, cutoff=NEXT)
    assert series.records == ()
    assert [item.reason for item in series.skipped] == [SkipReason.NOT_EVENT]


# ── C3 / C16 一半：续话 ─────────────────────────────────────────────────────


def test_c3_continue_lands_a_continuation_outside_the_series_and_leaves_the_record_it_continues_untouched(
    site: Site,
) -> None:
    site.record("conv-c3", at(DAY, 9), "帮我写一份项目方案", Word("s-k0004"))
    (first,) = site.session.tree.list_addresses(BehaviorKind.OCCURRENCE)
    path = site.session.tree.path_for(first)
    original = path.read_bytes()

    site.record("conv-c3", at(DAY, 9, 40), "方案写得怎么样了", Continue(said="帮我写一份项目方案"))

    tokens = sorted(marker(str(item["kind_token"])) for item in occurrences(site))
    assert tokens == ["s-k0004", "续话"]
    assert path.read_bytes() == original  # 被接的那条记录逐字节不变
    series = read_series(site.session.tree, cutoff=NEXT)
    assert [record.kind_token for record in series.records] == ["s-k0004"]
    (skipped,) = series.skipped
    # 续话有自己的跳过原因，不混成「非事件」或撞车重复
    assert skipped.reason not in (SkipReason.NOT_EVENT, SkipReason.DUPLICATE)
    assert site.router.count("pick") == 2


# ── C4 / C6 / C7 / C8：不合格就重问 ────────────────────────────────────────


def test_c4_continue_with_a_number_outside_the_list_is_invalid_and_ends_pending(site: Site) -> None:
    site.record("conv-c4", at(DAY, 9), "帮我写一份项目方案", Word("s-k0004"))
    site.router.picks.extend([Continue(number=99), Continue(number=99)])
    site.record("conv-c4", at(DAY, 9, 30), "接着弄", Continue(number=99))
    assert site.router.count("pick") == 1 + 3
    later = [item for item in occurrences(site) if item["kind_token"] != "s-k0004"]
    (fields,) = later
    assert marker(str(fields["kind_token"])) == "待定"
    pending = [record for record in read_series(site.session.tree, cutoff=NEXT).records if not record.classified]
    assert len(pending) == 1


def test_c5_missing_lands_an_out_of_vocabulary_record_that_stays_in_the_series_unclassified(site: Site) -> None:
    site.record("conv-c5", at(DAY, 11), "帮我给猫预约绝育手术", MISSING)
    assert marker(str(only_occurrence(site)["kind_token"])) == "表外"
    (record,) = read_series(site.session.tree, cutoff=NEXT).records
    assert record.classified is False


def test_c6_three_unparseable_answers_end_pending_after_exactly_three_pick_calls(site: Site) -> None:
    site.router.picks.extend([Raw("我觉得是写周报"), Raw("{坏的")])
    site.record("conv-c6", at(DAY, 12), "帮我写一份本周周报", Raw("还是不对"))
    assert site.router.count("pick") == 3
    assert marker(str(only_occurrence(site)["kind_token"])) == "待定"
    (record,) = read_series(site.session.tree, cutoff=NEXT).records
    assert record.classified is False


def test_c7_an_id_that_is_not_in_the_vocabulary_is_asked_again(site: Site) -> None:
    site.router.picks.append(Pick("s-k0999"))
    site.record("conv-c7", at(DAY, 13), "把这个季度的工作写个总结", Word("s-k0003"))
    assert site.router.count("pick") == 2
    assert only_occurrence(site)["kind_token"] == "s-k0003"


def test_c8_a_domain_heading_is_not_a_word_and_is_asked_again(site: Site) -> None:
    site.router.picks.append(Pick("文档"))
    site.record("conv-c8", at(DAY, 14), "帮我新写一份文档", Word("s-k0001"))
    assert site.router.count("pick") == 2
    assert only_occurrence(site)["kind_token"] == "s-k0001"


# ── C9 / C10：后台通知轮 ───────────────────────────────────────────────────


def _woken(site: Site, *, he_speaks: bool):  # type: ignore[no-untyped-def]
    prompt, completion = SessionMessageRole.PROMPT, SessionMessageRole.COMPLETION
    messages = [
        (0.0, prompt, "帮我在后台把全量测试跑一遍", False),
        (1.0, completion, "已经在后台跑了，跑完告诉你。", False),
        (5.0, prompt, "<task-notification>后台任务完成：全部测试通过</task-notification>", False),
    ]
    if he_speaks:
        messages.append((5.5, prompt, "那顺便把覆盖率也看一下", True))
    messages.append((6.0, completion, "测试全部通过了。", False))
    return site.turns("conv-wake", messages, at(DAY, 15))


def test_c9_a_wake_up_turn_without_his_words_is_a_not_event_written_by_code_without_any_model_call(
    site: Site,
) -> None:
    site.router.picks.append(Word("s-k0005"))
    first, woken = _woken(site, he_speaks=False)
    outcomes = site.consume(first, woken)
    assert (site.router.count("describe"), site.router.count("pick")) == (1, 1)  # 只有第一轮问了模型
    assert outcomes[1].disposition is TurnDisposition.RECORDED
    site.reduce()
    by_start = {
        address.started_at: dict(site.session.tree.read(address).fields)
        for address in site.session.tree.list_addresses(BehaviorKind.OCCURRENCE)
    }
    assert len(by_start) == 2
    wake = by_start[woken.instructed_at]
    assert marker(str(wake["kind_token"])) == "非事件"
    described = {str(fields["name"]) for fields in by_start.values() if fields is not wake}
    assert wake["name"] not in described and str(wake["name"]).strip()
    assert str(wake["summary"]).strip()
    # 同一轮重放：认得出写过，不再写、仍不问模型
    calls = len(site.router.calls)
    again = site.consume(first, woken)
    assert {item.disposition for item in again} == {TurnDisposition.ALREADY_RECORDED}
    assert len(site.router.calls) == calls
    site.reduce()
    assert len(site.session.tree.list_addresses(BehaviorKind.OCCURRENCE)) == 2


def test_c10_a_wake_up_turn_where_he_speaks_goes_the_ordinary_way(site: Site) -> None:
    site.router.picks.extend([Word("s-k0005"), Word("s-k0005")])
    first, woken = _woken(site, he_speaks=True)
    site.consume(first, woken)
    assert (site.router.count("describe"), site.router.count("pick")) == (2, 2)


# ── C11 / C12 / C12b / C13：失败与重做 ────────────────────────────────────


def test_c11_an_unavailable_model_while_picking_is_an_error_and_the_retry_redoes_describe_and_pick(
    site: Site,
) -> None:
    site.router.picks.extend([ModelAuthenticationError("down"), Word("s-k0002")])
    turn = site.turn("conv-c11", at(DAY, 16), "帮我写一份本周周报")
    (outcome,) = site.consume(turn)
    assert outcome.disposition is TurnDisposition.ERROR
    backlog = site.session.retry_backlog
    assert backlog is not None
    (item,) = backlog.snapshot().items
    assert (item.kind, item.cause) == (RetryKind.TURN, RetryCause.UNAVAILABLE)
    assert (site.router.count("describe"), site.router.count("pick")) == (1, 1)

    # 密钥错属于"要人处理"：熔断直接打开（裁定 108）；人改好之后到了试探的时候，重试才会真去问模型
    breaker = site.runtime.components.models.chat.breaker
    breaker._monotonic = lambda: time.monotonic() + 86_400.0
    asyncio.run(TurnHandler(site.session.fusion).attempt(item))
    assert (site.router.count("describe"), site.router.count("pick")) == (2, 2)
    site.reduce()
    tokens = [fields["kind_token"] for fields in occurrences(site)]
    assert tokens == ["s-k0002"]


def test_c12_a_refused_pick_is_an_error_not_retried_and_leaves_no_description_only_judgement(site: Site) -> None:
    site.router.picks.append(ModelContentSafetyError("blocked"))
    turn = site.turn("conv-c12", at(DAY, 17), "帮我写一份本周周报")
    (outcome,) = site.consume(turn)
    assert outcome.disposition is TurnDisposition.ERROR
    assert [judgement["behavior"] for judgement in site.session.judgements.list()] == [None]
    backlog = site.session.retry_backlog
    assert backlog is not None
    (item,) = backlog.snapshot().items
    assert item.cause is RetryCause.REFUSED


def test_c12b_a_turn_killed_after_its_judgement_landed_is_finished_without_asking_again(
    site: Site, monkeypatch: pytest.MonkeyPatch
) -> None:
    site.router.picks.extend([Word("s-k0005"), Word("s-k0006")])
    turn = site.turn("conv-c12b", at(DAY, 18), "帮我写一份调研报告")

    def killed(*_: object, **__: object) -> None:
        raise RuntimeError("进程被杀")

    with monkeypatch.context() as patched:
        patched.setattr(site.session.receipts, "put", killed)
        with pytest.raises(RuntimeError):
            site.consume(turn)
    assert (site.router.count("describe"), site.router.count("pick")) == (1, 1)

    (outcome,) = site.consume(turn)
    assert outcome.disposition is TurnDisposition.ALREADY_RECORDED
    assert (site.router.count("describe"), site.router.count("pick")) == (1, 1)
    site.reduce()
    assert only_occurrence(site)["kind_token"] == "s-k0005"


def test_c13_redoing_a_written_turn_asks_no_model_and_writes_nothing_twice(site: Site) -> None:
    site.router.picks.append(Word("s-k0002"))
    turn = site.turn("conv-c13", at(DAY, 19), "帮我写一份本周周报")
    site.consume(turn)
    judgements = len(site.session.judgements.list())
    asked = (site.router.count("describe"), site.router.count("pick"))
    assert asked == (1, 1)
    (again,) = site.consume(turn)
    assert again.disposition is TurnDisposition.ALREADY_RECORDED
    assert (site.router.count("describe"), site.router.count("pick")) == asked
    assert len(site.session.judgements.list()) == judgements
    site.reduce()
    (repeat,) = site.consume(turn)
    assert repeat.disposition is TurnDisposition.ALREADY_RECORDED
    assert (site.router.count("describe"), site.router.count("pick")) == asked
    assert len(occurrences(site)) == 1


# ── D1 / C22：提示词里的词表 ───────────────────────────────────────────────


def _session_pick_prompt(site: Site) -> str:
    site.record("conv-vocab", at(DAY, 20), "帮我写一份本周周报", Word("s-k0002"))
    (text,) = site.router.texts("pick")
    return text


def test_c22_d1_the_session_vocabulary_block_is_the_initial_279_word_table(site: Site) -> None:
    block = vocabulary_block(_session_pick_prompt(site))
    assert block.unparsed == ()
    assert len(block.words) == 279
    assert block.headings and "文档" in block.headings  # 领域标题行（只分组，不是词）
    levels = block.levels()
    assert sorted(levels.values()).count(1) == 135 and sorted(levels.values()).count(2) == 144
    ids = [word.id for word in block.words]
    assert len(set(ids)) == 279  # 编号不重复
    assert ids == [f"s-k{number:04d}" for number in range(1, 280)] or ids == [str(n) for n in range(1, 280)]
    seen: dict[str, int] = {}
    for word in block.words:
        level = levels[word.id]
        if level == 1:
            assert word.parent_id is None and word.indent == 0
        else:
            # 第 2 层缩进一级并带"（上级 的下级）"；上级在它前面、层数 = 上级层数 + 1
            assert word.parent_id in seen and seen[word.parent_id] == level - 1
            parent = next(item for item in block.words if item.id == word.parent_id)
            assert word.parent_name == parent.name
        assert word.definition.strip()
        seen[word.id] = level
    names = {word.id: word.name for word in block.words}
    for token, name in SESSION_WORDS.items():
        number = token if token in names else str(int(token.removeprefix("s-k")))
        assert names[number] == name


def test_c22_the_session_pick_prompt_keeps_the_fixed_choices_and_the_new_node_section(site: Site) -> None:
    text = _session_pick_prompt(site)
    for choice in ('"none"', '"continue"', '"missing"'):
        assert choice in text
    assert "【新节点：new】" in text  # 二乙加回来（逐字对照见 word_grow/test_prompt.py）
    assert "【最近的记录】" in text and "【这场对话里之前的几轮】" in text and "【这一轮】" in text


# ── C23：最近的记录 ────────────────────────────────────────────────────────


def test_c23_recent_records_are_the_last_eight_plus_the_last_three_of_this_conversation(site: Site) -> None:
    """10 条已有记录，第 2、6、9 条与这一轮同会话：给最近 8 条（3–10）加同会话最近 3 条（2、6、9），去重；
    v4（裁定 106）起分两段——"这场对话里的"（2、6、9）在前、"别的对话里的"（3、4、5、7、8、10）在后，各按时间排，
    编号从 1 连着排下来；第 1 条不在里面。"""

    long_said = (
        "请把下面这一大段会议录音整理成要点并按议题分组然后标出每个议题的负责人和截止时间"
        "最后再附一份待办清单发给我看看另外把上周那次评审会的结论也补进去顺便核对一下人名有没有写错"
    )
    assert len(long_said) > 61
    said = {index: f"第{index}件事请帮我整理材料编号{index:02d}" for index in range(1, 11)}
    said[10] = long_said
    same = {2, 6, 9}
    for index in range(1, 11):
        conversation = "conv-main" if index in same else f"conv-other-{index}"
        site.record(conversation, at(DAY, 8) + timedelta(minutes=10 * index), said[index], Word("s-k0005"))
    site.record("conv-main", at(DAY, 11), "这次换个事：帮我写一份项目方案", Word("s-k0004"))

    text = site.router.texts("pick")[-1]
    lines = record_lines(text)
    assert len(lines) == 9
    for index, line in zip((2, 6, 9, 3, 4, 5, 7, 8, 10), lines, strict=True):
        assert said[index][:60] in line
        assert "写报告" in line  # 归到的词名
    assert not any(said[1] in line for line in lines)
    assert long_said[:61] not in lines[-1]  # 他当时说的只给前 60 字
    numbers = [int(re.match(r"^\s*(\d+)", line)[1]) for line in lines]  # type: ignore[index]
    assert numbers == list(range(1, 10))
    block = section(text, "【最近的记录】")
    assert block[0] == "这场对话里的：" and block[4] == "别的对话里的："


def test_c23b_with_nothing_from_this_conversation_yet_its_part_says_so_and_a_continue_reaches_the_other_part(
    site: Site,
) -> None:
    """这场对话里还没有记录时"这场对话里的"一段写"（还没有）"；接别的对话里那一条的 continue 按它的编号照样落成续话。"""

    site.record("conv-a", at(DAY, 9), "帮我写一份本周周报", Word("s-k0005"))
    site.record("conv-b", at(DAY, 10), "周报写得怎么样了", Continue(said="帮我写一份本周周报"))
    text = site.router.texts("pick")[-1]
    block = section(text, "【最近的记录】")
    assert block[:3] == ["这场对话里的：", "（还没有）", "别的对话里的："]
    assert record_number(text, "帮我写一份本周周报") == 1
    assert sorted(marker(str(item["kind_token"])) for item in occurrences(site)) == ["s-k0005", "续话"]
