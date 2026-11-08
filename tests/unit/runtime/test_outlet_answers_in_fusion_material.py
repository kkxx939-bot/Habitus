"""出口已按固定选项写过回话的那次提问，不再交给会话 lane 的融合模型（v0.12 跑数主会话观察第 8、11 条；用户 2026-11-08 定）。

链路照真实的走：会话源消息（``ConversationMessage``）→ 真实翻译 ``host_message`` → 真实 ``ReminderOutlet.observe`` 写账本；
同一批消息 → 组合根真实翻译 ``_message`` → 真实 ``split_turns`` → 接了真实 ``ReminderResponses`` 桥（同一个账本）的会话 lane。
只断言交给模型的那段文字（照脚本回答的模型记下的 ``prompts``），不断言任何内部函数或中间结构。

场景：账本上另有一条早先被答"知道了"（= 中，按设计不关）的快递提醒，它在每一轮都该照样列在【还开着的提醒】里。
"""

from __future__ import annotations

import asyncio
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from habitus.behavior.fusion.lanes.session import SessionLane, SessionTurn, TurnDisposition, split_turns
from habitus.behavior.fusion.lanes.session.lane import TurnHandler
from habitus.behavior.fusion.lanes.session.material import ANSWER_PREFIX
from habitus.foresight.ledger import SPOKEN, Delivery, Expectation, Response
from habitus.pre.conversation import (
    QUESTION_REPLY_TAG,
    ConversationAdapterRegistry,
    ConversationMessage,
    ConversationMessageRole,
    ConversationToolResultContentMode,
    ConversationToolResultStatus,
)
from habitus.runtime.foresight_ledger import ForesightLedgerStore
from habitus.runtime.foresight_outlet import OPTIONS, OpenRequest, ReminderOutlet
from habitus.runtime.foresight_outlet_consumer import host_message
from habitus.runtime.foresight_outlet_store import OutletStore
from habitus.runtime.foresight_responses import ReminderResponses
from habitus.runtime.session_lane import _message, answered_question_tools
from tests.unit.behavior.scripted_provider import ScriptedProvider
from tests.unit.behavior.session_fixtures import Stores, description, lane

CST = timezone(timedelta(hours=8))
DAY = date(2026, 10, 22)
FORMED = datetime(2026, 10, 22, 14, 0, tzinfo=CST)
AWAITING_AT = FORMED + timedelta(minutes=100)  # 15:40 时窗末记「待取」
CLAIMED_AT = FORMED + timedelta(minutes=112)  # 15:52 他敲了一句话，hook 取走

#: 这一轮里 agent 问出口的那条提醒（真实跑数里的 s-k0014）。
TASK_ID = "e-task"
TASK_KIND = "s-k0014"
TASK_TEXT = "往常会确认任务是否完成，这次也确认一下。"
#: 早先答过"知道了"、一直开着的那条（真实跑数里的快递）。
PARCEL_ID = "e-parcel"
PARCEL_TEXT = "下午你一般会去取快递，今天的还没取。"

#: agent 自己的提问（选项不是我们那四个）。
OWN_TITLE = "用哪个数据库做本地缓存？"
OWN_OPTIONS = ("Postgres", "SQLite")

CLAUDE = ("claude_code", "claude-code", "claude-s1")
CODEX = ("codex_rollout", "codex", "codex-thread-1")

#: 接了预测层时，模型照提示词答的那一份（带着【还开着的提醒】就必须有 responses）。
JUDGED = {**description(), "responses": []}


def expectation(identity: str, kind: str, *, deadline_hours: int) -> Expectation:
    return Expectation(
        expectation_id=identity,
        kind_token=kind,
        day=DAY,
        formed_at=FORMED,
        source="日子",
        trigger="rel",
        trigger_instance=f"behavior://{identity}.md",
        window=(FORMED, FORMED + timedelta(hours=2)),
        deadline=FORMED + timedelta(hours=deadline_hours),
        basis=(),
        opportunities=3,
        hits=3,
        observed_from=FORMED,
    )


class World:
    """一套真实的账本 + 出口 + 行为侧存储；快递那条已经说出口、被答"知道了"；这一轮要问的那条已经被这个会话取走。"""

    def __init__(self, tmp_path: Path, conversation: tuple[str, str, str]) -> None:
        self.protocol, self.host, self.session = conversation
        self.ledger = ForesightLedgerStore(tmp_path / "foresight")
        self.outlet = ReminderOutlet(
            OutletStore(self.ledger.root / "outlet"),
            self.ledger,
            liveness=timedelta(hours=48),
            lease=timedelta(minutes=30),
            max_attempts=3,
            claim_limit=3,
            similarity=0.6,
        )
        self.behavior = Stores(tmp_path / "behavior")
        parcel = expectation(PARCEL_ID, "s-k0003", deadline_hours=8)
        self.ledger.record_expectation(parcel)
        self.ledger.record_delivery(
            DAY, Delivery(PARCEL_ID, FORMED + timedelta(minutes=20), 1, False, "待取", 1.0, PARCEL_TEXT)
        )
        said = FORMED + timedelta(minutes=30)
        self.ledger.record_delivery(
            DAY, Delivery(PARCEL_ID, said, 1, True, SPOKEN, 1.0, PARCEL_TEXT, via="claude-code", turn="toolu_0")
        )
        self.ledger.record_response(DAY, Response(PARCEL_ID, said + timedelta(minutes=1), "中", "toolu_0"))
        task = expectation(TASK_ID, TASK_KIND, deadline_hours=4)
        self.ledger.record_expectation(task)
        self.ledger.record_delivery(DAY, Delivery(TASK_ID, AWAITING_AT, 1, False, "待取", 1.0, TASK_TEXT))
        self.outlet.publish(
            [OpenRequest(TASK_ID, TASK_KIND, TASK_TEXT, task.deadline, task.formed_at, 1, DAY)], now=AWAITING_AT
        )
        claimed = self.outlet.claim(self.host, self.session, CLAIMED_AT, 3)
        assert [item.request_id for item in claimed] == [TASK_ID]

    def lane(self, bodies: list[dict[str, Any]], *, attach: bool = True) -> tuple[SessionLane, ScriptedProvider]:
        session, provider = lane(self.behavior, bodies)
        if attach:
            session.attach_reminders(ReminderResponses(self.ledger, limit=5))
        return session, provider

    def observe(self, messages: tuple[ConversationMessage, ...]) -> None:
        """同一份会话先过出口（真实翻译 + 真实 observe）。"""

        self.outlet.observe(self.session, tuple(host_message(item) for item in messages), host=self.host)

    def turns(self, messages: tuple[ConversationMessage, ...]) -> tuple[SessionTurn, ...]:
        """再按组合根的真实翻译拆轮。"""

        tools = answered_question_tools(ConversationAdapterRegistry.with_builtins())(self.protocol)
        return split_turns(self.session, tuple(_message(item) for item in messages), question_tools=tools)

    def task_responses(self) -> list[tuple[str, str]]:
        return [(item.polarity, item.turn) for item in self.ledger.responses_on(DAY) if item.expectation_id == TASK_ID]


# ── 造会话源消息 ───────────────────────────────────────────────────────────────


def at(minutes: float) -> datetime:
    return CLAIMED_AT + timedelta(minutes=minutes)


def said(sequence: int, text: str, minutes: float, *, mid_turn: bool = False) -> ConversationMessage:
    return ConversationMessage(
        message_id=f"m-{sequence}",
        sequence=sequence,
        role=ConversationMessageRole.PROMPT,
        occurred_at=at(minutes),
        content=text,
        mid_turn=mid_turn,
    )


def answered(sequence: int, text: str, minutes: float) -> ConversationMessage:
    return ConversationMessage(
        message_id=f"m-{sequence}",
        sequence=sequence,
        role=ConversationMessageRole.COMPLETION,
        occurred_at=at(minutes),
        content=text,
    )


def called(sequence: int, tool: str, arguments: object, call_id: str, minutes: float) -> ConversationMessage:
    return ConversationMessage(
        message_id=f"m-{sequence}",
        sequence=sequence,
        role=ConversationMessageRole.TOOL_CALL,
        occurred_at=at(minutes),
        content=arguments,
        tool_call_id=call_id,
        tool_name=tool,
    )


def returned(sequence: int, tool: str, text: str, call_id: str, minutes: float) -> ConversationMessage:
    return ConversationMessage(
        message_id=f"m-{sequence}",
        sequence=sequence,
        role=ConversationMessageRole.TOOL_RESULT,
        occurred_at=at(minutes),
        content=text,
        tool_call_id=call_id,
        tool_name=tool,
        tool_status=ConversationToolResultStatus.COMPLETED,
        content_mode=ConversationToolResultContentMode.INLINE,
    )


def claude_question(*questions: tuple[str, tuple[str, ...]]) -> dict[str, Any]:
    return {
        "questions": [
            {"question": title, "header": "提醒", "multiSelect": False, "options": [{"label": o} for o in options]}
            for title, options in questions
        ]
    }


def claude_answers(*pairs: tuple[str, str]) -> str:
    joined = ", ".join(f'"{title}"="{answer}"' for title, answer in pairs)
    return f"User has answered your questions: {joined}. You can now continue with the user's answers in mind."


def claude_round(question: dict[str, Any], answer: str) -> tuple[ConversationMessage, ...]:
    """Claude Code 的一轮：他开口、agent 用 AskUserQuestion 问（同步，作答在工具结果里）、再接着干、最后答复。"""

    return (
        said(0, "帮我看看这个报错", 0.1),
        called(1, "AskUserQuestion", question, "toolu_1", 0.5),
        returned(2, "AskUserQuestion", answer, "toolu_1", 1.5),
        called(3, "Bash", "uv pip list | grep httpx", "toolu_2", 2),
        returned(4, "Bash", "httpx 0.27.0", "toolu_2", 2.2),
        answered(5, "看完了，是依赖版本不对。", 4),
    )


def codex_question(*questions: tuple[str, tuple[str, ...]]) -> dict[str, Any]:
    return {"questions": [{"title": title, "options": list(options)} for title, options in questions]}


def desktop_reply(sequence: int, call_id: str, answer: str, minutes: float, *, mid_turn: bool) -> ConversationMessage:
    """Codex Desktop 晚来的作答：一条用户消息，标签里是 JSON 数组（与 tests/unit/behavior/test_session_lane.py 同形）。"""

    body = [
        {
            "questionItemId": json.dumps(["request_user_input_async", call_id, 0]),
            "question": TASK_TEXT,
            "answer": answer,
        }
    ]
    content = f"{QUESTION_REPLY_TAG}\n{json.dumps(body, ensure_ascii=False)}\n</send_user_message_question_reply>\n"
    return said(sequence, content, minutes, mid_turn=mid_turn)


def desktop_mid_turn(answer: str) -> tuple[ConversationMessage, ...]:
    """Codex Desktop：agent 用 request_user_input_async 问完接着干，他在这一轮中间作答（mid_turn），agent 做完答复。"""

    return (
        said(0, "帮我看看这个报错", 0.1),
        called(1, "request_user_input_async", codex_question((TASK_TEXT, OPTIONS)), "call_a1", 0.5),
        returned(2, "request_user_input_async", '{"accepted":true}', "call_a1", 0.6),
        called(3, "exec_command", '{"cmd": "uv pip list"}', "call_b1", 1),
        returned(4, "exec_command", "httpx 0.27.0", "call_b1", 1.2),
        desktop_reply(5, "call_a1", answer, 2, mid_turn=True),
        answered(6, "看完了，是依赖版本不对。", 4),
    )


def assert_parcel_still_listed(prompt: str) -> None:
    """【还开着的提醒】不受影响：答过"知道了"的快递照样列着。"""

    assert "【还开着的提醒】" in prompt
    assert PARCEL_TEXT in prompt


def assert_task_question_absent(prompt: str, *, tool: str) -> None:
    """出口写过回话的那次提问：题目原文、他的作答、那次调用都不在材料里。"""

    assert TASK_TEXT not in prompt
    assert "已经做过了" not in prompt
    assert tool not in prompt


# ── 场景 1：Claude Code 同步提问，出口按固定选项写过回话 ─────────────────────────────


def test_1_a_claude_question_the_outlet_answered_is_not_shown_to_the_model(tmp_path: Path) -> None:
    """场景 1：Claude Code 的 AskUserQuestion 问了我们那条提醒，他点了"已经做过了"，出口已写回话「已做」。交给模型的材料里
    没有那道题的原文、没有他的作答、没有那次调用；另一条答过"知道了"的快递提醒照样列在【还开着的提醒】里。"""

    world = World(tmp_path, CLAUDE)
    messages = claude_round(claude_question((TASK_TEXT, OPTIONS)), claude_answers((TASK_TEXT, "已经做过了")))
    world.observe(messages)
    assert world.task_responses() == [("已做", "toolu_1")]  # 前提：出口真的写过回话
    session, provider = world.lane([JUDGED])

    (outcome,) = asyncio.run(session.consume(world.turns(messages)))

    assert outcome.disposition is TurnDisposition.RECORDED and provider.calls == 1
    prompt = provider.prompts[-1]
    assert "他说：帮我看看这个报错" in prompt
    assert "Bash uv pip list | grep httpx" in prompt  # 这一轮别的事照旧
    assert "助手最后说：看完了，是依赖版本不对。" in prompt
    assert_task_question_absent(prompt, tool="AskUserQuestion")
    assert_parcel_still_listed(prompt)


# ── 场景 2：他对我们的提醒题自己打字作答 ──────────────────────────────────────────


def test_2_a_free_text_answer_to_our_question_still_goes_to_the_model(tmp_path: Path) -> None:
    """场景 2：同一道提醒题，他没点固定选项、自己打了"下周再弄吧"——出口不写回话；这次问答照旧进材料、交给模型判，
    这条提醒也照旧作为开着的提醒带给模型。"""

    world = World(tmp_path, CLAUDE)
    messages = claude_round(claude_question((TASK_TEXT, OPTIONS)), claude_answers((TASK_TEXT, "下周再弄吧")))
    world.observe(messages)
    assert world.task_responses() == []  # 前提：出口没写回话
    session, provider = world.lane([JUDGED])

    (outcome,) = asyncio.run(session.consume(world.turns(messages)))

    assert outcome.disposition is TurnDisposition.RECORDED
    prompt = provider.prompts[-1]
    assert f'{ANSWER_PREFIX}User has answered your questions: "{TASK_TEXT}"="下周再弄吧"' in prompt
    assert "AskUserQuestion" in prompt
    assert_parcel_still_listed(prompt)
    assert prompt.count(TASK_TEXT) >= 2  # 作答里一次，开着的提醒里一次


# ── 场景 3：agent 自己发起的提问 ───────────────────────────────────────────────────


def test_3_the_agents_own_question_still_goes_to_the_model(tmp_path: Path) -> None:
    """场景 3：这一轮 agent 用 AskUserQuestion 问的是它自己的事（选项不是我们那四个），出口不配；这次问答照旧进材料。"""

    world = World(tmp_path, CLAUDE)
    messages = claude_round(claude_question((OWN_TITLE, OWN_OPTIONS)), claude_answers((OWN_TITLE, "SQLite")))
    world.observe(messages)
    assert world.task_responses() == []
    session, provider = world.lane([JUDGED])

    (outcome,) = asyncio.run(session.consume(world.turns(messages)))

    assert outcome.disposition is TurnDisposition.RECORDED
    prompt = provider.prompts[-1]
    assert f'{ANSWER_PREFIX}User has answered your questions: "{OWN_TITLE}"="SQLite"' in prompt
    assert "AskUserQuestion" in prompt
    assert_parcel_still_listed(prompt)


# ── 场景 4：一次调用里同时有我们的题和 agent 自己的题 ─────────────────────────────


def test_4_a_mixed_call_whose_reminder_the_outlet_answered_is_left_out_whole(tmp_path: Path) -> None:
    """场景 4：一次 AskUserQuestion 里两题——我们的提醒题（他点了"已经做过了"，出口写过回话）和 agent 自己的题（他选了
    SQLite）。整次调用连同作答一起不进材料（用户已接受 agent 那题跟着丢）。"""

    world = World(tmp_path, CLAUDE)
    messages = claude_round(
        claude_question((TASK_TEXT, OPTIONS), (OWN_TITLE, OWN_OPTIONS)),
        claude_answers((TASK_TEXT, "已经做过了"), (OWN_TITLE, "SQLite")),
    )
    world.observe(messages)
    assert world.task_responses() == [("已做", "toolu_1")]
    session, provider = world.lane([JUDGED])

    (outcome,) = asyncio.run(session.consume(world.turns(messages)))

    assert outcome.disposition is TurnDisposition.RECORDED
    prompt = provider.prompts[-1]
    assert_task_question_absent(prompt, tool="AskUserQuestion")
    assert OWN_TITLE not in prompt and "SQLite" not in prompt
    assert "助手最后说：看完了，是依赖版本不对。" in prompt
    assert_parcel_still_listed(prompt)


# ── 场景 5：Codex Desktop 异步提问 ─────────────────────────────────────────────────


def test_5a_a_desktop_answer_mid_turn_the_outlet_answered_is_not_shown(tmp_path: Path) -> None:
    """场景 5（作答插在一轮中间）：Codex Desktop 用 request_user_input_async 问了我们那条，他在这一轮中间点了"已经做过了"
    （mid_turn 用户消息），出口经晚到作答写过回话。材料里没有那道题、没有作答、没有那次调用；这一轮别的事照旧。"""

    world = World(tmp_path, CODEX)
    messages = desktop_mid_turn("已经做过了")
    world.observe(messages)
    assert world.task_responses() == [("已做", "call_a1")]
    session, provider = world.lane([JUDGED])

    (outcome,) = asyncio.run(session.consume(world.turns(messages)))

    assert outcome.disposition is TurnDisposition.RECORDED and provider.calls == 1
    prompt = provider.prompts[-1]
    assert "他说：帮我看看这个报错" in prompt
    assert "exec_command" in prompt
    assert "助手最后说：看完了，是依赖版本不对。" in prompt
    assert_task_question_absent(prompt, tool="request_user_input_async")
    assert_parcel_still_listed(prompt)


def test_5b_a_desktop_answer_that_opens_its_own_turn_still_records_the_turn_without_the_answer(
    tmp_path: Path,
) -> None:
    """场景 5（作答在上一轮答完后才到、自己开一轮）：上一轮 agent 问了我们那条就答完了；他回来点了"已经做过了"，宿主为它
    开了新的一轮，助手接着升级依赖。出口写过回话：两轮的材料里都不出现那道题、作答和那次调用；作答开的那一轮照常出记录、
    照常问模型（助手接着做的事要描述）。"""

    world = World(tmp_path, CODEX)
    messages = (
        said(0, "帮我看看这个报错", 0.1),
        called(1, "request_user_input_async", codex_question((TASK_TEXT, OPTIONS)), "call_a1", 0.5),
        returned(2, "request_user_input_async", '{"accepted":true}', "call_a1", 0.6),
        answered(3, "报错是依赖版本不对。", 2),
        desktop_reply(4, "call_a1", "已经做过了", 6, mid_turn=False),
        called(5, "exec_command", '{"cmd": "uv pip install -U httpx"}', "call_b1", 6.5),
        returned(6, "exec_command", "Successfully installed httpx-0.28.1", "call_b1", 7),
        answered(7, "好，顺手把 httpx 升到了 0.28.1。", 8),
    )
    world.observe(messages)
    assert world.task_responses() == [("已做", "call_a1")]
    session, provider = world.lane([JUDGED, JUDGED])
    turns = world.turns(messages)

    outcomes = asyncio.run(session.consume(turns))

    assert [item.disposition for item in outcomes] == [TurnDisposition.RECORDED, TurnDisposition.RECORDED]
    assert provider.calls == 2
    asked, after = provider.prompts
    assert "他说：帮我看看这个报错" in asked and "报错是依赖版本不对。" in asked
    assert "uv pip install -U httpx" in after
    assert "好，顺手把 httpx 升到了 0.28.1。" in after
    for prompt in (asked, after):
        assert_task_question_absent(prompt, tool="request_user_input_async")
    assert_parcel_still_listed(after)


def test_5c_a_free_text_desktop_answer_still_goes_to_the_model(tmp_path: Path) -> None:
    """场景 5（自己打字）：Codex Desktop 那道提醒题他自己打了"下周再弄吧"，出口不写回话；作答照旧进材料、交给模型判。"""

    world = World(tmp_path, CODEX)
    messages = desktop_mid_turn("下周再弄吧")
    world.observe(messages)
    assert world.task_responses() == []
    session, provider = world.lane([JUDGED])

    (outcome,) = asyncio.run(session.consume(world.turns(messages)))

    assert outcome.disposition is TurnDisposition.RECORDED
    prompt = provider.prompts[-1]
    assert f"{ANSWER_PREFIX}{TASK_TEXT} → 下周再弄吧" in prompt
    assert "request_user_input_async" in prompt
    assert_parcel_still_listed(prompt)


# ── 场景 6：这一轮的记录身份不变 ──────────────────────────────────────────────────


def test_6_the_turn_keeps_its_identity_and_a_replay_is_already_recorded(tmp_path: Path) -> None:
    """场景 6：出口写过回话的那一轮，记录身份（轮的范围与时刻）与原来一样；同一份会话重放（先过出口、再进 lane）仍认作
    "已记录"、不重问模型、不多写判断。"""

    world = World(tmp_path, CLAUDE)
    messages = claude_round(claude_question((TASK_TEXT, OPTIONS)), claude_answers((TASK_TEXT, "已经做过了")))
    world.observe(messages)
    session, provider = world.lane([JUDGED, {**description("重放时换了个说法"), "responses": []}])

    (first,) = asyncio.run(session.consume(world.turns(messages)))
    assert first.disposition is TurnDisposition.RECORDED
    assert first.turn.key == f"{world.session}#0-5"
    assert (first.turn.instructed_at, first.turn.completed_at) == (at(0.1), at(4))

    world.observe(messages)
    (again,) = asyncio.run(session.consume(world.turns(messages)))

    assert again.disposition is TurnDisposition.ALREADY_RECORDED
    assert again.turn.key == first.turn.key
    assert provider.calls == 1
    assert len(world.behavior.judgements.list()) == 1
    assert world.task_responses() == [("已做", "toolu_1")]


# ── 场景 7：写成「错误」、夜里重试 ─────────────────────────────────────────────────


def _night_retry(world: World, messages: tuple[ConversationMessage, ...]) -> ScriptedProvider:
    """第一次模型答的格式两次都不合 → 这一轮写成「错误」、进重试堆；夜里经 ``TurnHandler``（轮经 encode/decode 进出堆）再试。"""

    bad = description(name="..")
    session, provider = world.lane([bad, bad, JUDGED])
    (outcome,) = asyncio.run(session.consume(world.turns(messages)))
    assert outcome.disposition is TurnDisposition.ERROR
    (item,) = world.behavior.backlog.snapshot().items
    calls_before = provider.calls
    assert asyncio.run(TurnHandler(session).attempt(item)).name == "RECOVERED"
    assert provider.calls == calls_before + 1
    return provider


def test_7a_a_night_retry_of_a_claude_turn_also_leaves_the_answered_question_out(tmp_path: Path) -> None:
    """场景 7（Claude Code）：出口写过回话的那一轮第一次写成「错误」，夜里重试再描述时，交给模型的材料同样不含那次问答。"""

    world = World(tmp_path, CLAUDE)
    messages = claude_round(claude_question((TASK_TEXT, OPTIONS)), claude_answers((TASK_TEXT, "已经做过了")))
    world.observe(messages)
    assert world.task_responses() == [("已做", "toolu_1")]

    provider = _night_retry(world, messages)

    retried = provider.prompts[-1]
    assert "他说：帮我看看这个报错" in retried
    assert_task_question_absent(retried, tool="AskUserQuestion")
    assert_parcel_still_listed(retried)


def test_7b_a_night_retry_of_a_desktop_turn_also_leaves_the_answered_question_out(tmp_path: Path) -> None:
    """场景 7（Codex Desktop，作答插在一轮中间）：同上；轮进出重试堆时消息只留编码过的那几个字段，重试的材料同样不含
    出口记过的那次问答。"""

    world = World(tmp_path, CODEX)
    messages = desktop_mid_turn("已经做过了")
    world.observe(messages)
    assert world.task_responses() == [("已做", "call_a1")]

    provider = _night_retry(world, messages)

    retried = provider.prompts[-1]
    assert "他说：帮我看看这个报错" in retried
    assert_task_question_absent(retried, tool="request_user_input_async")
    assert_parcel_still_listed(retried)


# ── 场景 8：没接预测层 ────────────────────────────────────────────────────────────


def test_8_without_the_foresight_layer_the_material_is_unchanged(tmp_path: Path) -> None:
    """场景 8：lane 没 attach_reminders（没接预测层）时，材料与现在一致——那次问答、那次调用照旧都在，没有【还开着的提醒】。
    （出口照样跑过、账本里有回话，lane 不认识它。）"""

    world = World(tmp_path, CLAUDE)
    messages = claude_round(claude_question((TASK_TEXT, OPTIONS)), claude_answers((TASK_TEXT, "已经做过了")))
    world.observe(messages)
    assert world.task_responses() == [("已做", "toolu_1")]
    session, provider = world.lane([description()], attach=False)

    (outcome,) = asyncio.run(session.consume(world.turns(messages)))

    assert outcome.disposition is TurnDisposition.RECORDED
    question = (
        '{"questions":[{"header":"提醒","multiSelect":false,"options":[{"label":"现在就做"},{"label":"这次不用"},'
        '{"label":"知道了"},{"label":"已经做过了"}],"question":"往常会确认任务是否完成，这次也确认一下。"}]}'
    )
    expected = "\n".join(
        (
            "【这一轮】2026-10-22 15:52",
            "他说：帮我看看这个报错",
            "助手干活时他又说（按先后）：",
            f"  - {ANSWER_PREFIX}{claude_answers((TASK_TEXT, '已经做过了'))}",
            "助手调用：",
            f"  - AskUserQuestion {question[:120]}",  # 调用参数只留开头 120 字（SessionLaneConfig.tool_argument_chars）
            "  - Bash uv pip list | grep httpx",
            "助手最后说：看完了，是依赖版本不对。",
        )
    )
    assert provider.prompts[-1] == expected
