"""他回"已经做过了"：提醒反馈与账两边都中性（裁定 91 第 3 条②，落实裁定 54 第 3 条③；由 ``37-材料/readme/repro-91`` 的复现
脚本改成，断言已翻转）。

链路同 ``test_reminder_outlet_chain.py``（时窗末「待取」→ 伪宿主取走 → 转写里 agent 用 AskUserQuestion 问、他作答 → 出口记
「说了」→ 这一轮带着开着的提醒进会话 lane → 结算写反馈），结算按组合根接法（``runtime/foresight.py``：带 ``opportunities``、
``tolerance``、``history``）。他早上真做了，但那次系统没看见（行为树上没有这一次）：

- 点第四个固定选项"已经做过了"：出口直接写「已做」（不经模型、关闭提醒，这一轮不再带给模型）；
- 自己填"早上浇过了"：交会话 lane 判；替身照提示词 v7 第二问"已做：他说这一次已经做过了"判「已做」（真实模型的判法见
  ``37-材料/readme/fix-91/report.md`` 的探针）；
- 到时限没看到做：结算过期待决，账上那一笔中性（强度 0、不算机会），提醒反馈不写，单元历史那一刻看不到；
- 对照：点"现在就做"却到时限也没做 → 坏（裁定 61 第 4 条）；回"已经做过了"而树上看到他在被问之后做了 → 照常「提醒后发生」、好。
"""

from __future__ import annotations

import asyncio
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pytest

from habitus.conditions import LocalCalendar
from habitus.conversation import (
    ConversationConsumerOutcomeState,
    ConversationSourceConsumer,
    ConversationSourceEnvelope,
    conversation_source_request_digest,
)
from habitus.foresight import Evidence, Tolerance
from habitus.foresight.clock import is_opportunity
from habitus.foresight.history import record_of
from habitus.foresight.ledger import Expectation
from habitus.pre.conversation import (
    ConversationBatch,
    ConversationMessage,
    ConversationMessageRole,
    ConversationToolResultContentMode,
    ConversationToolResultStatus,
)
from habitus.runtime.foresight import ForesightReadSide
from habitus.runtime.foresight_delivery import DeliveryStage, JudgeMaterials
from habitus.runtime.foresight_ledger import ForesightLedgerStore
from habitus.runtime.foresight_opportunities import OpportunityStore
from habitus.runtime.foresight_outlet import OPTIONS, ReminderOutlet
from habitus.runtime.foresight_outlet_consumer import OutletObservingConsumer
from habitus.runtime.foresight_outlet_store import OutletStore
from habitus.runtime.foresight_responses import ReminderResponses
from habitus.runtime.foresight_settlement import SettlementStage
from habitus.runtime.session_lane import BehaviorSessionConsumer
from tests.unit.behavior.session_fixtures import Stores, description, lane
from tests.unit.conversation.source_v2_helpers import delivery, stores
from tests.unit.kind_ids import kind_id
from tests.unit.runtime.test_foresight_delivery import Seeded
from tests.unit.scene.fixtures import SUBJECT, Site, at, publish

DAY = date(2026, 12, 10)
HOTEL = kind_id("订酒店")
SESSION = "claude-s1"
HOST = "claude-code"


def expectation() -> Expectation:
    formed = at(DAY, 14, 7)
    return Expectation(
        expectation_id="hotel",
        kind_token=HOTEL,
        day=DAY,
        formed_at=formed,
        source="日子",
        trigger="rel",
        trigger_instance="behavior://hotel.md",
        window=(formed + timedelta(minutes=4), formed + timedelta(minutes=51)),
        deadline=formed + timedelta(hours=3),  # 17:07
        basis=(),
        opportunities=8,
        hits=8,
        observed_from=formed,
    )


def message(
    sequence: int, role: ConversationMessageRole, content: object, when: datetime, **extra: Any
) -> ConversationMessage:
    return ConversationMessage(
        message_id=f"m-{sequence}", sequence=sequence, role=role, occurred_at=when, content=content, **extra
    )


def transcript(text: str, answer: str, asked_at: datetime) -> tuple[ConversationMessage, ...]:
    question = {
        "questions": [
            {"question": text, "header": "提醒", "multiSelect": False, "options": [{"label": o} for o in OPTIONS]}
        ]
    }
    return (
        message(0, ConversationMessageRole.PROMPT, "帮我看看这个报错", asked_at - timedelta(seconds=20)),
        message(
            1,
            ConversationMessageRole.TOOL_CALL,
            question,
            asked_at,
            tool_call_id="toolu_1",
            tool_name="AskUserQuestion",
        ),
        message(
            2,
            ConversationMessageRole.TOOL_RESULT,
            f'User has answered your questions: "{text}"="{answer}". You can now continue.',
            asked_at + timedelta(seconds=40),
            tool_call_id="toolu_1",
            tool_name="AskUserQuestion",
            tool_status=ConversationToolResultStatus.COMPLETED,
            content_mode=ConversationToolResultContentMode.INLINE,
        ),
        message(3, ConversationMessageRole.COMPLETION, "看完了，是依赖版本不对。", asked_at + timedelta(minutes=3)),
    )


def envelope(messages: tuple[ConversationMessage, ...]) -> ConversationSourceEnvelope:
    batch = ConversationBatch(SESSION, messages)
    digest = conversation_source_request_digest(
        conversation_id=SESSION,
        started_on=DAY,
        protocol="claude_code",
        batch=batch,
        after_turn=True,
        omit_tool_call_ids=frozenset(),
    )
    return ConversationSourceEnvelope.create(
        conversation_id=SESSION,
        started_on=DAY,
        protocol="claude_code",
        batch=batch,
        after_turn=True,
        omit_tool_call_ids=frozenset(),
        delivery_id=digest,
        request_digest=digest,
        recorded_at=messages[-1].occurred_at,
    )


SAMPLES = (
    # 作答, 做没做, 作答那一轮模型的回应, 预期结算结果, 预期反馈（好/坏/None）, 账上的回话方向, 账上那一笔（种类, 好, 强度）
    ("已经做过了", None, None, "过期待决", None, "已做", ("过期待决", False, 0.0)),
    ("早上浇过了", None, [{"reminder": 1, "polarity": "已做"}], "过期待决", None, "已做", ("过期待决", False, 0.0)),
    ("现在就做", None, None, "过期待决", False, "正", ("过期待决", False, 1.0)),
    ("已经做过了", (15, 40), None, "提醒后发生", True, "已做", ("提醒后发生", True, 0.0)),
)


@pytest.mark.parametrize(("answer", "done", "judged", "outcome", "feedback_good", "polarity", "account"), SAMPLES)
def test_an_already_done_reply_is_neutral_on_both_sides(  # noqa: PLR0913 - 四条样例的七个维度
    tmp_path: Path,
    answer: str,
    done: tuple[int, int] | None,
    judged: list[dict[str, Any]] | None,
    outcome: str,
    feedback_good: bool | None,
    polarity: str,
    account: tuple[str, bool, float],
) -> None:
    site = Site(tmp_path, now=at(DAY, 23, 0))
    ledger = ForesightLedgerStore(tmp_path / "foresight")
    seeded = Seeded(tmp_path / "foresight", ledger)
    seeded.earn(kind=HOTEL)  # 这条边攒够了把握：提醒档
    outlet = ReminderOutlet(
        OutletStore(ledger.root / "outlet"),
        ledger,
        liveness=timedelta(hours=48),
        lease=timedelta(minutes=30),
        max_attempts=3,
        claim_limit=3,
        similarity=0.6,
        kind_label=lambda token: "订酒店" if token == HOTEL else None,
    )
    tick = DeliveryStage(
        ledger,
        tolerance=Tolerance(line=2 / 3),
        evidence=Evidence(alpha=0.05, half_life=8.0),
        daily_cap=3,
        tick_seconds=900.0,
        outlet=outlet,
        history=seeded.store,
    )
    item = expectation()
    ledger.record_expectation(item)
    # 宿主之前来过（活性）：他上午在 Codex 里敲过话，hook 取过一次（什么也没取到）
    assert outlet.claim(HOST, SESSION, at(DAY, 9, 0), 3) == ()
    materials = JudgeMaterials(None, lambda uris: {})
    assert asyncio.run(tick.run_once(at(DAY, 14, 15), materials)).by_reason == {"等时窗末": 1}
    closing = asyncio.run(tick.run_once(at(DAY, 15, 0), materials))
    assert closing.by_reason == {"待取": 1} and closing.open_requests == 1  # 时窗末还没做、有宿主活着：话写好、等取
    (awaiting,) = [d for d in ledger.deliveries_on(DAY) if d.reason == "待取"]
    assert awaiting.text
    # 15:10 他敲了一句话：hook 取走
    (claimed,) = outlet.claim(HOST, SESSION, at(DAY, 15, 10), 3)
    assert claimed.text == awaiting.text and claimed.kind == "订酒店"
    # agent 先问、他作答、Stop 把转写送回：先过出口，再进会话 lane
    conversation_root = tmp_path / "conversation"
    _s, _o, _m, behavior_outputs = stores(conversation_root)
    behavior = Stores(tmp_path / "behavior")
    # 一轮 = 宿主的一轮：他开口、agent 先问、他作答、agent 接着答完，是同一轮、模型只看一次（用户 2026-10-22）
    only = description("看报错") if judged is None else description("看报错", responses=judged)
    session, provider = lane(behavior, [only])
    session.attach_reminders(ReminderResponses(ledger, limit=5))
    consumer = OutletObservingConsumer(BehaviorSessionConsumer(session, behavior_outputs), outlet)
    service, _memory, _behavior = delivery(conversation_root, behavior=consumer)  # type: ignore[arg-type]
    asked_at = at(DAY, 15, 11)

    async def deliver() -> None:
        source = service.sources.put(envelope(transcript(awaiting.text or "", answer, asked_at)))
        ensured = await service.ensure_outcome(source, ConversationSourceConsumer.BEHAVIOR_SESSION)
        assert ensured.outcome.state is ConversationConsumerOutcomeState.COMMITTED

    asyncio.run(deliver())
    (spoken,) = [d for d in ledger.deliveries_on(DAY) if d.spoken]
    assert (spoken.decided_at, spoken.via, spoken.turn) == (asked_at, HOST, "toolu_1")
    (response,) = ledger.responses_on(DAY)
    assert response.polarity == polarity
    assert len(provider.prompts) == 1
    # 出口先按固定选项写了回话：正 / 负关了、这一轮不再带；中（知道了）与自己填的还开着，这一轮带着、交给模型判
    assert ("【还开着的提醒】" in provider.prompts[0]) == (judged is not None)
    # 他那天开过会话（15:11 这一轮「看报错」，这里的会话 lane 写在另一棵树上，补一条到结算读的树）：会话 lane 一整天没有会话
    # 才算没在看（裁定 89 第 2 条），这里要的是"在看、没做"
    publish(site.behavior_tree, DAY, "看报错", 15, 11, kind="看报错")
    if done is not None:
        publish(site.behavior_tree, DAY, "订酒店", *done, kind="订酒店")
    # 之后每拍：说出口了就不再动它；过了最晚时限也不记「宿主没取」
    assert asyncio.run(tick.run_once(at(DAY, 15, 30), materials)).decided == 0
    assert asyncio.run(tick.run_once(at(DAY, 17, 15), materials)).decided == 0
    # 结算（最晚时限那天定稿之后）：按做没做 + 回话写提醒反馈（方案 30 第四节）
    settlement = SettlementStage(
        ledger,
        ForesightReadSide(site.behavior_tree, SUBJECT, LocalCalendar(), ZoneInfo("Asia/Shanghai")),
        closed_days=lambda: (DAY,),
        current=lambda token: (token,),
        migrating=lambda: False,
        evidence=Evidence(alpha=0.05, half_life=8.0),
        opportunities=OpportunityStore(tmp_path / "foresight"),
        lane="session",
        clock=lambda: at(DAY + timedelta(days=1), 4, 0).astimezone(UTC),
        tolerance=Tolerance(line=2 / 3),
        history=seeded.store,
    )
    report = settlement.run_once()
    assert report.outcomes == {outcome: 1}
    feedback = ledger.feedback_of(HOTEL)
    if feedback_good is None:
        assert feedback == ()  # 回了中又没做：不写、不算机会
    else:
        (fact,) = feedback
        assert (fact.good, fact.polarity, fact.expectation_id) == (feedback_good, polarity, "hotel")
    # 预测轴（账）上同一笔：回了已做又没看到做是中性（强度 0、不算机会），不是一笔坏（裁定 54 第 3 条③）
    (account_fact,) = [f for f in ledger.facts_of(item.account) if f.expectation_id == "hotel"]
    assert (account_fact.kind, account_fact.good, account_fact.strength) == account
    assert is_opportunity(account_fact) == (account != ("过期待决", False, 0.0))
    # 单元历史那一刻：中性的看不到（came=None），不当落空
    record = record_of(item, account_fact)
    assert record.came is (None if account_fact.strength == 0.0 else account_fact.good)
    # 中性的不进待裁：之后补看到做了也不改判成后补
    assert settlement.run_once().rulings == 0
