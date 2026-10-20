"""提醒出口包在会话 lane 消费者外面（方案 32 第 4.2 条）：同一份会话先过出口、再进 lane。

走真实的交付契约、真实的行为侧存储、真实的账本与出口；模型用照脚本回答的替身。一份含 agent 提问与他作答的会话跑完：
账本有「说了」（时刻 = 提问那一刻）；他点"这次不用"→ 出口直接写负、作答那一轮桥不再带这条；点"知道了"→ 出口写中、
作答那一轮桥仍带着它给模型。把 observe 挪到 lane 之后，第二条就会失败（作答那一轮进桥时这条提醒还没"开着"）。
"""

from __future__ import annotations

import asyncio
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from habitus.conversation import (
    ConversationConsumerOutcomeState,
    ConversationSourceConsumer,
    ConversationSourceEnvelope,
    conversation_source_request_digest,
)
from habitus.foresight.ledger import Delivery, Expectation
from habitus.pre.conversation import (
    ConversationBatch,
    ConversationMessage,
    ConversationMessageRole,
    ConversationToolResultContentMode,
    ConversationToolResultStatus,
)
from habitus.runtime.foresight_ledger import ForesightLedgerStore
from habitus.runtime.foresight_outlet import OPTIONS, OpenRequest, ReminderOutlet
from habitus.runtime.foresight_outlet_consumer import OutletObservingConsumer
from habitus.runtime.foresight_outlet_store import OutletStore
from habitus.runtime.foresight_responses import ReminderResponses
from habitus.runtime.session_lane import BehaviorSessionConsumer
from tests.unit.behavior.session_fixtures import Stores, description, lane
from tests.unit.conversation.source_v2_helpers import delivery, stores

CST = timezone(timedelta(hours=8))
DAY = date(2026, 10, 22)
FORMED = datetime(2026, 10, 22, 14, 0, tzinfo=CST)
AWAITING_AT = FORMED + timedelta(minutes=100)  # 15:40 时窗末记「待取」
CLAIMED_AT = FORMED + timedelta(minutes=112)  # 15:52 他敲了一句话，hook 取走
TEXT = "周四下午你一般会写周报，这周的还没写。"
SESSION = "claude-s1"


def expectation() -> Expectation:
    return Expectation(
        expectation_id="e1",
        kind_token="s-k0002",
        day=DAY,
        formed_at=FORMED,
        source="日子",
        trigger="rel",
        trigger_instance="behavior://e1.md",
        window=(FORMED, FORMED + timedelta(hours=2)),
        deadline=FORMED + timedelta(hours=4),
        basis=(),
        opportunities=3,
        hits=3,
        observed_from=FORMED,
    )


def message(sequence: int, role: ConversationMessageRole, content: object, at: datetime, **extra: Any) -> ConversationMessage:
    return ConversationMessage(message_id=f"m-{sequence}", sequence=sequence, role=role, occurred_at=at, content=content, **extra)


def conversation(answer: str) -> tuple[ConversationMessage, ...]:
    question = {"questions": [{"question": TEXT, "header": "提醒", "multiSelect": False, "options": [{"label": o} for o in OPTIONS]}]}
    return (
        message(0, ConversationMessageRole.PROMPT, "帮我看看这个报错", CLAIMED_AT + timedelta(seconds=5)),
        message(
            1,
            ConversationMessageRole.TOOL_CALL,
            question,
            CLAIMED_AT + timedelta(seconds=30),
            tool_call_id="toolu_1",
            tool_name="AskUserQuestion",
        ),
        message(
            2,
            ConversationMessageRole.TOOL_RESULT,
            f'User has answered your questions: "{TEXT}"="{answer}". You can now continue.',
            CLAIMED_AT + timedelta(minutes=1),
            tool_call_id="toolu_1",
            tool_name="AskUserQuestion",
            tool_status=ConversationToolResultStatus.COMPLETED,
            content_mode=ConversationToolResultContentMode.INLINE,
        ),
        message(3, ConversationMessageRole.COMPLETION, "看完了，是依赖版本不对。", CLAIMED_AT + timedelta(minutes=4)),
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
        recorded_at=CLAIMED_AT + timedelta(minutes=5),
    )


def wired(tmp_path: Path, bodies: list[dict[str, Any]]):  # type: ignore[no-untyped-def]
    conversation_root = tmp_path / "conversation"
    _sources, _outcomes, _memory_outputs, behavior_outputs = stores(conversation_root)
    behavior = Stores(tmp_path / "behavior")
    session, provider = lane(behavior, bodies)
    ledger = ForesightLedgerStore(tmp_path / "foresight")
    session.attach_reminders(ReminderResponses(ledger, limit=5))
    outlet = ReminderOutlet(
        OutletStore(ledger.root / "outlet"),
        ledger,
        liveness=timedelta(hours=48),
        lease=timedelta(minutes=30),
        max_attempts=3,
        claim_limit=3,
        similarity=0.6,
    )
    item = expectation()
    ledger.record_expectation(item)
    ledger.record_delivery(DAY, Delivery("e1", AWAITING_AT, 1, False, "待取", 1.0, TEXT))
    outlet.publish([OpenRequest("e1", item.kind_token, TEXT, item.deadline, item.formed_at, 1, DAY)], now=AWAITING_AT)
    assert [r.request_id for r in outlet.claim("claude-code", SESSION, CLAIMED_AT, 3)] == ["e1"]
    consumer = OutletObservingConsumer(BehaviorSessionConsumer(session, behavior_outputs), outlet)
    service, _memory, _behavior = delivery(conversation_root, behavior=consumer)  # type: ignore[arg-type]
    return service, ledger, provider


def run(tmp_path: Path, answer: str, second_body: dict[str, Any]):  # type: ignore[no-untyped-def]
    service, ledger, provider = wired(tmp_path, [description("看报错"), second_body])

    async def scenario() -> None:
        source = service.sources.put(envelope(conversation(answer)))
        ensured = await service.ensure_outcome(source, ConversationSourceConsumer.BEHAVIOR_SESSION)
        assert ensured.outcome.state is ConversationConsumerOutcomeState.COMMITTED

    asyncio.run(scenario())
    return ledger, provider


def test_a_fixed_no_is_written_by_the_outlet_and_the_answer_turn_no_longer_carries_the_reminder(tmp_path: Path) -> None:
    ledger, provider = run(tmp_path, "这次不用", description("回答提问"))
    (spoken,) = [d for d in ledger.deliveries_on(DAY) if d.spoken]
    assert (spoken.decided_at, spoken.text, spoken.via, spoken.turn) == (CLAIMED_AT + timedelta(seconds=30), TEXT, "claude-code", "toolu_1")
    (response,) = ledger.responses_on(DAY)
    assert (response.polarity, response.turn) == ("负", "toolu_1")
    assert provider.calls == 2  # 两轮：他的话那一轮、作答那一轮
    assert "【还开着的提醒】" not in provider.prompts[1]  # 负已经关了，作答那一轮模型不再见到这条
    assert ReminderResponses(ledger, limit=5).open(CLAIMED_AT + timedelta(minutes=2)) == ()


def test_a_fixed_noted_leaves_the_reminder_open_so_the_answer_turn_still_carries_it(tmp_path: Path) -> None:
    ledger, provider = run(tmp_path, "知道了", description("回答提问", responses=[]))
    (response,) = ledger.responses_on(DAY)
    assert response.polarity == "中"
    assert provider.calls == 2
    # 出口先于 lane：作答那一轮进桥时这条提醒已经"开着"（说出口的时刻 = 提问那一刻，早于他作答），模型见得到
    assert "【还开着的提醒】" in provider.prompts[1] and TEXT in provider.prompts[1]
    assert "【还开着的提醒】" not in provider.prompts[0]  # 他敲第一句话时还没问出口
    payload = json.loads(provider.prompts[1]) if provider.prompts[1].startswith("{") else None
    assert payload is None  # 材料是文字不是 JSON（只是确认提示词形状没变）
