"""会话源 → 会话 lane 这座桥：行为侧在会话源上的消费者，走真实的交付契约与真实的行为侧存储。

模型用照脚本回答的替身；其余（会话源存储、终态、租约、恢复、判断与回执存储）都是真的。
"""

from __future__ import annotations

import asyncio
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from habitus.behavior.fusion.lanes.session import SessionMessageRole
from habitus.conversation import (
    BehaviorSessionOutput,
    ConversationConsumerDeliveryState,
    ConversationConsumerOutcomeState,
    ConversationSourceConsumer,
    ConversationSourceEnvelope,
    ConversationSourceRecovery,
    conversation_source_request_digest,
)
from habitus.model_client.contracts import ModelStructuredOutputError
from habitus.pre.conversation import ConversationBatch, ConversationMessage, ConversationToolResultStatus
from habitus.runtime.session_lane import (
    SKIP_BEHAVIOR_DISABLED,
    SKIP_NO_TURNS,
    SKIP_SUBAGENT,
    BehaviorSessionConsumer,
    session_turns,
)
from tests.helpers import closed_turn, tool_turn
from tests.unit.behavior.session_fixtures import Stores, description, lane
from tests.unit.conversation.source_v2_helpers import delivery, stores

STARTED_ON = date(2026, 7, 1)


def envelope(conversation_id: str, messages: tuple[ConversationMessage, ...]) -> ConversationSourceEnvelope:
    batch = ConversationBatch(conversation_id, messages)
    request_digest = conversation_source_request_digest(
        conversation_id=conversation_id,
        started_on=STARTED_ON,
        protocol="normalized",
        batch=batch,
        after_turn=True,
        omit_tool_call_ids=frozenset(),
    )
    return ConversationSourceEnvelope.create(
        conversation_id=conversation_id,
        started_on=STARTED_ON,
        protocol="normalized",
        batch=batch,
        after_turn=True,
        omit_tool_call_ids=frozenset(),
        delivery_id=request_digest,
        request_digest=request_digest,
        recorded_at=datetime(2026, 7, 1, 9, 0, tzinfo=UTC),
    )


def wired(tmp_path: Path, bodies: list[dict[str, Any]] | None):  # type: ignore[no-untyped-def]
    """一套接好的交付：``bodies`` 为 None 表示行为侧没启用。"""

    conversation_root = tmp_path / "conversation"
    _sources, _outcomes, _memory_outputs, behavior_outputs = stores(conversation_root)
    behavior = Stores(tmp_path / "behavior")
    if bodies is None:
        consumer, provider = BehaviorSessionConsumer(None, behavior_outputs), None
    else:
        session, provider = lane(behavior, bodies)
        consumer = BehaviorSessionConsumer(session, behavior_outputs)
    service, _memory, _behavior = delivery(conversation_root, behavior=consumer)  # type: ignore[arg-type]
    return service, behavior, provider


def test_each_turn_in_a_source_becomes_one_judgement_and_the_source_gets_a_receipt(tmp_path: Path) -> None:
    async def scenario() -> None:
        service, behavior, provider = wired(tmp_path, [description("确认回答风格"), description("检查项目状态")])
        source = service.sources.put(envelope("two-turns", (*closed_turn(), *tool_turn(start_sequence=2))))

        ensured = await service.ensure_outcome(source, ConversationSourceConsumer.BEHAVIOR_SESSION)

        assert ensured.outcome.state is ConversationConsumerOutcomeState.COMMITTED
        receipt = service.restore_terminal(source, ConversationSourceConsumer.BEHAVIOR_SESSION)
        assert isinstance(receipt, BehaviorSessionOutput)
        assert [(turn.start_sequence, turn.end_sequence, turn.recorded) for turn in receipt.turns] == [
            (0, 1, True),
            (2, 5, True),
        ]
        assert provider.calls == 2  # 每轮恰好一次模型调用
        assert sorted(item["behavior"] for item in behavior.judgements.list()) == ["检查项目状态", "确认回答风格"]
        assert all(item["relations"] == [] for item in behavior.judgements.list())
        # 回执里没有对话内容。
        assert "请记住我喜欢简洁回答" not in str(receipt.to_dict())

    asyncio.run(scenario())


def test_without_behaviour_every_source_is_skipped_and_nothing_is_written(tmp_path: Path) -> None:
    async def scenario() -> None:
        service, behavior, _provider = wired(tmp_path, None)
        source = service.sources.put(envelope("behaviour-off", closed_turn()))

        ensured = await service.ensure_outcome(source, ConversationSourceConsumer.BEHAVIOR_SESSION)

        assert ensured.outcome.state is ConversationConsumerOutcomeState.SKIPPED
        assert ensured.outcome.skip_reason == SKIP_BEHAVIOR_DISABLED
        assert behavior.judgements.list() == ()

    asyncio.run(scenario())


def test_subagent_conversations_are_not_the_person_speaking(tmp_path: Path) -> None:
    async def scenario() -> None:
        service, behavior, provider = wired(tmp_path, [description()])
        source = service.sources.put(envelope("3f2a.subagent.reviewer", closed_turn()))

        ensured = await service.ensure_outcome(source, ConversationSourceConsumer.BEHAVIOR_SESSION)

        assert ensured.outcome.skip_reason == SKIP_SUBAGENT
        assert provider.calls == 0
        assert behavior.judgements.list() == ()
        assert session_turns(source) == ()

    asyncio.run(scenario())


def test_a_source_without_any_instruction_has_no_turns(tmp_path: Path) -> None:
    async def scenario() -> None:
        service, _behavior, provider = wired(tmp_path, [description()])
        leftover = closed_turn()[1:]  # 只有助手的一条答复：上一轮剩下的
        source = service.sources.put(envelope("leftover-only", leftover))

        ensured = await service.ensure_outcome(source, ConversationSourceConsumer.BEHAVIOR_SESSION)

        assert ensured.outcome.skip_reason == SKIP_NO_TURNS
        assert provider.calls == 0

    asyncio.run(scenario())


def test_a_failed_turn_leaves_the_source_pending_and_recovery_does_not_repeat_finished_turns(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        bad = description(name="..")  # 两次都答成不合格的名字：这一轮判失败
        service, behavior, provider = wired(tmp_path, [description("确认回答风格"), bad, bad, description("检查项目状态")])
        source = service.sources.put(envelope("fails-once", (*closed_turn(), *tool_turn(start_sequence=2))))

        try:
            await service.ensure_outcome(source, ConversationSourceConsumer.BEHAVIOR_SESSION)
        except ModelStructuredOutputError:
            pass
        else:  # pragma: no cover - 上面必须抛
            raise AssertionError("the failing turn must surface")
        # 不落终态、不写空白；第一轮已经写下的判断留着。
        assert service.inspect(source, ConversationSourceConsumer.BEHAVIOR_SESSION).state is (
            ConversationConsumerDeliveryState.PENDING
        )
        assert [item["behavior"] for item in behavior.judgements.list()] == ["确认回答风格"]

        results = await ConversationSourceRecovery(service.sources, service, batch_size=10).recover_pending()

        assert [result.error for result in results if result.consumer is ConversationSourceConsumer.BEHAVIOR_SESSION] == [
            None
        ]
        assert service.inspect(source, ConversationSourceConsumer.BEHAVIOR_SESSION).state is (
            ConversationConsumerDeliveryState.COMMITTED
        )
        # 重做时第一轮被认出来，没有再问模型、没有再写一条。
        assert provider.calls == 4
        assert sorted(item["behavior"] for item in behavior.judgements.list()) == ["检查项目状态", "确认回答风格"]

    asyncio.run(scenario())


def test_tool_calls_and_failures_cross_the_bridge_in_the_lane_s_own_shape() -> None:
    (turn,) = session_turns(envelope("tool-turn", tool_turn(status=ConversationToolResultStatus.ERROR)))

    roles = [item.role for item in turn.messages]
    assert roles == [
        SessionMessageRole.PROMPT,
        SessionMessageRole.TOOL_CALL,
        SessionMessageRole.TOOL_RESULT,
        SessionMessageRole.COMPLETION,
    ]
    tool_call, tool_result = turn.messages[1], turn.messages[2]
    assert (tool_call.tool_name, tool_call.content) == ("workspace.inspect", '{"path":"."}')  # 参数对象转成一行文字
    assert tool_result.failed
    assert turn.answered
