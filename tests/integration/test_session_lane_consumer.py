"""会话源 → 会话 lane 这座桥：行为侧在会话源上的消费者，走真实的交付契约与真实的行为侧存储。

模型用照脚本回答的替身；其余（会话源存储、终态、租约、恢复、判断与回执存储）都是真的。
"""

from __future__ import annotations

import asyncio
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from habitus.behavior.fusion.lanes.session import SessionMessageRole
from habitus.conversation import (
    BehaviorSessionOutput,
    ConversationConsumerDeliveryState,
    ConversationConsumerOutcomeState,
    ConversationSourceConsumer,
    ConversationSourceEnvelope,
    conversation_source_request_digest,
)
from habitus.conversation.source.recovery import recovery_key
from habitus.foundation.lifecycle import DeterministicFailure
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
from tests.unit.conversation.source_v2_helpers import NOW, delivery, stores
from tests.unit.conversation.source_v2_helpers import recovery as source_recovery

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
        # 覆盖记录带着所属会话源：这份源的 Behavior 到终态之前不删（26 号清单 6.1）
        import json

        records = [
            json.loads(path.read_text()) for path in (tmp_path / "behavior" / "fusion" / "coverage").rglob("*.json")
        ]
        assert len(records) == 2 and all(item["source_refs"] == [source.source_id] for item in records)

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
        service, behavior, provider = wired(
            tmp_path, [description("确认回答风格"), bad, bad, description("检查项目状态")]
        )
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

        results = await source_recovery(service).recover_pending()

        assert [
            result.error for result in results if result.consumer is ConversationSourceConsumer.BEHAVIOR_SESSION
        ] == [None]
        assert service.inspect(source, ConversationSourceConsumer.BEHAVIOR_SESSION).state is (
            ConversationConsumerDeliveryState.COMMITTED
        )
        # 重做时第一轮被认出来，没有再问模型、没有再写一条。
        assert provider.calls == 4
        assert sorted(item["behavior"] for item in behavior.judgements.list()) == ["检查项目状态", "确认回答风格"]

    asyncio.run(scenario())


def _model_down(service, monkeypatch: pytest.MonkeyPatch) -> list[object]:  # type: ignore[no-untyped-def]
    """模型连不上（像额度用完）：每次描述都抛可重试的错。返回调用记录。"""

    calls: list[object] = []

    async def down(turn: object, reminders: object = ()) -> object:
        calls.append(turn)
        raise ConnectionError("model quota exhausted")

    consumer = service.consumers[ConversationSourceConsumer.BEHAVIOR_SESSION]
    monkeypatch.setattr(consumer.lane.describer, "describe", down)
    return calls


def test_a_source_the_retry_ledger_gives_up_on_is_written_as_unreadable_and_reaches_a_terminal_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """模型连着调不通到重试账放弃：还没写进树的轮写成没读懂空白，这份源对 Behavior 落终态（23 号第一节第 6 条）——
    不然下游把"看了没读懂"读成"没做"，信封也永远放不了。"""

    async def scenario() -> None:
        service, behavior, _provider = wired(tmp_path, [description()])
        source = service.sources.put(envelope("quota-out", (*closed_turn(), *tool_turn(start_sequence=2))))
        calls = _model_down(service, monkeypatch)
        now = [NOW]
        recovering = source_recovery(service, max_attempts=2, clock=lambda: now[0])

        await recovering.recover_pending()
        assert service.inspect(source, ConversationSourceConsumer.BEHAVIOR_SESSION).state is (
            ConversationConsumerDeliveryState.PENDING
        )
        now[0] += timedelta(hours=2)
        await recovering.recover_pending()  # 第二次失败：账判放弃 → 当场写没读懂

        assert len(calls) == 2  # 放弃之后不再问模型
        assert service.inspect(source, ConversationSourceConsumer.BEHAVIOR_SESSION).state is (
            ConversationConsumerDeliveryState.COMMITTED
        )
        receipt = service.restore_terminal(source, ConversationSourceConsumer.BEHAVIOR_SESSION)
        assert isinstance(receipt, BehaviorSessionOutput) and all(turn.recorded for turn in receipt.turns)
        judgements = behavior.judgements.list()
        assert len(judgements) == 2 and all(item["behavior"] is None for item in judgements)  # 两段没读懂
        assert (
            recovering.retries.entry(recovery_key(source.source_id, ConversationSourceConsumer.BEHAVIOR_SESSION))
            is None
        )

    asyncio.run(scenario())


def test_a_give_up_that_never_became_unreadable_is_finished_on_the_next_round(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """账里已经判了放弃、没读懂还没写（放弃那一刻进程退出）：下一轮恢复补写，不等人工。"""

    async def scenario() -> None:
        service, behavior, _provider = wired(tmp_path, [description()])
        source = service.sources.put(envelope("given-up", closed_turn()))
        calls = _model_down(service, monkeypatch)
        recovering = source_recovery(service)
        key = recovery_key(source.source_id, ConversationSourceConsumer.BEHAVIOR_SESSION)
        recovering.retries.record_failure(key, _Doomed("given up before the crash"))

        await recovering.recover_pending()

        assert calls == []
        assert service.inspect(source, ConversationSourceConsumer.BEHAVIOR_SESSION).state is (
            ConversationConsumerDeliveryState.COMMITTED
        )
        assert [item["behavior"] for item in behavior.judgements.list()] == [None]

    asyncio.run(scenario())


class _Doomed(DeterministicFailure):
    pass


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


def test_the_session_lane_consumer_skips_sources_that_were_already_released() -> None:
    """2.1.5：会话 lane 的消费者声明"已释放的源重投时跳过"，Memory 不声明（它靠自己的 ingress 去重）（L10.14 补测）。"""

    from habitus.memory.workflow.conversation_consumer import MemoryConversationConsumer
    from habitus.runtime.session_lane import BehaviorSessionConsumer

    assert BehaviorSessionConsumer.skips_released_sources is True
    assert MemoryConversationConsumer.skips_released_sources is False
