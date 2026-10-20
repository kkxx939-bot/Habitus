"""会话源 → 会话 lane 的桥：行为侧在会话源上的消费者。

这是一座桥，所以住在组合根：会话源不认识行为侧，行为侧的会话 lane（``behavior/fusion/lanes/session``）
也不认识会话源。桥上做三件事：

1. 把一份会话源信封里的消息翻成会话 lane 的输入形状，按"人说的话"拆成一轮一轮；
2. 一轮一轮交给会话 lane（描述 + 写判断）；
3. 给这份源留一份回执（``conversation/behavior_session``），供会话源的交付契约判定"处理完了"。

失败怎么办：某一轮的模型调用失败就让异常抛出去，这份源不落终态，由会话源的恢复按重试账之后重做；重做时已经写过判断的
轮会被认出来，不重复写。重试账放弃了（怎么失败的都一样）就走 ``abandon``：还没写进树的轮写成没读懂空白、落终态
（23 号第一节第 6 条：缺失必须记录，否则下游把"看了没读懂"读成"没做"）。

TODO(BHV-SESSION-LANE-001)：只收本人在交互界面里说话的会话（裁定 37）——现在只挡得住子代理的会话
（会话身份带 ``.subagent.``）。脚本调起、定时触发的会话在信封上没有任何标识，要插件与服务入口先把来源带上来。
"""

from __future__ import annotations

from dataclasses import asdict

from habitus.behavior.fusion.lanes.session import (
    SESSION_FUSION_VERSION,
    SessionLane,
    SessionLaneConfig,
    SessionMessage,
    SessionMessageRole,
    SessionTurn,
    TurnDisposition,
    TurnOutcome,
    split_turns,
)
from habitus.conversation.behavior_session import (
    BEHAVIOR_SESSION_OUTPUT_SCHEMA_VERSION,
    BehaviorSessionOutput,
    BehaviorSessionOutputStore,
    BehaviorSessionTurnRecord,
)
from habitus.conversation.source import (
    ConversationConsumerExecutionLease,
    ConversationConsumerRunDisposition,
    ConversationConsumerRunResult,
    ConversationSourceConsumer,
    ConversationSourceEnvelope,
)
from habitus.foundation.integrity import canonical_digest, canonical_json
from habitus.pre.conversation.messages import (
    ConversationMessage,
    ConversationMessageRole,
    ConversationToolResultStatus,
)

_FINGERPRINT_SCHEMA = "conversation_behavior_session_processor_fingerprint_v1"
# 子代理的会话不是这个人说的话。
_SUBAGENT_MARK = ".subagent."

SKIP_BEHAVIOR_DISABLED = "BEHAVIOR_DISABLED"
SKIP_SUBAGENT = "SUBAGENT_CONVERSATION"
SKIP_NO_TURNS = "NO_TURNS"


def session_turns(
    envelope: ConversationSourceEnvelope,
    *,
    question_tools: tuple[str, ...] = SessionLaneConfig().question_tools,
) -> tuple[SessionTurn, ...]:
    """把一份会话源信封拆成一轮一轮。子代理的会话一轮也不出。

    信封开头若是上一轮剩下的消息（没有人说的话打头），它们不属于这份信封里的任何一轮，丢掉。
    """

    if _SUBAGENT_MARK in envelope.conversation_id:
        return ()
    return split_turns(
        envelope.conversation_id,
        tuple(_message(item) for item in envelope.batch.messages),
        question_tools=question_tools,
    )


def _message(message: ConversationMessage) -> SessionMessage:
    return SessionMessage(
        sequence=message.sequence,
        occurred_at=message.occurred_at,
        role=SessionMessageRole(ConversationMessageRole(message.role).value),
        content=message.content if isinstance(message.content, str) else canonical_json(message.content),
        tool_name=message.tool_name,
        tool_call_id=message.tool_call_id,
        failed=message.tool_status is ConversationToolResultStatus.ERROR,
    )


class BehaviorSessionConsumer:
    """会话源的消费者：把一份源交给会话 lane，留一份回执。行为侧没启用时 ``lane`` 为 None，每份源都落"跳过"。"""

    consumer = ConversationSourceConsumer.BEHAVIOR_SESSION
    # 一轮一条、互不相认：同一个会话的几份源谁先处理都一样。
    ordered_within_conversation = False
    # 已释放的源被重投时不再执行：覆盖记录会随源释放，重放就会写出重复的判断。
    skips_released_sources = True

    def __init__(
        self,
        lane: SessionLane | None,
        store: BehaviorSessionOutputStore,
    ) -> None:
        if lane is not None and not isinstance(lane, SessionLane):
            raise TypeError("lane must be SessionLane or None")
        if not isinstance(store, BehaviorSessionOutputStore):
            raise TypeError("store must be BehaviorSessionOutputStore")
        self.lane = lane
        self.store = store
        self.output_store = store
        config = SessionLaneConfig() if lane is None else lane.describer.config
        # 提示词、材料上限或回执形状变了，同一份源就该算另一种处理结果。
        self.processor_fingerprint = canonical_digest(
            {
                "schema_version": _FINGERPRINT_SCHEMA,
                "output_schema_version": BEHAVIOR_SESSION_OUTPUT_SCHEMA_VERSION,
                "fusion_version": SESSION_FUSION_VERSION,
                "config": asdict(config),
                "enabled": lane is not None,
            }
        )

    async def execute(
        self,
        envelope: ConversationSourceEnvelope,
        lease: ConversationConsumerExecutionLease,
    ) -> ConversationConsumerRunResult:
        turns = self._turns(envelope)
        if isinstance(turns, ConversationConsumerRunResult):
            return turns
        assert self.lane is not None
        # 一轮一次模型调用，可能很久：每轮之前确认租约还在，丢了就停，别和接手的那一方各写一遍。
        outcomes = await self.lane.consume(turns, before_each=lease.require_alive, sources=(envelope.source_id,))
        return await self._receipt(envelope, lease, outcomes)

    async def abandon(
        self,
        envelope: ConversationSourceEnvelope,
        lease: ConversationConsumerExecutionLease,
        *,
        reason: str,
    ) -> ConversationConsumerRunResult:
        """重试账放弃了这份源：不再问模型，还没写进树的轮写成没读懂，照常留回执、落终态（信封因此可以释放）。"""

        turns = self._turns(envelope)
        if isinstance(turns, ConversationConsumerRunResult):
            return turns
        assert self.lane is not None
        outcomes = await self.lane.abandon(
            turns, reason=reason, before_each=lease.require_alive, sources=(envelope.source_id,)
        )
        return await self._receipt(envelope, lease, outcomes)

    def _turns(self, envelope: ConversationSourceEnvelope) -> tuple[SessionTurn, ...] | ConversationConsumerRunResult:
        """这份源要处理的轮；没有要处理的（行为侧没启用、子代理的会话、没有一轮）给出"跳过"。"""

        if self.lane is None:
            return _skipped(SKIP_BEHAVIOR_DISABLED)
        if _SUBAGENT_MARK in envelope.conversation_id:
            return _skipped(SKIP_SUBAGENT)
        turns = session_turns(envelope, question_tools=self.lane.describer.config.question_tools)
        return turns if turns else _skipped(SKIP_NO_TURNS)

    async def _receipt(
        self,
        envelope: ConversationSourceEnvelope,
        lease: ConversationConsumerExecutionLease,
        outcomes: tuple[TurnOutcome, ...],
    ) -> ConversationConsumerRunResult:
        records = [
            BehaviorSessionTurnRecord(
                start_sequence=outcome.turn.start_sequence,
                end_sequence=outcome.turn.end_sequence,
                instructed_at=outcome.turn.instructed_at,
                completed_at=outcome.turn.completed_at,
                recorded=outcome.disposition is not TurnDisposition.UNANSWERED,
            )
            for outcome in outcomes
        ]
        output = BehaviorSessionOutput.create(
            source=envelope,
            processor_fingerprint=self.processor_fingerprint,
            turns=tuple(records),
        )
        stored = await lease.run_fenced(lambda: self.store.put(envelope, output))
        return ConversationConsumerRunResult(
            disposition=ConversationConsumerRunDisposition.OUTPUT_WRITTEN,
            output_ref=self.store.ref(stored),
            skip_reason=None,
            runtime_result=stored,
        )


def _skipped(reason: str) -> ConversationConsumerRunResult:
    return ConversationConsumerRunResult(
        disposition=ConversationConsumerRunDisposition.SKIPPED,
        output_ref=None,
        skip_reason=reason,
        runtime_result=None,
    )


__all__ = [
    "SKIP_BEHAVIOR_DISABLED",
    "SKIP_NO_TURNS",
    "SKIP_SUBAGENT",
    "BehaviorSessionConsumer",
    "session_turns",
]
