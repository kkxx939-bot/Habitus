"""从 SourceEnvelope.batch 生成无模型、无 Claim 的确定性 Behavior 投影。

TODO(CONV-PROJECTION-001): 当前投影不足以支撑 Event Fusion，缺两块且互为两半——
一块让融合读不出语义，一块让融合取不到、排不了序。两者都要提 Projector 版本，
因此应当一次性落地，不要分两次各提一次版本。

## 缺口 A：Assistant 的 COMPLETION 被整条丢弃

- 现状：``_ROLE_PROJECTIONS`` 把 COMPLETION 映射为跳过，Agent 自己的输出完全不进
  Outbox；Outbox 里只有"做了什么"，没有"为什么这么做"。
- 具体场景：一轮真实排查共 5 条消息——用户提问、Assistant"我怀疑是 token 刷新的
  并发问题，先看 auth.py"、``Read auth.py``、工具结果、Assistant"确认了
  refresh_token 没有加锁"——只投影出 3 条，被丢掉的恰好是意图与结论。剩下的三条
  只能说明"读了 auth.py 且成功"，读不出它在验证什么假设、读完得到了什么判断。
- 影响大小：高。Event Schema 的 ``trigger``、``goal``、``onset_semantics`` 和
  ``semantic_summary`` 都明确要求"有证据支持"，而这些证据只存在于 COMPLETION。
  不解决这一块，融合只能把这些字段留空，或者从工具调用序列反推——后者正是
  Schema 想禁止的推测。
- 改造方案（两条路，动手前必须先确认选哪条）：
  1. 投影层维持无内容的确定性指纹层，由融合自己回读 ``memory/conversation`` history
     层的不可变原文，Outbox 只作为确定性骨架（身份、tool_call/tool_result 配对、
     顺序、摘要绑定）。注意这条路线里回读原文的是融合模块，本模块仍然只读
     ``envelope.batch``，不得因此引入对会话片段类型的依赖。
  2. 把 COMPLETION 也投影进来，Outbox 自带语义内容。
  倾向方案 1：原文在 ``memory/conversation`` 的 live/history 已有唯一真相源，把
  内容搬进 Outbox 等于维护第二份原文，也会让本模块失去"无内容"这一性质。

## 缺口 B：投影项与批次缺少定位和排序所需的系统字段

- 现状：投影项只有 ``projection_item_id``、``source_id``、``source_message_id``、
  ``source_message_digest``、``occurred_at``、``projection_kind`` 和 ``payload``；
  批次只有身份、指纹、版本、条目和 ``recorded_at``。``ConversationMessage.sequence``
  以及 Envelope 的 ``conversation_id``、``protocol``、``after_turn`` 都没有带过来。
- 具体场景：同一会话的两个连续批次加上另一个会话的一个批次，只看投影输出无法判断
  前两者是否同属一个会话，无法判断先后（``occurred_at`` 允许并列，消息校验用的是
  ``>`` 而不是 ``>=``），无法知道该逻辑轮次是否已经收尾，也分不出来源是 claude_code
  还是 codex。批内顺序尚可由数组下标还原，跨批次则完全断掉。
- 影响大小：中。纯机械缺口、不影响正确性，但它是融合的前置条件：融合第一步就是
  "按会话取一段时序行为信号"，现在这一步做不了；而 ``after_turn`` 恰好是"逻辑轮次
  完整闭合"的唯一信号，也就是 Event 边界最自然的候选。目前只能回读 Envelope 才能
  补齐这些字段，而 ``ConversationSourceStore.list()`` 是全局扁平枚举，没有按会话的
  索引，回读成本随来源总量线性增长。
- 改造方案：投影项增加 ``sequence``；批次增加 ``conversation_id``、``protocol`` 和
  ``after_turn``。四项全部从 Envelope 确定性派生，仍然不引入任何模型判断。同时需要
  一条按会话有序枚举投影批次的路径（在投影侧建索引，或在 Source 侧补按会话枚举）。

## 共同约束与时机

改动映射或批次结构必须提升 ``CONVERSATION_BEHAVIOR_PROJECTOR_VERSION``，指纹随之
变化，新来源会产生新的 ``output_id``。按 ``model`` 模块记录的混版契约这是允许的
（旧来源保留旧语义、不重投影），但必须是有意识的动作，并确认下游按
``projector_version`` 分治。落地时机与 Event Fusion（``TODO(BHV-FUSION-003)``）的
方案确认同步；缺口 A 的路线一旦定下，两块一起改、只提一次版本。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from conversation.projection.behavior.model import (
    BEHAVIOR_PROJECTION_OUTPUT_SCHEMA_VERSION,
    ConversationBehaviorProjectionBatch,
    ConversationBehaviorProjectionItem,
    ConversationBehaviorProjectionKind,
)
from conversation.source.model import ConversationSourceEnvelope, ConversationSourceError
from foundation.integrity import canonical_digest
from pre.conversation import ConversationMessage, ConversationMessageRole

# 改动映射表、payload 字段或摘要语义时必须同步提升本版本；它进入
# processor_fingerprint，进而决定 output_id，是新旧投影语义的唯一分界。
CONVERSATION_BEHAVIOR_PROJECTOR_VERSION = "conversation_behavior_projector_v1"
_FINGERPRINT_SCHEMA = "conversation_behavior_processor_fingerprint_v1"
_SKIPPED_ROLE = "skip"


def _user_conversation_input_payload(message: ConversationMessage) -> dict[str, Any]:
    return {"content": message.content}


def _agent_tool_call_payload(message: ConversationMessage) -> dict[str, Any]:
    return {
        "tool_call_id": message.tool_call_id,
        "tool_name": message.tool_name,
        "arguments_digest": canonical_digest(message.content),
    }


def _tool_execution_result_payload(message: ConversationMessage) -> dict[str, Any]:
    return {
        "tool_call_id": message.tool_call_id,
        "tool_name": message.tool_name,
        "tool_status": message.tool_status.value if message.tool_status is not None else None,
        "content_mode": message.content_mode.value if message.content_mode is not None else None,
        "result_digest": canonical_digest(message.content),
        "source_ref": message.source_ref,
        "original_size_bytes": message.original_size_bytes,
        "original_sha256": message.original_sha256,
    }


@dataclass(frozen=True)
class _RoleProjection:
    """一个 Conversation 角色的投影规则；``kind`` 为 None 表示显式跳过。"""

    kind: ConversationBehaviorProjectionKind | None
    build_payload: Callable[[ConversationMessage], dict[str, Any]] | None

    def __post_init__(self) -> None:
        if (self.kind is None) != (self.build_payload is None):
            raise ValueError("a projected role needs a payload builder; a skipped role forbids one")

    @property
    def fingerprint_value(self) -> str:
        return _SKIPPED_ROLE if self.kind is None else self.kind.value


# 角色映射的唯一真相源：``_project_message`` 按它分派，``processor_fingerprint``
# 也从它派生。两者共用同一张表，改了分派却漏改指纹在物理上不再可能发生。
_ROLE_PROJECTIONS: Mapping[ConversationMessageRole, _RoleProjection] = {
    ConversationMessageRole.PROMPT: _RoleProjection(
        ConversationBehaviorProjectionKind.USER_CONVERSATION_INPUT,
        _user_conversation_input_payload,
    ),
    # TODO(CONV-PROJECTION-001) 缺口 A：跳过 COMPLETION 使 Outbox 只有"做了什么"，
    # 没有"为什么"；Event 的 trigger/goal/onset_semantics 因此无证据可依。
    ConversationMessageRole.COMPLETION: _RoleProjection(None, None),
    ConversationMessageRole.TOOL_CALL: _RoleProjection(
        ConversationBehaviorProjectionKind.AGENT_TOOL_CALL,
        _agent_tool_call_payload,
    ),
    ConversationMessageRole.TOOL_RESULT: _RoleProjection(
        ConversationBehaviorProjectionKind.TOOL_EXECUTION_RESULT,
        _tool_execution_result_payload,
    ),
}


class ConversationBehaviorProjector:
    """只读取 SourceEnvelope.batch；COMPLETION 在第一版明确跳过。

    投影是来源的纯函数：同一个 Source 无论何时、由谁投影，都必须产出逐字节
    相同的批次。因此这里没有时钟——``recorded_at`` 取自来源本身，而不是投影
    发生的墙上时间。若取墙上时钟，并发执行会产出 ``output_id`` 相同却
    ``output_record_digest`` 不同的两份结果，后写入者被误报为内容冲突；重算
    比对也会因此必须先信任被验证文件里的时间字段。
    """

    def __init__(self) -> None:
        self.projector_version = CONVERSATION_BEHAVIOR_PROJECTOR_VERSION
        self.processor_fingerprint = canonical_digest(
            {
                "schema_version": _FINGERPRINT_SCHEMA,
                "projector_version": self.projector_version,
                "output_schema_version": BEHAVIOR_PROJECTION_OUTPUT_SCHEMA_VERSION,
                "mappings": {
                    role.value: projection.fingerprint_value
                    for role, projection in _ROLE_PROJECTIONS.items()
                },
            }
        )

    def project(self, envelope: ConversationSourceEnvelope) -> ConversationBehaviorProjectionBatch | None:
        if not isinstance(envelope, ConversationSourceEnvelope):
            raise TypeError("envelope must be ConversationSourceEnvelope")
        items = tuple(
            projected
            for message in envelope.batch.messages
            if (projected := self._project_message(envelope.source_id, message)) is not None
        )
        if not items:
            return None
        return ConversationBehaviorProjectionBatch.create(
            source=envelope,
            processor_fingerprint=self.processor_fingerprint,
            projector_version=self.projector_version,
            items=items,
            recorded_at=envelope.recorded_at,
        )

    def _project_message(
        self, source_id: str, message: ConversationMessage
    ) -> ConversationBehaviorProjectionItem | None:
        projection = _ROLE_PROJECTIONS.get(message.role)
        if projection is None:  # pragma: no cover - ConversationMessageRole 是封闭枚举。
            raise ConversationSourceError("unsupported Conversation role")
        if projection.kind is None or projection.build_payload is None:
            return None
        return ConversationBehaviorProjectionItem.create(
            source_id=source_id,
            message=message,
            projection_kind=projection.kind,
            payload=projection.build_payload(message),
        )


__all__ = [
    "CONVERSATION_BEHAVIOR_PROJECTOR_VERSION",
    "ConversationBehaviorProjector",
]
