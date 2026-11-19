"""会话中断时的收尾（裁定 107 第 2 条）：照通行的"修补悬空的工具调用"（LangChain deepagents ``PatchToolCallsMiddleware``、
Eino ``PatchToolCalls``），闲置到点、封存之前由系统如实补上没完成的地方，之后 live 就全是完整的轮、走现有的封存与抽取。

- 没有结果的 tool_call：补一条同名的终态 tool_result，状态记失败，内容说明对话中断、没有结果；
- 最后一轮没有完整的最终回复：补一条最终回复，内容只说没有最终回复、对话中断，不替助手编答案。

只补 live 里的；补出的消息接在 live 末尾，序号接着排、时刻是收尾的那一刻。只有最后一轮可能没闭合（更早的轮没闭合时
afterTurn 规划会整批拒收），所以补在末尾就落在它里面。
"""

from __future__ import annotations

from datetime import datetime

from habitus.foundation.integrity import canonical_digest
from habitus.pre.conversation.messages import (
    ConversationBatch,
    ConversationMessage,
    ConversationMessageRole,
    ConversationToolResultContentMode,
    ConversationToolResultStatus,
)

INTERRUPTED_TOOL_RESULT = "[系统记录] 没有结果：对话在这次调用之后中断，宿主没有送来它的结果。"
INTERRUPTED_COMPLETION = "[系统记录] 没有最终回复：对话在这里中断。"
_ID_PREFIX = "habitus-interrupted-"


def interruption_messages(live: ConversationBatch, *, at: datetime) -> tuple[ConversationMessage, ...]:
    """这场对话停在 ``at`` 时，要接在 live 末尾补上的系统收尾消息；live 本来就全是闭合的轮时为空。"""

    messages = live.messages
    answered = {item.tool_call_id for item in messages if item.role is ConversationMessageRole.TOOL_RESULT}
    sequence = messages[-1].sequence + 1
    patched: list[ConversationMessage] = []
    for call in messages:
        if call.role is not ConversationMessageRole.TOOL_CALL or call.tool_call_id in answered:
            continue
        patched.append(
            ConversationMessage(
                message_id=_message_id(live.conversation_id, "tool_result", call.message_id),
                sequence=sequence,
                role=ConversationMessageRole.TOOL_RESULT,
                occurred_at=at,
                content=INTERRUPTED_TOOL_RESULT,
                tool_call_id=call.tool_call_id,
                tool_name=call.tool_name,
                tool_status=ConversationToolResultStatus.ERROR,
                content_mode=ConversationToolResultContentMode.INLINE,
            )
        )
        sequence += 1
    final = patched[-1] if patched else messages[-1]
    if final.role is not ConversationMessageRole.COMPLETION or not final.completes_logical_message:
        patched.append(
            ConversationMessage(
                message_id=_message_id(live.conversation_id, "completion", final.message_id),
                sequence=sequence,
                role=ConversationMessageRole.COMPLETION,
                occurred_at=at,
                content=INTERRUPTED_COMPLETION,
            )
        )
    return tuple(patched)


def _message_id(conversation_id: str, kind: str, after: str) -> str:
    digest = canonical_digest(
        {"schema": "conversation_interruption_v1", "conversation": conversation_id, "kind": kind, "after": after}
    )
    return f"{_ID_PREFIX}{kind}-{digest[:24]}"


def is_interruption_message(message: ConversationMessage) -> bool:
    """这条是不是系统补的收尾消息（编号前缀与固定内容都对上）。"""

    return message.message_id.startswith(_ID_PREFIX) and message.content in (
        INTERRUPTED_TOOL_RESULT,
        INTERRUPTED_COMPLETION,
    )


__all__ = ["INTERRUPTED_COMPLETION", "INTERRUPTED_TOOL_RESULT", "interruption_messages", "is_interruption_message"]
