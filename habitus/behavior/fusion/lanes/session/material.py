"""从会话消息里取出一轮一轮的材料（确定性整理，不调模型）。

一轮 = 人说的一句话，加上到他下一句话之前助手做的全部事。给模型的材料除了工具结果的正文，其余尽量都留
（裁定 33）：人的指令全文、助手的全部答复、工具调用每次一行（工具名 + 参数开头一段，不做归类）、工具结果只留
状态（失败的留报错开头几行）。整轮有一个总字数上限，超出时先削工具调用的中段，再削中间答复，再削最后一条
答复的中段，人的指令最后才削。材料给模型之前先抹一道密钥（裁定 37）：每一段文字在截短之前就抹——先截再抹的话，
被截掉后半截的密钥不再是密钥的形状，开头那几位会漏出去。
"""

from __future__ import annotations

from collections.abc import Sequence

from habitus.behavior.fusion.lanes.session.config import SessionLaneConfig
from habitus.behavior.fusion.lanes.session.model import SessionMessage, SessionMessageRole, SessionTurn
from habitus.behavior.fusion.lanes.session.redaction import redact

# 宿主注入的合成消息：不是人说的话，不开一轮。
SYNTHETIC_PREFIXES = (
    "<environment_context",
    "<recommended_plugins",
    "<turn_aborted",
    "<skill>",
    "<in-app-browser",
    "<external_codex",
    "# AGENTS.md",
    "<user_instructions",
    "<user_shell_command",
    "<subagent_notification",
    "<task-notification",
    "<local-command",
    "<command-name>",
    "<system-reminder",
    "This session is being continued",
    "<artifact-content",
    "Another Claude session",
    "[Cross-session",
    "[Image: original",
    "[Your previous response",
    "[Request interrupted",
    "Caveat:",
)
# 把停下来的助手重新叫起来的合成消息（后台任务做完了、别的会话发来消息）。助手已经说完话之后被它叫醒
# 又做的那一段，本质上还是在这场对话里处理任务：上一轮到它停下为止，被叫醒的那一段自己算一轮、自己出一条记录。
WAKE_PREFIXES = (
    "<task-notification",
    "<subagent_notification",
    "Another Claude session",
    "[Cross-session",
)
# 被叫醒的那一轮没有人说的话，"他说"的位置写这个开头，后面跟叫醒它的那条通知的开头。
WAKE_LABEL = "（没有新的话；后台任务回来，助手接着做）"
WAKE_NOTICE_CHARS = 300
# 人对助手提问的作答，翻成人说的话时带的开头。
ANSWER_PREFIX = "（回答助手的提问）"
# 同一秒里成批出现的"人说的话"是重放，不是他说的。
REPLAY_BURST = 3


def is_synthetic(text: str) -> bool:
    return text.lstrip().startswith(SYNTHETIC_PREFIXES)


def split_turns(
    conversation_id: str,
    messages: Sequence[SessionMessage],
    *,
    question_tools: Sequence[str] = SessionLaneConfig().question_tools,
) -> tuple[SessionTurn, ...]:
    """把一个会话的消息按"人说的话"拆成一轮一轮；合成消息与重放不开轮。

    人对助手提问的作答（``question_tools`` 的工具结果）也是人说的话：助手发问时上一轮到此为止，
    作答开新的一轮。提问没成（参数不合法）或被他拒绝的不算作答——拒绝之后他会另说一句话，那句话自己开轮。

    助手说完话停下之后，被后台任务通知或别的会话的消息叫醒又做的那一段，不算进上一轮，自己算一轮；
    助手还在干活（上一条不是它的答复）时到的通知不算叫醒，这一轮照常继续。被叫醒之后什么也没做的，和别的
    没答复的轮一样不出记录。
    """

    asking = frozenset(question_tools)

    # 只数人说的话：宿主在同一秒里写下的合成消息、叫醒通知不算数，否则人真说的那一句会被它们连累丢掉。
    burst: dict[int, int] = {}
    for message in messages:
        if message.role is SessionMessageRole.PROMPT and _spoken(message.content):
            second = int(message.occurred_at.timestamp())
            burst[second] = burst.get(second, 0) + 1
    replayed = {second for second, count in burst.items() if count >= REPLAY_BURST}

    turns: list[SessionTurn] = []
    current: list[SessionMessage] = []

    def close() -> None:
        if not current:
            return
        head = current[0]
        turns.append(
            SessionTurn(
                conversation_id=conversation_id,
                start_sequence=head.sequence,
                end_sequence=current[-1].sequence,
                instructed_at=head.occurred_at,
                completed_at=max(item.occurred_at for item in current),
                instruction=head.content.strip(),
                messages=tuple(current),
            )
        )

    for message in messages:
        if message.role is SessionMessageRole.PROMPT:
            text = message.content.strip()
            if _wakes(text) and (not current or current[-1].role is SessionMessageRole.COMPLETION):
                close()
                current = [_wake_as_prompt(message)]  # 上一轮到助手停下为止；被叫醒的这一段自己算一轮
                continue
            if not text or is_synthetic(text) or int(message.occurred_at.timestamp()) in replayed:
                continue  # 不是人说的话：丢掉，也不结束当前这一轮
            close()
            current = [message]
        elif _is_answer(message, asking):
            close()
            current = [_answer_as_prompt(message)]
        elif current:
            current.append(message)
    close()
    return tuple(turns)


def _spoken(text: str) -> bool:
    return bool(text.strip()) and not is_synthetic(text)


def _wakes(text: str) -> bool:
    return text.lstrip().startswith(WAKE_PREFIXES)


def _wake_as_prompt(message: SessionMessage) -> SessionMessage:
    notice = redact(" ".join(message.content.split()))[:WAKE_NOTICE_CHARS]
    return SessionMessage(
        sequence=message.sequence,
        occurred_at=message.occurred_at,
        role=SessionMessageRole.PROMPT,
        content=f"{WAKE_LABEL}{notice}",
    )


def _is_answer(message: SessionMessage, asking: frozenset[str]) -> bool:
    return (
        message.role is SessionMessageRole.TOOL_RESULT
        and message.tool_name in asking
        and not message.failed
        and bool(message.content.strip())
    )


def _answer_as_prompt(message: SessionMessage) -> SessionMessage:
    """把人对提问的作答翻成一句人说的话（工具结果里本来就写着问了什么、他选了什么）。"""

    return SessionMessage(
        sequence=message.sequence,
        occurred_at=message.occurred_at,
        role=SessionMessageRole.PROMPT,
        content=f"{ANSWER_PREFIX}{message.content.strip()}",
    )


def render_turn(turn: SessionTurn, *, config: SessionLaneConfig | None = None) -> str:
    """把一轮整理成给模型看的文字。每一段在截短之前已经抹过密钥；出门之前整篇再抹一遍。"""

    return redact(_render(turn, config or SessionLaneConfig()))


def _render(turn: SessionTurn, limits: SessionLaneConfig) -> str:
    instruction = _head_tail(redact(turn.instruction), limits.max_instruction_chars)
    replies = [
        redact(item.content.strip())
        for item in turn.messages[1:]
        if item.role is SessionMessageRole.COMPLETION and item.content.strip()
    ]
    middle, last = (replies[:-1], replies[-1]) if replies else ([], "")
    calls = _tool_lines(turn, limits)

    def build(call_lines: list[str], middle_replies: list[str], final: str) -> str:
        lines = [f"他说：{instruction}"]
        if middle_replies:
            lines.append("助手中途说：")
            lines.extend(f"  - {text}" for text in middle_replies)
        if call_lines:
            lines.append("助手调用：")
            lines.extend(f"  - {line}" for line in call_lines)
        lines.append(f"助手最后说：{final}" if final else "助手最后说：（没有答复）")
        return "\n".join(lines)

    text = build(calls, middle, last)
    if len(text) <= limits.max_turn_chars:
        return text
    # 超出上限：先削工具调用的中段
    if len(calls) > limits.kept_tool_lines:
        half = limits.kept_tool_lines // 2
        calls = [*calls[:half], f"……（中间省略 {len(calls) - 2 * half} 次调用）", *calls[-half:]]
        text = build(calls, middle, last)
    # 再削中间答复：条数太多就只留头尾几条（逐条各留一点仍会超，最后落到整段盲截）
    if len(text) > limits.max_turn_chars and len(middle) > limits.kept_reply_lines:
        half = limits.kept_reply_lines // 2
        middle = [*middle[:half], f"……（中间省略 {len(middle) - 2 * half} 条）", *middle[-half:]]
        text = build(calls, middle, last)
    if len(text) > limits.max_turn_chars and middle:
        budget = max(0, limits.max_turn_chars - len(build(calls, [], last)))
        each = max(60, budget // len(middle))
        middle = [_head_tail(item, each) for item in middle]
        text = build(calls, middle, last)
    # 再削最后一条答复的中段
    if len(text) > limits.max_turn_chars and last:
        budget = max(400, limits.max_turn_chars - len(build(calls, middle, "")))
        last = _head_tail(last, budget)
        text = build(calls, middle, last)
    return text if len(text) <= limits.max_turn_chars else _head_tail(text, limits.max_turn_chars)


def _tool_lines(turn: SessionTurn, limits: SessionLaneConfig) -> list[str]:
    results = {
        item.tool_call_id: item
        for item in turn.messages
        if item.role is SessionMessageRole.TOOL_RESULT and item.tool_call_id
    }
    lines: list[str] = []
    counts: list[int] = []
    for item in turn.messages:
        if item.role is not SessionMessageRole.TOOL_CALL:
            continue
        line = f"{item.tool_name} {_arguments(item.content, limits.tool_argument_chars)}".strip()
        result = results.get(item.tool_call_id or "")
        if result is not None and result.failed:
            line += f" → 失败：{redact(_one_line(result.content))[: limits.tool_error_chars]}"
        if lines and lines[-1] == line:
            counts[-1] += 1  # 连续完全相同的合成一行
            continue
        lines.append(line)
        counts.append(1)
    return [line if count == 1 else f"{line}（连续 {count} 次）" for line, count in zip(lines, counts, strict=True)]


def _arguments(content: str, limit: int) -> str:
    return redact(_one_line(content))[:limit]


def _one_line(text: str) -> str:
    return " ".join(text.split())


def _head_tail(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    half = max(1, (limit - 20) // 2)
    return f"{text[:half]}……（中间省略 {len(text) - 2 * half} 字）……{text[-half:]}"


__all__ = ["ANSWER_PREFIX", "WAKE_LABEL", "WAKE_PREFIXES", "is_synthetic", "render_turn", "split_turns"]
