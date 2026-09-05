"""从完整 ConversationSegment 构造有界的旧记忆语义查询。"""

from __future__ import annotations

from habitus.foundation.integrity import canonical_json
from habitus.memory.editor.retrieval.model import MemoryRetrievalConfig
from habitus.pre.conversation import (
    ConversationMessage,
    ConversationMessageRole,
    ConversationSegment,
)


class ConversationSegmentQueryBuilder:
    """保留消息角色和工具身份，并优先使用用户 prompt 作为召回信号。"""

    def __init__(self, config: MemoryRetrievalConfig | None = None) -> None:
        if config is not None and not isinstance(config, MemoryRetrievalConfig):
            raise TypeError("config must be MemoryRetrievalConfig")
        self.config = config or MemoryRetrievalConfig()

    def build(self, segment: ConversationSegment) -> str:
        """从原始片段生成搜索文本；Conversation Summary 不参与此过程。"""

        if not isinstance(segment, ConversationSegment):
            raise TypeError("segment must be a ConversationSegment")
        messages = segment.messages
        rendered = tuple(self._render_message(message) for message in messages)
        prompt_indexes = [
            index
            for index, message in enumerate(messages)
            if message.role is ConversationMessageRole.PROMPT
        ]
        anchor = prompt_indexes[-1] if prompt_indexes else len(messages) - 1
        selected = {anchor}
        used = len(rendered[anchor])
        if used > self.config.max_query_chars:
            return self._head_tail(rendered[anchor], self.config.max_query_chars)

        units = self._selection_units(messages, anchor=anchor)
        for unit in units:
            addition = sum(len(rendered[index]) for index in unit) + 2 * len(unit)
            if used + addition > self.config.max_query_chars:
                continue
            selected.update(unit)
            used += addition
        # 选择阶段从最新事实向前预算；渲染阶段恢复原始顺序，角色和工具因果不失真。
        return "\n\n".join(rendered[index] for index in sorted(selected))

    @staticmethod
    def _selection_units(
        messages: tuple[ConversationMessage, ...],
        *,
        anchor: int,
    ) -> tuple[tuple[int, ...], ...]:
        """按最新优先选择消息，并把 tool_call/tool_result 作为不可拆分单元。"""

        tool_indexes: dict[str, list[int]] = {}
        for index, message in enumerate(messages):
            if message.tool_call_id is not None:
                tool_indexes.setdefault(message.tool_call_id, []).append(index)

        seen: set[int] = {anchor}
        prompt_units: list[tuple[int, ...]] = []
        supporting_units: list[tuple[int, ...]] = []
        for index in range(len(messages) - 1, -1, -1):
            if index in seen:
                continue
            message = messages[index]
            if message.tool_call_id is not None:
                unit = tuple(tool_indexes[message.tool_call_id])
            else:
                unit = (index,)
            seen.update(unit)
            target = (
                prompt_units
                if any(messages[item].role is ConversationMessageRole.PROMPT for item in unit)
                else supporting_units
            )
            target.append(unit)
        return tuple((*prompt_units, *supporting_units))

    def _render_message(self, message: ConversationMessage) -> str:
        content = message.content if isinstance(message.content, str) else canonical_json(message.content)
        limit = self._message_limit(message.role)
        body = self._truncate(content, limit)
        header = f"[{message.sequence}][{message.role.value}]"
        if message.tool_name is not None:
            header += f"[tool={message.tool_name}]"
        if message.tool_call_id is not None:
            header += f"[call={message.tool_call_id}]"
        if message.tool_status is not None:
            header += f"[status={message.tool_status.value}]"
        if message.content_mode is not None:
            header += f"[content={message.content_mode.value}]"
        return f"{header}: {body}"

    def _message_limit(self, role: ConversationMessageRole) -> int:
        if role is ConversationMessageRole.PROMPT:
            return self.config.max_prompt_chars
        if role is ConversationMessageRole.COMPLETION:
            return self.config.max_completion_chars
        return self.config.max_tool_message_chars

    @staticmethod
    def _truncate(value: object, limit: int) -> str:
        normalized = " ".join(str(value or "").split())
        if len(normalized) <= limit:
            return normalized
        if limit <= 3:
            return normalized[:limit]
        return normalized[: limit - 3].rstrip() + "..."

    @staticmethod
    def _head_tail(value: str, limit: int) -> str:
        """单条消息超过总预算时同时保留开头身份和末尾最终修正。"""

        if len(value) <= limit:
            return value
        if limit <= 3:
            return value[:limit]
        remaining = limit - 3
        head = (remaining + 1) // 2
        tail = remaining - head
        return value[:head].rstrip() + "..." + (value[-tail:].lstrip() if tail else "")


__all__ = ["ConversationSegmentQueryBuilder"]
