"""为 Agent 检索读取当前 Conversation 的活跃摘要前沿和最近消息。"""

from __future__ import annotations

from dataclasses import dataclass

from foundation.integrity import canonical_json
from memory.conversation import (
    ConversationAddress,
    ConversationMessageJournal,
    ConversationSummaryCompactor,
)
from memory.retrieval.model import MemorySearchError, MemorySearchServiceConfig
from pre.conversation import ConversationMessage


@dataclass(frozen=True)
class ConversationSearchContext:
    """只用于查询规划、不会代替 Conversation 原文或写回记忆的临时上下文。"""

    conversation_id: str
    summary_context: str
    recent_messages: tuple[ConversationMessage, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.conversation_id, str) or not self.conversation_id.strip():
            raise ValueError("search context conversation_id must be non-empty")
        if not isinstance(self.summary_context, str):
            raise TypeError("search context summary_context must be text")
        if not isinstance(self.recent_messages, tuple) or any(
            not isinstance(message, ConversationMessage) for message in self.recent_messages
        ):
            raise TypeError("search context recent_messages are invalid")

    @property
    def empty(self) -> bool:
        return not self.summary_context and not self.recent_messages


class ConversationSearchContextReader:
    """按照当前不可变摘要前沿和 live 尾部构造有界查询规划上下文。"""

    def __init__(
        self,
        journal: ConversationMessageJournal,
        compactor: ConversationSummaryCompactor,
        *,
        config: MemorySearchServiceConfig | None = None,
    ) -> None:
        if not isinstance(journal, ConversationMessageJournal):
            raise TypeError("journal must be ConversationMessageJournal")
        if not isinstance(compactor, ConversationSummaryCompactor):
            raise TypeError("compactor must be ConversationSummaryCompactor")
        if compactor.journal is not journal:
            raise ValueError("search context reader must share the Conversation journal")
        if config is not None and not isinstance(config, MemorySearchServiceConfig):
            raise TypeError("config must be MemorySearchServiceConfig")
        self.journal = journal
        self.compactor = compactor
        self.config = config or MemorySearchServiceConfig()

    def read(self, address: ConversationAddress) -> ConversationSearchContext:
        """读取当前活跃摘要和 live 尾部；损坏数据不得伪装成空上下文。"""

        if not isinstance(address, ConversationAddress):
            raise TypeError("address must be ConversationAddress")
        try:
            frontier = self.compactor.frontier(address)
            live = self.journal.read_live(address)
        except Exception as exc:
            raise MemorySearchError("failed to read Conversation context for memory search") from exc
        summary_context = self._summary_context(frontier.active)
        recent_messages = () if live is None else live.messages[-self.config.max_recent_messages :]
        return ConversationSearchContext(
            conversation_id=address.conversation_id,
            summary_context=summary_context,
            recent_messages=recent_messages,
        )

    def _summary_context(self, summaries: tuple[object, ...]) -> str:
        maximum = self.config.max_summary_context_chars
        selected: list[str] = []
        used = 0
        for summary in reversed(summaries):
            to_dict = getattr(summary, "to_dict", None)
            if not callable(to_dict):
                raise MemorySearchError("Conversation summary frontier contains an invalid value")
            rendered = canonical_json(to_dict())
            separator = 2 if selected else 0
            remaining = maximum - used - separator
            if remaining <= 0:
                break
            if len(rendered) > remaining:
                if not selected:
                    selected.append(_truncate(rendered, remaining))
                break
            selected.append(rendered)
            used += len(rendered) + separator
        return "\n\n".join(reversed(selected))


def render_recent_messages(
    messages: tuple[ConversationMessage, ...],
    *,
    max_message_chars: int,
) -> str:
    """严格保留角色、工具名和调用 ID，同时限制每条规划输入大小。"""

    rendered: list[str] = []
    for message in messages:
        content = canonical_json(message.content) if not isinstance(message.content, str) else message.content
        header = f"[{message.sequence}][{message.role.value}]"
        if message.tool_name is not None:
            header += f"[tool={message.tool_name}]"
        if message.tool_call_id is not None:
            header += f"[call={message.tool_call_id}]"
        if message.tool_status is not None:
            header += f"[status={message.tool_status.value}]"
        rendered.append(f"{header}: {_truncate(content, max_message_chars)}")
    return "\n".join(rendered)


def _truncate(value: str, maximum: int) -> str:
    if len(value) <= maximum:
        return value
    if maximum <= 3:
        return value[:maximum]
    return value[: maximum - 3].rstrip() + "..."


__all__ = [
    "ConversationSearchContext",
    "ConversationSearchContextReader",
    "render_recent_messages",
]
