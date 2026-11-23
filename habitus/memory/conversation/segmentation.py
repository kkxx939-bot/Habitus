"""超大 Conversation 的文本分块。"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import replace

from habitus.pre.conversation import (
    ConversationBatch,
    ConversationMessage,
    ConversationMessageRole,
)

logger = logging.getLogger(__name__)

SEMANTIC_CHUNKER_VERSION = "conversation_semantic_chunker_v1"


# TODO(conversation-source): 修改此拆分算法语义时，必须同步提升
# memory/workflow/conversation_consumer.py 中 text_chunker_version。
class ConversationMessageChunker:
    """在写入 live 前把单条超长文本拆成可重组的物理消息。"""

    def __init__(
        self,
        *,
        max_message_tokens: int,
        token_estimator: Callable[[ConversationMessage], int],
    ) -> None:
        if max_message_tokens <= 0:
            raise ValueError("max_message_tokens must be a positive integer")
        self.max_message_tokens = max_message_tokens
        self.token_estimator = token_estimator

    def normalize(self, batch: ConversationBatch) -> ConversationBatch:
        """确定性分块并重新分配当前批次内的连续物理序号。"""

        expanded: list[ConversationMessage] = []
        next_sequence = batch.start_sequence
        for source in batch.messages:
            parts = self._parts(source)
            count = len(parts)
            logical_id = source.logical_message_id or source.message_id
            for index, content in enumerate(parts):
                split = count > 1
                message = replace(
                    source,
                    message_id=(f"{logical_id}~part-{index + 1}-of-{count}" if split else source.message_id),
                    sequence=next_sequence,
                    content=content,
                    logical_message_id=logical_id if split else source.logical_message_id,
                    logical_part_index=index if split else source.logical_part_index,
                    logical_part_count=count if split else source.logical_part_count,
                )
                expanded.append(message)
                next_sequence += 1
        return ConversationBatch(batch.conversation_id, tuple(expanded))

    def _parts(self, message: ConversationMessage) -> tuple[object, ...]:
        if message.logical_part_count is not None:
            if self.token_estimator(message) > self.max_message_tokens:
                raise ValueError("an existing logical message part exceeds max_message_tokens")
            return (message.content,)
        if message.role not in {
            ConversationMessageRole.PROMPT,
            ConversationMessageRole.COMPLETION,
        }:
            return (message.content,)
        if self.token_estimator(message) <= self.max_message_tokens:
            return (message.content,)
        assert isinstance(message.content, str)
        parts = self._split_text(message, message.content, level=0)
        if len(parts) < 2 or "".join(parts) != message.content:
            raise ValueError("conversation text splitter did not preserve the original content")
        return tuple(parts)

    def _split_text(
        self,
        source: ConversationMessage,
        text: str,
        *,
        level: int,
    ) -> list[str]:
        if self._fits(source, text):
            return [text]
        splitters: tuple[Callable[[str], list[str]], ...] = (
            _split_markdown_fences,
            lambda value: _split_preserving(value, r"(\n{2,})"),
            lambda value: _split_preserving(value, r"(\n)"),
            lambda value: _split_after(value, r"(?<=[。！？.!?])"),
            lambda value: _split_preserving(value, r"(\s+)"),
        )
        if level < len(splitters):
            units = splitters[level](text)
            if len(units) > 1:
                resolved: list[str] = []
                for block in self._greedy_blocks(source, units):
                    resolved.extend(self._split_text(source, block, level=level + 1))
                return resolved
            return self._split_text(source, text, level=level + 1)
        return self._hard_split(source, text)

    def _greedy_blocks(self, source: ConversationMessage, units: list[str]) -> list[str]:
        """从左往右把相邻单元贪心并成块：每块从一个单元起（单个放不下的交给下一级再拆），并到再多一个就放不下为止。

        放不放得下随追加文本单调，所以每块的终点按倍增再二分去找，每块只估算 O(log 单元数) 次、每次估的文本不超过
        块长的两倍；不逐个单元重估整块（那样一块之内是平方的，几十 KB 没有换行的单行要跑上百秒）。
        """

        offsets = [0]
        for unit in units:
            offsets.append(offsets[-1] + len(unit))
        text = "".join(units)
        count = len(units)
        blocks: list[str] = []
        start = 0
        while start < count:
            # [start, good) 放得下；终点到 bad 就放不下（count + 1 表示还没见到放不下的）
            good, bad, step = start + 1, count + 1, 1
            while good + 1 < bad:
                candidate = min(good + step, count) if bad > count else (good + bad) // 2
                if self._fits(source, text[offsets[start] : offsets[candidate]]):
                    good = candidate
                    step *= 2
                else:
                    bad = candidate
            blocks.append(text[offsets[start] : offsets[good]])
            start = good
        return blocks

    def _hard_split(self, source: ConversationMessage, text: str) -> list[str]:
        result: list[str] = []
        remaining = text
        while remaining:
            low, high = 1, len(remaining)
            best = 0
            while low <= high:
                middle = (low + high) // 2
                if self._fits(source, remaining[:middle]):
                    best = middle
                    low = middle + 1
                else:
                    high = middle - 1
            if best == 0:
                raise ValueError("max_message_tokens cannot fit one text character and message metadata")
            result.append(remaining[:best])
            remaining = remaining[best:]
        return result

    def _fits(self, source: ConversationMessage, text: str) -> bool:
        probe = replace(
            source,
            message_id=f"{source.message_id}~part-999999-of-999999",
            content=text,
            logical_message_id=source.logical_message_id or source.message_id,
            logical_part_index=999_998,
            logical_part_count=999_999,
        )
        value = self.token_estimator(probe)
        if value < 0:
            raise ValueError("token_estimator must return a non-negative integer")
        return value <= self.max_message_tokens


def _split_markdown_fences(value: str) -> list[str]:
    return [item for item in re.split(r"(```[\s\S]*?```)", value) if item]


def _split_preserving(value: str, pattern: str) -> list[str]:
    return [item for item in re.split(pattern, value) if item]


def _split_after(value: str, pattern: str) -> list[str]:
    points = [match.end() for match in re.finditer(pattern, value)]
    if not points:
        return [value]
    result: list[str] = []
    start = 0
    for end in points:
        if end > start:
            result.append(value[start:end])
            start = end
    if start < len(value):
        result.append(value[start:])
    return result


__all__ = [
    "SEMANTIC_CHUNKER_VERSION",
    "ConversationMessageChunker",
]
