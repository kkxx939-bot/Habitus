"""超大 Conversation 的文本分块与可选语义边界评分。"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Protocol, cast

from foundation.integrity import canonical_json
from pre.conversation import (
    ConversationBatch,
    ConversationMessage,
    ConversationMessageRole,
)

logger = logging.getLogger(__name__)

SEMANTIC_CHUNKER_VERSION = "conversation_semantic_chunker_v1"


class ConversationBoundaryVector(Protocol):
    """边界评分只依赖归一化向量值，不绑定具体模型客户端。"""

    values: tuple[float, ...]


class ConversationBoundaryEmbedder(Protocol):
    """供 Conversation 切段使用的最小异步 Embedding 契约。"""

    async def embed_documents(
        self,
        texts: Sequence[str],
    ) -> Sequence[ConversationBoundaryVector]: ...


@dataclass(frozen=True)
class ConversationSemanticBoundary:
    """一个安全消息边界两侧的语义距离。"""

    after_sequence: int
    distance: float

    def __post_init__(self) -> None:
        if (
            isinstance(self.after_sequence, bool)
            or not isinstance(self.after_sequence, int)
            or self.after_sequence < 0
        ):
            raise ValueError("after_sequence must be a non-negative integer")
        if (
            isinstance(self.distance, bool)
            or not isinstance(self.distance, int | float)
            or not 0.0 <= float(self.distance) <= 2.0
        ):
            raise ValueError("semantic boundary distance must be between zero and two")
        object.__setattr__(self, "distance", float(self.distance))


@dataclass(frozen=True)
class ConversationBoundaryHints:
    """与一个 live 快照严格绑定的可选语义提示。"""

    source_digest: str
    boundaries: tuple[ConversationSemanticBoundary, ...]
    embedding_fingerprint: str | None
    fallback_reason: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.source_digest, str) or len(self.source_digest) != 64:
            raise ValueError("source_digest must be a SHA-256 digest")
        if not isinstance(self.boundaries, tuple) or any(
            not isinstance(item, ConversationSemanticBoundary) for item in self.boundaries
        ):
            raise TypeError("boundaries must contain ConversationSemanticBoundary values")
        sequences = tuple(item.after_sequence for item in self.boundaries)
        if sequences != tuple(sorted(set(sequences))):
            raise ValueError("semantic boundaries must be unique and ordered")
        if self.embedding_fingerprint is not None and (
            not isinstance(self.embedding_fingerprint, str)
            or not self.embedding_fingerprint.strip()
        ):
            raise ValueError("embedding_fingerprint must be non-empty text or None")
        if self.fallback_reason is not None and (
            not isinstance(self.fallback_reason, str)
            or not self.fallback_reason.strip()
        ):
            raise ValueError("fallback_reason must be non-empty text or None")

    def distance_after(self, sequence: int) -> float | None:
        for item in self.boundaries:
            if item.after_sequence == sequence:
                return item.distance
        return None


class ConversationSemanticBoundaryScorer:
    """用相邻安全单元的向量距离辅助选点，失败时返回结构化降级结果。"""

    def __init__(
        self,
        embedder: object,
        *,
        embedding_fingerprint: str,
        max_unit_chars: int,
    ) -> None:
        if not callable(getattr(embedder, "embed_documents", None)):
            raise TypeError("embedder must implement embed_documents")
        if not isinstance(embedding_fingerprint, str) or not embedding_fingerprint.strip():
            raise ValueError("embedding_fingerprint must be non-empty text")
        if (
            isinstance(max_unit_chars, bool)
            or not isinstance(max_unit_chars, int)
            or max_unit_chars < 256
        ):
            raise ValueError("max_unit_chars must be at least 256")
        self.embedder = cast(ConversationBoundaryEmbedder, embedder)
        self.embedding_fingerprint = embedding_fingerprint.strip()
        self.max_unit_chars = max_unit_chars

    async def score(self, live: ConversationBatch | None) -> ConversationBoundaryHints | None:
        """只为安全结构边界生成提示；任何模型失败都不阻断 Conversation。"""

        if live is None:
            return None
        if not isinstance(live, ConversationBatch):
            raise TypeError("live must be ConversationBatch or None")
        units = _safe_semantic_units(live.messages)
        if len(units) < 2:
            return ConversationBoundaryHints(live.digest, (), None, "fewer than two safe units")
        texts = tuple(self._bounded_text(text) for _sequence, text in units)
        try:
            vectors = await self.embedder.embed_documents(texts)
            if len(vectors) != len(units):
                raise ValueError("embedding result count does not match semantic units")
            boundaries = tuple(
                ConversationSemanticBoundary(
                    after_sequence=units[index][0],
                    distance=max(
                        0.0,
                        min(
                            2.0,
                            1.0
                            - sum(
                                left * right
                                for left, right in zip(
                                    vectors[index].values,
                                    vectors[index + 1].values,
                                    strict=True,
                                )
                            ),
                        ),
                    ),
                )
                for index in range(len(units) - 1)
            )
            return ConversationBoundaryHints(
                live.digest,
                boundaries,
                self.embedding_fingerprint,
            )
        except Exception as exc:
            reason = f"{type(exc).__name__}: {' '.join(str(exc).split())}"[:500]
            logger.warning(
                "Conversation 语义边界评分失败，退回确定性结构边界：%s",
                type(exc).__name__,
            )
            return ConversationBoundaryHints(live.digest, (), None, reason)

    def _bounded_text(self, value: str) -> str:
        if len(value) <= self.max_unit_chars:
            return value
        head = self.max_unit_chars * 2 // 3
        tail = self.max_unit_chars - head
        return value[:head] + value[-tail:]


class ConversationMessageChunker:
    """在写入 live 前把单条超长文本拆成可重组的物理消息。"""

    def __init__(
        self,
        *,
        max_message_tokens: int,
        token_estimator: Callable[[ConversationMessage], int],
    ) -> None:
        if (
            isinstance(max_message_tokens, bool)
            or not isinstance(max_message_tokens, int)
            or max_message_tokens <= 0
        ):
            raise ValueError("max_message_tokens must be a positive integer")
        if not callable(token_estimator):
            raise TypeError("token_estimator must be callable")
        self.max_message_tokens = max_message_tokens
        self.token_estimator = token_estimator

    def normalize(self, batch: ConversationBatch) -> ConversationBatch:
        """确定性分块并重新分配当前批次内的连续物理序号。"""

        if not isinstance(batch, ConversationBatch):
            raise TypeError("batch must be ConversationBatch")
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
                    message_id=(
                        f"{logical_id}~part-{index + 1}-of-{count}"
                        if split
                        else source.message_id
                    ),
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
        splitters = (
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
                current = ""
                for unit in units:
                    candidate = current + unit
                    if current and not self._fits(source, candidate):
                        resolved.extend(self._split_text(source, current, level=level + 1))
                        current = unit
                    else:
                        current = candidate
                if current:
                    resolved.extend(self._split_text(source, current, level=level + 1))
                return resolved
            return self._split_text(source, text, level=level + 1)
        return self._hard_split(source, text)

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
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("token_estimator must return a non-negative integer")
        return value <= self.max_message_tokens


def _safe_semantic_units(
    messages: tuple[ConversationMessage, ...],
) -> tuple[tuple[int, str], ...]:
    units: list[tuple[int, str]] = []
    buffered: list[ConversationMessage] = []
    open_tool_calls: set[str] = set()
    for index, message in enumerate(messages):
        buffered.append(message)
        if message.role is ConversationMessageRole.TOOL_CALL:
            assert message.tool_call_id is not None
            open_tool_calls.add(message.tool_call_id)
        elif message.role is ConversationMessageRole.TOOL_RESULT:
            assert message.tool_call_id is not None
            open_tool_calls.discard(message.tool_call_id)
        if open_tool_calls:
            continue
        following = messages[index + 1] if index + 1 < len(messages) else None
        if (
            message.role is ConversationMessageRole.COMPLETION
            and message.completes_logical_message
            and following is not None
            and not (
                following.role is ConversationMessageRole.PROMPT
                and not following.is_logical_continuation
            )
        ):
            continue
        rendered = "\n".join(
            f"{item.role.value}: {canonical_json(item.content)}" for item in buffered
        )
        units.append((message.sequence, rendered))
        buffered = []
    return tuple(units)


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
    "ConversationBoundaryHints",
    "ConversationMessageChunker",
    "ConversationSemanticBoundary",
    "ConversationSemanticBoundaryScorer",
]
