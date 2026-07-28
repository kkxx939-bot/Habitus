"""把真实 SearchService 结果转换为有界、可追踪的回答上下文。"""

from __future__ import annotations

from dataclasses import dataclass

from memory.retrieval import MemorySearchResult
from memory.uri import MemoryURI


@dataclass(frozen=True)
class BenchmarkAnswerContext:
    """回答模型实际看到的内容及其与完整召回结果的差异。"""

    text: str
    memory_uris: tuple[str, ...]
    related_uris: tuple[str, ...]
    summary_ids: tuple[str, ...]
    skipped_ids: tuple[str, ...]
    content_chars: int


@dataclass(frozen=True)
class _ContextBlock:
    identity: str
    kind: str
    content: str


def select_answer_context(result: MemorySearchResult, *, max_content_chars: int) -> BenchmarkAnswerContext:
    """按直接 Memory、关系一跳、Summary 后备的优先级整块选入上下文。"""

    if not isinstance(result, MemorySearchResult):
        raise TypeError("result must be MemorySearchResult")
    if isinstance(max_content_chars, bool) or not isinstance(max_content_chars, int) or max_content_chars <= 0:
        raise ValueError("max_content_chars must be a positive integer")

    direct_blocks: list[_ContextBlock] = []
    related_blocks: list[_ContextBlock] = []
    summary_blocks: list[_ContextBlock] = []
    direct_uris = {str(memory.uri) for memory in result.memories}
    for memory in result.memories:
        uri = str(memory.uri)
        timestamp = memory.document.metadata.updated_at.isoformat().replace("+00:00", "Z")
        direct_blocks.append(
            _ContextBlock(
                identity=uri,
                kind="memory",
                content=(
                    f'<memory uri="{uri}" kind="{memory.document.kind.value}" updated_at="{timestamp}">\n'
                    f"{memory.document.markdown_body.strip()}\n"
                    "</memory>"
                ),
            )
        )
        for related in memory.related:
            related_uri = str(MemoryURI.from_address(related.document.address))
            if related_uri in direct_uris or any(block.identity == related_uri for block in related_blocks):
                continue
            timestamp = related.document.metadata.updated_at.isoformat().replace("+00:00", "Z")
            related_blocks.append(
                _ContextBlock(
                    identity=related_uri,
                    kind="related",
                    content=(
                        f'<related_memory seed_uri="{uri}" uri="{related_uri}" '
                        f'kind="{related.document.kind.value}" '
                        f'link_type="{related.relation.link_type.value}" updated_at="{timestamp}">\n'
                        f"{related.document.markdown_body.strip()}\n"
                        "</related_memory>"
                    ),
                )
            )
    for summary in result.summary_fallbacks:
        identity = summary.reference.identity
        started_at = summary.started_at.isoformat().replace("+00:00", "Z")
        ended_at = summary.ended_at.isoformat().replace("+00:00", "Z")
        summary_blocks.append(
            _ContextBlock(
                identity=identity,
                kind="summary",
                content=(
                    f'<conversation_summary identity="{identity}" started_at="{started_at}" '
                    f'ended_at="{ended_at}">\n{summary.content.strip()}\n</conversation_summary>'
                ),
            )
        )

    blocks = (*direct_blocks, *related_blocks, *summary_blocks)
    selected: list[_ContextBlock] = []
    skipped: list[str] = []
    used = 0
    for block in blocks:
        size = len(block.content) + (2 if selected else 0)
        if used + size > max_content_chars:
            skipped.append(block.identity)
            continue
        selected.append(block)
        used += size
    return BenchmarkAnswerContext(
        text="\n\n".join(block.content for block in selected),
        memory_uris=tuple(block.identity for block in selected if block.kind == "memory"),
        related_uris=tuple(block.identity for block in selected if block.kind == "related"),
        summary_ids=tuple(block.identity for block in selected if block.kind == "summary"),
        skipped_ids=tuple(skipped),
        content_chars=used,
    )


__all__ = ["BenchmarkAnswerContext", "select_answer_context"]
