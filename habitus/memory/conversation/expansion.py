"""按不可变 source_refs/digest 临时展开 Summary 来源链，不改变活跃前沿。"""

from __future__ import annotations

from dataclasses import replace

from habitus.memory.conversation.compaction import ConversationRangeSummaryStore
from habitus.memory.conversation.indexing.model import ConversationSummaryMatch
from habitus.memory.conversation.summary import ConversationSummaryStore
from habitus.pre.conversation import (
    ConversationRangeSummary,
    ConversationRangeSummaryStage,
    ConversationSegmentSummary,
    ConversationSummarySourceKind,
)


class ConversationSummaryExpansionError(RuntimeError):
    """Summary 来源链缺失、摘要被篡改或超过展开边界。"""


class ConversationSummaryExpander:
    """召回后按需补充子摘要语义；不恢复为永久活跃索引记录。"""

    def __init__(
        self,
        segment_store: ConversationSummaryStore,
        range_store: ConversationRangeSummaryStore,
        *,
        max_source_reads: int = 10_000,
    ) -> None:
        if not isinstance(segment_store, ConversationSummaryStore):
            raise TypeError("segment_store must be ConversationSummaryStore")
        if not isinstance(range_store, ConversationRangeSummaryStore):
            raise TypeError("range_store must be ConversationRangeSummaryStore")
        if segment_store.layout.root != range_store.layout.root:
            raise ValueError("Summary expander stores must share one Conversation root")
        if (
            isinstance(max_source_reads, bool)
            or not isinstance(max_source_reads, int)
            or not 1 <= max_source_reads <= 100_000
        ):
            raise ValueError("Summary expansion max_source_reads is outside its supported range")
        self.segment_store = segment_store
        self.range_store = range_store
        self.max_source_reads = max_source_reads

    def expand(
        self,
        match: ConversationSummaryMatch,
        *,
        max_chars: int,
    ) -> ConversationSummaryMatch:
        if not isinstance(match, ConversationSummaryMatch):
            raise TypeError("match must be ConversationSummaryMatch")
        if isinstance(max_chars, bool) or not isinstance(max_chars, int) or max_chars <= 0:
            raise ValueError("Summary expansion max_chars must be positive")
        if isinstance(match.summary, ConversationSegmentSummary):
            return match
        blocks = [match.content.strip()]
        used = len(blocks[0])
        seen: set[tuple[str, str]] = set()
        reads = [0]
        for source in self._walk(match.reference.address, match.summary, seen, reads):
            block = self._block(source)
            required = len(block) + 1
            if used + required > max_chars:
                break
            blocks.append(block)
            used += required
        if len(blocks) == 1:
            return match
        return replace(match, content="\n".join(blocks))

    def _walk(self, address, parent: ConversationRangeSummary, seen, reads):
        for reference in parent.source_refs:
            if reads[0] >= self.max_source_reads:
                raise ConversationSummaryExpansionError(
                    "Summary expansion exceeds max_source_reads"
                )
            key = (reference.kind.value, reference.summary_id)
            if key in seen:
                raise ConversationSummaryExpansionError("Summary source graph contains a cycle or duplicate")
            seen.add(key)
            source: ConversationSegmentSummary | ConversationRangeSummary | None
            if reference.kind is ConversationSummarySourceKind.SEGMENT:
                source = self.segment_store.try_read_by_id(address, reference.summary_id)
            elif reference.kind is ConversationSummarySourceKind.RANGE:
                source = self.range_store.try_read(
                    address,
                    ConversationRangeSummaryStage.RANGE,
                    reference.summary_id,
                )
            else:
                raise ConversationSummaryExpansionError("Summary source reference kind is unsupported")
            if source is None:
                raise ConversationSummaryExpansionError("Summary source recovery artifact is missing")
            reads[0] += 1
            if source.digest != reference.digest:
                raise ConversationSummaryExpansionError("Summary source digest does not match its parent")
            yield source
            if isinstance(source, ConversationRangeSummary):
                yield from self._walk(address, source, seen, reads)

    @staticmethod
    def _block(summary: ConversationSegmentSummary | ConversationRangeSummary) -> str:
        identifier = summary.segment_id if isinstance(summary, ConversationSegmentSummary) else summary.range_id
        stage = "segment" if isinstance(summary, ConversationSegmentSummary) else summary.stage.value
        lines = [
            f'<conversation_summary_source stage="{stage}" summary_id="{identifier}">',
            f"overview: {summary.overview}",
            "chronology:",
            *(f"- {item}" for item in summary.chronology),
        ]
        if summary.corrections:
            lines.extend(("corrections:", *(f"- {item}" for item in summary.corrections)))
        lines.append(f"ending_state: {summary.ending_state}")
        if summary.open_threads:
            lines.extend(("open_threads:", *(f"- {item}" for item in summary.open_threads)))
        lines.append("</conversation_summary_source>")
        return "\n".join(lines)


__all__ = ["ConversationSummaryExpander", "ConversationSummaryExpansionError"]
