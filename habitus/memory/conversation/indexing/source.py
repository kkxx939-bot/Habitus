"""从 Conversation 当前活跃 Summary 前沿生成可重建索引源。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

from habitus.memory.conversation.compaction import ConversationSummaryCompactor
from habitus.memory.conversation.indexing.config import ConversationSummaryVectorIndexConfig
from habitus.memory.conversation.indexing.model import (
    ConversationSummary,
    ConversationSummaryIndexSource,
    ConversationSummaryReference,
    summary_reference,
)
from habitus.memory.conversation.layout import ConversationAddress
from habitus.memory.conversation.messages import ConversationMessageJournal
from habitus.pre.conversation import ConversationRangeSummaryStage


class ConversationSummaryRetirementFilter(Protocol):
    def hidden(self, reference: ConversationSummaryReference) -> bool: ...


class ConversationSummaryIndexSourceReader:
    """只读取 Summary 真相源，不生成向量或修改生命周期。"""

    def __init__(
        self,
        journal: ConversationMessageJournal,
        compactor: ConversationSummaryCompactor,
        *,
        config: ConversationSummaryVectorIndexConfig,
        retirement_store: ConversationSummaryRetirementFilter | None = None,
    ) -> None:
        if not isinstance(journal, ConversationMessageJournal):
            raise TypeError("journal must be ConversationMessageJournal")
        if not isinstance(compactor, ConversationSummaryCompactor):
            raise TypeError("compactor must be ConversationSummaryCompactor")
        if compactor.journal is not journal:
            raise ValueError("summary source reader must share the Conversation journal")
        if not isinstance(config, ConversationSummaryVectorIndexConfig):
            raise TypeError("config must be ConversationSummaryVectorIndexConfig")
        self.journal = journal
        self.compactor = compactor
        self.config = config
        if retirement_store is not None and not callable(getattr(retirement_store, "hidden", None)):
            raise TypeError("retirement_store must implement hidden(reference)")
        self.retirement_store = retirement_store

    def active(self, address: ConversationAddress) -> tuple[ConversationSummaryIndexSource, ...]:
        """读取单个 Conversation 当前不重叠的唯一活跃摘要前沿。"""

        if not isinstance(address, ConversationAddress):
            raise TypeError("address must be ConversationAddress")
        summaries = self.compactor.frontier(address).active
        if len(summaries) > self.config.max_records_per_conversation:
            raise ValueError("active Conversation Summary frontier exceeds its index bound")
        return tuple(
            self._source(address, summary)
            for summary in summaries
            if self.retirement_store is None
            or not self.retirement_store.hidden(summary_reference(address, summary))
        )

    def all_references(self, address: ConversationAddress) -> tuple[ConversationSummaryReference, ...]:
        """枚举仍在物理保存的全部摘要身份，供活跃前沿切换时删除旧记录。"""

        if not isinstance(address, ConversationAddress):
            raise TypeError("address must be ConversationAddress")
        values: list[ConversationSummary] = [*self.compactor.segment_store.list(address)]
        values.extend(
            self.compactor.range_store.list(address, ConversationRangeSummaryStage.RANGE)
        )
        values.extend(
            self.compactor.range_store.list(address, ConversationRangeSummaryStage.ARCHIVE)
        )
        if len(values) > self.config.max_records_per_conversation:
            raise ValueError("physical Conversation summaries exceed their index reconciliation bound")
        references = tuple(summary_reference(address, summary) for summary in values)
        if len({reference.identity for reference in references}) != len(references):
            raise ValueError("Conversation Summary stores contain duplicate identities")
        return tuple(sorted(references, key=lambda item: item.identity))

    def walk(self) -> tuple[ConversationSummaryIndexSource, ...]:
        """有界遍历全部 Conversation 的活跃 Summary，供完整重建使用。"""

        sources: dict[str, ConversationSummaryIndexSource] = {}
        for address in self.journal.list_addresses():
            for source in self.active(address):
                if source.identity in sources:
                    raise ValueError("Conversation Summary index identity is duplicated")
                sources[source.identity] = source
                self._require_total_bound(sources)
        return tuple(sources[identity] for identity in sorted(sources))

    def resolve_active(
        self,
        reference: ConversationSummaryReference,
    ) -> ConversationSummaryIndexSource | None:
        """以真相源重新验证远程命中仍属于当前活跃前沿。"""

        if not isinstance(reference, ConversationSummaryReference):
            raise TypeError("reference must be ConversationSummaryReference")
        return {source.identity: source for source in self.active(reference.address)}.get(
            reference.identity
        )

    def _source(
        self,
        address: ConversationAddress,
        summary: ConversationSummary,
    ) -> ConversationSummaryIndexSource:
        reference = summary_reference(address, summary)
        lines = [
            "[conversation_summary]",
            f"conversation_id: {summary.conversation_id}",
            f"stage: {reference.stage.value}",
            f"summary_id: {reference.summary_id}",
            f"time_range: {summary.started_at.isoformat()} -> {summary.ended_at.isoformat()}",
            f"overview: {summary.overview}",
            "chronology:",
            *(f"- {item}" for item in summary.chronology),
        ]
        if summary.corrections:
            lines.extend(("corrections:", *(f"- {item}" for item in summary.corrections)))
        lines.extend((f"ending_state: {summary.ending_state}",))
        if summary.open_threads:
            lines.extend(("open_threads:", *(f"- {item}" for item in summary.open_threads)))
        content = self._bounded("\n".join(lines))
        return ConversationSummaryIndexSource(reference, summary, content)

    def _bounded(self, value: str) -> str:
        maximum = self.config.max_record_chars
        if len(value) <= maximum:
            return value
        if maximum <= 3:
            return value[:maximum]
        return value[: maximum - 3].rstrip() + "..."

    def _require_total_bound(
        self,
        values: Mapping[str, ConversationSummaryIndexSource],
    ) -> None:
        if len(values) > self.config.max_records:
            raise ValueError("Conversation Summary vector rebuild exceeded its record bound")


__all__ = ["ConversationSummaryIndexSourceReader"]
