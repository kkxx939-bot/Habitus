"""读取各 Conversation 当前有效的 Summary 前沿，供检索按日期或关键词查找。

Summary 文件是唯一真相源；这里只读，不建索引、不改生命周期。没有索引，一次查找要读全部当前有效的摘要。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from habitus.memory.conversation.compaction import ConversationSummaryCompactor
from habitus.memory.conversation.layout import ConversationAddress
from habitus.memory.conversation.messages import ConversationMessageJournal
from habitus.memory.conversation.summary_reference import (
    ConversationSummary,
    ConversationSummaryMatch,
    ConversationSummaryReference,
    summary_reference,
)


class ConversationSummaryRetirementFilter(Protocol):
    def hidden(self, reference: ConversationSummaryReference) -> bool: ...


@dataclass(frozen=True)
class ConversationSummaryReaderConfig:
    """限制一次遍历读多少份摘要、每份交给模型的文字多长。"""

    max_records: int = 100_000
    max_records_per_conversation: int = 30_000
    max_record_chars: int = 16_000

    def __post_init__(self) -> None:
        for name, value, minimum, maximum in (
            ("max_records", self.max_records, 1, 100_000_000),
            ("max_records_per_conversation", self.max_records_per_conversation, 1, 1_000_000),
            ("max_record_chars", self.max_record_chars, 1, 1_000_000),
        ):
            if not minimum <= value <= maximum:
                raise ValueError(f"summary reader {name} must be between {minimum} and {maximum}")
        if self.max_records_per_conversation > self.max_records:
            raise ValueError("summary records per Conversation cannot exceed total records")


class ConversationSummaryReader:
    """按 Conversation 读当前不重叠的活跃摘要前沿；退休中的 Archive 不出现。"""

    def __init__(
        self,
        journal: ConversationMessageJournal,
        compactor: ConversationSummaryCompactor,
        *,
        config: ConversationSummaryReaderConfig | None = None,
        retirement_store: ConversationSummaryRetirementFilter | None = None,
    ) -> None:
        if compactor.journal is not journal:
            raise ValueError("summary reader must share the Conversation journal")
        self.journal = journal
        self.compactor = compactor
        self.config = config or ConversationSummaryReaderConfig()
        self.retirement_store = retirement_store

    def active(self, address: ConversationAddress) -> tuple[ConversationSummaryMatch, ...]:
        """读取单个 Conversation 当前不重叠的活跃摘要前沿。"""

        summaries = self.compactor.frontier(address).active
        if len(summaries) > self.config.max_records_per_conversation:
            raise ValueError("active Conversation Summary frontier exceeds its reader bound")
        return tuple(
            self._match(address, summary)
            for summary in summaries
            if self.retirement_store is None or not self.retirement_store.hidden(summary_reference(address, summary))
        )

    def walk(self) -> tuple[ConversationSummaryMatch, ...]:
        """有界遍历全部 Conversation 的活跃摘要。"""

        matches: dict[str, ConversationSummaryMatch] = {}
        for address in self.journal.list_addresses():
            for match in self.active(address):
                if match.reference.identity in matches:
                    raise ValueError("Conversation Summary identity is duplicated")
                matches[match.reference.identity] = match
                if len(matches) > self.config.max_records:
                    raise ValueError("Conversation Summary walk exceeded its record bound")
        return tuple(matches[identity] for identity in sorted(matches))

    def _match(self, address: ConversationAddress, summary: ConversationSummary) -> ConversationSummaryMatch:
        reference = summary_reference(address, summary)
        lines = [
            f"overview: {summary.overview}",
            "chronology:",
            *(f"- {item}" for item in summary.chronology),
        ]
        if summary.corrections:
            lines.extend(("corrections:", *(f"- {item}" for item in summary.corrections)))
        lines.append(f"ending_state: {summary.ending_state}")
        if summary.open_threads:
            lines.extend(("open_threads:", *(f"- {item}" for item in summary.open_threads)))
        return ConversationSummaryMatch(reference, summary, self._bounded("\n".join(lines)))

    def _bounded(self, value: str) -> str:
        maximum = self.config.max_record_chars
        if len(value) <= maximum:
            return value
        if maximum <= 3:
            return value[:maximum]
        return value[: maximum - 3].rstrip() + "..."


__all__ = [
    "ConversationSummaryReader",
    "ConversationSummaryReaderConfig",
    "ConversationSummaryRetirementFilter",
]
