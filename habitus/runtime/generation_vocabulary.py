"""换代里"数在用的词"那一步的取材料与落盘（词表方案 2026-11-14 第 5、7 节）：判法在 ``behavior.kinds.active``（纯计数），
这里读这一代的序列、调它、把在用状态写回盘（CAS），把数出来的集合交给这一代的语义树。

只数会话一支（换代只接会话那一支）。在用只增不减、重数结果一样：在用状态写成了、后面的语义树一步没成也不要紧，下一代同步词表
按在用的词核对时自然补上。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from habitus.behavior.kinds.active import count_active
from habitus.behavior.kinds.config import BehaviorKindConfig
from habitus.behavior.kinds.store import BehaviorKindStore
from habitus.runtime.scene_vocabulary import catalog_words
from habitus.scene.concepts.catalog import CatalogWords
from habitus.series import EventSeries


@dataclass(frozen=True)
class Counted:
    """数一遍的结果：这一代在用的词（交给语义树）、这一代新转在用的与够条件却因满额等着的个数。"""

    words: CatalogWords
    added: int
    waiting: int


class VocabularyCounter:
    def __init__(self, store: BehaviorKindStore, config: BehaviorKindConfig) -> None:
        self.store = store
        self.config = config

    def revision(self) -> int:
        """在用状态此刻的修订号（下游记的"词表版本"）。"""

        return self.store.read().revision

    def count(self, series: EventSeries, today: date) -> Counted:
        """按这一代的序列数一遍、写回盘；在用集合没变不写。"""

        vocabulary = self.store.read()
        counted = count_active(
            vocabulary,
            ((record.kind_token, record.day) for record in series.records),
            today=today,
            min_days=self.config.active_min_days,
            cap=self.config.active_cap,
        )
        updated = self.store.replace(counted.vocabulary, expected_revision=vocabulary.revision)
        return Counted(catalog_words(updated), len(counted.added), len(counted.waiting))


__all__ = ["Counted", "VocabularyCounter"]
