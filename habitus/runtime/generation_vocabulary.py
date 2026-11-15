"""换代里"数词表"那一步的取材料与落盘（词表方案 2026-11-14 第 5、7 节）：判法在 ``behavior.kinds.active``（纯计数），
这里读这一代的序列、调它、把词表状态写回盘（读、改、写回整段持词表状态的跨进程租约，和选词时长节点的写者互不丢修改），把数出来的在用集合
交给这一代的语义树。

只数会话一支（换代只接会话那一支）。每一代先退出、删掉 20 天没被选中的，再转在用。在用状态写成了、后面的语义树一步没成也
不要紧：下一代同步词表按在用的词核对时自然补上（停用不在用的、新建在用的）。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from habitus.behavior.kinds.active import ActiveCount, count_active
from habitus.behavior.kinds.config import BehaviorKindConfig
from habitus.behavior.kinds.model import Vocabulary
from habitus.behavior.kinds.store import BehaviorKindStore
from habitus.runtime.scene_vocabulary import catalog_words
from habitus.scene.concepts.catalog import CatalogWords
from habitus.series import EventSeries


@dataclass(frozen=True)
class Counted:
    """数一遍的结果：这一代在用的词（交给语义树），这一代新转在用的、够条件却因满额等着的、删掉的、退出在用的个数，
    与数完以后此刻的候选（自己长的、活着的、不在用的节点）个数。"""

    words: CatalogWords
    added: int
    waiting: int
    deleted: int
    exited: int
    candidates: int


class VocabularyCounter:
    def __init__(self, store: BehaviorKindStore, config: BehaviorKindConfig) -> None:
        self.store = store
        self.config = config

    def revision(self) -> int:
        """在用状态此刻的修订号（下游记的"词表版本"）。"""

        return self.store.read().revision

    def count(self, series: EventSeries, today: date) -> Counted:
        """按这一代的序列数一遍、写回盘；什么都没变不写。"""

        picks = tuple((record.kind_token, record.day) for record in series.records)
        found: list[ActiveCount] = []

        def change(vocabulary: Vocabulary) -> Vocabulary:
            found.append(
                count_active(
                    vocabulary,
                    picks,
                    today=today,
                    min_days=self.config.active_min_days,
                    cap=self.config.active_cap,
                    retire_days=self.config.retire_days,
                )
            )
            return found[0].vocabulary

        updated = self.store.update(change)
        counted = found[0]
        return Counted(
            catalog_words(updated),
            added=len(counted.added),
            waiting=len(counted.waiting),
            deleted=len(counted.deleted),
            exited=len(counted.exited),
            candidates=sum(1 for word in updated.words if word.grown and word.id not in updated.active),
        )


__all__ = ["Counted", "VocabularyCounter"]
