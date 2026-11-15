"""词表口的实现：用基础词表的在用状态装出语义树要的 ``ClassCatalog``（裁定 20；词表方案 2026-11-14）。

scene 不 import 词表包，所以"在用状态怎么读成在用的词"住在组合根（occurrence 上的编号怎么读归事件序列 ``series.reader``）。
"""

from __future__ import annotations

from habitus.behavior.kinds.model import Vocabulary
from habitus.behavior.kinds.store import BehaviorKindStore
from habitus.scene.concepts.catalog import CatalogWord, CatalogWords


def catalog_words(vocabulary: Vocabulary) -> CatalogWords:
    """一份词表的在用的词与修订号。"""

    return CatalogWords(
        version=vocabulary.revision,
        words=tuple(
            CatalogWord(id=str(word.id), name=word.name, definition=word.definition, lane=word.lane.value)
            for word in vocabulary.in_use()
        ),
    )


class VocabularyCatalog:
    """按 ``ClassCatalog`` 协议实现：每次调用现读在用状态（文件很小；只在换代里变）。"""

    def __init__(self, store: BehaviorKindStore) -> None:
        self.store = store

    def current(self) -> CatalogWords:
        return catalog_words(self.store.read())


__all__ = ["VocabularyCatalog", "catalog_words"]
