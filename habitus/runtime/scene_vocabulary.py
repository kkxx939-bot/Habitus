"""词表口的实现：用基础词表的在用状态装出语义树要的 ``ClassCatalog``（裁定 20；词表方案 2026-11-14），读行为树上
``kind_token`` 的换算口（``series.model.KindResolution``：删掉的节点换成最近的活着的上级，词表方案二乙），以及预测层的上下级口
（``foresight.hierarchy.KindHierarchyPort``：每个活着的词的全部下级、在用的词，丙一）。

scene、序列、预测层都不 import 词表状态，所以"在用状态怎么读成在用的词""编号怎么换算"住在组合根（occurrence 上的编号
怎么分流归事件序列 ``series.reader``）。
"""

from __future__ import annotations

from collections.abc import Callable

from habitus.behavior.kinds.ids import ClassId, KindIdError
from habitus.behavior.kinds.model import Vocabulary
from habitus.behavior.kinds.store import BehaviorKindStore
from habitus.foresight.hierarchy import KindHierarchy, KindHierarchyPort
from habitus.scene.concepts.catalog import CatalogWord, CatalogWords


def catalog_words(vocabulary: Vocabulary) -> CatalogWords:
    """一份词表的在用的词与修订号；每个词带上它全部活着的下级（按编号）。"""

    return CatalogWords(
        version=vocabulary.revision,
        words=tuple(
            CatalogWord(
                id=str(word.id),
                name=word.name,
                definition=word.definition,
                lane=word.lane.value,
                below=tuple(str(item) for item in sorted(all_below(vocabulary, word.id))),
            )
            for word in vocabulary.in_use()
        ),
    )


def all_below(vocabulary: Vocabulary, word_id: ClassId) -> frozenset[ClassId]:
    """一个词全部活着的下级（下级的下级也算），不含自己。"""

    found: set[ClassId] = set()
    pending = [word_id]
    while pending:
        for child in vocabulary.children(pending.pop()):
            found.add(child.id)
            pending.append(child.id)
    return frozenset(found)


class VocabularyCatalog:
    """按 ``ClassCatalog`` 协议实现：每次调用现读在用状态（文件很小；只在换代里变）。"""

    def __init__(self, store: BehaviorKindStore) -> None:
        self.store = store

    def current(self) -> CatalogWords:
        return catalog_words(self.store.read())


def kind_resolution(store: BehaviorKindStore) -> Callable[[], Callable[[str], str]]:
    """这一支的换算口：每次调用读一次此刻的词表状态，给出它的 编号 → 编号 换算（``Vocabulary.resolve_token``）。"""

    def resolution() -> Callable[[str], str]:
        return store.read().resolve_token

    return resolution


def kind_hierarchy(store: BehaviorKindStore) -> KindHierarchyPort:
    """这一支的上下级口（丙一 2026-11-15）：每次调用读一次此刻的词表状态，给出每个活着的词的全部下级与在用的词。预测层每拍开头
    取一份、整拍用同一份。"""

    def hierarchy() -> KindHierarchy:
        vocabulary = store.read()
        return KindHierarchy(
            below={
                str(word.id): frozenset(str(item) for item in all_below(vocabulary, word.id))
                for word in vocabulary.words
            },
            in_use=frozenset(str(word_id) for word_id in vocabulary.active),
        )

    return hierarchy


def kind_label(store: BehaviorKindStore) -> Callable[[str], str | None]:
    """词编号 → 词表里给人看的词名（不是编号、不是活着的词的给 None）。"""

    def label(token: str) -> str | None:
        try:
            word_id = ClassId.parse(token)
            vocabulary = store.read()
        except (KindIdError, ValueError):  # 词名只是出口配题目的辅助：不是编号、词表读不出来都不拦取
            return None
        return vocabulary.get(word_id).name if word_id in vocabulary else None

    return label


__all__ = ["VocabularyCatalog", "all_below", "catalog_words", "kind_hierarchy", "kind_label", "kind_resolution"]
