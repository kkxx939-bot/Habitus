"""词表的值对象：词、一条 lane 的词表（只读的初始词 + 在用状态）。

只做自洽校验（编号与 lane 一致、编号与名字不重复、上级在表里且层数 = 上级层数 + 1、在用的词在表里），
不规定现实里的行为该长什么样（词表方案 2026-11-14 第 2 节）。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from types import MappingProxyType

from habitus.behavior.kinds.ids import ClassId, Lane

#: 词表最多几层（词表方案第 6 节：ATUS、ICATUS、Anthropic Economic Index 都是 3 层）。
MAX_LEVEL = 3


class BehaviorKindError(ValueError):
    pass


def _text(value: object, field_name: str) -> str:
    """词表条目里的一段文字：非空、首尾无空白、单行可打印。"""

    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise BehaviorKindError(f"{field_name} must be non-empty text without surrounding whitespace")
    if any(not character.isprintable() for character in value):
        raise BehaviorKindError(f"{field_name} must be a single printable line")
    return value


@dataclass(frozen=True)
class Word:
    """词表里的一个词。``domain`` 只是提示词里分组的标题，不是词。"""

    id: ClassId
    level: int
    name: str
    parent: ClassId | None
    definition: str
    domain: str

    def __post_init__(self) -> None:
        # 字段类型在边界（``words.py`` 读数据文件）已经核过，这里只核自洽
        if not 1 <= self.level <= MAX_LEVEL:
            raise BehaviorKindError(f"word {self.id} level must be between 1 and {MAX_LEVEL}")
        if (self.parent is None) != (self.level == 1):
            raise BehaviorKindError(f"word {self.id}: only level-1 words have no parent")
        if self.parent is not None and self.parent.lane is not self.id.lane:
            raise BehaviorKindError(f"word {self.id} parent must be a word of the same lane")
        _text(self.name, "word name")
        _text(self.definition, "word definition")
        _text(self.domain, "word domain")

    @property
    def lane(self) -> Lane:
        return self.id.lane


def reminder_sentence(word: Word) -> str:
    """给人看的提醒句：词表里没有提醒句字段，代码拿词的名字套句式（词表方案第 2 节）。"""

    return f"该{word.name}了"


@dataclass(frozen=True)
class Vocabulary:
    """一条 lane 的词表：按源文件顺序的全部词，加在用状态（哪些词在用、从哪天起）与它的修订号。

    修订号是在用状态的版本：在用集合每变一次加一；下游记的"词表版本"就是它。
    """

    lane: Lane
    words: Sequence[Word]
    active: Mapping[ClassId, date] = field(default_factory=dict)
    revision: int = 0
    _by_id: Mapping[ClassId, Word] = field(init=False, repr=False, compare=False, default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "lane", Lane(self.lane))
        if self.revision < 0:
            raise BehaviorKindError("vocabulary revision must be a non-negative integer")
        by_id: dict[ClassId, Word] = {}
        names: set[str] = set()
        for word in self.words:
            if word.lane is not self.lane:
                raise BehaviorKindError(f"word {word.id} does not belong to lane {self.lane.value}")
            if word.id in by_id:
                raise BehaviorKindError(f"word id repeats: {word.id}")
            if word.name in names:
                raise BehaviorKindError(f"word name repeats: {word.name}")
            by_id[word.id] = word
            names.add(word.name)
        for word in self.words:
            if word.parent is None:
                continue
            parent = by_id.get(word.parent)
            if parent is None:
                raise BehaviorKindError(f"word {word.id} points to an unknown parent {word.parent}")
            if word.level != parent.level + 1:
                raise BehaviorKindError(f"word {word.id} level {word.level} does not follow its parent {parent.id}")
        for word_id in self.active:
            if word_id not in by_id:
                raise BehaviorKindError(f"active word is not in the vocabulary: {word_id}")
        object.__setattr__(self, "words", tuple(self.words))
        object.__setattr__(self, "active", MappingProxyType(dict(sorted(self.active.items()))))
        object.__setattr__(self, "_by_id", MappingProxyType(by_id))

    def get(self, word_id: ClassId) -> Word:
        try:
            return self._by_id[word_id]
        except KeyError:
            raise BehaviorKindError(f"unknown word: {word_id}") from None

    def __contains__(self, word_id: object) -> bool:
        return word_id in self._by_id

    def in_use(self) -> tuple[Word, ...]:
        """在用的词（按编号）。"""

        return tuple(self._by_id[word_id] for word_id in self.active)

    def with_active(self, active: Mapping[ClassId, date]) -> Vocabulary:
        """换一份在用集合；集合变了修订号加一，没变原样返回。"""

        if dict(active) == dict(self.active):
            return self
        return Vocabulary(self.lane, self.words, active, self.revision + 1)


__all__ = [
    "MAX_LEVEL",
    "BehaviorKindError",
    "Vocabulary",
    "Word",
    "reminder_sentence",
]
