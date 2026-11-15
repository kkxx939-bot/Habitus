"""词表的值对象：词、一条 lane 的词表（初始词 + 这个用户自己长的、还活着的节点 + "已删"的记号 + 在用状态）。

只做自洽校验（编号与 lane 一致、编号不重复、活着的节点名字不重复、上级活着且层数 = 上级层数 + 1、在用与退出过的词活着、
已删的记号一路往上能落到活着的节点），不规定现实里的行为该长什么样（词表方案 2026-11-14 第 2、4、5 节）。
层数上限、名字长度是配置（``config.py``），由读状态文件的边界与长节点时查，不在值对象里。
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from types import MappingProxyType

from habitus.behavior.kinds.ids import ClassId, KindIdError, Lane


class BehaviorKindError(ValueError):
    pass


class WordOrigin(str, Enum):
    """词从哪来：随代码发布的初始词，或这个用户的选词时模型写出来的（自己长的）。"""

    INITIAL = "initial"
    GROWN = "grown"


def _text(value: object, field_name: str) -> str:
    """词表条目里的一段文字：非空、首尾无空白、单行可打印。"""

    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise BehaviorKindError(f"{field_name} must be non-empty text without surrounding whitespace")
    if any(not character.isprintable() for character in value):
        raise BehaviorKindError(f"{field_name} must be a single printable line")
    return value


def _aware(value: datetime | None, field_name: str) -> None:
    if value is not None and value.utcoffset() is None:
        raise BehaviorKindError(f"{field_name} must be a timezone-aware datetime")


@dataclass(frozen=True)
class Word:
    """词表里的一个词。``domain`` 只是提示词里分组的标题，不是词；自己长的节点跟它的上级。

    自己长的节点带 ``created_at``（哪一刻写的）与 ``last_picked``（最后一次被直接选中的时刻，没被选中过是 None）；
    初始词两样都没有。
    """

    id: ClassId
    level: int
    name: str
    parent: ClassId | None
    definition: str
    domain: str
    origin: WordOrigin = WordOrigin.INITIAL
    created_at: datetime | None = None
    last_picked: datetime | None = None

    def __post_init__(self) -> None:
        # 字段类型在边界（``words.py`` 读数据文件、``store.py`` 读状态文件）已经核过，这里只核自洽
        object.__setattr__(self, "origin", WordOrigin(self.origin))
        if self.level < 1:
            raise BehaviorKindError(f"word {self.id} level must be a positive integer")
        if (self.parent is None) != (self.level == 1):
            raise BehaviorKindError(f"word {self.id}: only level-1 words have no parent")
        if self.parent is not None and self.parent.lane is not self.id.lane:
            raise BehaviorKindError(f"word {self.id} parent must be a word of the same lane")
        if self.grown:
            if self.created_at is None:
                raise BehaviorKindError(f"grown word {self.id} needs the moment it was written")
            if self.parent is None:
                raise BehaviorKindError(f"grown word {self.id} needs a parent")
        elif self.created_at is not None or self.last_picked is not None:
            raise BehaviorKindError(f"initial word {self.id} has no written or picked moment")
        _aware(self.created_at, f"word {self.id} created_at")
        _aware(self.last_picked, f"word {self.id} last_picked")
        _text(self.name, "word name")
        _text(self.definition, "word definition")
        _text(self.domain, "word domain")

    @property
    def lane(self) -> Lane:
        return self.id.lane

    @property
    def grown(self) -> bool:
        return self.origin is WordOrigin.GROWN

    @property
    def seen(self) -> datetime | None:
        """自己长的节点最近一次"露面"：被直接选中的时刻，没被选中过就是写出它的时刻。初始词为 None。"""

        return self.last_picked if self.last_picked is not None else self.created_at


@dataclass(frozen=True)
class DeletedWord:
    """删掉的自己长的节点留下的记号：编号、上级、哪天删的。树上挂在它上面的旧记录按它往上读到最近的活着的上级。"""

    id: ClassId
    parent: ClassId
    deleted_on: date


def reminder_sentence(word: Word) -> str:
    """给人看的提醒句：词表里没有提醒句字段，代码拿词的名字套句式（词表方案第 2 节）。"""

    return f"该{word.name}了"


@dataclass(frozen=True)
class Vocabulary:
    """一条 lane 的词表：活着的词（初始词按源文件顺序，后面是自己长的、按编号）、"已删"的记号、在用状态
    （哪些词在用、从哪天起）与它的修订号，以及退出过在用的词哪天退出的（之后只数退出那天之后的日子）。

    修订号是在用状态的版本：在用集合每变一次加一；下游记的"词表版本"就是它。长节点、删候选不动它。
    """

    lane: Lane
    words: Sequence[Word]
    active: Mapping[ClassId, date] = field(default_factory=dict)
    revision: int = 0
    deleted: Sequence[DeletedWord] = ()
    exited: Mapping[ClassId, date] = field(default_factory=dict)
    _by_id: Mapping[ClassId, Word] = field(init=False, repr=False, compare=False, default_factory=dict)
    _by_name: Mapping[str, Word] = field(init=False, repr=False, compare=False, default_factory=dict)
    _gone: Mapping[ClassId, DeletedWord] = field(init=False, repr=False, compare=False, default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "lane", Lane(self.lane))
        if self.revision < 0:
            raise BehaviorKindError("vocabulary revision must be a non-negative integer")
        by_id: dict[ClassId, Word] = {}
        by_name: dict[str, Word] = {}
        for word in self.words:
            if word.lane is not self.lane:
                raise BehaviorKindError(f"word {word.id} does not belong to lane {self.lane.value}")
            if word.id in by_id:
                raise BehaviorKindError(f"word id repeats: {word.id}")
            if word.name in by_name:
                raise BehaviorKindError(f"word name repeats: {word.name}")
            by_id[word.id] = word
            by_name[word.name] = word
        for word in self.words:
            if word.parent is None:
                continue
            parent = by_id.get(word.parent)
            if parent is None:
                raise BehaviorKindError(f"word {word.id} points to an unknown parent (not a live word): {word.parent}")
            if word.level != parent.level + 1:
                raise BehaviorKindError(f"word {word.id} level {word.level} does not follow its parent {parent.id}")
        gone: dict[ClassId, DeletedWord] = {}
        for mark in self.deleted:
            if mark.id.lane is not self.lane or mark.parent.lane is not self.lane:
                raise BehaviorKindError(f"deleted word {mark.id} does not belong to lane {self.lane.value}")
            if mark.id in by_id or mark.id in gone:
                raise BehaviorKindError(f"word id repeats: {mark.id}")
            gone[mark.id] = mark
        for mark in gone.values():
            # 一路往上必须落到活着的节点（初始词永远活着），不能断、不能绕圈
            seen = {mark.id}
            upward = mark.parent
            while upward not in by_id:
                if upward not in gone:
                    raise BehaviorKindError(f"deleted word {mark.id} points to an unknown parent {upward}")
                if upward in seen:
                    raise BehaviorKindError(f"deleted word {mark.id} has a parent loop")
                seen.add(upward)
                upward = gone[upward].parent
        for label, marks in (("active", self.active), ("exited", self.exited)):
            for word_id in marks:
                if word_id not in by_id:
                    raise BehaviorKindError(f"{label} word is not a live word of the vocabulary: {word_id}")
        both = set(self.active) & set(self.exited)
        if both:
            raise BehaviorKindError(f"a word is both active and exited: {min(both)}")
        object.__setattr__(self, "words", tuple(self.words))
        object.__setattr__(self, "deleted", tuple(sorted(gone.values(), key=lambda mark: mark.id)))
        object.__setattr__(self, "active", MappingProxyType(dict(sorted(self.active.items()))))
        object.__setattr__(self, "exited", MappingProxyType(dict(sorted(self.exited.items()))))
        object.__setattr__(self, "_by_id", MappingProxyType(by_id))
        object.__setattr__(self, "_by_name", MappingProxyType(by_name))
        object.__setattr__(self, "_gone", MappingProxyType(gone))

    # ── 查 ────────────────────────────────────────────────────────────────────

    def get(self, word_id: ClassId) -> Word:
        try:
            return self._by_id[word_id]
        except KeyError:
            raise BehaviorKindError(f"unknown word: {word_id}") from None

    def __contains__(self, word_id: object) -> bool:
        """活着的词（初始词与活着的自己长的节点）；已删的不算。"""

        return word_id in self._by_id

    def named(self, name: str) -> Word | None:
        """叫这个名字的活着的词。"""

        return self._by_name.get(name)

    def in_use(self) -> tuple[Word, ...]:
        """在用的词（按编号）。"""

        return tuple(self._by_id[word_id] for word_id in self.active)

    def children(self, word_id: ClassId) -> tuple[Word, ...]:
        """活着的直接下级：初始下级按源文件顺序，后面是自己长的、按编号。"""

        return tuple(word for word in self.words if word.parent == word_id)

    def candidates(self, word_id: ClassId) -> tuple[Word, ...]:
        """这个上级下面的候选：自己长的、活着的、不在用的直接下级。"""

        return tuple(word for word in self.children(word_id) if word.grown and word.id not in self.active)

    def next_id(self) -> ClassId:
        """下一个编号：这条 lane 用过的最大号加一（删掉的也算，编号不重用）。"""

        used = [word.id.number for word in self.words] + [mark.id.number for mark in self.deleted]
        return ClassId(self.lane, max(used, default=0) + 1)

    def resolve(self, word_id: ClassId) -> ClassId:
        """一个编号此刻落在哪个活着的词上：活着的是它自己；删掉的一层层往上找最近的活着的上级。不认识的编号报错。"""

        current = word_id
        while current not in self._by_id:
            mark = self._gone.get(current)
            if mark is None:
                raise BehaviorKindError(f"unknown word: {word_id}")
            current = mark.parent
        return current

    def resolve_token(self, token: str) -> str:
        """``kind_token`` 的换算：这条 lane 已删节点的编号换成最近的活着的上级的编号，别的（活着的词、占位标记、别的 lane、
        不认识的编号）原样返回——不认识的由读者自己的校验报出来。"""

        try:
            word_id = ClassId.parse(token)
        except KindIdError:
            return token
        if word_id.lane is not self.lane or word_id not in self._gone:
            return token
        return str(self.resolve(word_id))

    def ordered(self) -> Iterator[Word]:
        """给模型和人看的顺序：初始词按源文件顺序；自己长的节点紧跟在它上级的初始子树后面（按编号，各自的下级紧跟其后）。

        初始词的源文件不一定是深度优先排的（物理那份就不是），所以"上级的初始子树后面"按源文件里那棵子树最后出现的位置算：
        没有自己长的节点时，顺序就是源文件顺序。
        """

        initial = [word for word in self.words if not word.grown]
        position = {word.id: index for index, word in enumerate(initial)}
        last: dict[ClassId, int] = {}
        for index, word in enumerate(initial):
            current: ClassId | None = word.id
            while current is not None:
                last[current] = index
                current = self._by_id[current].parent
        grown: dict[ClassId, list[Word]] = {}
        for word in sorted((word for word in self.words if word.grown), key=lambda word: word.id):
            if word.parent is not None:
                grown.setdefault(word.parent, []).append(word)

        def subtree(parent: ClassId) -> Iterator[Word]:
            for word in grown.get(parent, ()):
                yield word
                yield from subtree(word.id)

        after: dict[int, list[Word]] = {}
        for key in grown:
            if key in position:
                after.setdefault(last[key], []).append(self._by_id[key])
        for index, word in enumerate(initial):
            yield word
            # 同一个位置收尾的几棵子树：深的先（那个位置的词自己的子树在它的上级的子树里面）
            for top in sorted(after.get(index, ()), key=lambda item: -item.level):
                yield from subtree(top.id)

    # ── 改（都返回新的一份）─────────────────────────────────────────────────────

    def with_active(self, active: Mapping[ClassId, date], exited: Mapping[ClassId, date] | None = None) -> Vocabulary:
        """换一份在用集合（与退出记录）；在用集合变了修订号加一，都没变原样返回。"""

        exited = self.exited if exited is None else exited
        if dict(active) == dict(self.active) and dict(exited) == dict(self.exited):
            return self
        revision = self.revision + (dict(active) != dict(self.active))
        return Vocabulary(self.lane, self.words, active, revision, self.deleted, exited)

    def with_word(self, word: Word) -> Vocabulary:
        """加一个自己长的节点，或换掉同编号的那个（记被选中的时刻）。在用状态不变。"""

        words = [item for item in self.words if item.id != word.id]
        words.append(word)
        initial = [item for item in words if not item.grown]
        grown = sorted((item for item in words if item.grown), key=lambda item: item.id)
        return Vocabulary(self.lane, (*initial, *grown), self.active, self.revision, self.deleted, self.exited)

    def without(self, word_id: ClassId, *, on: date) -> Vocabulary:
        """删掉一个自己长的、下面没有活着的下级的节点，留"已删"的记号。它在用的话在用集合变了，修订号加一。"""

        word = self.get(word_id)
        if not word.grown or word.parent is None:
            raise BehaviorKindError(f"only grown words can be deleted: {word_id}")
        if self.children(word_id):
            raise BehaviorKindError(f"word {word_id} still has live children")
        active = {key: value for key, value in self.active.items() if key != word_id}
        exited = {key: value for key, value in self.exited.items() if key != word_id}
        return Vocabulary(
            self.lane,
            tuple(item for item in self.words if item.id != word_id),
            active,
            self.revision + (word_id in self.active),
            (*self.deleted, DeletedWord(word_id, word.parent, on)),
            exited,
        )


__all__ = [
    "BehaviorKindError",
    "DeletedWord",
    "Vocabulary",
    "Word",
    "WordOrigin",
    "reminder_sentence",
]
