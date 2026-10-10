"""词表的值对象：行为类与某一版的词表。

只做自洽校验（编号与 lane 一致、同 lane 在用类不重名、「不含」与去向指向同 lane 的已知类），
不规定现实里的行为该长什么样。词表怎么一版版变过来见 ``changes.py``。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType

from habitus.behavior.kinds.ids import ClassId, Lane


class BehaviorKindError(ValueError):
    pass


class ClassStatus(str, Enum):
    ACTIVE = "active"
    RETIRED = "retired"


class ClassOrigin(str, Enum):
    """这一类从哪来：待定池转正、拆分出来。词表从空开始，类只从用户自己的数据里长出来（裁定 29：没有预置清单）。"""

    PROMOTED = "promoted"
    SPLIT = "split"


def _text(value: object, field_name: str) -> str:
    """词表条目里的一段文字：非空、首尾无空白、单行可打印。

    类名、判据、例子多半是模型写的；混进 U+2028 / U+2029 / U+0085 这类"换行"字符时，变更日志按行读会把一条记录切断，
    整份词表从此读不出来（第二轮评审 C5）。所以在条目上就拒。
    """

    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise BehaviorKindError(f"{field_name} must be non-empty text without surrounding whitespace")
    if any(not character.isprintable() for character in value):
        raise BehaviorKindError(f"{field_name} must be a single printable line")
    return value


@dataclass(frozen=True)
class Exclusion:
    """「不含」的一条：这类事不算本类，应该去 ``goes_to`` 那一类（没有对应类时为空）。"""

    text: str
    goes_to: ClassId | None = None

    def __post_init__(self) -> None:
        _text(self.text, "exclusion text")
        if self.goes_to is not None and not isinstance(self.goes_to, ClassId):
            raise BehaviorKindError("exclusion target must be a ClassId or None")


@dataclass(frozen=True)
class BehaviorClass:
    """词表里的一条。"""

    id: ClassId
    name: str
    criterion: str
    reminder: str
    excludes: tuple[Exclusion, ...] = ()
    examples: tuple[str, ...] = ()
    status: ClassStatus = ClassStatus.ACTIVE
    origin: ClassOrigin = field(kw_only=True)

    def __post_init__(self) -> None:
        if not isinstance(self.id, ClassId):
            raise BehaviorKindError("class id must be a ClassId")
        _text(self.name, "class name")
        _text(self.criterion, "class criterion")
        _text(self.reminder, "class reminder")
        if not isinstance(self.excludes, tuple) or not all(isinstance(item, Exclusion) for item in self.excludes):
            raise BehaviorKindError("class excludes must be a tuple of Exclusion")
        if not isinstance(self.examples, tuple):
            raise BehaviorKindError("class examples must be a tuple of text")
        for example in self.examples:
            _text(example, "class example")
        if len(set(self.examples)) != len(self.examples):
            raise BehaviorKindError(f"class examples repeat: {self.id}")
        if any(item.goes_to == self.id for item in self.excludes):
            raise BehaviorKindError(f"class excludes point back to itself: {self.id}")
        object.__setattr__(self, "status", ClassStatus(self.status))
        object.__setattr__(self, "origin", ClassOrigin(self.origin))

    @property
    def lane(self) -> Lane:
        return self.id.lane

    @property
    def active(self) -> bool:
        return self.status is ClassStatus.ACTIVE


@dataclass(frozen=True)
class Vocabulary:
    """某一版的词表：全部行为类（含停用的）+ 去向（合并、拆分时条目交给了谁）。

    ``branches[a]`` = a 把条目交出去的那些类：合并 a→b 记 ``a: (b,)``，从 a 拆出 c 记 ``a: (c,)``。
    "编号 → 现在对应哪些编号"顺着它往下找（``descendants``）。
    """

    version: int = 0
    classes: Mapping[ClassId, BehaviorClass] = field(default_factory=dict)
    branches: Mapping[ClassId, tuple[ClassId, ...]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.version, bool) or not isinstance(self.version, int) or self.version < 0:
            raise BehaviorKindError("vocabulary version must be a non-negative integer")
        classes = dict(self.classes)
        for key, item in classes.items():
            if not isinstance(item, BehaviorClass) or item.id != key:
                raise BehaviorKindError("vocabulary classes must be keyed by their own id")
        names: dict[tuple[Lane, str], ClassId] = {}
        for item in classes.values():
            if not item.active:
                continue
            name_key = (item.lane, item.name.casefold())
            if name_key in names:
                raise BehaviorKindError(f"active class name repeats in lane {item.lane.value}: {item.name}")
            names[name_key] = item.id
            for exclusion in item.excludes:
                self._require_same_lane(classes, item.id, exclusion.goes_to)
        branches: dict[ClassId, tuple[ClassId, ...]] = {}
        for source, targets in self.branches.items():
            if source not in classes or not targets or len(set(targets)) != len(targets):
                raise BehaviorKindError(f"vocabulary branch is malformed: {source}")
            for target in targets:
                if target == source:
                    raise BehaviorKindError(f"vocabulary branch points back to itself: {source}")
                self._require_same_lane(classes, source, target)
            branches[source] = tuple(targets)
        object.__setattr__(self, "classes", MappingProxyType(dict(sorted(classes.items()))))
        object.__setattr__(self, "branches", MappingProxyType(dict(sorted(branches.items()))))

    @staticmethod
    def _require_same_lane(classes: Mapping[ClassId, BehaviorClass], source: ClassId, target: ClassId | None) -> None:
        if target is None:
            return
        if target not in classes:
            raise BehaviorKindError(f"{source} points to an unknown class {target}")
        if target.lane is not source.lane:
            raise BehaviorKindError(f"{source} points across lanes to {target}")

    def get(self, class_id: ClassId) -> BehaviorClass:
        try:
            return self.classes[class_id]
        except KeyError:
            raise BehaviorKindError(f"unknown behavior class: {class_id}") from None

    def active(self, lane: Lane) -> tuple[BehaviorClass, ...]:
        resolved = Lane(lane)
        return tuple(item for item in self.classes.values() if item.active and item.lane is resolved)

    def next_id(self, lane: Lane) -> ClassId:
        resolved = Lane(lane)
        used = [item.number for item in self.classes if item.lane is resolved]
        return ClassId(resolved, max(used, default=0) + 1)

    def descendants(self, class_id: ClassId) -> tuple[ClassId, ...]:
        """这个编号的条目现在落在哪些在用类上（自己仍在用就含自己）。"""

        found: list[ClassId] = []
        seen: set[ClassId] = set()
        stack = [class_id]
        while stack:
            current = stack.pop()
            if current in seen or current not in self.classes:
                continue
            seen.add(current)
            if self.classes[current].active:
                found.append(current)
            stack.extend(reversed(self.branches.get(current, ())))
        return tuple(sorted(found))


__all__ = [
    "BehaviorClass",
    "BehaviorKindError",
    "ClassOrigin",
    "ClassStatus",
    "Exclusion",
    "Vocabulary",
]
