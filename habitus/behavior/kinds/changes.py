"""变更日志：词表一版一版怎么变过来。

每一版是一组有序操作，加上"哪些 occurrence 从哪个 token 改到哪个 token"（迁移），留着可以撤销。
词表不单独存真相：从空词表依次应用每一版就得到当前词表（``replay``）。

操作只有四种原子形状，常见的变动由它们组合：
- 新增一类：``AddClass``；
- 合并 a→b：``RetireClass(a)`` + ``Branch(a, b)``；
- 从 a 拆出 c：``AddClass(c)`` + ``Branch(a, c)``（a 往往还会 ``ReviseClass`` 收窄判据）；
- 改判据 / 名称 / 提醒句 / 不含 / 例子：``ReviseClass``；
- 停用：``RetireClass``。
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace
from datetime import datetime
from enum import Enum

from habitus.behavior.kinds.ids import ClassId, KindIdError, lane_of_token
from habitus.behavior.kinds.model import BehaviorClass, BehaviorKindError, ClassStatus, Vocabulary


class ChangeReason(str, Enum):
    """这一版是谁改的：每晚新增、定期拆改。"""

    NIGHTLY = "nightly"
    REVISION = "revision"


@dataclass(frozen=True)
class AddClass:
    item: BehaviorClass


@dataclass(frozen=True)
class ReviseClass:
    item: BehaviorClass


@dataclass(frozen=True)
class RetireClass:
    class_id: ClassId


@dataclass(frozen=True)
class Branch:
    source: ClassId
    target: ClassId


Operation = AddClass | ReviseClass | RetireClass | Branch


@dataclass(frozen=True)
class Move:
    """一条 occurrence 的 ``kind_token`` 从 ``source`` 改到 ``target``。"""

    occurrence: str
    source: str
    target: str

    def __post_init__(self) -> None:
        for name in ("occurrence", "source", "target"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise BehaviorKindError(f"move {name} must be non-empty text")
        if self.source == self.target:
            raise BehaviorKindError(f"move does not change the token: {self.occurrence}")
        for name in ("source", "target"):
            try:
                lane_of_token(getattr(self, name))
            except KindIdError as exc:
                raise BehaviorKindError(
                    f"move {name} is neither a class id nor a marker: {getattr(self, name)!r}"
                ) from exc


@dataclass(frozen=True)
class VersionRecord:
    """变更日志的一版。"""

    version: int
    at: datetime
    reason: ChangeReason
    operations: tuple[Operation, ...]
    moves: tuple[Move, ...] = ()
    note: str = ""

    def __post_init__(self) -> None:
        if isinstance(self.version, bool) or not isinstance(self.version, int) or self.version <= 0:
            raise BehaviorKindError("version must be a positive integer")
        if not isinstance(self.at, datetime) or self.at.utcoffset() is None:
            raise BehaviorKindError("version time must be timezone-aware")
        object.__setattr__(self, "reason", ChangeReason(self.reason))
        if not self.operations and not self.moves:
            raise BehaviorKindError("a version must change something")
        occurrences = [move.occurrence for move in self.moves]
        if len(set(occurrences)) != len(occurrences):
            raise BehaviorKindError(f"version {self.version} moves an occurrence twice")


def apply(vocabulary: Vocabulary, record: VersionRecord) -> Vocabulary:
    """把一版应用到上一版词表上；版本号必须正好接上。"""

    if record.version != vocabulary.version + 1:
        raise BehaviorKindError(f"version {record.version} does not follow {vocabulary.version}")
    classes = dict(vocabulary.classes)
    branches = {key: list(value) for key, value in vocabulary.branches.items()}
    for operation in record.operations:
        if isinstance(operation, AddClass):
            if operation.item.id in classes:
                raise BehaviorKindError(f"class already exists: {operation.item.id}")
            classes[operation.item.id] = operation.item
        elif isinstance(operation, ReviseClass):
            current = _existing(classes, operation.item.id)
            if not current.active or not operation.item.active:
                raise BehaviorKindError(f"only an active class can be revised: {operation.item.id}")
            classes[operation.item.id] = operation.item
        elif isinstance(operation, RetireClass):
            current = _existing(classes, operation.class_id)
            if not current.active:
                raise BehaviorKindError(f"class is already retired: {operation.class_id}")
            classes[operation.class_id] = replace(current, status=ClassStatus.RETIRED)
        elif isinstance(operation, Branch):
            _existing(classes, operation.source)
            if not _existing(classes, operation.target).active:
                raise BehaviorKindError(f"branch target must be an active class: {operation.target}")
            targets = branches.setdefault(operation.source, [])
            if operation.target in targets:
                raise BehaviorKindError(f"branch repeats: {operation.source} -> {operation.target}")
            if _reaches(branches, operation.target, operation.source):
                raise BehaviorKindError(f"branch would form a cycle: {operation.source} -> {operation.target}")
            targets.append(operation.target)
        else:  # pragma: no cover - 类型系统已穷尽
            raise BehaviorKindError(f"unknown operation: {operation!r}")
    return Vocabulary(
        version=record.version,
        classes=classes,
        branches={key: tuple(value) for key, value in branches.items()},
    )


def replay(records: Iterable[VersionRecord]) -> Vocabulary:
    vocabulary = Vocabulary()
    for record in records:
        vocabulary = apply(vocabulary, record)
    return vocabulary


def merge_operations(source: ClassId, target: ClassId) -> tuple[Operation, ...]:
    return (RetireClass(source), Branch(source, target))


def split_operations(source: BehaviorClass, new: BehaviorClass) -> tuple[Operation, ...]:
    """从 ``source`` 拆出 ``new``；``source`` 传入的是收窄判据之后的样子。"""

    return (AddClass(new), Branch(source.id, new.id), ReviseClass(source))


def _reaches(branches: dict[ClassId, list[ClassId]], start: ClassId, goal: ClassId) -> bool:
    stack, seen = [start], set()
    while stack:
        current = stack.pop()
        if current == goal:
            return True
        if current not in seen:
            seen.add(current)
            stack.extend(branches.get(current, ()))
    return False


def _existing(classes: dict[ClassId, BehaviorClass], class_id: ClassId) -> BehaviorClass:
    try:
        return classes[class_id]
    except KeyError:
        raise BehaviorKindError(f"unknown behavior class: {class_id}") from None


__all__ = [
    "AddClass",
    "Branch",
    "ChangeReason",
    "Move",
    "Operation",
    "RetireClass",
    "ReviseClass",
    "VersionRecord",
    "apply",
    "merge_operations",
    "replay",
    "split_operations",
]
