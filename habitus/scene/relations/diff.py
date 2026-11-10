"""两晚关系表之间的变化（只给观测：换代的每代报告与 ``scene.b4`` span 的计数）。纯函数，不读盘、不改状态。

- **新增**：上一晚的表里没有这一行（样本不够的不逐条存，它们变成已检、候选……也算新增）；
- **消失**：上一晚有、这一晚没有（退回样本不够、或行被清掉）；
- **状态变**：两晚都有、状态不同。

族与进族路径从两张表推（表里不存检验族）：长跨度、不带条件的短跨度、带条件的三族；带条件的这一行当晚进了族时，父关系
（少一个条件的键）当晚过了第 1 道是"父关系"，父关系在上一晚是候选 / 成立（跨夜带着）是"带着"，都不是就是"按组"。
跨夜带着的候选 / 成立本身不进族（状态折叠各管各的），记"不进族"。
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from habitus.scene.relations.engine import RelationKey, Verdict
from habitus.scene.relations.state import LaneState, Relation, Status

ADDED, REMOVED, STATUS = "added", "removed", "status"


@dataclass(frozen=True)
class RelationChange:
    key: RelationKey
    kind: str
    before: Status | None
    after: Status | None
    #: 这一晚的那一行（消失的是 None）：当晚读数从它的 ``tonight`` 取。
    relation: Relation | None


def diff_states(before: LaneState | None, after: LaneState) -> tuple[RelationChange, ...]:
    """``before`` 是这条 lane 上一晚的表（没有就是第一晚），``after`` 是这一晚的；按键的文本排序。"""

    old = {} if before is None else {relation.key: relation for relation in before.relations}
    new = {relation.key: relation for relation in after.relations}
    changes: list[RelationChange] = []
    for key in sorted(set(old) | set(new), key=lambda item: item.text):
        previous, current = old.get(key), new.get(key)
        if previous is None and current is not None:
            changes.append(RelationChange(key, ADDED, None, current.status, current))
        elif current is None and previous is not None:
            changes.append(RelationChange(key, REMOVED, previous.status, None, None))
        elif previous is not None and current is not None and previous.status is not current.status:
            changes.append(RelationChange(key, STATUS, previous.status, current.status, current))
    return tuple(changes)


def family_of(key: RelationKey) -> str:
    """这条关系属于哪一族：``long`` 长跨度、``plain`` 不带条件的短跨度、``conditioned`` 带条件的短跨度。"""

    if key.segment.long:
        return "long"
    return "conditioned" if key.condition else "plain"


def entry_paths(keys: Iterable[RelationKey], before: LaneState | None, after: LaneState) -> dict[RelationKey, str]:
    """这些关系当晚怎么进的族：``parent`` / ``carried`` / ``group``（只对带条件的），``family``（另两族直接进），``outside``（跨夜
    带着的候选 / 成立，不进族），``none``（当晚没进族）。"""

    tonight = {relation.key: relation for relation in after.relations}
    held = (
        set()
        if before is None
        else {key for key, row in before.carried().items() if row.status in (Status.CANDIDATE, Status.ESTABLISHED)}
    )
    passed = {
        key for key, row in tonight.items() if row.tonight is not None and row.tonight.verdict is Verdict.SIGNIFICANT
    }
    return {key: _path(key, tonight.get(key), held, passed) for key in keys}


def _path(key: RelationKey, current: Relation | None, held: set[RelationKey], passed: set[RelationKey]) -> str:
    if key in held:
        return "outside"
    if current is None or current.tonight is None or current.tonight.verdict is Verdict.SPARSE:
        return "none"
    if not key.condition or key.segment.long:
        return "family"
    parents = key.parents()
    if any(parent in passed for parent in parents):
        return "parent"
    if any(parent in held for parent in parents):
        return "carried"
    return "group"


__all__ = ["ADDED", "REMOVED", "STATUS", "RelationChange", "diff_states", "entry_paths", "family_of"]
