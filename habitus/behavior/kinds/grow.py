"""模型选完词以后，代码怎么收它写的新节点（裁定 101、102）。纯函数，不落盘。

按 first、second 的顺序一格一格收；一格没建成，后面的格不建，记录挂在已经建成（或复用到）的最深一层；一格都没建成就挂在
选中的词上。每一格依次查：

| 检查 | 不过 |
|---|---|
| 名字（去掉首尾空白）超过 ``max_name_chars`` 个字 | 不建 |
| 建了以后层数超过 ``max_level`` | 不建 |
| 和同一个上级下面一个活着的节点同名 | 不新建，挂到那个上，接着收下一格 |
| 和词表里别处一个活着的节点同名 | 不建（一条 lane 里活着的节点名字不重复） |
| 这个上级下面的候选已有 ``candidate_slots`` 个 | 挤掉最久没被选中的（同时刻按编号小的）、下面没有活着的下级的那个；都挤不掉就不建 |

记录最终挂的节点是自己长的，就记下它这一刻被直接选中（比已记的更晚才改）。模型选中的词在它作答期间已被删掉（换代同时在数）：落到它最近的
活着的上级，新节点不建、不复活它。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import datetime

from habitus.behavior.kinds.config import BehaviorKindConfig
from habitus.behavior.kinds.ids import ClassId
from habitus.behavior.kinds.model import Vocabulary, Word, WordOrigin


@dataclass(frozen=True)
class Proposal:
    """模型写的一格：名字与一句定义（原样，首尾空白收的时候去掉）。"""

    name: str
    definition: str


def settle(
    vocabulary: Vocabulary,
    picked: ClassId,
    proposals: Sequence[Proposal],
    *,
    at: datetime,
    config: BehaviorKindConfig,
) -> tuple[Vocabulary, ClassId]:
    """收一轮的选词结果：返回收完的词表（没有变化时原样返回同一份）与记录最终挂的节点。"""

    if picked not in vocabulary:
        return vocabulary, vocabulary.resolve(picked)
    current = picked
    for proposal in proposals:
        name, definition = proposal.name.strip(), proposal.definition.strip()
        parent = vocabulary.get(current)
        if len(name) > config.max_name_chars or parent.level + 1 > config.max_level:
            break
        same = vocabulary.named(name)
        if same is not None:
            if same.parent != current:
                break
            current = same.id
            continue
        room = _make_room(vocabulary, current, config=config, on=at)
        if room is None:
            break
        word = Word(
            id=room.next_id(),
            level=parent.level + 1,
            name=name,
            parent=current,
            definition=definition,
            domain=parent.domain,
            origin=WordOrigin.GROWN,
            created_at=at,
        )
        vocabulary = room.with_word(word)
        current = word.id
    final = vocabulary.get(current)
    # "最后被选中"取时刻最晚的那一次：重试堆补回的一轮处理得晚、时刻却早，不能把它倒拨回去
    if final.grown and (final.last_picked is None or final.last_picked < at):
        vocabulary = vocabulary.with_word(replace(final, last_picked=at))
    return vocabulary, current


def _make_room(
    vocabulary: Vocabulary, parent: ClassId, *, config: BehaviorKindConfig, on: datetime
) -> Vocabulary | None:
    """候选位置满了就挤：最久没被选中的（同时刻按编号小的）、下面没有活着的下级的候选变成"已删"。挤不出位置是 None。"""

    while len(vocabulary.candidates(parent)) >= config.candidate_slots:
        evictable = [word for word in vocabulary.candidates(parent) if not vocabulary.children(word.id)]
        if not evictable:
            return None
        victim = min(evictable, key=lambda word: (word.seen, word.id))
        vocabulary = vocabulary.without(victim.id, on=on.date())
    return vocabulary


__all__ = ["Proposal", "settle"]
