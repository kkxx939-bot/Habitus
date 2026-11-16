"""数在用的词（换代里，纯计数、不调模型；裁定 101 第 7 条）。每一代先退出、后转在用：

1. **退出**：一个词"``retire_days`` 天没被选中"= 这一代的日子减去最后一次被直接选中的日子 ≥ ``retire_days``
   （自己长的、从没被选中过的从写出它的那天算）。先数第 3 层、再数第 2 层、再第 1 层：下级这一代删掉以后，上级在同一代里
   就算没有下级了。

   | 词 | 到了 |
   |---|---|
   | 自己长的，下面没有活着的下级 | 删掉（不管在不在用），留"已删"的记号 |
   | 自己长的，下面还有活着的下级 | 不删；在用的话退出在用，回到候选 |
   | 初始词，在用 | 退出在用 |
   | 初始词，不在用 | 没有变化 |

   退出在用的词记下退出那天，之前攒的日子作废，以后只数退出那天之后的日子。
2. **转在用**：一个词在 ``min_days`` 个不同的本地日被**直接**选中（记录的 ``kind_token`` 换算以后就是它；同一天两次不算，
   不要求相邻），这一代起在用，记"哪一天转的在用"= 这一代的日子。上级不因下级被选中而在用；被删节点的旧记录换算到上级，
   算上级的。**例外一条**（裁定 101 第 7 条，用户 2026-11-15 定）：记录挂在**候选**（自己长的、不在用；在用以前的那些日子也算）上时，
   这一天也算给它最近的不是候选的上级（初始词，或自己长的、在用的；上级也是候选就接着往上找）——只算"转在用"的日子；退出、
   删除、"最后被选中"仍只看直接记录。上级靠下级的日子转了在用、自己从没被直接选中过的，20 天从转在用那天算。每条 lane 在用的词最多 ``cap`` 个：满了够条件的词等着、不挤别人；等着的按"凑够天数的那一天"先后进，同一天按编号。
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date

from habitus.behavior.kinds.ids import ClassId, KindIdError
from habitus.behavior.kinds.model import Vocabulary


@dataclass(frozen=True)
class ActiveCount:
    """数一遍的结果：数完的词表（在用集合变了修订号加一）、这一代转在用的、够条件却因满额等着的、删掉的、退出在用的。"""

    vocabulary: Vocabulary
    waiting: tuple[ClassId, ...]
    added: tuple[ClassId, ...]
    deleted: tuple[ClassId, ...] = ()
    exited: tuple[ClassId, ...] = ()


def count_active(
    vocabulary: Vocabulary,
    picks: Iterable[tuple[str, date]],
    *,
    today: date,
    min_days: int,
    cap: int,
    retire_days: int,
) -> ActiveCount:
    """``picks`` 是序列里算数的记录：（``kind_token``, 本地日）。不是这条 lane 的词编号的（标记、别的 lane）不算；
    已删节点的编号按词表换算到最近的活着的上级。"""

    picks = tuple(picks)
    before = vocabulary
    deleted: list[ClassId] = []
    exited: list[ClassId] = []
    for level in sorted({word.level for word in vocabulary.words}, reverse=True):
        days = _days(vocabulary, picks)
        for word in [word for word in vocabulary.words if word.level == level]:
            last = _last_seen(vocabulary, word.id, days)
            if last is None or (today - last).days < retire_days:
                continue
            if word.grown and not vocabulary.children(word.id):
                vocabulary = vocabulary.without(word.id, on=today)
                deleted.append(word.id)
            elif word.id in vocabulary.active:
                active = {key: value for key, value in vocabulary.active.items() if key != word.id}
                vocabulary = vocabulary.with_active(active, {**vocabulary.exited, word.id: today})
                exited.append(word.id)
    days = _entering_days(vocabulary, picks)
    left = vocabulary.exited
    ready = sorted(
        (counted[min_days - 1], word_id)
        for word_id, seen in days.items()
        if word_id not in vocabulary.active
        and len(counted := sorted(day for day in seen if word_id not in left or day > left[word_id])) >= min_days
    )
    room = max(0, cap - len(vocabulary.active))
    added = tuple(word_id for _day, word_id in ready[:room])
    waiting = tuple(word_id for _day, word_id in ready[room:])
    vocabulary = vocabulary.with_active(
        {**vocabulary.active, **dict.fromkeys(added, today)},
        {key: value for key, value in left.items() if key not in added},
    )
    # 修订号是在用状态的版本：这一代在用集合变了只加一次
    revision = before.revision + (dict(vocabulary.active) != dict(before.active))
    if vocabulary.revision != revision:
        vocabulary = Vocabulary(
            vocabulary.lane, vocabulary.words, vocabulary.active, revision, vocabulary.deleted, vocabulary.exited
        )
    return ActiveCount(vocabulary, waiting, added, tuple(deleted), tuple(exited))


def _last_seen(vocabulary: Vocabulary, word_id: ClassId, days: dict[ClassId, set[date]]) -> date | None:
    """最后一次被直接选中的日子：序列里换算后落在它上面的记录；自己长的还看状态里记的时刻。从没被直接选中过的：在用的从转在用
    那天算（初始词读不到那么早，或靠下级的日子转了在用），不在用的自己长的从写出它的那天算。"""

    word = vocabulary.get(word_id)
    found = [*days.get(word_id, ())]
    if word.last_picked is not None:
        found.append(word.last_picked.date())
    if not found and word_id in vocabulary.active:
        # 从没被直接选中过、在用的（初始词读不到那么早，或靠下级的日子转了在用，3.5）：从转在用那天算
        found.append(vocabulary.active[word_id])
    elif not found and word.created_at is not None:
        found.append(word.created_at.date())
    return max(found, default=None)


def _entering_days(vocabulary: Vocabulary, picks: tuple[tuple[str, date], ...]) -> dict[ClassId, set[date]]:
    """数"转在用"用的日子：直接选中的日子，加上 3.5 那一条——落在候选上的日子也算给它最近的不是候选的上级。候选 = 自己长的、
    此刻不在用的；在用的自己长的节点，转在用那天以前的日子仍按候选算（算过给上级的不收回）；退出在用、回到候选的，只借退出那天
    之后的日子（裁定 101 第 7 条：取与"退出之前的日子作废"同一口径）。"""

    days = {word_id: set(seen) for word_id, seen in _days(vocabulary, picks).items()}
    for word_id, seen in _days(vocabulary, picks).items():
        since, left = vocabulary.active.get(word_id), vocabulary.exited.get(word_id)
        # 在用那一段的日子只算自己；退出在用、回到候选的，只借退出那天之后的（退出之前的日子本来就作废）
        lent = (
            {day for day in seen if (since is None or day < since) and (left is None or day > left)}
            if vocabulary.get(word_id).grown
            else set()
        )
        if not lent:
            continue
        upward = vocabulary.get(word_id).parent
        while upward is not None and _candidate(vocabulary, upward):
            upward = vocabulary.get(upward).parent
        if upward is not None:
            days.setdefault(upward, set()).update(lent)
    return days


def _candidate(vocabulary: Vocabulary, word_id: ClassId) -> bool:
    word = vocabulary.get(word_id)
    return word.grown and word_id not in vocabulary.active


def _days(vocabulary: Vocabulary, picks: tuple[tuple[str, date], ...]) -> dict[ClassId, set[date]]:
    """每个活着的词被直接选中的本地日（编号先按词表换算）。"""

    days: dict[ClassId, set[date]] = {}
    for token, day in picks:
        try:
            word_id = ClassId.parse(vocabulary.resolve_token(token))
        except KindIdError:
            continue
        if word_id.lane is vocabulary.lane and word_id in vocabulary:
            days.setdefault(word_id, set()).add(day)
    return days


__all__ = ["ActiveCount", "count_active"]
