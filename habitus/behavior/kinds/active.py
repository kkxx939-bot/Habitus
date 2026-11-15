"""数在用的词（换代里，纯计数、不调模型；词表方案 2026-11-14 第 5 节）。

- 一个词在 ``min_days`` 个不同的本地日被**直接**选中（记录的 ``kind_token`` 就是它；同一天两次不算，不要求相邻），
  这一代起在用，记"哪一天转的在用"= 这一代的日子。上级不因下级被选中而在用。
- 每条 lane 在用的词最多 ``cap`` 个：满了够条件的词等着、不挤别人；等着的按"凑够天数的那一天"先后进，同一天按编号。
- 二甲里在用的词不退出：在用只增不减，重数结果一样。
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date

from habitus.behavior.kinds.ids import ClassId, KindIdError
from habitus.behavior.kinds.model import Vocabulary


@dataclass(frozen=True)
class ActiveCount:
    """数一遍的结果：数完的词表（在用集合变了修订号加一）与够条件却因满额等着的词。"""

    vocabulary: Vocabulary
    waiting: tuple[ClassId, ...]
    added: tuple[ClassId, ...]


def count_active(
    vocabulary: Vocabulary,
    picks: Iterable[tuple[str, date]],
    *,
    today: date,
    min_days: int,
    cap: int,
) -> ActiveCount:
    """``picks`` 是序列里算数的记录：（``kind_token``, 本地日）。不是这条 lane 的词编号的（标记、别的 lane）不算。"""

    days: dict[ClassId, set[date]] = {}
    for token, day in picks:
        try:
            word_id = ClassId.parse(token)
        except KindIdError:
            continue
        if word_id.lane is vocabulary.lane and word_id in vocabulary:
            days.setdefault(word_id, set()).add(day)
    ready = sorted(
        (sorted(seen)[min_days - 1], word_id)
        for word_id, seen in days.items()
        if word_id not in vocabulary.active and len(seen) >= min_days
    )
    room = max(0, cap - len(vocabulary.active))
    added = tuple(word_id for _day, word_id in ready[:room])
    waiting = tuple(word_id for _day, word_id in ready[room:])
    active = {**vocabulary.active, **dict.fromkeys(added, today)}
    return ActiveCount(vocabulary.with_active(active), waiting, added)


__all__ = ["ActiveCount", "count_active"]
