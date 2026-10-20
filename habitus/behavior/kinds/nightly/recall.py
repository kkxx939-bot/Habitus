"""每晚新增的召回（裁定 60 第 2 条）：新条目之外，从池里旧条目按字符二元组重合度各挑几条一起交给模型聚组。

不引入新的模型依赖：只比条目里模型提议的名字与给聚组看的内容（``PendingEntry.proposed`` / ``content``）的字符二元组
（Jaccard）。两件不同月份、措辞相近的同一件事（"订去杭州的机票""订杭州出差的机票"）靠它碰到一起；一条旧条目只进一组
（只被召回一次），没有任何重合的不召回。纯函数。
"""

from __future__ import annotations

import unicodedata
from collections.abc import Sequence

from habitus.behavior.kinds.pending import PendingEntry


def recall(
    new_entries: Sequence[PendingEntry], old_entries: Sequence[PendingEntry], *, per_entry: int
) -> tuple[PendingEntry, ...]:
    """每条新条目从 ``old_entries`` 里挑最像的至多 ``per_entry`` 条（重合度相同按地址），已被挑走的不再挑。"""

    if isinstance(per_entry, bool) or not isinstance(per_entry, int) or per_entry < 0:
        raise ValueError("per_entry must be a non-negative integer")
    taken: set[str] = set()
    found: list[PendingEntry] = []
    grams = {entry.occurrence: _bigrams(entry) for entry in old_entries}
    for entry in sorted(new_entries, key=lambda item: (item.day, item.occurrence)):
        mine = _bigrams(entry)
        ranked = sorted(
            (
                (_similarity(mine, grams[candidate.occurrence]), candidate)
                for candidate in old_entries
                if candidate.occurrence not in taken
            ),
            key=lambda pair: (-pair[0], pair[1].occurrence),
        )
        for score, candidate in ranked[:per_entry]:
            if score <= 0.0:
                break
            taken.add(candidate.occurrence)
            found.append(candidate)
    return tuple(found)


def _bigrams(entry: PendingEntry) -> frozenset[str]:
    text = "".join(
        char
        for char in unicodedata.normalize("NFKC", f"{entry.proposed} {entry.content}").casefold()
        if not char.isspace() and unicodedata.category(char)[0] not in "PZ"
    )
    return frozenset(text[index : index + 2] for index in range(len(text) - 1))


def _similarity(left: frozenset[str], right: frozenset[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


__all__ = ["recall"]
