"""先验的一对（前因, 后果）归哪一题（实时化阶段 2「先验混合问法」）。纯值，不调模型、不读盘。

**一对只归一题，按两个概念谁先出现分**：

- 后果比前因晚出现 → 归"这个后果"那一题，旧问法（固定后果、列前因）；
- 否则（前因更晚、或同一批）→ 归"这个前因"那一题，新问法（固定前因、列后果）。

"谁先出现"按概念创建时刻的当地日期比；还没进概念集的（同步词表将要生成的基础概念、待生效的概念）算比所有已有的都晚、
彼此同一批。所以同一对在任何时候问都归同一题：只看两个概念出现的先后，不看队列做到哪——一个新概念出现恰好两题（它当前因
新问法一题，列出现不晚于它的后果；它当后果旧问法一题，列比它早出现的前因），题里的清单之后再问也不变。

每一题只问归它、且还没有答案的那些对子（``Questions``）。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date


def consequent_is_later(antecedent_seen: date | None, consequent_seen: date | None) -> bool:
    """后果比前因晚出现吗。``None`` = 还没进概念集（比所有已有的都晚，彼此同一批）。"""

    if consequent_seen is None:
        return antecedent_seen is not None
    return antecedent_seen is not None and consequent_seen > antecedent_seen


@dataclass(frozen=True)
class Questions:
    """这条 lane 现在要问的题：旧问法按后果（后果 → 归它、还没答案的前因），新问法按前因（前因 → 归它、还没答案的后果）。
    清单里的先后是调用方给的概念先后。"""

    by_consequent: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    by_antecedent: Mapping[str, tuple[str, ...]] = field(default_factory=dict)

    @property
    def count(self) -> int:
        return len(self.by_consequent) + len(self.by_antecedent)


def assign(
    names: Sequence[str],
    *,
    comparable: Callable[[str, str], bool],
    seen: Callable[[str], date | None],
    answered: Callable[[str, str], bool],
) -> Questions:
    """把还没有答案的对子归到题上。``names`` 是这条 lane 的行为概念（先后就是题里清单的先后），``comparable(前因, 后果)``
    是这一对要不要比，``seen`` 给概念出现的那一天，``answered(前因, 后果)`` 是这一对有没有答案（两种问法的缓存里任一份有）。"""

    first = {name: seen(name) for name in names}
    by_consequent: dict[str, list[str]] = {}
    by_antecedent: dict[str, list[str]] = {}
    for consequent in names:
        for antecedent in names:
            if not comparable(antecedent, consequent) or answered(antecedent, consequent):
                continue
            if consequent_is_later(first[antecedent], first[consequent]):
                by_consequent.setdefault(consequent, []).append(antecedent)
    for antecedent in names:
        for consequent in names:
            if not comparable(antecedent, consequent) or answered(antecedent, consequent):
                continue
            if not consequent_is_later(first[antecedent], first[consequent]):
                by_antecedent.setdefault(antecedent, []).append(consequent)
    return Questions(
        {name: tuple(found) for name, found in by_consequent.items()},
        {name: tuple(found) for name, found in by_antecedent.items()},
    )


__all__ = ["Questions", "assign", "consequent_is_later"]
