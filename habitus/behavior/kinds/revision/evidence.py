"""定期拆改的证据规则（裁定 17 ①）：只看用户自己的数据。

拆分：切出来的两边都要在近期跨天反复出现（各至少 ``min_count`` 条、跨 ``min_days`` 个不同日子），而且提醒句不同——
否则就是一次性的事，或者两边其实是同一件事。合并的证据（不同周期都被提出）在 ``schedule.JobState`` 里。
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from habitus.behavior.kinds.model import BehaviorClass
from habitus.behavior.kinds.revision.facts import Member

_NOISE = re.compile(r"[\s，。、,.!！？?「」\"'“”]+")


def split_shortfall(
    new: BehaviorClass,
    narrowed: BehaviorClass,
    new_members: Sequence[Member],
    kept_members: Sequence[Member],
    *,
    min_count: int,
    min_days: int,
) -> str | None:
    """拆分不够格时说出差在哪；够格返回 ``None``。"""

    if _same_sentence(new.reminder, narrowed.reminder):
        return f"reminders are the same ({new.reminder!r})"
    for side, members in (("split-out", new_members), ("remaining", kept_members)):
        days = len({member.day for member in members})
        if len(members) < min_count or days < min_days:
            return f"{side} side has {len(members)} entries over {days} days"
    return None


def _same_sentence(left: str, right: str) -> bool:
    return _NOISE.sub("", left).casefold() == _NOISE.sub("", right).casefold()


__all__ = ["split_shortfall"]
