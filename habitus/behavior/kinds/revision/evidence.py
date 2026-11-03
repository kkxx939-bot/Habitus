"""定期拆改的证据规则（裁定 17 ①）：只看用户自己的数据。

拆分：切出来的两边都要在近期跨天反复出现（各至少 ``min_count`` 条、跨 ``min_days`` 个不同日子），而且提醒句不同——
否则就是一次性的事，或者两边其实是同一件事。合并的证据（不同周期都被提出）在 ``schedule.JobState`` 里。
"""

from __future__ import annotations

from collections.abc import Sequence

from habitus.behavior.kinds.model import BehaviorClass, sentence_key
from habitus.behavior.kinds.revision.facts import Member


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

    # 分组已按同一把尺子把提醒句相同的组并成一组，切出的类到这里不会与收窄的原类同句：这一条是防御，挡住别的造提案路径
    if sentence_key(new.reminder) == sentence_key(narrowed.reminder):
        return f"reminders are the same ({new.reminder!r})"
    for side, members in (("split-out", new_members), ("remaining", kept_members)):
        days = len({member.day for member in members})
        if len(members) < min_count or days < min_days:
            return f"{side} side has {len(members)} entries over {days} days"
    return None


__all__ = ["split_shortfall"]
