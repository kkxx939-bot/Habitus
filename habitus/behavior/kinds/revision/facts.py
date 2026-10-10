"""定期拆改的输入事实：由归约侧从树上现读后交进来，词表这一层不读树。

- ``Member``：一条 occurrence 现在的 ``kind_token``、日子与内容（证据规则、封闭集合重判、锚点自检都看它）。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

from habitus.behavior.kinds.classify.request import OccurrenceContent
from habitus.behavior.kinds.ids import Lane
from habitus.behavior.kinds.model import BehaviorKindError


@dataclass(frozen=True)
class Member:
    occurrence: str
    token: str
    lane: Lane
    day: date
    content: OccurrenceContent

    def __post_init__(self) -> None:
        if not isinstance(self.occurrence, str) or not self.occurrence:
            raise BehaviorKindError("member occurrence must be non-empty text")
        if not isinstance(self.token, str) or not self.token:
            raise BehaviorKindError("member token must be non-empty text")
        object.__setattr__(self, "lane", Lane(self.lane))
        if isinstance(self.day, bool) or not isinstance(self.day, date):
            raise BehaviorKindError("member day must be a date")
        if not isinstance(self.content, OccurrenceContent):
            raise BehaviorKindError("member content must be OccurrenceContent")


def members_of(members: Sequence[Member], token: str) -> tuple[Member, ...]:
    return tuple(member for member in members if member.token == token)


__all__ = ["Member", "members_of"]
