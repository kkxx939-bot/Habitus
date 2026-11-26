"""宿主来取提醒的那一刻再判一次（裁定 104）：这一次新交出之前，按当下所知看这件事做没做。

一条规则，两支各看自己的：这一支有已经交进来、还没判完的（失败退避中的也算），判完之前不知道那一轮做的是哪件事，这一支这一次
全部先不给；都判完了，按递送同一个判法（``foresight.delivery.done_since``）看这一条做没做，做了的不给。被拦下的不写任何东西：
还在可取清单上，下一次再取；做了的由下一拍记「已发生」。已经交到宿主手里的不经过这里（``ReminderOutlet.claim``）。

这里只取材料、调判法。读材料出错照常抛出：这一次取提醒失败，宿主那边就是这一次没有提醒。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime

from habitus.foresight.delivery import done_since
from habitus.runtime.foresight_forming import ExpectationForming, Recent
from habitus.runtime.foresight_outlet import Hold, OpenRequest


@dataclass(frozen=True)
class ClaimBranch:
    """一支再判一次要的两样：读最近几天与账本的那一份、这一支有没有还没判完的。"""

    forming: ExpectationForming
    undecided: Callable[[], bool]


class ClaimCheck:
    def __init__(self, branches: Mapping[str, ClaimBranch]) -> None:
        self.branches = dict(branches)

    def for_one_claim(self) -> Hold:
        """一次 ``claim`` 用的判定：每支的材料问到才读、这一次只读一遍（``None`` 是这一支还有没判完的）。"""

        readings: dict[str, Recent | None] = {}

        def hold(request: OpenRequest, at: datetime) -> bool:
            branch = self.branches.get(request.lane)
            if branch is None:
                return False
            if request.lane not in readings:
                # 先看判完没有、再读行：看的时候判完了的，读行时一定读得到
                local = at.astimezone(branch.forming.read_side.zone)
                readings[request.lane] = None if branch.undecided() else branch.forming.snapshot(local)
            recent = readings[request.lane]
            if recent is None:
                return True
            item = branch.forming.ledger.expectation(request.day, request.request_id)
            return item is not None and done_since(item, recent.seen, now=recent.now, kinds=recent.kinds)

        return hold


__all__ = ["ClaimBranch", "ClaimCheck"]
