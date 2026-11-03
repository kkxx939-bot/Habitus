"""时间区间上的小判断（纯函数）。

观测空白的读法收在这里一处（``standing_pieces``）：事件序列、按天读树的视图、行为层的当日叙述、预测层白天那一拍都用它，
行为层读不了 ``series``，所以规则放在两边都能 import 的这一层。
"""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Callable, Sequence
from datetime import date, datetime, time, timedelta


def starts_within(begin: datetime, end: datetime, starts: Sequence[datetime]) -> bool:
    """``starts``（升序）里有没有一个落在 ``[begin, end)``。"""

    stamps = [moment.timestamp() for moment in starts]
    index = bisect_left(stamps, begin.timestamp())
    return index < len(stamps) and stamps[index] < end.timestamp()


def day_pieces(begin: datetime, end: datetime) -> tuple[tuple[datetime, datetime], ...]:
    """把 ``[begin, end)`` 按本地日切开（日界按 ``begin`` 自己的偏移算），每一天一段；零宽的留在它那一天。"""

    if end <= begin:
        return ((begin, end),)
    pieces: list[tuple[datetime, datetime]] = []
    start_of_day = datetime.combine(begin.date(), time(), tzinfo=begin.tzinfo)
    while start_of_day < end:
        following = start_of_day + timedelta(days=1)
        piece = (max(begin, start_of_day), min(end, following))
        if piece[1] > piece[0]:
            pieces.append(piece)
        start_of_day = following
    return tuple(pieces)


def standing_pieces(
    begin: datetime,
    end: datetime,
    *,
    voidable: bool,
    starts_on: Callable[[date], Sequence[datetime]],
) -> tuple[tuple[datetime, datetime], ...]:
    """一段观测空白里还站得住的那几段。

    - 先按本地日切开（``day_pieces``）：每一天那一段只由**那一天**的记录判。这样第 N 晚读到的空白只取决于 N 之前的日子，
      之后追加的记录改不了已封口那天的读法（13 号方案第四节的截止日不变量）；跨午夜的空白前一天那半段与后一天那半段各判各的。
    - ``voidable``（在看却没读懂）：这一段里读出了一条行为的起点（``starts_on(那一天)`` 给那一天算数的行为起点，升序），
      这一段整段作废——它断言的正是"这段读不出行为"，读出来就被证伪了。只挖掉"那一瞬"是零测度、等于没挖，挖"可见跨度"又要
      编造宽度，整段作废是唯一不用编宽度的规则。
    - 不可作废的（没在看）原样保留：里面出现了记录是上游自相矛盾，这里不消解。
    """

    pieces = day_pieces(begin, end)
    if not voidable:
        return pieces
    return tuple(piece for piece in pieces if not starts_within(*piece, starts_on(piece[0].date())))


__all__ = ["day_pieces", "standing_pieces", "starts_within"]
