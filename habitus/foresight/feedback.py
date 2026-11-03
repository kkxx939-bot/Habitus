"""单元的提醒反馈（裁定 61 第 4 条；方案 30 第四节）：说 + 做合起来，一个行为一份，单位权重。纯函数。

| 情况 | 提醒反馈 |
|---|---|
| 说了 → 之后做了（「提醒后发生」，说什么都一样） | 好 |
| 说了、可他在说之前就做了（结算是准时 / 早到 / 晚了但做了） | 不写、不算机会：行为识别滞后、提醒出口认出提问时他已经做完（裁定 65）——功劳不是提醒的 |
| 说了 → 负 / 没回应 → 没做 | 坏（提醒了他没做就是一次误提醒；没回应不算中性，不然误提醒率永远降不下来） |
| 说了 → 中 → 时限前做了 | 好 |
| 说了 → 中 → 没做 | 不写、不算机会（中说的是时机，不说明会不会做；兜底：进来的边按做照样掉） |
| 说了 → 已做 → 没看到做 | 不写、不算机会（他说做了、我们没看到：观测的缺口，不是误提醒；账上那一笔同样改判中性，裁定 91 第 3 条②） |
| 没说出口 | 不写（影子期没有"说"） |

读数与边同一种算法：只读最近 M 刻、按名次遗忘、Beta 上下界；上界掉到线下，单元的档就回影子（``units.unit_level``）。
名次按这个行为单元历史里的每一次机会推进（裁定 91 第 3 条①，改裁定 60 第 4 条"只数到时窗末还没做的"）：回了影子以后没有"说"、
不再写反馈，但他之后的每一次机会——做了的、没做的——都把名次往前推一格（分量 0，不改计数，只让旧的那几笔往后退、被遗忘），
遗忘把上界带回线上：降档是扣分、可恢复（裁定 54 第 2 条），天天做的事两笔坏约 4 次机会、三笔坏约 8 次回到线上。
不算机会的（作废、观测空白里的中性、被提醒才做的）不在单元历史里，也不推。
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import datetime

from habitus.foresight.edges import PastMoment, fold_sequence
from habitus.foresight.history import Moment
from habitus.foresight.ledger.model import SAID_DONE, Expectation, Response, Settlement, UnitFeedback
from habitus.foresight.reliability import Reliability


def reminder_feedback(
    item: Expectation, result: Settlement, *, spoken_at: datetime | None, responses: Sequence[Response]
) -> UnitFeedback | None:
    """依据那条预期结算时给单元写的一笔；没说出口、或回了中又没做的不写。``responses`` 是这条提醒收到的全部回话。"""

    if spoken_at is None:
        return None
    polarity = latest_polarity(responses)
    if result.outcome == "提醒后发生":
        return UnitFeedback(item.kind_token, item.formed_at, True, item.expectation_id, polarity)
    if result.outcome != "过期待决":
        return None  # 说之前就做了（结算只把说出之后的发生记成「提醒后发生」）：这次机会不算提醒的（裁定 65）
    if polarity in ("中", SAID_DONE):
        return None
    return UnitFeedback(item.kind_token, item.formed_at, False, item.expectation_id, polarity)


def latest_polarity(responses: Iterable[Response]) -> str | None:
    """这条提醒判出的方向：正 / 负关闭、之后不再判，所以取时刻最早的那一条关闭的（两份会话源乱序融合时，晚判的中不能盖掉
    早说的负）；没有关闭的才取时刻最晚的中。"""

    ordered = sorted(responses, key=lambda item: item.responded_at)
    closing = next((item for item in ordered if item.closes), None)
    if closing is not None:
        return closing.polarity
    return ordered[-1].polarity if ordered else None


def said_done(responses: Iterable[Response]) -> bool:
    """这条提醒判出的方向是不是「已做」（他说这一次已经做过了）：结算没看到做时，账上那一笔与提醒反馈都按中性（裁定 91
    第 3 条②）。与写提醒反馈读同一个方向（``latest_polarity``），同一组回话只有一种读法。"""

    return latest_polarity(responses) == SAID_DONE


def feedback_reading(
    facts: Iterable[UnitFeedback], *, moments: Iterable[Moment] = (), alpha: float, half_life: float, window: int
) -> Reliability:
    """单元此刻的提醒反馈读数（只读最近 M 刻、按名次遗忘；单位权重）。``moments`` 是这个行为的单元历史：其中每一刻（来了的、
    没来的；没有作为反馈记过的）各占一个名次、分量 0——之后的每一次机会推着时钟走，坏笔才会被遗忘（裁定 91 第 3 条①）。
    推时钟的那几刻只当名次、不当证据：记成"没来、分量 0"，不打断也不加长末尾那串连错（``edges.fold_sequence``）。"""

    rows = sorted(facts, key=lambda item: (item.at, item.expectation_id))
    written = {item.expectation_id for item in rows}
    sequence = [PastMoment(item.at, frozenset({item.expectation_id}), item.good, 1.0) for item in rows]
    sequence += [
        PastMoment(item.past.at, item.past.signature, False, 0.0)
        for item in moments
        if written.isdisjoint(item.expectation_ids)
    ]
    sequence.sort(key=lambda item: (item.at, sorted(item.signature)))
    return fold_sequence(sequence, alpha=alpha, half_life=half_life, window=window).reading


__all__ = ["feedback_reading", "latest_polarity", "reminder_feedback", "said_done"]
