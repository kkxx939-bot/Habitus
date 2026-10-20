"""单元的提醒反馈（裁定 61 第 4 条；方案 30 第四节）：说 + 做合起来，一个行为一份，单位权重。纯函数。

| 情况 | 提醒反馈 |
|---|---|
| 说了 → 做了（说什么都一样） | 好 |
| 说了 → 负 / 没回应 → 没做 | 坏（提醒了他没做就是一次误提醒；没回应不算中性，不然误提醒率永远降不下来） |
| 说了 → 中 → 时限前做了 | 好 |
| 说了 → 中 → 没做 | 不写、不算机会（中说的是时机，不说明会不会做；兜底：进来的边按做照样掉） |
| 没说出口 | 不写（影子期没有"说"） |

读数与边同一种算法：只读最近 M 刻、按名次遗忘、Beta 上下界；上界掉到线下，单元的档就回影子（``units.unit_level``）。
名次按这个行为"到时窗末还没做"的机会推进（裁定 60 第 4 条）：回了影子以后没有"说"、不再写反馈，但他每一次没做的机会照样
把名次往前推一格（分量 0，不改计数，只让旧的那几笔往后退、被遗忘），遗忘把上界带回线上，单元才有再开口的一天。
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import datetime

from habitus.foresight.edges import PastMoment, fold_sequence
from habitus.foresight.history import Moment
from habitus.foresight.ledger.model import Expectation, Response, Settlement, UnitFeedback
from habitus.foresight.reliability import Reliability


def reminder_feedback(
    item: Expectation, result: Settlement, *, spoken_at: datetime | None, responses: Sequence[Response]
) -> UnitFeedback | None:
    """依据那条预期结算时给单元写的一笔；没说出口、或回了中又没做的不写。``responses`` 是这条提醒收到的全部回话。"""

    if spoken_at is None:
        return None
    polarity = latest_polarity(responses)
    done = result.outcome != "过期待决"
    if done:
        return UnitFeedback(item.kind_token, item.formed_at, True, item.expectation_id, polarity)
    if polarity == "中":
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


def feedback_reading(
    facts: Iterable[UnitFeedback], *, moments: Iterable[Moment] = (), alpha: float, half_life: float, window: int
) -> Reliability:
    """单元此刻的提醒反馈读数（只读最近 M 刻、按名次遗忘；单位权重）。``moments`` 是这个行为的单元历史：其中没来的那几刻
    （没有作为反馈记过的）各占一个名次、分量 0——影子期的机会推着时钟走。"""

    rows = sorted(facts, key=lambda item: (item.at, item.expectation_id))
    written = {item.expectation_id for item in rows}
    sequence = [PastMoment(item.at, frozenset({item.expectation_id}), item.good, 1.0) for item in rows]
    sequence += [
        PastMoment(item.past.at, item.past.signature, False, 0.0)
        for item in moments
        if not item.past.came and written.isdisjoint(item.expectation_ids)
    ]
    sequence.sort(key=lambda item: (item.at, sorted(item.signature)))
    return fold_sequence(sequence, alpha=alpha, half_life=half_life, window=window).reading


__all__ = ["feedback_reading", "latest_polarity", "reminder_feedback"]
