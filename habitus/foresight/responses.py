"""开着的提醒（方案 30 第五节；裁定 61 第 5 条）：说出口、还没到最晚时限、还没判出正 / 负的那几条——每段会话进融合时带给模型，
判这段会话在回应哪一条、正 / 中 / 负。纯函数。

- 判出正 / 负就关，不再判；判出中不关——后面的会话照样判（先说"改天"、后来说"算了"记负），做也一直看，到最晚时限为止。
- 提示词里最多带 N 条（按截止由近到远；配置 ``open_reminders_judged``，启动值 5）；没带进去的只按做结算。
- 到最晚时限没判出来的按"没回应"结算（``feedback.reminder_feedback``：没回应又没做就是坏）。
- 只有会话 lane 有"说"：物理 lane 的提醒只看做没做。
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime

from habitus.foresight.errors import ForesightError
from habitus.foresight.ledger.model import Delivery, Expectation, Response


@dataclass(frozen=True)
class OpenReminder:
    """一条还开着的提醒：哪条预期、什么时候说的、说了什么、最晚到什么时候。"""

    expectation_id: str
    kind_token: str
    said_at: datetime
    text: str
    deadline: datetime

    def __post_init__(self) -> None:
        for label in ("expectation_id", "kind_token"):
            if not isinstance(getattr(self, label), str) or not getattr(self, label):
                raise ForesightError(f"open reminder {label} must be non-empty text")
        if not isinstance(self.text, str):
            raise ForesightError("open reminder text must be text")
        for label in ("said_at", "deadline"):
            value = getattr(self, label)
            if not isinstance(value, datetime) or value.utcoffset() is None:
                raise ForesightError(f"open reminder {label} must be a timezone-aware datetime")


def open_reminders(
    expectations: Iterable[Expectation],
    deliveries: Iterable[Delivery],
    responses: Iterable[Response],
    *,
    at: datetime,
    limit: int,
) -> tuple[OpenReminder, ...]:
    """``at`` 那一刻开着的提醒：说出口（在 ``at`` 之前）、最晚时限还没到、没有在 ``at`` 之前判出正 / 负的。按截止由近到远，
    最多 ``limit`` 条。"""

    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ForesightError("limit must be a positive integer")
    if not isinstance(at, datetime) or at.utcoffset() is None:
        raise ForesightError("at must be a timezone-aware datetime")
    closed = {item.expectation_id for item in responses if item.closes and item.responded_at <= at}
    by_id = {item.expectation_id: item for item in expectations}
    found: list[OpenReminder] = []
    for delivery in sorted(deliveries, key=lambda item: item.decided_at):
        item = by_id.get(delivery.expectation_id)
        if item is None or not delivery.spoken or delivery.decided_at > at:
            continue
        if item.deadline <= at or item.expectation_id in closed:
            continue
        if any(found_item.expectation_id == item.expectation_id for found_item in found):
            continue
        found.append(
            OpenReminder(item.expectation_id, item.kind_token, delivery.decided_at, delivery.text or "", item.deadline)
        )
    found.sort(key=lambda item: (item.deadline, item.said_at, item.expectation_id))
    return tuple(found[:limit])


def closes_after(responses: Sequence[Response]) -> bool:
    """这条提醒已经判出正 / 负（之后的会话不再判）。"""

    return any(item.closes for item in responses)


__all__ = ["OpenReminder", "closes_after", "open_reminders"]
