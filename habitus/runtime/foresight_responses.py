"""开着的提醒与回话的桥（方案 30 第五节；裁定 61 第 5 条、62 第 2 条）：会话 lane 每轮问"此刻还开着哪些提醒"、把模型判出的
回应写回账本。住在组合根：行为侧不认识预测层，这座桥实现会话 lane 的 ``ReminderFeedback`` 接口、读写预测层的账本。

- **开着的提醒**：账本里还没结完的日子上说出口的递送，预期的最晚时限还没到、还没判出正 / 负的；按截止由近到远，最多
  ``limit`` 条（``config.foresight.open_reminders_judged``）。
- **回应**：按轮的身份记一条（``Response.turn``），同一轮同一条提醒重放时内容相同、幂等；写不进去（账本冲突、读不到预期在哪一天、
  盘坏了）都不让这一轮失败，只记观测——回应丢了结算按"没回应"处理。
- **省读**：先只读递送目录，这一天没有说出口的就不读预期与回话；``at`` 可能是 UTC 写法而预期的日子是主体本地日，预筛留一天余量。
"""

from __future__ import annotations

from datetime import datetime, timedelta

from habitus.behavior.fusion.lanes.session.reminders import PendingReminder, ReminderResponse
from habitus.foresight import ForesightError
from habitus.foresight.ledger import Delivery, Expectation, Response
from habitus.foresight.responses import open_reminders
from habitus.foundation.observability import NullObserver, ObservationEvent, ObservationStatus, Observer
from habitus.runtime.foresight_ledger import ForesightLedgerStore


class ReminderResponses:
    def __init__(self, ledger: ForesightLedgerStore, *, limit: int, observer: Observer | None = None) -> None:
        if not isinstance(ledger, ForesightLedgerStore):
            raise TypeError("ledger must be a ForesightLedgerStore")
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("limit must be a positive integer")
        self.ledger = ledger
        self.limit = limit
        self.observer: Observer = observer if observer is not None else NullObserver()

    def open(self, at: datetime) -> tuple[PendingReminder, ...]:
        """``at`` 那一刻还开着的提醒。只看账本里还没结完的日子（说中的范围还没过完的预期都在那里）。"""

        expectations: list[Expectation] = []
        deliveries: list[Delivery] = []
        responses: list[Response] = []
        for day in self.ledger.open_days():
            if day > at.date() + timedelta(days=1):
                continue
            spoken = [item for item in self.ledger.deliveries_on(day) if item.spoken]
            if not spoken:
                continue  # 这一天一句都没说出口（送达没接上时每一天都是）：预期与回话不必读
            deliveries.extend(spoken)
            expectations.extend(self.ledger.expectations_on(day))
            responses.extend(self.ledger.responses_on(day))
        found = open_reminders(expectations, deliveries, responses, at=at, limit=self.limit)
        return tuple(PendingReminder(item.expectation_id, item.said_at, item.text, item.deadline) for item in found)

    def respond(self, turn: str, response: ReminderResponse, *, at: datetime) -> None:
        try:
            day = self.ledger.day_of(response.reminder_id)
            if day is None:
                self._observe(ObservationStatus.FAILURE, {"error": "unknown_reminder"})
                return
            self.ledger.record_response(day, Response(response.reminder_id, at, response.polarity, turn))
        except (ForesightError, OSError) as exc:
            # 同一轮对同一条提醒重放出了不同的方向、索引读不了、盘写不进：账本只增不改，这一次的不写；回应丢了结算按
            # "没回应"处理。这一轮的记录已经落定，不让它因为回应写不进去再失败一次
            self._observe(ObservationStatus.FAILURE, {"error": type(exc).__name__})
            return
        self._observe(ObservationStatus.SUCCESS, {"polarity": response.polarity})

    def _observe(self, status: ObservationStatus, attributes: dict[str, str | int | float | bool]) -> None:
        try:
            self.observer.record(
                ObservationEvent(
                    category="foresight",
                    operation="response",
                    status=status,
                    duration_seconds=0.0,
                    attributes=attributes,
                )
            )
        except Exception:  # noqa: BLE001 - 观测不许影响融合
            pass


__all__ = ["ReminderResponses"]
