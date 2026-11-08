"""开着的提醒与回话的桥（方案 30 第五节；裁定 61 第 5 条、62 第 2 条）：会话 lane 每轮问"此刻还开着哪些提醒"、把模型判出的
回应写回账本。住在组合根：行为侧不认识预测层，这座桥实现会话 lane 的 ``ReminderFeedback`` 接口、读写预测层的账本。

- **开着的提醒**：账本里还没结完的日子上说出口的递送，预期的最晚时限还没到、还没判出正 / 负 / 已做的；按截止由近到远，最多
  ``limit`` 条（``config.foresight.open_reminders_judged``）。
- **回应**：按轮的身份记一条（``Response.turn``），同一轮同一条提醒重放时内容相同、幂等；写不进去（账本冲突、读不到预期在哪一天、
  盘坏了）都不让这一轮失败，只记观测——回应丢了结算按"没回应"处理。
- **出口记过的提问**：出口按固定选项写回话时 ``Response.turn`` 记的是那次提问调用的 call_id（会话 lane 判的记轮的身份）；
  会话 lane 问"这几次调用里哪些出口记过"，据此把那次问答剔出给模型的材料。
- **省读**：先只读递送目录，这一天没有说出口的就不读预期与回话；``at`` 可能是 UTC 写法而预期的日子是主体本地日，预筛留一天余量。
"""

from __future__ import annotations

from datetime import datetime, timedelta

from habitus.behavior.fusion.lanes.session.reminders import PendingReminder, ReminderResponse
from habitus.foresight import ForesightError
from habitus.foresight.ledger import Delivery, Expectation, Response
from habitus.foresight.responses import open_reminders
from habitus.foundation.observability import (
    NullObserver,
    ObservationStatus,
    Observer,
    record_observation,
)
from habitus.runtime.foresight_ledger import ForesightLedgerStore


class ReminderResponses:
    def __init__(self, ledger: ForesightLedgerStore, *, limit: int, observer: Observer | None = None) -> None:
        if limit < 1:
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

    def answered(self, calls: frozenset[str], at: datetime) -> frozenset[str]:
        """``calls`` 里出口已按固定选项写过回话的提问调用：还没结完的日子上、回话的 ``turn`` 就是这次调用的。"""

        if not calls:
            return frozenset()
        found: set[str] = set()
        for day in self.ledger.open_days():
            if day > at.date() + timedelta(days=1):
                continue
            if not any(item.spoken for item in self.ledger.deliveries_on(day)):
                continue  # 回话只写在说出口的提醒上
            found.update(item.turn for item in self.ledger.responses_on(day) if item.turn in calls)
        return frozenset(found)

    def respond(self, turn: str, response: ReminderResponse, *, at: datetime) -> None:
        # TODO(裁定 64 第 8 条)：模型判正 / 中 / 负 / 已做这条路保留、不删。他点固定选项的回话由出口直接写（``runtime/foresight_outlet.py``，
        # 不经模型）；走到这里的是他自己填的话。将来推式宿主（方案 32 第六节的本地通知）或别的 harness 的回复不是提问工具的作答时，
        # 也要靠这条路判——那时按钮作答仍该由出口 / 回执直接写，这里只管"话"。
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
        record_observation(self.observer, "foresight", "response", status, attributes=attributes)


__all__ = ["ReminderResponses"]
