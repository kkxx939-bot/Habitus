"""一条候选的草稿：触发出来、还没附反向前因、没有身份的预期。对 scene 零知识。

三种触发各自产草稿，装配时统一补身份、前因清单、出处（预测树的代、机会表那一晚、词表版本）。时刻一律带时区。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta

from habitus.foresight.errors import ForesightError


@dataclass(frozen=True)
class Draft:
    """``kind_token`` 是候选的类；``source`` / ``trigger`` / ``trigger_instance`` 与账本预期同义；``window`` 来自历史那几次的
    实际时刻（可以早于此刻：到了钟点才建，形成时他可能已经做了）；``deadline`` 是这条触发自己那一段的边界；``basis`` 是命中的
    那几次（URI）；``observed_from`` 是从什么时候起的发生算这条说中的（钟面 / 日子：当天零点；事件：前因做完那一刻）。"""

    kind_token: str
    source: str
    trigger: str
    trigger_instance: str
    window: tuple[datetime, datetime]
    deadline: datetime
    basis: tuple[str, ...]
    opportunities: int
    hits: int
    observed_from: datetime

    def __post_init__(self) -> None:
        start, end = self.window
        if end < start:
            raise ForesightError("draft window must not end before it starts")
        if self.deadline < end:
            raise ForesightError("draft deadline must not come before the window ends")
        if len(self.basis) > self.hits or self.hits > self.opportunities:
            raise ForesightError("draft basis, hits and opportunities are inconsistent")
        if self.observed_from > self.deadline:
            raise ForesightError("a draft cannot count occurrences only after its deadline")


def instance_at(started_at: datetime) -> str:
    """一条记录作为触发实例的身份：它的开始时刻（UTC）。未封口的判断封口后 URI 才定下来，开始时刻前后不变，
    所以事件触发的实例用它，一条链不会因为前因封口而被当成第二个实例。"""

    return started_at.astimezone(UTC).isoformat()


def midnight_of(day: date, like: datetime) -> datetime:
    """``day`` 开始的那一刻（零点），用 ``like`` 的时区写。"""

    return datetime.combine(day, time(0), tzinfo=like.tzinfo)


def midnight_after(day: date, like: datetime) -> datetime:
    """``day`` 过完的那一刻（次日零点），用 ``like`` 的时区写。"""

    return datetime.combine(day + timedelta(days=1), time(0), tzinfo=like.tzinfo)


def at_time_of(day: date, moment: datetime, like: datetime) -> datetime:
    """``day`` 这一天、``moment`` 的本地时分秒，用 ``like`` 的时区写（历史记录各带自己的偏移，比较前先落到此刻的时区）。"""

    local = moment.astimezone(like.tzinfo)
    return datetime.combine(day, local.time().replace(tzinfo=None), tzinfo=like.tzinfo)


def clamp(
    start: datetime,
    end: datetime,
    boundary: datetime,
    *,
    now: datetime,
    slot_minutes: int,
    open_until: datetime | None = None,
    keep: bool = False,
    caught_from: datetime | None = None,
) -> tuple[tuple[datetime, datetime], datetime] | None:
    """时窗至少一槽宽（单样本的时刻只是一个点，一拍一槽的拍点落不上去）；上一次看的时候（``caught_from``，默认此刻）就已经
    过了时窗（``open_until`` 给了就取两者较晚的，事件触发用它多宽限一槽）是退场，返回 None——除非 ``keep``（这件事这回已经
    做了：那是一次命中，照建、照结算，裁定 49）。按上一次看的时刻判，拍点晚了一点时没做的那次也照建：不然账上只丢没做的、
    把账往上抬。时窗保留历史那几次的实际时刻，不挪到此刻；最晚时限不早于终点，也不早于此刻。"""

    end = max(end, start + timedelta(minutes=slot_minutes))
    still_open = end if open_until is None else max(end, open_until)
    looked = now if caught_from is None else min(caught_from, now)
    if looked > still_open and not keep:
        return None
    return (start, end), max(boundary, end, now)


__all__ = ["Draft", "at_time_of", "clamp", "instance_at", "midnight_after", "midnight_of"]
