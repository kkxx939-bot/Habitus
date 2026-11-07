"""行为树读侧的两种值：时间轴上的一条原子行为（``FlowRow``）与一段观测空白（``ObservationGap``）。

全部是从行为树机械读出来的**观测事实**，缺什么就是空，不推测、不打分。行为之间的关系由语义树的关系表提供，
这里只剩行为树自己的读侧。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class FlowRow:
    """时间轴上的一条原子行为：够判断者认出它是什么、几点开始、最后所见几点、说了什么。

    ``day_count`` 是这个 kind **在这一行自己那天**全天发生的次数——"与某人交谈(100)"一眼就看得出是底噪。
    跨日的行各带各天的数，不借别的天的。
    """

    uri: str
    name: str
    kind_token: str
    at: datetime
    last_observed_at: datetime
    summary: str
    day_count: int
    #: 这次怎么结束的（行为树 occurrence 的 ``status``：ongoing / completed / interrupted / abandoned）；None = 没读到。
    status: str | None = None
    #: 谁发起的（行为树 ``prompted_by``：system / host_agent / self）；没写是 None。
    prompted_by: str | None = None
    #: 地点与同在的人（行为树的结构字段；上游没给时为空）。预测层白天判带条件关系的条件要看它们。
    place: str | None = None
    subjects: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.last_observed_at < self.at:
            raise ValueError("a flow row cannot be last seen before it started")
        if self.place is not None and not self.place:
            raise ValueError("a flow row's place is non-empty text or None")
        if any(not item for item in self.subjects):
            raise ValueError("a flow row's subjects are a tuple of names")
        if self.day_count < 1:
            raise ValueError("a flow row's own kind occurred at least once that day")


@dataclass(frozen=True)
class ObservationGap:
    """一段观测空白（行为树的 gap 节点，已按天裁开）：``kind`` 是上游词表里的两个取值之一（没读懂 / 未观测）。

    本层不翻译它——"这段时间没看到 X"到底是"没在看"还是"看了没读懂"，交给读它的人分辨；
    但取值只认上游词表（``behavior.schema.vocabulary.GAP_KINDS``），上游改名时在读入处硬失败。
    """

    started_at: datetime
    ended_at: datetime
    kind: str

    def __post_init__(self) -> None:
        for name in ("started_at", "ended_at"):
            value = getattr(self, name)
            if value.utcoffset() is None:
                raise ValueError(f"observation gap {name} must be a timezone-aware datetime")
        if self.ended_at < self.started_at:
            raise ValueError("observation gap must not end before it starts")
        if not self.kind:
            raise ValueError("observation gap kind must be non-empty text")


__all__ = ["FlowRow", "ObservationGap"]
