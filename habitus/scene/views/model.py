"""行为上下文视图：一条行为周围是什么样，全部是机械投影出来的**观测事实**。

槽位都**按预测树算候选的维度对齐**（2026-09-12 定：树用哪些维度算候选，语义侧就沿同样的维度
取回上下文）：``first_of_day``（树的危险率与累积率只描述"当天第一次"）、``preceding`` /
``following``（树的转移边看的是时间上**紧邻**的上一条 / 下一条）、``gaps``（树的曝光分母：这段
时间在不在看）、``day_note``（当地日历对这一天的说法）。缺什么就是空，不推测、不打分。

原来还有"所在的事"、角色、事里更早成员留下的前提——那是按天归组的产物，已随日情景树删掉。
规律级的上下文（一句"当时是什么情况"、两类边、情境、前提）按行为 URI 取记录，读口在 ``gloss``。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

# 与预测树 ``transition_window_seconds`` 同一个上限（prediction/config.py）：视图只读相邻一天，
# 窗口超过一天会静默截断，所以在这里同样硬拒。
MAX_TRANSITION_WINDOW_SECONDS = 86_400.0


@dataclass(frozen=True)
class ActionRef:
    """一条行为的引用：URI + 名字 + kind_token（对比按 kind_token，展示用名字）。"""

    uri: str
    name: str
    kind_token: str


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

    def __post_init__(self) -> None:
        if self.last_observed_at < self.at:
            raise ValueError("a flow row cannot be last seen before it started")
        if isinstance(self.day_count, bool) or not isinstance(self.day_count, int) or self.day_count < 1:
            raise ValueError("a flow row's own kind occurred at least once that day")


@dataclass(frozen=True)
class LastTime:
    """同 kind 的上一次：哪一条、距今几天。

    原来还带"当时所在的事"，那是按天归组的概念；换成规律级之后，那一次的上下文按行为 URI
    直接取记录就有，不需要在这里带一个中间容器。
    """

    uri: str
    days_ago: int


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
            if not isinstance(value, datetime) or value.utcoffset() is None:
                raise ValueError(f"observation gap {name} must be a timezone-aware datetime")
        if self.ended_at < self.started_at:
            raise ValueError("observation gap must not end before it starts")
        if not isinstance(self.kind, str) or not self.kind:
            raise ValueError("observation gap kind must be non-empty text")


@dataclass(frozen=True)
class Neighbour:
    """转移窗口内紧邻的一条行为——三态，与预测树的转移边同一组答案：

    - ``action`` 非空：找到了（树上的一条转移边）；
    - ``action`` 为空且 ``censored`` 为假：窗口内**确认没有**（树上的 ∅ 边——"做完就收工"）；
    - ``censored`` 为真：窗口内有观测空洞，不知道（树上的删失，既不算边也不算 ∅）。

    "没算"（调用方没给窗口）不在这里表达，那是视图字段本身为 None。
    """

    action: ActionRef | None
    censored: bool = False

    def __post_init__(self) -> None:
        if self.action is not None and self.censored:
            raise ValueError("a found neighbour cannot also be censored")


@dataclass(frozen=True)
class ContextView:
    """一条行为（``occurrence_uri`` 非空）或此刻（``occurrence_uri`` 为空）的上下文。

    ``today_kinds`` 只在此刻视图上有意义：今日已发生的 kind，供"历史前提今天成立了没有"的对比，
    它不是前提本身。

    按预测树维度对齐的几个字段：``first_of_day`` 只对历史视图有意义（此刻视图为 None）；
    ``preceding`` / ``following`` 只在调用方给了转移窗口时才算（没给即 None，不是"没有"——"没有"
    是三态里的第二态：找过了、确实没有）；
    ``gaps`` 只对此刻视图有意义。

    ``day_note`` 是当地日历对这一天的说法（"补班日，按周一上班"），没有日历数据时恒为空。它是
    摆在判断者面前的一个事实，不参与本层的任何筛选与排序——"今天像周几"由判断者自己判。
    """

    kind_token: str
    at: datetime
    occurrence_uri: str | None = None
    name: str | None = None
    causes: tuple[ActionRef, ...] = ()
    last_time: LastTime | None = None
    concurrent: tuple[ActionRef, ...] = ()
    subjects: tuple[str, ...] = ()
    summary: str | None = None
    first_of_day: bool | None = None
    preceding: Neighbour | None = None
    following: Neighbour | None = None
    day_note: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind_token, str) or not self.kind_token:
            raise ValueError("context view kind_token must be non-empty text")
        if not isinstance(self.at, datetime) or self.at.utcoffset() is None:
            raise ValueError("context view at must be a timezone-aware datetime")

    @property
    def day(self) -> date:
        return self.at.date()

    @property
    def weekday(self) -> int:
        return self.at.weekday()



def transition_window(value: float | None) -> float | None:
    """转移窗口的唯一守卫：None 即"不算"；否则必须是正数且不超过一天（视图只读相邻一天）。"""

    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float) or value <= 0:
        raise ValueError("transition_window_seconds must be a positive number or None")
    if value > MAX_TRANSITION_WINDOW_SECONDS:
        raise ValueError("transition_window_seconds must not exceed one day")
    return float(value)


__all__ = [
    "MAX_TRANSITION_WINDOW_SECONDS",
    "ActionRef",
    "ContextView",
    "FlowRow",
    "LastTime",
    "Neighbour",
    "ObservationGap",
    "transition_window",
]
