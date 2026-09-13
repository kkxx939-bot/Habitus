"""行为上下文视图：一条行为（或此刻）周围是什么样，全部是机械投影出来的**观测事实**。

槽位是定稿的语义维度（不是时间统计——那是预测树的）：时间、所在的事、此前步骤、前提（双来源）、
起因、上一次、同时在做、和谁、本条事实。缺什么就是空，不推测、不打分。此刻视图的"所在的事"
永远为空——今天还没有情景文档，本层不替判决层推断它。

另有几个**按预测树的维度对齐**的事实（2026-09-12 定：树用哪些维度算候选，语义侧就沿同样的维度
取回上下文）：``first_of_day``（树的危险率与累积率只描述"当天第一次"）、``preceding`` /
``following``（树的转移边看的是时间上**紧邻**的上一条 / 下一条，不是同一件事里的成员——两个口径
都给、分开标）、``next_steps``（同一件事里更晚的成员，"相似的那几次接下来发生了什么"）、
``gaps``（树的曝光分母：这段时间在不在看）。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

# 与预测树 ``transition_window_seconds`` 同一个上限（prediction/config.py）：视图只读相邻一天，
# 窗口超过一天会静默截断，所以在这里同样硬拒。
MAX_TRANSITION_WINDOW_SECONDS = 86_400.0

# 转移边的"无后继"哨兵，与预测树 ``edges.NO_SUCCESSOR`` 同字——对比表与画像把"确认没有"当一个可比的值。
NO_NEIGHBOUR = "∅"


@dataclass(frozen=True)
class ActionRef:
    """一条行为的引用：URI + 名字 + kind_token（对比按 kind_token，展示用名字）。"""

    uri: str
    name: str
    kind_token: str


@dataclass(frozen=True)
class SceneRef:
    """一件事的引用；``kinds`` 是它留下的待用前提的产生方 kind（作为起因/前提被对比时落到 kind 上；
    没记待用前提就是空——"它留下了什么"没有记录，不用成员冒充）。"""

    uri: str
    label: str
    kinds: tuple[str, ...] = ()


@dataclass(frozen=True)
class Precondition:
    """一条前提。``source`` 是它来自哪一路：``needs``（所在的事以 needs 指向的目标）、``pending``
    （所在的事里更早的成员留下的待用前提，或此刻的待用清单项）；两路同时收，抵消归组粒度的抖动。
    ``target_kinds`` 是它落到 kind 上的样子（对比按这个），空即没有可比的形式。"""

    source: str
    text: str
    target_uri: str
    target_kinds: tuple[str, ...] = ()


@dataclass(frozen=True)
class LastTime:
    """同 kind 的上一次：距今天数、当时所在的事。"""

    uri: str
    days_ago: int
    scene: SceneRef | None


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

    @property
    def absent(self) -> bool:
        return self.action is None and not self.censored

    @property
    def value(self) -> str | None:
        """对比与计数用的值：kind_token、∅、或 None（删失，不可比）。"""

        if self.censored:
            return None
        return NO_NEIGHBOUR if self.action is None else self.action.kind_token


@dataclass(frozen=True)
class ContextView:
    """一条行为（``occurrence_uri`` 非空）或此刻（``occurrence_uri`` 为空）的上下文。

    ``covered`` 是"那一天有情景文档"这个机械事实：没覆盖的日子 scene 为空是"语义层没处理"，
    覆盖了仍为空才是"模型没把它归进任何事"。``today_kinds`` 只在此刻视图上有意义：今日已发生的
    kind，供"历史前提今天成立了没有"的对比，它不是前提本身。

    按预测树维度对齐的几个字段：``first_of_day`` 只对历史视图有意义（此刻视图为 None）；
    ``preceding`` / ``following`` 只在调用方给了转移窗口时才算（没给即 None，不是"没有"——"没有"
    是 ``Neighbour.absent``）；``next_steps`` 与 ``prior_steps`` 同口径（同一件事里的成员）；
    ``gaps`` 只对此刻视图有意义。

    ``day_note`` 是当地日历对这一天的说法（"补班日，按周一上班"），没有日历数据时恒为空。它是
    摆在判断者面前的一个事实，不参与本层的任何筛选与排序——"今天像周几"由判断者自己判。
    """

    kind_token: str
    at: datetime
    occurrence_uri: str | None = None
    name: str | None = None
    covered: bool = False
    scene: SceneRef | None = None
    role: str | None = None
    prior_steps: tuple[ActionRef, ...] = ()
    preconditions: tuple[Precondition, ...] = ()
    causes: tuple[ActionRef | SceneRef, ...] = ()
    last_time: LastTime | None = None
    concurrent: tuple[ActionRef, ...] = ()
    subjects: tuple[str, ...] = ()
    summary: str | None = None
    today_kinds: frozenset[str] = frozenset()
    first_of_day: bool | None = None
    preceding: Neighbour | None = None
    following: Neighbour | None = None
    next_steps: tuple[ActionRef, ...] = ()
    gaps: tuple[ObservationGap, ...] = ()
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

    @property
    def minute_of_day(self) -> int:
        return self.at.hour * 60 + self.at.minute


def transition_window(value: float | None) -> float | None:
    """转移窗口的唯一守卫：None 即"不算"；否则必须是正数且不超过一天（视图只读相邻一天）。"""

    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float) or value <= 0:
        raise ValueError("transition_window_seconds must be a positive number or None")
    if value > MAX_TRANSITION_WINDOW_SECONDS:
        raise ValueError("transition_window_seconds must not exceed one day")
    return float(value)


SLOT_NAMES: tuple[str, ...] = (
    "time",
    "scene",
    "prior_steps",
    "preceding",
    "preconditions",
    "causes",
    "last_time",
    "concurrent",
    "subjects",
    "fact",
)

# 对比表只比语义槽与树维度的对齐槽：时间（周几/槽位）是预测树的键，不在这里复述。
COMPARED_SLOTS: tuple[str, ...] = tuple(name for name in SLOT_NAMES if name != "time")

SLOT_LABELS: dict[str, str] = {
    "time": "时间",
    "scene": "所在的事",
    "prior_steps": "此前步骤",
    "preceding": "紧邻上一条",
    "preconditions": "前提",
    "causes": "起因",
    "last_time": "上一次",
    "concurrent": "同时在做",
    "subjects": "和谁",
    "fact": "本条事实",
}


__all__ = [
    "COMPARED_SLOTS",
    "MAX_TRANSITION_WINDOW_SECONDS",
    "NO_NEIGHBOUR",
    "SLOT_LABELS",
    "SLOT_NAMES",
    "ActionRef",
    "ContextView",
    "LastTime",
    "Neighbour",
    "ObservationGap",
    "Precondition",
    "SceneRef",
    "transition_window",
]
