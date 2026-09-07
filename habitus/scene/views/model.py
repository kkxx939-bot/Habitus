"""行为上下文视图：一条行为（或此刻）周围是什么样，九个槽位，全部是机械投影出来的**观测事实**。

槽位是定稿的语义维度（不是时间统计——那是预测树的）：时间、所在的事、此前步骤、前提（双来源）、
起因、上一次、同时在做、和谁、本条事实。缺什么就是空，不推测、不打分。此刻视图的"所在的事"
永远为空——今天还没有情景文档，本层不替判决层推断它。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime


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
class ContextView:
    """一条行为（``occurrence_uri`` 非空）或此刻（``occurrence_uri`` 为空）的上下文。

    ``covered`` 是"那一天有情景文档"这个机械事实：没覆盖的日子 scene 为空是"语义层没处理"，
    覆盖了仍为空才是"模型没把它归进任何事"。``today_kinds`` 只在此刻视图上有意义：今日已发生的
    kind，供"历史前提今天成立了没有"的对比，它不是前提本身。
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


SLOT_NAMES: tuple[str, ...] = (
    "time",
    "scene",
    "prior_steps",
    "preconditions",
    "causes",
    "last_time",
    "concurrent",
    "subjects",
    "fact",
)

# 对比表只比语义槽：时间（周几/槽位）是预测树的键，不在这里复述。
COMPARED_SLOTS: tuple[str, ...] = tuple(name for name in SLOT_NAMES if name != "time")

SLOT_LABELS: dict[str, str] = {
    "time": "时间",
    "scene": "所在的事",
    "prior_steps": "此前步骤",
    "preconditions": "前提",
    "causes": "起因",
    "last_time": "上一次",
    "concurrent": "同时在做",
    "subjects": "和谁",
    "fact": "本条事实",
}


__all__ = ["COMPARED_SLOTS", "SLOT_LABELS", "SLOT_NAMES", "ActionRef", "ContextView", "LastTime", "Precondition", "SceneRef"]
