"""待关联清单：预测树算出候选之后，语义层还欠哪些 (候选, 日期) 的上下文。

**为什么由预测树驱动**：语义关联是为候选服务的，而候选与"这个数是哪几天攒出来的"只有预测树
知道。范围因此不是任何日历窗口，而是树上的出处日——周频行为的相邻两次正好隔 7 天，按日历回看
的窗口永远卡在边界上（旧实现的 ``lookback_days: 7`` 就是这个毛病）。

**为什么用差集而不是比较新旧两代树**：老代会被 ``published_generations`` 清掉，清了就比不出来。
差集只问"树上有哪些天"减"语义层已经关联过哪些天"，于是天然幂等（重跑不会重复关联）、漏跑几天
会自动补上、也不关心树重建过几次。

**为什么必须按时间升序处理**：规律级是**增量叠加**的——先关联 3/18 那次，再关联 3/25 那次，
"朋友教 → 约朋友"这个演化顺序才对。乱序或者"先做最近的"会让叠加出来的情境顺序错乱。

但升序只在**候选内部**是正确性要求：规律级按 ``kind_token`` 归档，候选与候选之间不存在叠加顺序
约束。所以预算是**按候选配额**发的，不是全局按日期截断——后者会让最老的那个候选把预算整段吃光
（实测：一个候选 60 个待关联日、另一个 1 个，全局截断下后者连跑六轮才排得上），而"低频但重要"
的行为正是最需要上下文的那一类。

**为什么还要一个"跳过"的口子**：待办集只有"树上有"减"做完了"两种状态，没有第三种。一件永久
失败的任务（输入本身让模型答不出来、或者超出单次调用上限）永远不会被标完成，于是每一轮都排在
那个候选的最前面，把配额里的一个名额永久吃掉——``per_candidate=1`` 时就是全部。候选内部因此
head-of-line 堵死，而整条链路没有任何地方会报错。所以 ``blocked`` 是注入的第二个事实源，
由编排层的失败记账实现；不给就是"什么都没被挡住"。

零 IO 纯函数：树由调用方钉住一代，"已关联的日期"与"被挡住的日期"都由调用方注入
（第一期还没有生产实现）。
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date, datetime
from typing import Protocol

from habitus.prediction.model import PredictionTree, SlotKey


def slot_order(slot: SlotKey) -> tuple[int, int]:
    """格子的全序。``SlotKey`` 自己没有序，而这里的"按槽位线性推进"必须是确定的。"""

    return (slot.weekday, slot.slot)


@dataclass(frozen=True)
class AssociationTask:
    """一件待关联的事：这个候选在这一天发生过，语义层还没关联它。

    ``slots`` 是它那天落在哪几个格子上——同一天做了两次、落在不同槽，仍然只是**一件**待关联的
    事（关联的单位是"这个候选的这一天"），但格子要带上：关联时要按槽位线性推进，也要让下游知道
    这次发生对应的是钟面上的哪一格。
    """

    kind_token: str
    day: date
    slots: tuple[SlotKey, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.kind_token, str) or not self.kind_token:
            raise ValueError("kind_token must be non-empty text")
        # ``datetime`` 是 ``date`` 的子类：放它过去的话，这个 day 与树上的 ``date`` 永远不相等，
        # 这件任务会每一轮重新出现。
        if isinstance(self.day, datetime) or not isinstance(self.day, date):
            raise TypeError("day must be a date without a time")
        if not isinstance(self.slots, tuple):
            # frozen dataclass 要可哈希：传 list 会在 hash() 时才炸，离出错点太远。
            raise TypeError("slots must be a tuple")
        if not self.slots:
            raise ValueError("an association task must name the cells it came from")
        keys = [slot_order(slot) for slot in self.slots]
        if any(later <= earlier for earlier, later in zip(keys, keys[1:], strict=False)):
            raise ValueError("association task slots must be sorted and unique")


class AssociatedDays(Protocol):
    """语义层已经关联**完成**哪些日期。按候选问，返回日期集合。

    生产实现是 ``scene.regularity.RegularityTree.days_for``——它按**完成标记**回答，不是按日目录
    在不在：目录在工作开始前就存在，中断过的那一天会被永久跳过。

    返回空集就是"这个候选一天都没关联过"，不是错误——新候选本来如此。
    """

    def days_for(self, kind_token: str) -> frozenset[date]: ...


class AssociationLedger(AssociatedDays, Protocol):
    """``AssociatedDays`` 加上"按哪个版本算的"：读关联记录的人按同一个版本读，两边才对得上。"""

    @property
    def version(self) -> str: ...


class BlockedDays(Protocol):
    """哪些日期这一轮不要再取了。按候选问，返回日期集合。

    与 ``AssociatedDays`` 是两件不同的事实："关联完成了"是产物，"被挡住了"是失败记账；一件被挡住
    的任务将来可能解封（输入变了、上限调了），所以它不能被写成"已完成"。
    """

    def days_for(self, kind_token: str) -> frozenset[date]: ...


class _NothingBlocked:
    """默认实现：什么都没被挡住。"""

    def days_for(self, kind_token: str) -> frozenset[date]:
        return frozenset()


def backlog(
    tree: PredictionTree,
    associated: AssociatedDays,
    *,
    per_candidate: int,
    limit: int,
    blocked: BlockedDays | None = None,
) -> tuple[AssociationTask, ...]:
    """树上的出处日减去已关联完成的日期，返回待关联的事。

    ``per_candidate`` 是每个候选这一轮最多取几件（各自取**最早**的那几件），``limit`` 是整轮的
    总上限。两级预算的理由见模块开头：候选内部必须升序，候选之间不必，而全局截断会饿死新候选。
    ``blocked`` 里的日期这一轮直接不取，让被挡住的那件事后面的活能往前走。

    返回顺序仍是时间升序——处理顺序线性向前，同一天内按槽位推进。
    """

    if not isinstance(tree, PredictionTree):
        raise TypeError("tree must be a PredictionTree")
    barrier: BlockedDays = _NothingBlocked() if blocked is None else blocked
    for name, source in (("associated", associated), ("blocked", barrier)):
        if not hasattr(source, "days_for") or not callable(source.days_for):
            raise TypeError(f"{name} must provide days_for(kind_token)")
    for name, value in (("per_candidate", per_candidate), ("limit", limit)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    cells = _cells_by_action(tree)
    tasks: list[AssociationTask] = []
    for action in sorted(cells):
        skip = _day_set(associated, action, "days_for") | _day_set(barrier, action, "blocked days_for")
        owed = [
            AssociationTask(kind_token=action, day=day, slots=tuple(sorted(slots, key=slot_order)))
            for day, slots in sorted(cells[action].items())
            if day not in skip
        ]
        # 每个候选只取自己最早的那几件：候选内部的升序是叠加的正确性要求，配额不能破坏它。
        tasks.extend(owed[:per_candidate])
    # 同一天内按候选**最晚**的那一格排。按最早的排会让"07:00 做早饭、20:00 做晚饭"这个候选
    # 整批排在"09:00 买菜"前面，于是买菜那天留下的前提，当天晚上那次做饭一条都看不见。
    # 第三级（候选名）今天是冗余的，留着是为了让顺序**不依赖**插入顺序。
    tasks.sort(key=lambda task: (task.day, slot_order(task.slots[-1]), task.kind_token))
    return tuple(tasks[:limit])


def _day_set(source: AssociatedDays | BlockedDays, action: str, label: str) -> frozenset[date]:
    """每个候选**只问一次**；元素类型也要查。

    只查容器类型是不够的：返回 ``frozenset[datetime]`` 能通过，而 ``day in done``（``date`` 比
    ``datetime``）永远为假，于是同一批日子每一轮都被重新关联一遍。
    """

    days = source.days_for(action)
    if not isinstance(days, frozenset | set):
        raise TypeError(f"{label} must return a set of dates")
    if any(isinstance(item, datetime) or not isinstance(item, date) for item in days):
        raise TypeError(f"{label} must return plain dates, not datetimes")
    return frozenset(days)


def _cells_by_action(tree: PredictionTree) -> Mapping[str, dict[date, set[SlotKey]]]:
    """把树上的出处日按 (候选, 日期) 归并；一天落在多个格子的合成一件事。

    ``NodeStatistics.days`` 记的是这个动作在这一格**实际起过头**的日子（``earlier_days`` 那种
    "当天更早已经做过"不在里面），所以这里数出来的就是"这个候选那天真的发生了"。
    """

    grouped: dict[str, dict[date, set[SlotKey]]] = {}
    for (slot, action), statistics in tree.nodes.items():
        days = grouped.setdefault(action, {})
        for day in statistics.days:
            days.setdefault(day, set()).add(slot)
    return grouped


@dataclass(frozen=True)
class CauseFact:
    """一个候选前因的两个数字，**取自树、原样传递**。

    ``rarity`` 是这个动作不看时刻的整体发生率（``tree.baselines``），越低越罕见——前因更可能是
    "跟朋友通话"这种少见事件，不是每天都做的吃饭看手机。``transitions`` 是树上"它 → 这个候选"
    这条边的计数，没有这条边就是 None。

    两个数只决定**哪些行为值得挑出来给模型看**，不进提示词：把它们摆给模型会让它拿转移计数倒推
    语义前因，统计被洗成语义再喂回预测层，形成自证。语义层的维度必须是语义的表现。
    """

    action: str
    rarity: float
    transitions: float | None = None

    @property
    def order(self) -> tuple[float, float, str]:
        """排序键：先看树上有没有指过来的边（有的在前，计数多的在前），再看越罕见越前。"""

        return (0.0 if self.transitions is None else -self.transitions, self.rarity, self.action)


class CauseFacts:
    """一棵钉住的预测树折出来的前因排序依据。

    **它是 scene 侧认识预测树的唯一产物**：编排与输入装配只拿这些事实，完全不认识 ``PredictionTree``。
    这样"语义层不重算数字"由类型保证，而不是靠自觉——架构测试钉死了读预测树的模块只有本文件。
    """

    def __init__(self, tree: PredictionTree) -> None:
        if not isinstance(tree, PredictionTree):
            raise TypeError("tree must be a PredictionTree")
        self._baselines = dict(tree.baselines)
        self._transitions: dict[tuple[str, str], float] = {
            (source, target): statistics.count for (source, target), statistics in tree.edges.items()
        }

    def ranked_for(self, kind_token: str, candidates: Iterable[str]) -> tuple[CauseFact, ...]:
        """把一批候选前因动作按"值得看的程度"排好。未知动作按基线 1.0 收尾，不丢。"""

        seen: dict[str, CauseFact] = {}
        for action in candidates:
            if action in seen:
                continue
            seen[action] = CauseFact(
                action=action,
                rarity=float(self._baselines.get(action, 1.0)),
                transitions=self._transitions.get((action, kind_token)),
            )
        return tuple(sorted(seen.values(), key=lambda fact: fact.order))


__all__ = [
    "AssociatedDays",
    "AssociationLedger",
    "AssociationTask",
    "BlockedDays",
    "CauseFact",
    "CauseFacts",
    "backlog",
    "slot_order",
]
