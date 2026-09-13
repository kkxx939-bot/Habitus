"""四层拆解：一个候选在这一刻的数字，以及每一层各自是哪几天攒出来的。

树把候选的率算成一条收缩链——本槽 ← 邻域 ← 跨周几 ← 全天——但发布出来的只是链末端那一个数。
判断者要的是链本身：「本槽 2/4、邻域 6/20、跨周几 0.1、全天 0.05」这样能自己掂量的来龙去脉。
本模块从已发布的树上把这四层恢复出来，**一个数都不重算**。

同时恢复每一层的**出处日**。出处直接取树上的 ``cell_days``，不由本层按覆盖日重推：树读整棵
行为树、按覆盖扣、按衰减加权，重推等于同一批上游记录被两个消费者读成两种事实。四层的日子
从同一份出处并出来，并法必须照着那一层实际是怎么算的：

- 本槽：就是该格。
- 邻域：``pool_indexes`` 那几格的并集——池化是**无权重的环形窗口求和**，所以并集与它严格对应。
- 跨周几：七个周几 × **同一个 ±k 邻域**的并集。链上这一层是
  ``Σ_w pooled_top[w][slot] / Σ_w pooled_bottom[w][slot]``（``_dense_chain`` 里的 ``cross``），
  它只把周几从 1 扩到 7，**时间窗保持不变**。
  **不要用 ``weekday_baselines``**：那是 ``per_slot_cross_weekday``，按精确槽归并、带 Laplace，
  是 ``lift_周几`` 的分母，不是收缩链的任何一层（``nodes.py`` 里那句"也是收缩链倒数第二层的
  先验"是错的，``_dense_chain`` 从来没收到过它）。真实数据实测：12 周每周一 19:00 打球，
  链上第三层 0.0286、``weekday_baselines`` 0.1479，差 5.18 倍——前者让邻域显出 7.00 倍的
  周规律，后者只剩 1.35 倍，恰好把"每周二打球"这类规律抹平成"这个时段大家都忙"。
- 全天：该动作全部格子的并集。

零 IO、零模型：输入是钉住的一代树与一个"这天归组了没有"的谓词，输出是纯数据。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date

from habitus.foresight.errors import ForesightError
from habitus.foresight.model import Layer, Provenance
from habitus.prediction.errors import PredictionTreeError
from habitus.prediction.model import WEEKDAYS, NodeStatistics, PredictionTree, SlotKey, slot_count
from habitus.prediction.query import neighbourhood


@dataclass(frozen=True)
class CellIndex:
    """树上的格子按动作归好，**一次装配建一次**。

    全天那一层要并这个动作的全部格子；不建索引就得为每个候选扫一遍 ``tree.nodes``，
    真实体量下是"格子数 × 候选数"次遍历（十万级 × 数百）。索引本身只是一次线性扫。
    """

    tree: PredictionTree
    by_action: Mapping[str, tuple[tuple[SlotKey, NodeStatistics], ...]]

    @classmethod
    def of(cls, tree: PredictionTree) -> CellIndex:
        if not isinstance(tree, PredictionTree):
            raise ForesightError("tree must be a PredictionTree")
        grouped: dict[str, list[tuple[SlotKey, NodeStatistics]]] = {}
        for (slot, action), statistics in tree.nodes.items():
            grouped.setdefault(action, []).append((slot, statistics))
        return cls(
            tree=tree,
            by_action={
                action: tuple(sorted(cells, key=lambda item: (item[0].weekday, item[0].slot)))
                for action, cells in grouped.items()
            },
        )

    def cells(self, action: str) -> tuple[tuple[SlotKey, NodeStatistics], ...]:
        return self.by_action.get(action, ())


def provenance(
    cells: CellIndex,
    action: str,
    slot: SlotKey,
    *,
    half_width: int,
    grouped: Callable[[date], bool],
) -> Provenance:
    """一个候选在 ``slot`` 上的四层拆解。

    ``grouped(day)`` 回答"情景树归组过这一天没有"——没归组的日子有数、没背景，进
    ``Layer.ungrouped`` 如实说出来，而不是让它在取背景时悄悄少掉几条。这个谓词由调用方注入
    （生产实现是情景树的指针查询），所以本模块不认识情景树。
    """

    tree = cells.tree
    if not isinstance(action, str) or not action:
        raise ForesightError("action must be non-empty text")
    if not isinstance(slot, SlotKey):
        raise ForesightError("slot must be a SlotKey")
    if not callable(grouped):
        raise ForesightError("grouped must be a predicate on dates")
    # ``SlotKey`` 自己不查槽的上界，而 ``pool_indexes`` 会静默取模——槽 500 会安静地落到
    # 槽 18–22 那个**不相干的邻域**上，然后在读曲线时崩在一个看不出原因的 IndexError。
    if not 0 <= slot.slot < slot_count(tree.slot_minutes):
        raise ForesightError("slot falls outside this tree's clock face")
    try:
        pool = neighbourhood(tree, slot, half_width)
    except PredictionTreeError as exc:
        # 本层把下层的失败归一成自己的类型（见 errors.py）：调用方只该认识 ForesightError。
        raise ForesightError(str(exc)) from exc
    # 只扩周几、不动时间窗——这正是链上 ``cross`` 的构造。
    across = tuple(SlotKey(weekday=weekday, slot=key.slot) for weekday in range(WEEKDAYS) for key in pool)
    whole_day = tuple(key for key, _statistics in cells.cells(action))
    return Provenance(
        slot=_counted(cells, action, (slot,), name="slot", grouped=grouped),
        pool=_counted(cells, action, pool, name="pool", grouped=grouped),
        cross_weekday=_counted(cells, action, across, name="cross_weekday", grouped=grouped),
        all_day=_published(
            cells, action, whole_day, name="all_day", value=tree.baselines.get(action, 0.0), grouped=grouped
        ),
    )


def _counted(
    cells: CellIndex, action: str, keys: tuple[SlotKey, ...], *, name: str, grouped: Callable[[date], bool]
) -> Layer:
    """有裸账本的那两层：分子分母都摊在同一批格子上，比值不收缩、不平滑。"""

    tree = cells.tree
    hits = 0.0
    exposure = 0.0
    for key in keys:
        statistics = tree.nodes.get((key, action))
        if statistics is not None:
            hits += statistics.counts.occurred_days
        seen = tree.exposure.get(key)
        if seen is not None:
            exposure += seen.observed_days
    days = _days(tree, action, keys)
    return Layer(
        name=name,
        value=hits / exposure if exposure > 0.0 else 0.0,
        days=days,
        ungrouped=_ungrouped(days, grouped),
        hits=hits,
        exposure=exposure,
    )


def _published(
    cells: CellIndex,
    action: str,
    keys: tuple[SlotKey, ...],
    *,
    name: str,
    value: float,
    grouped: Callable[[date], bool],
) -> Layer:
    """只发布了率的那两层：树上就没有对应的裸账本，所以 hits / exposure 留空而不是编一个。"""

    days = _days(cells.tree, action, keys)
    return Layer(name=name, value=value, days=days, ungrouped=_ungrouped(days, grouped))


def _days(tree: PredictionTree, action: str, keys: tuple[SlotKey, ...]) -> tuple[date, ...]:
    """这几个格子的出处日并集（升序去重）。"""

    collected: set[date] = set()
    for key in keys:
        statistics = tree.nodes.get((key, action))
        if statistics is not None:
            collected.update(statistics.days)
    return tuple(sorted(collected))


def _ungrouped(days: tuple[date, ...], grouped: Callable[[date], bool]) -> tuple[date, ...]:
    return tuple(day for day in days if not grouped(day))


__all__ = ["CellIndex", "provenance"]
