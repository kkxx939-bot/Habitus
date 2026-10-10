"""节律口的实现：从预测树算出每个行为概念"典型一天"的那几个机会。

节律口答"这个概念平常一天有几次机会、几点、多大概率"——把七个周几的曲线**逐槽平均**再取峰。读它的有两处：
触点①写概念（它写的是不分周几的常识判断，给七条曲线只会让提示词变噪音），常态按峰分份（``occurrences.baselines``）。

**为什么在 runtime 而不在 scene**：``scene`` 不许 import ``prediction``（架构测试按传递闭包钉死），所以这座桥
住在组合根，scene 只见到 ``RhythmProvider`` 协议与 ``Rhythm`` 值。

**概念 → 曲线**（裁定 20）：树按类编号存曲线；概念记着自己的类（``ConceptSet.classes_of``），那几条曲线**逐槽相加、
截到 1**。基础概念与细分概念是自己那一个类的曲线，汇总概念是成员类相加（成员在同步词表时已跟着拆分 / 合并改写成现编号）。
相加的依据是 B13（同一槽近似互斥）。

平均而不是挑某一天：挑"峰最多的那天"会让工作日型的行为按最忙的周三报，给出的节律对周末不成立；
平均之后周末没做的那几天会把峰压低，峰还在、概率更老实。

``days_with_peaks`` 数的是**逐日**有没有峰（不是平均曲线上的），因为分型问的正是"是不是天天有机会"。
``recurrence_hours`` 取这个概念那几个类里**最短**的复发间隔中位数：汇总概念的任一成员类发生它就发生，
所以真实间隔不会长于最短那个——这是上界估计。
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from habitus.prediction.model import MINUTES_PER_DAY, PredictionTree
from habitus.scene.concepts.model import ConceptSet
from habitus.scene.concepts.rhythm import MAX_RHYTHM_PEAKS, Rhythm, RhythmPeak

WEEKDAYS = tuple(range(7))
#: 一个峰至少要有多大概率才算得上"机会"。**没有这条下限，"高于当日均值"在稀疏曲线上会退化**：
#: 2026-09-29 探针实测，一个 45 天里 97 次的行为在 96 槽的钟面上均值只有 0.0107，于是 6 个**单槽**
#: （质量 0.033–0.046）跟着两个真峰（质量各 0.31，08:00–10:00 与 12:00–13:45）一起被当成峰——
#: 一天报出 8 个"机会"，12 个概念合计 119 个——常态按峰分份时每份都切得太碎，触点①读到的节律也是噪音。
#: 0.10 是**参考档**（探针数据上刚好滤掉全部单槽伪峰、留下两个真峰），真实数据上按"一天真有几次"复核。
MIN_PEAK_MASS = 0.10


@dataclass(frozen=True)
class Peak:
    """曲线上的一个峰：起止槽（含两端）、中心槽、峰内 marginal 之和。

    **跨午夜的峰** ``last_slot``（可能还有 ``centre_slot``）**≥ 槽数**：就寝常态在 23:00–01:00 时，当天末尾的质量接着
    **次日周几曲线**开头的质量算一个峰，归给它**开始**的那一天，尾巴落在次日（按日历日各取各的会切成两半，
    每半够不上 ``MIN_PEAK_MASS`` 的整个消失）。
    """

    first_slot: int
    last_slot: int
    centre_slot: int
    mass: float


def class_curves(concepts: ConceptSet) -> dict[str, tuple[str, ...]]:
    """桥：每个行为概念 → 要相加的那几条曲线的键（类编号）。情境概念没有曲线。

    基础 / 细分概念是自己那一个类；汇总概念是它的成员类——成员在同步词表时已改写成现编号（拆出来的新类加进来、
    被并掉的换成并入的类），桥与命中聚合读同一份成员。
    """

    return {identity: concepts.classes_of(identity) for identity in concepts.behaviors()}


def merged_marginal(tree: PredictionTree, weekday: int, kinds: Sequence[str]) -> tuple[float, ...] | None:
    """这个概念在这个周几的曲线：它那几个类的 ``marginal`` 逐槽相加、截到 1。

    一个类都没有曲线（这个周几从没做过）→ ``None``，调用方跳过这一天。
    """

    curves = [tree.curves[(weekday, kind)] for kind in kinds if (weekday, kind) in tree.curves]
    if not curves:
        return None
    slots = len(curves[0].marginal)
    return tuple(min(1.0, sum(curve.marginal[slot] for curve in curves)) for slot in range(slots))


def peaks_of(
    curve: Sequence[float],
    *,
    floor: float = MIN_PEAK_MASS,
    following: Sequence[float] | None = None,
    preceding: Sequence[float] | None = None,
) -> tuple[Peak, ...]:
    """连续一段高于当日均值、**且峰内质量够得上 ``floor``** 的槽 = 一个峰。

    两道判据各管一件事：高于均值管"这一段比别处更可能"，``floor`` 管"它值得叫一次机会"。
    只有前者时稀疏曲线会把每个略高于均值的单槽都算成机会（见 ``MIN_PEAK_MASS`` 的实测）。
    均值为 0（整条曲线都是 0）时没有峰。

    **跨午夜**（2026-09-30 二-4）：给了 ``following``（**真正的次日**那个周几的曲线）就把两天拼成一条 2×槽数 的连续轴，
    当天末尾高于均值的一段接着次日开头高于均值的一段算**一个**峰，``last_slot ≥ 槽数`` 表示尾巴在次日；
    次日开头那一段不再单独成峰（它属于前一天开始的那个峰）。没给 ``following`` = 次日没数据 → **不猜**：峰截在 24:00，
    不拿当天开头绕回来补（"没看过就当没看到"）。

    对称地，给了 ``preceding``（前一天的曲线）且它的末尾高于它的均值时，当天从 00:00 开始的那一段是**前一天那个峰的尾巴**，
    这里不再算成当天的峰；没给 ``preceding`` 就当它是当天自己的峰。
    """

    average = sum(curve) / len(curve) if curve else 0.0
    if average <= 0.0:
        return ()
    size = len(curve)
    axis = list(curve)
    if following is not None:
        if len(following) != size:
            raise ValueError("the following day's curve must have the same number of slots")
        axis += list(following)
    found: list[Peak] = []
    start: int | None = None
    for slot in range(len(axis)):
        if axis[slot] > average:
            start = slot if start is None else start
            continue
        if start is not None:
            found.append(_peak(axis, start, slot - 1))
            start = None
    if start is not None:
        found.append(_peak(axis, start, len(axis) - 1))
    tail_of_yesterday = preceding is not None and bool(preceding) and preceding[-1] > sum(preceding) / len(preceding)
    kept = []
    for peak in found:
        if peak.first_slot >= size:
            continue  # 次日自己的峰：归次日那天算，这里只取当天开始的
        if peak.first_slot == 0 and tail_of_yesterday:
            continue  # 前一天那个峰的尾巴，前一天已经把它算进去了
        if peak.mass >= floor:
            kept.append(peak)
    return tuple(kept)


def _peak(axis: Sequence[float], first: int, last: int) -> Peak:
    centre = max(range(first, last + 1), key=lambda slot: (axis[slot], -slot))
    return Peak(first_slot=first, last_slot=last, centre_slot=centre, mass=sum(axis[first : last + 1]))


class TreeRhythms:
    """按 ``RhythmProvider`` 协议实现的节律口。组合根在**重建树之后**构造它。

    ``kinds`` 是 ``class_curves`` 的产物（概念 → 要相加的那几条曲线的键）；
    ``labels`` 是给模型看的名字（基础概念的身份是类编号，显示类名）。
    """

    def __init__(
        self, tree: PredictionTree, kinds: Mapping[str, Sequence[str]], labels: Mapping[str, str] | None = None
    ) -> None:
        if not isinstance(tree, PredictionTree):
            raise TypeError("tree must be a PredictionTree")
        self.tree = tree
        self.kinds = {concept: tuple(tokens) for concept, tokens in kinds.items()}
        self.labels = dict(labels or {})

    def rhythms(self, among: Iterable[str] | None = None) -> Mapping[str, Rhythm]:
        wanted = tuple(self.kinds) if among is None else tuple(among)
        return MappingProxyType({concept: self.rhythm_of(concept) for concept in wanted})

    def rhythm_of(self, concept: str) -> Rhythm:
        label = self.labels.get(concept)
        kinds = self.kinds.get(concept, ())
        if not kinds:
            return Rhythm(concept=concept, label=label)
        curves = [
            curve for curve in (merged_marginal(self.tree, weekday, kinds) for weekday in WEEKDAYS) if curve is not None
        ]
        if not curves:
            return Rhythm(concept=concept, label=label)
        # 七天平均的曲线，"次日"还是它自己：跨午夜的峰接自己的开头即可（逐日数峰时才接真正的次日）
        average = _average(curves)
        days_with_peaks = sum(
            1
            for weekday_curve, preceding, following in _with_neighbours(kinds, self.tree)
            if peaks_of(weekday_curve, following=following, preceding=preceding)
        )
        peaks = peaks_of(average, following=average, preceding=average)[:MAX_RHYTHM_PEAKS]
        if not peaks or not days_with_peaks:
            return Rhythm(concept=concept, recurrence_hours=self._recurrence_hours(kinds), label=label)
        minutes = MINUTES_PER_DAY // len(curves[0])
        return Rhythm(
            concept=concept,
            peaks=tuple(
                RhythmPeak(
                    ordinal=ordinal,
                    start_minute=peak.first_slot * minutes,
                    end_minute=(peak.last_slot + 1) * minutes,
                    probability=min(1.0, peak.mass),
                )
                for ordinal, peak in enumerate(peaks, start=1)
            ),
            days_with_peaks=days_with_peaks,
            recurrence_hours=self._recurrence_hours(kinds),
            label=label,
        )

    def _recurrence_hours(self, kinds: Sequence[str]) -> float | None:
        medians = [
            self.tree.recurrences[kind].intervals.p50 / 3600.0
            for kind in kinds
            if kind in self.tree.recurrences and self.tree.recurrences[kind].intervals.p50 > 0
        ]
        return min(medians) if medians else None


def _with_neighbours(
    kinds: Sequence[str], tree: PredictionTree
) -> tuple[tuple[tuple[float, ...], tuple[float, ...] | None, tuple[float, ...] | None], ...]:
    """每个周几的曲线配上真正的前一天与次日曲线（周日接周一），给"逐日有没有峰"用。"""

    found = []
    for weekday in WEEKDAYS:
        curve = merged_marginal(tree, weekday, kinds)
        if curve is None:
            continue
        found.append(
            (curve, merged_marginal(tree, (weekday - 1) % 7, kinds), merged_marginal(tree, (weekday + 1) % 7, kinds))
        )
    return tuple(found)


def _average(curves: Sequence[Sequence[float]]) -> tuple[float, ...]:
    slots = len(curves[0])
    return tuple(sum(curve[slot] for curve in curves) / len(curves) for slot in range(slots))


__all__ = ["MIN_PEAK_MASS", "WEEKDAYS", "Peak", "TreeRhythms", "class_curves", "merged_marginal", "peaks_of"]
