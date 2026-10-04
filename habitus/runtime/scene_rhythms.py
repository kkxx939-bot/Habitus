"""节律口的实现：从预测树算出每个行为概念"典型一天"的那几个机会。

与 ``scene_opportunities`` 是同一座桥的两头，共用 ``merged_marginal`` / ``peaks_of``：

- **机会口**答"这个后件在锚之后的前几次机会是哪几个时刻"——按**具体那一天的周几**取峰，供账本开承诺；
- **节律口**答"这个后件平常一天有几次机会、几点、多大概率"——把七个周几的曲线**逐槽平均**再取峰，
  供基准生成的两个 LLM 触点读（它们写的是不分周几的常识判断，给七条曲线只会让提示词变噪音）。

平均而不是挑某一天：挑"峰最多的那天"会让工作日型的行为按最忙的周三报，写出来的假设对周末不成立；
平均之后周末没做的那几天会把峰压低，峰还在、概率更老实。

``days_with_peaks`` 数的是**逐日**有没有峰（不是平均曲线上的），因为分型问的正是"是不是天天有机会"。
``recurrence_hours`` 取这个概念命中过的那些 kind 里**最短**的复发间隔中位数：概念比 kind 粗，任一 kind
发生它就发生，所以真实间隔不会长于最短那个——这是上界估计（将来 A4 拿它缩放 ``censor_after`` 时按上界更保守；那一条还没做）。
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from types import MappingProxyType

from habitus.prediction.model import MINUTES_PER_DAY, PredictionTree
from habitus.runtime.scene_opportunities import merged_marginal, peaks_of
from habitus.scene.concepts.rhythm import MAX_RHYTHM_PEAKS, Rhythm, RhythmPeak

WEEKDAYS = tuple(range(7))


class TreeRhythms:
    """按 ``RhythmProvider`` 协议实现的节律口。与机会口一样，组合根在**重建树之后**构造它。

    ``kinds`` 是 ``scene.views.kinds.kinds_by_concept`` 的产物（概念 → 它命中过的那几条曲线的键）。
    """

    def __init__(self, tree: PredictionTree, kinds: Mapping[str, Sequence[str]]) -> None:
        if not isinstance(tree, PredictionTree):
            raise TypeError("tree must be a PredictionTree")
        self.tree = tree
        self.kinds = {concept: tuple(tokens) for concept, tokens in kinds.items()}

    def rhythms(self, among: Iterable[str] | None = None) -> Mapping[str, Rhythm]:
        wanted = tuple(self.kinds) if among is None else tuple(among)
        return MappingProxyType({concept: self.rhythm_of(concept) for concept in wanted})

    def rhythm_of(self, concept: str) -> Rhythm:
        kinds = self.kinds.get(concept, ())
        if not kinds:
            return Rhythm(concept=concept)
        curves = [curve for curve in (merged_marginal(self.tree, weekday, kinds) for weekday in WEEKDAYS) if curve is not None]
        if not curves:
            return Rhythm(concept=concept)
        # 七天平均的曲线，"次日"还是它自己：跨午夜的峰接自己的开头即可（每个周几各自的曲线在机会口那边才接真正的次日）
        average = _average(curves)
        days_with_peaks = sum(1 for weekday_curve, preceding, following in _with_neighbours(kinds, self.tree) if peaks_of(weekday_curve, following=following, preceding=preceding))
        peaks = peaks_of(average, following=average, preceding=average)[:MAX_RHYTHM_PEAKS]
        if not peaks or not days_with_peaks:
            return Rhythm(concept=concept, recurrence_hours=self._recurrence_hours(kinds))
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
        found.append((curve, merged_marginal(tree, (weekday - 1) % 7, kinds), merged_marginal(tree, (weekday + 1) % 7, kinds)))
    return tuple(found)


def _average(curves: Sequence[Sequence[float]]) -> tuple[float, ...]:
    slots = len(curves[0])
    return tuple(sum(curve[slot] for curve in curves) / len(curves) for slot in range(slots))


__all__ = ["TreeRhythms", "WEEKDAYS"]
