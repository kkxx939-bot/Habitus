"""安慰剂（文档里叫"假配对"）的读数：同一套口径在**随手配的前件**上读出什么样的数。

这是七c ⑭ 那把尺子的刻度盘。安慰剂假设（前件换成一条算法挑的、常识没选过的行为）和真假设一样开账、一样结算、
一样出读数。

**2026-09-30 裁定六：不按标签判真假，按数据判。**安慰剂只是"随手配的"，不保证真的无关——这个人可能就是打球前爱喝咖啡，
那是有价值的个人规律。所以这里**不再下"误报率 x%、真假设高出 y%"那种判词**（评审 C-2 实测：两边真实显著率一样时它仍有
一半概率印"高出"）。这里只给**参照分布**：同一次机会那一档里，整批安慰剂的读数落在哪（p50 / p90），真假设里有几条超过
p90——碰巧该有约十分之一。三条都过才算"有关"（样本按独立天数够下限 · 超过分位线 · 双向验证成立），下限与分位线等
重跑后看数再定；过了的安慰剂不算误报，升为"个人规律候选"。

**未校准的不进来**（裁定五）：无节律型的读数没有平时概率可比，两边都必然"显著"，量不出东西。

样本不够时不给数：一条安慰剂都没攒够的时候说"误报率 0%"是假的确定。
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from habitus.scene.hypotheses.model import Hypothesis
from habitus.scene.views.relations import RelationReading

#: 一档里至少要有这么多条攒够样本的安慰剂才给参照分位。少于它只报"还没攒够"。**待定值**。
MIN_PLACEBOS = 5
#: 参照分位线：真假设的读数要超过整批安慰剂的第几分位才算"差得够"。**待定值**（裁定六，重跑后定）。
REFERENCE_QUANTILE = 0.9


@dataclass(frozen=True)
class StepReference:
    """同一次机会那一档的参照：安慰剂读数的分位、真假设里超过分位线的条数。"""

    step: int | None
    placebo_tested: int
    real_tested: int
    placebo_p50: float | None
    placebo_reference: float | None
    real_above_reference: int

    @property
    def ready(self) -> bool:
        return self.placebo_tested >= MIN_PLACEBOS

    def render(self) -> str:
        label = "无节律型" if self.step is None else f"第 {self.step} 次机会"
        if not self.ready or self.placebo_p50 is None or self.placebo_reference is None:
            return f"{label}：安慰剂 {self.placebo_tested} 条攒够样本（要 {MIN_PLACEBOS} 条才给参照）"
        expected = self.real_tested * (1.0 - REFERENCE_QUANTILE)
        mark = f"p{round(REFERENCE_QUANTILE * 100)}"
        return (
            f"{label}：安慰剂 {self.placebo_tested} 条，读数 p50 {self.placebo_p50 * 100:+.0f}pp、{mark} {self.placebo_reference * 100:+.0f}pp"
            f"；真假设 {self.real_tested} 条里超过 {mark} 的 {self.real_above_reference} 条（碰巧该约 {expected:.1f} 条）"
        )


@dataclass(frozen=True)
class PlaceboReport:
    """真假设与安慰剂各自攒够了几条、其中几条显著，加按机会分档的参照分布。

    ``real_significant`` / ``placebo_significant`` 只是计数，**不再拿来下判词**（裁定六）。
    """

    real_tested: int
    real_significant: int
    placebo_tested: int
    placebo_significant: int
    references: tuple[StepReference, ...] = ()

    @property
    def ready(self) -> bool:
        return self.placebo_tested >= MIN_PLACEBOS

    @property
    def false_alarm_rate(self) -> float | None:
        """安慰剂里显著的占比。**只是一个计数比**——安慰剂不保证真的无关，所以它不是"误报率"（裁定六）。没攒够就是 None。"""

        return self.placebo_significant / self.placebo_tested if self.ready else None

    def render(self) -> str:
        head = f"安慰剂参照：真假设 {self.real_significant}/{self.real_tested} 条显著，安慰剂 {self.placebo_significant}/{self.placebo_tested} 条显著（判词待裁定六的数值，不下结论）"
        if not self.references:
            return head
        return head + "；" + "；".join(item.render() for item in self.references)


def placebo_report(readings: Iterable[RelationReading], hypotheses: Mapping[str, Hypothesis]) -> PlaceboReport:
    """按来源把读数分成两边数，再按第几次机会分档给参照分位。假设集里没有的读数跳过（作废过的旧账）；未校准的跳过。"""

    counts = {True: [0, 0], False: [0, 0]}  # is_placebo -> [攒够的, 显著的]
    points: dict[int | None, dict[bool, list[float]]] = {}
    for reading in readings:
        hypothesis = hypotheses.get(reading.hypothesis_identity)
        if hypothesis is None or reading.uncalibrated or not reading.strength.sufficient:
            continue
        placebo = hypothesis.source.origin.is_placebo
        slot = counts[placebo]
        slot[0] += 1
        slot[1] += int(reading.significant)
        if reading.strength.interval is not None:
            points.setdefault(hypothesis.consequent_peak, {True: [], False: []})[placebo].append(reading.strength.interval.point)
    references = tuple(_reference(step, table[True], table[False]) for step, table in sorted(points.items(), key=lambda item: (item[0] is None, item[0] or 0)))
    return PlaceboReport(
        real_tested=counts[False][0],
        real_significant=counts[False][1],
        placebo_tested=counts[True][0],
        placebo_significant=counts[True][1],
        references=references,
    )


def _reference(step: int | None, placebo: list[float], real: list[float]) -> StepReference:
    if len(placebo) < MIN_PLACEBOS:
        return StepReference(step, len(placebo), len(real), None, None, 0)
    p50 = _quantile(placebo, 0.5)
    line = _quantile(placebo, REFERENCE_QUANTILE)
    return StepReference(step, len(placebo), len(real), p50, line, sum(1 for value in real if value > line))


def _quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = q * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


__all__ = ["MIN_PLACEBOS", "REFERENCE_QUANTILE", "PlaceboReport", "StepReference", "placebo_report"]
