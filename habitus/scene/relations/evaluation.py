"""一晚的关系怎么量（「怎么量」）：每条关系的读数、两族加权 BH（第 1 道）、按块看稳（第 2 道）与区间够大（第 3 道），
以及状态折叠在子集上要的读数（前向验证、维持检验）。

要量哪些在 ``candidates``；链、层、块这些材料由 ``LaneTests`` 一晚只算一次，这里只读它们。
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping
from dataclasses import replace
from datetime import date
from typing import TYPE_CHECKING

from habitus.foundation.integrity import canonical_digest
from habitus.scene.concepts.calendar_situations import calendar_key
from habitus.scene.concepts.model import ConceptKind
from habitus.scene.relations import longspan, stats
from habitus.scene.relations.calendar import calendar_measure
from habitus.scene.relations.candidates import RelationCandidates
from habitus.scene.relations.conditions import ModeratedMeasure, evaluator, ready_to_test, split
from habitus.scene.relations.engine import (
    LaneNight,
    RelationKey,
    RelationTest,
    Subset,
    Verdict,
    gone,
    large_enough,
)
from habitus.scene.relations.measures import ChainMeasure, Measure, PairedMeasure
from habitus.scene.relations.spans import Pairing, Segment

if TYPE_CHECKING:  # 只给类型检查看：``lane_tests`` 在运行时导入本模块
    from habitus.scene.relations.lane_tests import LaneTests

#: 一条关系当晚的读数：（量法, p, 最小可达 p, 朝上）。
_Reading = tuple[Measure, float, float, bool]


class RelationEvaluator:
    """读数、检验族与子集读数。互相调用（量、种子、要量哪些、上下界、第 2/3 道）都经 ``LaneTests`` 门面，与拆前同一条分派
    （门面上的方法可以按实例替换）。"""

    def __init__(self, tests: LaneTests, candidates: RelationCandidates) -> None:
        self.tests = tests
        self.candidates = candidates

    def night(
        self,
        weights: Mapping[RelationKey, float] | None = None,
        *,
        since: Mapping[RelationKey, date] | None = None,
        outside: frozenset[RelationKey] = frozenset(),
    ) -> LaneNight:
        """当晚的检验族。

        ``since``：这些关系只看那天及以后的数据（失效、前向没复现之后，要在新数据上重新显著才算"又回来了"——
        不然旧数据里那段强关系会让它每晚都"又过第 1 道"）。``outside``：这些关系照算读数、但不进检验族
        （候选、成立的由状态折叠各管各的，成立之后不再进全局 FDR）。
        """

        config = self.tests.config
        weights = weights or {}
        keys = self.tests.keys()
        since = {**self.candidates.proposed, **(since or {})}
        readings, blinded = self._readings(keys, since)
        texts = {key.text: key for key in readings}
        inside = [key for key in readings if key not in outside and key not in blinded]
        family, passed, limits = _families(readings, inside, weights, q=config.thresholds.fdr)
        tests: list[RelationTest] = []
        for text in sorted(texts):
            key = texts[text]
            measure, p_value, minimum, upward = readings[key]
            blocks = self.tests.blocks(key.antecedent)
            base = RelationTest(
                key=key,
                verdict=Verdict.SIGNIFICANT
                if text in passed
                else Verdict.TESTED
                if text in family or (key in outside and measure.units > 0)
                else Verdict.SPARSE,
                antecedents=measure.units,
                blocks=measure.blocks,
                block_days=blocks.days if blocks is not None else 1.0,
                effect=measure.effect(),
                p_value=p_value,
                minimum_p=minimum,
                upward=upward,
                weight=weights.get(key, 1.0),
            )
            if base.verdict is Verdict.SPARSE:
                tests.append(
                    replace(base, more_needed=_more_needed(measure, config.thresholds.fdr / limits.get(text, 1)))
                )
            elif base.verdict is Verdict.SIGNIFICANT:
                tests.append(self.tests.assess(base, measure))
            else:
                tests.append(base)
        return LaneNight(lane=self.tests.line.lane, cutoff=self.tests.cutoff, tests=tuple(tests), family=len(family))

    def _readings(
        self, keys: tuple[RelationKey, ...], since: Mapping[RelationKey, date]
    ) -> tuple[dict[RelationKey, _Reading], set[RelationKey]]:
        """每条关系当晚的读数；判不了的太多、不进检验族的另列（``blinded``）。"""

        config = self.tests.config
        readings: dict[RelationKey, _Reading] = {}
        blinded: set[RelationKey] = set()
        for key in keys:
            measure = self.tests.measure(key, since=since.get(key))
            upper, lower = measure.tails(exact_limit=config.thresholds.exact_strata_limit)
            if self.tests.refined_consequent(key):
                bounded = self._bounded_reading(key, since.get(key), measure)
                if bounded is None:
                    # 判不了的太多：留一条读数、不进检验族（该回去改触点①的区别句）
                    readings[key] = (measure, 1.0, 1.0, upper <= lower)
                    blinded.add(key)
                else:
                    readings[key] = bounded
            elif (
                key.condition
                and isinstance(measure, ModeratedMeasure)
                and not ready_to_test(measure, alpha=config.thresholds.forward_alpha)
            ):
                # 两边全分开也到不了 α：不进检验族（裁定 60 第 9 条，取代固定的前因 ≥ 40）
                readings[key] = (measure, 1.0, 1.0, upper <= lower)
            elif key.segment is Segment.CHAIN and not key.condition:
                # 同一条链只检"更多"（裁定 27 第 1 条）：一件接一件做事时，"接下来做什么"此消彼长——对照是别的事做完之后，
                # 一条强关系（讨论方案 → 修改代码）会让别的前因之后的修改代码都显得"更少"；时长差异也只会造出"更少"。
                # "做完 A 之后更不会接着做 B"在这里多半是别处关系的影子，不是一条独立的发现，对"接下来可能做什么"也没用。
                readings[key] = (measure, upper, measure.minimum_toward(upward=True), True)
            else:
                readings[key] = (measure, stats.two_sided(upper, lower), measure.minimum_p(), upper <= lower)
        return readings, blinded

    def _chain_measure(self, key: RelationKey, *, since: date | None = None, blind_as: int | None = None) -> Measure:
        """不带条件的短跨度关系怎么量：同一条链两道对照成对（``PairedMeasure``），别的段一道。"""

        first = ChainMeasure(self.tests.strata(key, since=since, blind_as=blind_as))
        if key.segment is not Segment.CHAIN:
            return first
        second = ChainMeasure(self.tests.strata(key, since=since, blind_as=blind_as, pairing=Pairing.SAME_SLOT))
        return PairedMeasure(first=first, second=second)

    def _bounded_reading(self, key: RelationKey, since: date | None, measure: Measure) -> _Reading | None:
        """细分概念当后果的第 1 道：后果"判不了"当 0、当 1 各算一遍，两遍都显著、方向一致才算过（p 取两遍里大的）。
        判不了的窗口占比超过 ``max_blind_share`` 的返回 None（不进检验族）。窗口里"有一条命中就记 1、判不了就丢"会把
        「A → 返工」读成「A → 修改代码」又量一遍（第四轮评审人工智能 3：窗口里两条修改代码时读出 97% "返工"）。"""

        thresholds = self.tests.config.thresholds
        low = self._chain_measure(key, since=since, blind_as=0)
        high = self._chain_measure(key, since=since, blind_as=1)
        if low.units and 1.0 - measure.units / low.units > thresholds.max_blind_share:
            return None
        found = []
        for bound in (low, high):
            upper, lower = bound.tails(exact_limit=thresholds.exact_strata_limit)
            if key.segment is Segment.CHAIN:
                # 同一条链只检"更多"（裁定 27 第 1 条），细分概念当后果也一样
                found.append((upper, bound.minimum_toward(upward=True), True))
            else:
                found.append((stats.two_sided(upper, lower), bound.minimum_p(), upper <= lower))
        (p_low, min_low, up_low), (p_high, min_high, up_high) = found
        if up_low != up_high:
            return measure, 1.0, max(min_low, min_high), up_low  # 两头方向都不一致：说不了
        return measure, max(p_low, p_high), max(min_low, min_high), up_low

    def refined_consequent(self, key: RelationKey) -> bool:
        """后果是细分概念的短跨度、不带条件的关系：它的"判不了"来自映射器，要做上下界检验。"""

        item = self.tests.concepts.get(key.consequent)
        return item is not None and item.kind is ConceptKind.REFINEMENT and not key.condition and not key.segment.long

    def evaluate(
        self, key: RelationKey, *, upward: bool, since: date | None = None, last_blocks: int | None = None, purpose: str
    ) -> Subset:
        """子集上的读数：``since`` 之后的新数据（前向验证、维持检验），或最近 ``last_blocks`` 个块。p 是朝 ``upward`` 的单侧。"""

        thresholds = self.tests.config.thresholds
        measure = self.tests.measure(key).subset(since=since, last_blocks=last_blocks)
        upper, lower = measure.tails(exact_limit=thresholds.exact_strata_limit)
        spread, relative = measure.interval(
            level=thresholds.interval_level,
            rounds=thresholds.bootstrap_rounds,
            seed=self.tests.seed(key, purpose),
        )
        return Subset(
            antecedents=measure.units,
            blocks=measure.blocks,
            effect=measure.effect(),
            p_value=upper if upward else lower,
            interval=spread,
            relative_interval=relative,
            minimum_p=measure.minimum_toward(upward=upward),
        )

    def measure(self, key: RelationKey, *, since: date | None = None) -> Measure:
        """这个关系怎么量：长跨度按状态（``longspan``），带条件的把层按条件分两组（``conditions``），其余按层。"""

        line = self.tests.line
        measure: Measure
        rule = calendar_key(self.tests.concepts, key.antecedent) if key.segment is Segment.SITUATION else None
        if rule is not None:
            # 日历情境按维度取对照（``relations.calendar``，裁定 59 第 7 条）；模型写的情境仍和前后紧挨着的平常日子比
            measure = calendar_measure(line, key.consequent, *rule)
        elif key.segment.long:
            measure = longspan.measure_of(line, key.consequent, self.tests.episodes(key.antecedent, key.segment))
        elif key.condition:
            window = self.tests.config.transition_window_slots
            judges = [evaluator(line, condition, window=window) for condition in key.conditions]
            # 带条件的只按第 1 道对照分组比：两组的钟点偏差一样，分组之间抵掉；几个条件合起来时每个条件各比一次——"全成立"对
            # "其余成立、它不成立"，按最难过的那组读（裁定 63 第 1 条）
            measure = split(ChainMeasure(self.tests.strata(key)), judges)
        else:
            measure = self._chain_measure(key)
        return measure if since is None else measure.subset(since=since)

    def seed(self, key: RelationKey, purpose: str) -> int:
        """重抽样的种子：（关系身份, 第几晚, 用途）的摘要——同一晚重跑结果逐字相同。"""

        night = self.tests.cutoff.isoformat()
        return int(canonical_digest({"key": key.text, "night": night, "purpose": purpose})[:16], 16)

    def assess(
        self, test: RelationTest, measure: Measure | None = None, *, previously_large: bool | None = None
    ) -> RelationTest:
        """第 2、3 道与几个只报告的读数，按 ``test.upward`` 那个方向判（候选在某一晚没显著时，状态折叠也要用它）。
        ``previously_large`` 是上一晚判的"影响够大"：给了就带滞回——进用近端过线，退用远端也掉线（``gone``），中间维持上一晚
        （裁定 63 第 3 条：成立关系每晚判，线附近的别一晚进一晚出）。"""

        config = self.tests.config
        if measure is None:
            measure = self.tests.measure(test.key)
        spread, relative = measure.interval(
            level=config.thresholds.interval_level,
            rounds=config.thresholds.bootstrap_rounds,
            seed=self.tests.seed(test.key, "interval"),
        )
        heterogeneity = measure.heterogeneity()
        leave_one_out = measure.leave_one_out(upward=test.upward)
        effect = measure.effect()
        consequent_count = len(self.tests.line.outcomes(test.key.consequent))
        excess = effect.antecedents * effect.difference
        return replace(
            test,
            heterogeneity_p=heterogeneity,
            leave_one_out=leave_one_out,
            stable=leave_one_out and (heterogeneity is None or heterogeneity >= config.thresholds.heterogeneity_alpha),
            interval=spread,
            relative_interval=relative,
            excess_share=excess / consequent_count if consequent_count else None,
            same_event_share=measure.same_event_share(),
            large_enough=large_enough(spread, relative, upward=test.upward, config=config)
            or (previously_large is True and not gone(spread, relative, upward=test.upward, config=config)),
        )


def _families(
    readings: Mapping[RelationKey, _Reading],
    inside: list[RelationKey],
    weights: Mapping[RelationKey, float],
    *,
    q: float,
) -> tuple[set[str], set[str], dict[str, int]]:
    """长跨度自成一族（裁定 27 第 4 条，S4）：它与短跨度问的是两类问题，同一个 q 各控各的；混在一族里，
    几十条长跨度要和几百条短跨度分摊误发现率，门槛被压到 q / 300 上下，永远进不了族。返回（族, 过了第 1 道的, 各自族的大小）。"""

    family: set[str] = set()
    passed: set[str] = set()
    limits: dict[str, int] = {}
    for members in ([key for key in inside if not key.segment.long], [key for key in inside if key.segment.long]):
        group, group_passed = stats.tarone_bh(
            {key.text: readings[key][1] for key in members},
            {key.text: readings[key][2] for key in members},
            {key.text: weights.get(key, 1.0) for key in members},
            q=q,
        )
        family |= group
        passed |= group_passed
        limits.update({key.text: max(1, len(group)) for key in members})
    return family, passed, limits


def _more_needed(measure: Measure, threshold: float) -> int | None:
    """照现在的样子，还要几个全朝一个方向的单位才可能把最小可达 p 压到 ``threshold`` 以下（粗估）。"""

    if isinstance(measure, PairedMeasure):
        # 两道对照都要过：按更难过的那道估
        needed = [_more_needed(part, threshold) for part in measure.parts]
        return None if any(item is None for item in needed) else max(item for item in needed if item is not None)
    if not isinstance(measure, ChainMeasure):
        return None
    informative = [item.null_probability for item in measure.strata if 0.0 < item.null_probability < 1.0]
    if not informative:
        return None
    minimum = measure.minimum_p()
    if minimum <= threshold:
        return 0
    typical = statistics.median(min(p, 1.0 - p) for p in informative)
    if typical <= 0.0:
        return None
    return math.ceil(math.log(threshold / minimum) / math.log(max(typical, 1e-12)))


__all__ = ["RelationEvaluator"]
