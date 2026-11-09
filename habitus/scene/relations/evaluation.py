"""一晚的关系怎么量（「怎么量」）：每条关系的读数、三族加权 BH（第 1 道）、按块看稳（第 2 道）与区间够大（第 3 道），
以及状态折叠在子集上要的读数（前向验证、维持检验）。

要量哪些在 ``candidates``；链、层、块这些材料由 ``LaneTests`` 一晚只算一次，这里只读它们。
"""

from __future__ import annotations

import math
import statistics
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import date
from typing import TYPE_CHECKING

from habitus.foundation.integrity import canonical_digest
from habitus.scene.concepts.calendar_situations import calendar_key
from habitus.scene.concepts.model import ConceptKind
from habitus.scene.relations import longspan, stats
from habitus.scene.relations.calendar import calendar_measure
from habitus.scene.relations.candidates import RelationCandidates
from habitus.scene.relations.conditions import ModeratedMeasure, ready_to_test, split
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
        keep: frozenset[RelationKey] = frozenset(),
    ) -> LaneNight:
        """当晚的检验族（三族、带条件的按父关系或按（前因, 后果, 段）一组进族，见 ``_families``）。

        ``since``：这些关系只看那天及以后的数据（失效、前向没复现之后，要在新数据上重新显著才算"又回来了"——
        不然旧数据里那段强关系会让它每晚都"又过第 1 道"）。``outside``：这些关系照算读数、但不进检验族
        （候选、成立的由状态折叠各管各的，成立之后不再进全局 FDR）。``keep``：这些关系每晚都要完整读数（跨夜带着的），
        不走"整组到不了开测门就不逐条量"的捷径。开测门第 1 道整组到不了的带条件关系（``LaneTests.sides_reachable``）不逐条量，
        只按段计数（``LaneNight.skipped``），归一照计（``_families``）。给了权重时同一份读数再不加权判一次第 1 道
        （``plain_significant``），状态折叠据它打"靠先验"。"""

        config = self.tests.config
        q = config.thresholds.fdr
        weights = weights or {}
        keys = self.tests.keys()
        since = {**self.candidates.proposed, **(since or {})}
        readings, blinded, skipped = self._readings(keys, since, keep | outside)
        inside = [key for key in readings if key not in outside and key not in blinded]
        # 带条件的权重恒为 1 时（先验只给不带条件的），带条件那一族的归一系数就是 1，跳过的键既不改归一、也进不了族
        weighted = any(key.condition for key in weights)
        try:
            family, passed, limits, parts = _families(
                readings, inside, weights, q=q, outside=outside, skipped=skipped if weighted else ()
            )
        except _SkippedMayEnter:
            more, more_blinded, _none = self._readings(skipped, since, frozenset(skipped))
            readings.update(more)
            blinded |= more_blinded
            skipped = ()
            inside = [key for key in readings if key not in outside and key not in blinded]
            family, passed, limits, parts = _families(readings, inside, weights, q=q, outside=outside)
        texts = {key.text: key for key in readings}
        plain = passed if not weights else _families(readings, inside, {}, q=q, outside=outside)[1]
        ordered = [texts[text] for text in sorted(texts)]
        tests = [
            self._test(
                key, readings[key], weights, _verdict(key, readings, passed, family, outside), limits.get(key.text, 1)
            )
            for key in ordered
        ]
        return LaneNight(
            lane=self.tests.line.lane,
            cutoff=self.tests.cutoff,
            tests=tuple(tests),
            family=len(family),
            skipped=dict(Counter(key.segment.value for key in skipped)),
            plain_significant=frozenset(texts[text] for text in plain),
            breakdown={**parts, "measured": len(readings), "outside": len(outside), "blinded": len(blinded)},
        )

    def _test(
        self, key: RelationKey, reading: _Reading, weights: Mapping[RelationKey, float], verdict: Verdict, limit: int
    ) -> RelationTest:
        """一条关系当晚的检验：稀疏的补"还要几次"，过了第 1 道的接着判第 2、3 道。"""

        measure, p_value, minimum, upward = reading
        blocks = self.tests.blocks(key.antecedent)
        base = RelationTest(
            key=key,
            verdict=verdict,
            antecedents=measure.units,
            blocks=measure.blocks,
            block_days=blocks.days if blocks is not None else 1.0,
            effect=measure.effect(),
            p_value=p_value,
            minimum_p=minimum,
            upward=upward,
            weight=weights.get(key, 1.0),
        )
        if verdict is Verdict.SPARSE:
            return replace(base, more_needed=_more_needed(measure, self.tests.config.thresholds.fdr / limit))
        if verdict is Verdict.SIGNIFICANT:
            return self.tests.assess(base, measure)
        return base

    def _readings(
        self, keys: Sequence[RelationKey], since: Mapping[RelationKey, date], keep: frozenset[RelationKey]
    ) -> tuple[dict[RelationKey, _Reading], set[RelationKey], tuple[RelationKey, ...]]:
        """每条关系当晚的读数；判不了的太多、不进检验族的另列（``blinded``）；开测门第 1 道整组到不了、不逐条量的另列
        （``skipped``，``keep`` 里的不跳）。"""

        config = self.tests.config
        readings: dict[RelationKey, _Reading] = {}
        blinded: set[RelationKey] = set()
        skipped: list[RelationKey] = []
        for key in keys:
            if key.condition and key not in keep and not self.tests.sides_reachable(key):
                skipped.append(key)
                continue
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
                and isinstance(measure, ModeratedMeasure | PairedMeasure)
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
        return readings, blinded, tuple(skipped)

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
            judges = [self.tests.condition_judge(key.antecedent, condition) for condition in key.conditions]
            # 每个条件各比一次、按最难过的那组读（裁定 63 第 1 条）；同一条链两道对照各分一次组、两道都显著才算（裁定 95 第 4 条）
            measure = split(ChainMeasure(self.tests.strata(key)), judges)
            if key.segment is Segment.CHAIN:
                slot = split(ChainMeasure(self.tests.strata(key, pairing=Pairing.SAME_SLOT)), judges)
                measure = PairedMeasure(first=measure, second=slot)
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


def _verdict(
    key: RelationKey,
    readings: Mapping[RelationKey, _Reading],
    passed: set[str],
    family: set[str],
    outside: frozenset[RelationKey],
) -> Verdict:
    if key.text in passed:
        return Verdict.SIGNIFICANT
    if key.text in family or (key in outside and readings[key][0].units > 0):
        return Verdict.TESTED
    return Verdict.SPARSE


class _SkippedMayEnter(Exception):
    """没逐条量的带条件键在带条件那一族归一之后可能进族：要逐条量回来再判。"""


#: 只给观测的分解里固定有的键：没有就写 0，看 trace 的人分得清"是 0"还是"没记"。
_BREAKDOWN_KEYS = (
    *(f"{family}_{part}" for family in ("plain", "long", "conditioned", "groups") for part in ("family", "passed")),
    *(f"entry_{path}" for path in ("parent", "carried", "group")),
)


def _families(
    readings: Mapping[RelationKey, _Reading],
    inside: list[RelationKey],
    weights: Mapping[RelationKey, float],
    *,
    q: float,
    outside: frozenset[RelationKey],
    skipped: Sequence[RelationKey] = (),
) -> tuple[set[str], set[str], dict[str, int], dict[str, int]]:
    """第 1 道分三族，同一个 q 各控各的（裁定 95 第 1 条）：不带条件的短跨度、长跨度（裁定 27 第 4 条：两类问题，混在一族里
    几十条长跨度要和几百条短跨度分摊误发现率）、带条件的短跨度。带条件的不和不带条件的同族：几十万条带条件的会把族撑大、门槛压低，
    不带条件的一条都进不了族（v0.12 跑数 13 晚 0 条）。

    带条件那一族按层级放人（裁定 95 第 2 条，Yekutieli 2008），两条路任一：至少一条直接父关系（少一个条件的键）当晚在不带条件
    那一族过了第 1 道、或跨夜带着为候选 / 成立（``outside``）；或它所在的（前因, 后果, 段）这一组过了组间检验（``_pairs_passed``，
    只在条件下成立、整体看不出的关系走这条）。其余这一晚不进任何族。``skipped``：没逐条量的带条件键（开测门第 1 道
    整组到不了，读数必是"到不了"），放得进来的那些按各自权重计入带条件那一族的归一；归一之后可能进族时抛 ``_SkippedMayEnter``。
    返回（族, 过了第 1 道的, 各自族的大小, 只给观测的分解：各族大小与过第 1 道数、各条进族路的条数）。"""

    family: set[str] = set()
    passed: set[str] = set()
    limits, breakdown = dict[str, int](), dict.fromkeys(_BREAKDOWN_KEYS, 0)

    def test(name: str, members: list[RelationKey], untestable: Sequence[float] = ()) -> frozenset[str]:
        group, group_passed = stats.tarone_bh(
            {key.text: readings[key][1] for key in members},
            {key.text: readings[key][2] for key in members},
            {key.text: weights.get(key, 1.0) for key in members},
            q=q,
            untestable=untestable,
        )
        family.update(group)
        passed.update(group_passed)
        limits.update({key.text: max(1, len(group)) for key in members})
        breakdown[f"{name}_family"], breakdown[f"{name}_passed"] = len(group), len(group_passed)
        return group_passed

    parents_passed = test("plain", [key for key in inside if not key.segment.long and not key.condition])
    test("long", [key for key in inside if key.segment.long])

    def by_parent(key: RelationKey) -> bool:
        return any(parent.text in parents_passed or parent in outside for parent in key.parents())

    pairs_passed = _pairs_passed(
        [key for key in inside if not key.segment.long and key.condition and not by_parent(key)],
        readings,
        breakdown,
        q=q,
    )

    def admitted(key: RelationKey) -> bool:
        return by_parent(key) or _pair_of(key) in pairs_passed

    moderated = [key for key in inside if not key.segment.long and key.condition and admitted(key)]
    untestable = [weights.get(key, 1.0) for key in skipped if admitted(key)]
    if untestable and stats.untestable_may_enter([weights.get(key, 1.0) for key in moderated], untestable, q=q):
        raise _SkippedMayEnter
    test("conditioned", moderated, untestable)
    breakdown.update(Counter(f"entry_{_entry_path(key, parents_passed, outside)}" for key in moderated))
    return family, passed, limits, breakdown


def _entry_path(key: RelationKey, parents_passed: frozenset[str], outside: frozenset[RelationKey]) -> str:
    """带条件的键从哪条路进的族（只给观测）：父关系当晚过了第 1 道、父关系跨夜带着（候选 / 成立），否则是按组。"""

    if any(parent.text in parents_passed for parent in key.parents()):
        return "parent"
    if any(parent in outside for parent in key.parents()):
        return "carried"
    return "group"


def _pair_of(key: RelationKey) -> str:
    return f"{key.antecedent}|{key.consequent}|{key.segment.value}"


def _pairs_passed(
    keys: Sequence[RelationKey],
    readings: Mapping[RelationKey, _Reading],
    breakdown: dict[str, int],
    *,
    q: float,
) -> frozenset[str]:
    """父关系没放行的带条件键的第二条进族路（裁定 95 第 2 条）：按（前因, 后果, 段）分组，组的 p 是组内键的 Simes 合并、
    组的最小可达 p 是同一公式代入各键的最小可达 p（组 p 的下界，供 Tarone 修剪）；组与组之间做一次 Tarone + BH（同一个 q）。
    开测门到不了的键（最小可达 p = 1）不进组：它们不可能显著，算进来只会抬高门槛。只在条件下成立、整体看不出的关系从这里进族；
    定点作息巧合已由同一条链的第 2 道对照扣掉（裁定 95 第 4 条），不会从这里回来。"""

    groups: dict[str, list[RelationKey]] = {}
    for key in keys:
        if readings[key][2] < 1.0:
            groups.setdefault(_pair_of(key), []).append(key)
    if not groups:
        return frozenset()
    p_values = {name: stats.simes([readings[key][1] for key in members]) for name, members in groups.items()}
    minimum = {name: stats.simes([readings[key][2] for key in members]) for name, members in groups.items()}
    family, passed = stats.tarone_bh(p_values, minimum, {}, q=q)
    breakdown["groups_family"], breakdown["groups_passed"] = len(family), len(passed)
    return passed


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
